"""
train.py — Training loop for the multilayer-diaphragm PINN.

Physics:  pure energy minimisation (Föppl–von Kármán, variational form)
Data:     optional COMSOL anchor in the first WARMUP_STEPS steps only

Run
---
    python train.py                          # full 50 k steps
    python train.py --no-warmup             # λ_anchor=0 from step 0 (sanity check 1)
    python train.py --steps 5000            # quick smoke test
    python train.py --checkpoint my.pt      # resume from checkpoint
"""

import argparse
import os
import math
import glob
import time
import random

import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lamination  import compute_laminate_stiffness
from model       import DiaphragmPINN, InputNormaliser
from energy      import compute_energy

# ── hyper-parameters ─────────────────────────────────────────────────────────
TOTAL_STEPS    = 20_000
WARMUP_STEPS   = 2_000
LR_INIT        = 1e-3
LR_FINAL       = 1e-4

GEO_BATCH      = 64      # geometry samples per step
P_BATCH        = 8       # pressure samples per geometry
N_QUAD         = 32      # GL quadrature nodes (32 nodes sufficient for smooth profiles)

KAPPA_INIT     = 1e4
KAPPA_MAX      = 1e10
KAPPA_RAMP     = 2_000   # steps between ×10 ramps

CKPT_INTERVAL  = 4_000
LOG_INTERVAL   = 200

# geometry training ranges (SI units)
GEO_RANGES = {
    "t1": (0.8e-6,  1.2e-6),
    "t2": (0.15e-6, 0.25e-6),
    "t3": (3.0e-6,  5.0e-6),
    "a":  (280e-6,  320e-6),
    "ag": (4.0e-6,  6.0e-6),
}
P_RANGE = (10.0, 20e3)   # Pa  (log-uniform; lower bound avoids log(0))

COMSOL_DIR = "COMSOL_Deflection_Outputs"
CKPT_DIR   = "checkpoints"


# ── helpers ───────────────────────────────────────────────────────────────────

def sample_geometries(n, device, dtype):
    """Uniform samples from the 5-D geometry box."""
    def _u(lo, hi):
        return lo + (hi - lo) * torch.rand(n, device=device, dtype=dtype)
    t1 = _u(*GEO_RANGES["t1"])
    t2 = _u(*GEO_RANGES["t2"])
    t3 = _u(*GEO_RANGES["t3"])
    a  = _u(*GEO_RANGES["a"])
    ag = _u(*GEO_RANGES["ag"])
    return t1, t2, t3, a, ag


def sample_pressures_log(n, device, dtype):
    """Log-uniform samples in [P_lo, P_hi]."""
    lo, hi = math.log(P_RANGE[0]), math.log(P_RANGE[1])
    return torch.exp(lo + (hi - lo) * torch.rand(n, device=device, dtype=dtype))


def expand_geo_x_pressure(t1, t2, t3, a, ag, P):
    """
    Cross-product: replicate each geometry across all pressures.
    (G,) × (Np,) → (G*Np,)
    """
    G, Np = t1.shape[0], P.shape[0]

    def _rep(v):
        return v.unsqueeze(1).expand(G, Np).reshape(G * Np)

    return (_rep(t1), _rep(t2), _rep(t3), _rep(a), _rep(ag),
            P.unsqueeze(0).expand(G, Np).reshape(G * Np))


def lr_schedule(step, total, warmup, lr_init, lr_final):
    """Cosine decay from lr_init to lr_final after warmup."""
    if step < warmup:
        return lr_init
    progress = (step - warmup) / max(total - warmup, 1)
    return lr_final + 0.5 * (lr_init - lr_final) * (1 + math.cos(math.pi * progress))


def lambda_anchor(step, warmup):
    """Linearly anneal from 1 → 0 over warmup steps."""
    if step >= warmup:
        return 0.0
    return 1.0 - step / warmup


# ── COMSOL warmup data ────────────────────────────────────────────────────────

def load_comsol_warmup(device, dtype, max_files=200):
    """
    Load a subset of COMSOL files for the warmup anchor.
    Returns list of dicts with keys: t1,t2,t3,a,ag,P,r,w_comsol
    Only uses pressures in [0,20] kPa and files where geometry is within
    2× the training box (broader to help warmup).
    """
    files = sorted(glob.glob(os.path.join(COMSOL_DIR, "*.csv")))
    random.shuffle(files)
    records = []
    for fpath in files[:max_files]:
        fname = os.path.basename(fpath)
        try:
            # parse geometry from filename
            parts = fname.replace(".csv", "").split("_")
            geo = {}
            for p in parts[1:]:
                for key in ["rad", "t1", "t2", "t3", "ag"]:
                    if p.startswith(key):
                        geo[key] = float(p[len(key):])
                        break
            a_  = geo["rad"] * 1e-6
            t1_ = geo["t1"]  * 1e-6
            t2_ = geo["t2"]  * 1e-6
            t3_ = geo["t3"]  * 1e-6
            ag_ = geo["ag"]  * 1e-6

            df = pd.read_csv(fpath)
            r_um = df["r_um"].values

            # iterate over pressure columns
            for col in df.columns[1:]:
                P_pa = float(col.split("w_")[1].replace("Pa_um", ""))
                if P_pa < 10.0:
                    continue
                w_um = df[col].values
                # clip COMSOL touch values are already at -ag
                records.append({
                    "t1": t1_, "t2": t2_, "t3": t3_,
                    "a": a_, "ag": ag_, "P": P_pa,
                    "r": r_um * 1e-6,
                    "w": w_um * 1e-6,
                })
        except Exception:
            continue
    print(f"[warmup] loaded {len(records)} COMSOL pressure profiles")
    return records


def comsol_anchor_loss(model, normaliser, records, n_samples=4, device=None, dtype=None):
    """
    MSE between network w(r) and COMSOL w(r) on a random subset of records.
    Only used during warmup.
    """
    chosen = random.sample(records, min(n_samples, len(records)))
    total_mse = torch.tensor(0.0, device=device, dtype=dtype)
    count = 0
    for rec in chosen:
        r   = torch.tensor(rec["r"],  dtype=dtype, device=device)
        w_c = torch.tensor(rec["w"],  dtype=dtype, device=device)
        a   = torch.tensor(rec["a"],  dtype=dtype, device=device)
        ag  = torch.tensor(rec["ag"], dtype=dtype, device=device)
        t1  = torch.tensor(rec["t1"], dtype=dtype, device=device)
        t2  = torch.tensor(rec["t2"], dtype=dtype, device=device)
        t3  = torch.tensor(rec["t3"], dtype=dtype, device=device)
        P   = torch.tensor(rec["P"],  dtype=dtype, device=device)

        xi  = r / a
        n   = xi.shape[0]

        # build input — xi does NOT need grad here (anchor loss, not energy)
        x_in = normaliser.normalise(
            xi,
            P.expand(n),
            t1.expand(n), t2.expand(n), t3.expand(n),
            a.expand(n),  ag.expand(n),
        )
        w_pred, _ = model(x_in)
        w_pred = w_pred.squeeze(-1)

        total_mse = total_mse + torch.mean((w_pred - w_c)**2)
        count += 1

    return total_mse / max(count, 1)


# ── training loop ─────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    print(f"Device: {device}  |  dtype: {dtype}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    model       = DiaphragmPINN(n_hidden=5, hidden_dim=128).to(device).to(dtype)
    normaliser  = InputNormaliser()
    optimizer   = optim.Adam(model.parameters(), lr=LR_INIT)

    start_step = 0
    kappa      = KAPPA_INIT
    loss_history = []

    # ── resume from checkpoint ────────────────────────────────────────────────
    if args.checkpoint and os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step   = ckpt["step"]
        kappa        = ckpt["kappa"]
        loss_history = ckpt.get("loss_history", [])
        print(f"Resumed from {args.checkpoint} at step {start_step}")

    # ── load COMSOL warmup data ───────────────────────────────────────────────
    comsol_records = []
    use_warmup = (not args.no_warmup) and (WARMUP_STEPS > 0)
    if use_warmup:
        comsol_records = load_comsol_warmup(device, dtype, max_files=300)

    warmup_steps = WARMUP_STEPS if use_warmup else 0

    t0 = time.time()

    for step in range(start_step, args.steps):

        # ── learning rate update ───────────────────────────────────────────────
        lr = lr_schedule(step, args.steps, warmup_steps, LR_INIT, LR_FINAL)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── kappa ramp ────────────────────────────────────────────────────────
        if step > 0 and step % KAPPA_RAMP == 0:
            kappa = min(kappa * 10.0, KAPPA_MAX)

        # ── sample batch ──────────────────────────────────────────────────────
        t1g, t2g, t3g, ag_, agg = sample_geometries(GEO_BATCH, device, dtype)
        Pg = sample_pressures_log(P_BATCH, device, dtype)

        t1b, t2b, t3b, ab, agb, Pb = expand_geo_x_pressure(
            t1g, t2g, t3g, ag_, agg, Pg
        )

        # ── laminate stiffness ────────────────────────────────────────────────
        D_star, A11, A12, _ = compute_laminate_stiffness(t1b, t2b, t3b)

        # ── physics loss (energy functional) ─────────────────────────────────
        optimizer.zero_grad()

        L_energy = compute_energy(
            model, Pb, t1b, t2b, t3b, ab, agb,
            D_star, A11, A12,
            normaliser, n_quad=N_QUAD, kappa=kappa,
        )

        # ── optional warmup anchor ────────────────────────────────────────────
        lam = lambda_anchor(step, warmup_steps)
        if lam > 0.0 and comsol_records:
            L_data = comsol_anchor_loss(
                model, normaliser, comsol_records,
                n_samples=4, device=device, dtype=dtype,
            )
            L_total = L_energy + lam * L_data
        else:
            L_total = L_energy

        L_total.backward()

        # gradient clip to prevent early instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        loss_val = L_energy.item()
        loss_history.append(loss_val)

        # ── logging ───────────────────────────────────────────────────────────
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | L_energy={loss_val:.4e} | "
                  f"kappa={kappa:.0e} | lam={lam:.3f} | "
                  f"lr={lr:.2e} | {elapsed:.1f}s")

        # ── checkpoint ────────────────────────────────────────────────────────
        if step > 0 and step % CKPT_INTERVAL == 0:
            path = os.path.join(CKPT_DIR, f"ckpt_step{step:06d}.pt")
            torch.save({
                "step":         step,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "kappa":        kappa,
                "loss_history": loss_history,
            }, path)
            print(f"  → saved {path}")

    # ── final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(CKPT_DIR, "ckpt_final.pt")
    torch.save({
        "step":         args.steps,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "kappa":        kappa,
        "loss_history": loss_history,
    }, final_path)
    print(f"Training complete. Final checkpoint: {final_path}")

    # ── loss curve ────────────────────────────────────────────────────────────
    _plot_loss(loss_history, args.steps)


def _plot_loss(history, total_steps):
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = np.arange(len(history))
    # log-scale; shift any non-positive values
    vals = np.array(history)
    pos  = vals[vals > 0]
    if pos.size > 0:
        vals = np.clip(vals, pos.min() * 0.1, None)
    ax.semilogy(steps, vals, linewidth=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Π (energy loss)")
    ax.set_title("Physics energy loss vs. training step")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()
    print("Loss curve saved to loss_curve.png")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",      type=int,  default=TOTAL_STEPS)
    parser.add_argument("--no-warmup",  action="store_true",
                        help="λ_anchor=0 from step 0 (sanity check: pure physics)")
    parser.add_argument("--checkpoint", type=str,  default=None,
                        help="Path to resume checkpoint")
    args = parser.parse_args()
    train(args)
