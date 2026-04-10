"""
validate.py — Post-training validation against COMSOL deflection data.

Usage
-----
    python validate.py --checkpoint checkpoints/ckpt_final.pt
    python validate.py --checkpoint checkpoints/ckpt_final.pt --max-files 50

What this does
--------------
1. Loads COMSOL CSV files (all or a subset).
2. For each file × pressure column, runs the network on the same r-points.
3. Produces overlay plots of w(r): COMSOL vs. PINN.
4. Computes RMSE table per geometry.
5. Computes C(P) from both COMSOL and PINN deflections using the
   capacitance integral from Eroglu et al. (eqs. 4 & 5).
6. Overlays C(P) curves.

COMSOL data is NOT used in training after the warmup; any agreement here
is because the network learned the underlying physics, not the data.
"""

import argparse
import os
import glob
import re

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lamination import compute_laminate_stiffness
from model      import DiaphragmPINN, InputNormaliser

# ── constants ─────────────────────────────────────────────────────────────────
EPS0   = 8.854e-12   # F/m
EPS_P  = 3.15        # relative permittivity of parylene-C
COMSOL_DIR = "COMSOL_Deflection_Outputs"
OUT_DIR    = "validation_outputs"


# ── capacitance integrals (Eroglu et al. eqs. 4 & 5) ─────────────────────────

def capacitance(r, w, a, ag, t3, t4=0.0):
    """
    Numerical capacitance from the deflection profile w(r).

    Non-touch mode (eq. 4):
        C = ∫₀ᵃ  2πr ε₀ / [h − w(r)]  dr
        h = ag + (t3 + t4) / ε_p

    Touch mode (eq. 5):
        contact radius b where w(b) = −ag
        C = ∫₀ᵇ 2πr ε₀ / (t3+t4)/ε_p dr  +  ∫ᵦᵃ  same as non-touch

    Parameters
    ----------
    r  : 1-D array  [m]   radial coordinates (increasing)
    w  : 1-D array  [m]   deflection at those radii (negative downward)
    a  : float  [m]       diaphragm radius
    ag : float  [m]       initial air gap
    t3 : float  [m]       top parylene thickness
    t4 : float  [m]       bottom parylene thickness (default 0 for this sensor config)

    Returns
    -------
    C : float  [F]
    """
    h_diel = (t3 + t4) / EPS_P        # effective dielectric gap [m]
    h_eff  = ag + h_diel               # h in Eroglu eq. (6)

    # Determine contact radius b (where w hits −ag)
    touch = w <= -ag + 1e-15          # boolean mask
    if np.any(touch):
        b_idx = np.argmax(touch)
        b     = r[b_idx]
        # Touch region: gap = h_diel only
        r_t = r[:b_idx+1]
        C_t = np.trapz(2 * np.pi * r_t * EPS0 / h_diel, r_t)
        # Non-touch annulus
        r_nt = r[b_idx:]
        w_nt = w[b_idx:]
        gap_nt = h_eff - w_nt         # h - w(r), w is negative so h+|w|
        gap_nt = np.maximum(gap_nt, h_diel)   # floor at dielectric
        C_nt = np.trapz(2 * np.pi * r_nt * EPS0 / gap_nt, r_nt)
        return C_t + C_nt
    else:
        gap = h_eff - w               # h - w(r)
        gap = np.maximum(gap, h_diel)
        return np.trapz(2 * np.pi * r * EPS0 / gap, r)


# ── parse geometry from filename ─────────────────────────────────────────────

def parse_geo(fname):
    """Extract (a,t1,t2,t3,ag) in metres from CSV filename."""
    name = os.path.basename(fname).replace(".csv", "")
    def _get(key):
        m = re.search(rf"_{key}([\d.]+)", name)
        return float(m.group(1)) * 1e-6 if m else None
    return {
        "a":  _get("rad"),
        "t1": _get("t1"),
        "t2": _get("t2"),
        "t3": _get("t3"),
        "ag": _get("ag"),
    }


# ── run network on COMSOL r-points ───────────────────────────────────────────

def predict_w(model, normaliser, r_arr, P_val, geo, device, dtype):
    """
    Run the PINN on the COMSOL r-points for a single (geometry, pressure).

    Returns w_pred in metres (numpy array).
    """
    a  = geo["a"];  ag = geo["ag"]
    t1 = geo["t1"]; t2 = geo["t2"]; t3 = geo["t3"]
    n  = len(r_arr)

    xi = torch.tensor(r_arr / a, dtype=dtype, device=device)
    P  = torch.tensor(P_val,     dtype=dtype, device=device).expand(n)
    t1t = torch.tensor(t1, dtype=dtype, device=device).expand(n)
    t2t = torch.tensor(t2, dtype=dtype, device=device).expand(n)
    t3t = torch.tensor(t3, dtype=dtype, device=device).expand(n)
    at  = torch.tensor(a,  dtype=dtype, device=device).expand(n)
    agt = torch.tensor(ag, dtype=dtype, device=device).expand(n)

    x_in = normaliser.normalise(xi, P, t1t, t2t, t3t, at, agt)

    with torch.no_grad():
        w_pred, _ = model(x_in)

    return w_pred.squeeze(-1).cpu().numpy()


# ── main validation ───────────────────────────────────────────────────────────

def validate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float64

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    model      = DiaphragmPINN(n_hidden=5, hidden_dim=128).to(device).to(dtype)
    normaliser = InputNormaliser()

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from step {ckpt.get('step','?')}")

    # ── collect COMSOL files ──────────────────────────────────────────────────
    files = sorted(glob.glob(os.path.join(COMSOL_DIR, "*.csv")))
    if args.max_files:
        files = files[:args.max_files]
    print(f"Validating on {len(files)} COMSOL files …")

    rmse_rows  = []
    P_levels   = np.arange(0, 20001, 500)   # 0 to 20 kPa in 500 Pa steps

    pdf_path   = os.path.join(OUT_DIR, "deflection_overlays.pdf")
    cap_path   = os.path.join(OUT_DIR, "capacitance_CP.pdf")

    with PdfPages(pdf_path) as pdf_w, PdfPages(cap_path) as pdf_c:
        for fpath in files:
            geo   = parse_geo(fpath)
            if any(v is None for v in geo.values()):
                continue

            df    = pd.read_csv(fpath)
            r_um  = df["r_um"].values
            r_m   = r_um * 1e-6

            # ── deflection overlay ────────────────────────────────────────────
            fig_w, axes_w = plt.subplots(2, 4, figsize=(16, 8))
            axes_w = axes_w.flatten()

            rmse_list = []
            C_comsol  = []
            C_pinn    = []
            P_for_cap = []

            ax_idx = 0
            for col in df.columns[1:]:
                P_pa = float(col.split("w_")[1].replace("Pa_um", ""))

                w_comsol = df[col].values * 1e-6   # → metres

                # network prediction
                w_pred   = predict_w(model, normaliser, r_m, P_pa, geo, device, dtype)

                # RMSE (µm for readability)
                rmse = np.sqrt(np.mean((w_pred - w_comsol)**2)) * 1e6
                rmse_list.append(rmse)

                # capacitance
                C_cos = capacitance(r_m, w_comsol, geo["a"], geo["ag"], geo["t3"])
                C_pin = capacitance(r_m, w_pred,   geo["a"], geo["ag"], geo["t3"])
                C_comsol.append(C_cos * 1e15)    # fF
                C_pinn.append(C_pin * 1e15)
                P_for_cap.append(P_pa)

                # plot every 4th pressure on the overlay figure
                if ax_idx < len(axes_w) and P_pa % 2000 == 0:
                    ax = axes_w[ax_idx]
                    ax.plot(r_um, w_comsol * 1e6, "k-",  label="COMSOL", lw=1.5)
                    ax.plot(r_um, w_pred   * 1e6, "r--", label="PINN",   lw=1.5)
                    ax.axhline(-geo["ag"] * 1e6, color="gray", ls=":", lw=0.8, label="−ag")
                    ax.set_title(f"P={P_pa/1e3:.1f} kPa", fontsize=8)
                    ax.set_xlabel("r (µm)", fontsize=7)
                    ax.set_ylabel("w (µm)", fontsize=7)
                    ax.legend(fontsize=6)
                    ax_idx += 1

            fname_short = os.path.basename(fpath).replace(".csv", "")
            fig_w.suptitle(fname_short, fontsize=9)
            plt.tight_layout()
            pdf_w.savefig(fig_w)
            plt.close(fig_w)

            # ── C(P) overlay ──────────────────────────────────────────────────
            fig_c, ax_c = plt.subplots(figsize=(6, 4))
            ax_c.plot(np.array(P_for_cap) / 1e3, C_comsol, "k-",  lw=1.5, label="COMSOL")
            ax_c.plot(np.array(P_for_cap) / 1e3, C_pinn,   "r--", lw=1.5, label="PINN")
            ax_c.set_xlabel("Pressure (kPa)")
            ax_c.set_ylabel("Capacitance (fF)")
            ax_c.set_title(fname_short[:50], fontsize=8)
            ax_c.legend()
            plt.tight_layout()
            pdf_c.savefig(fig_c)
            plt.close(fig_c)

            mean_rmse = float(np.mean(rmse_list))
            rmse_rows.append({
                "file":      fname_short,
                "a_um":      geo["a"]  * 1e6,
                "t1_um":     geo["t1"] * 1e6,
                "t2_um":     geo["t2"] * 1e6,
                "t3_um":     geo["t3"] * 1e6,
                "ag_um":     geo["ag"] * 1e6,
                "mean_rmse_um": mean_rmse,
                "max_rmse_um":  float(np.max(rmse_list)),
            })

    # ── RMSE table ────────────────────────────────────────────────────────────
    rmse_df = pd.DataFrame(rmse_rows).sort_values("mean_rmse_um")
    rmse_csv = os.path.join(OUT_DIR, "rmse_table.csv")
    rmse_df.to_csv(rmse_csv, index=False)
    print(f"\nRMSE summary (µm):")
    print(rmse_df[["file", "mean_rmse_um", "max_rmse_um"]].to_string(index=False))
    print(f"\nAll outputs written to  {OUT_DIR}/")
    print(f"  deflection_overlays.pdf")
    print(f"  capacitance_CP.pdf")
    print(f"  rmse_table.csv")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/ckpt_final.pt")
    parser.add_argument("--max-files",  type=int, default=None,
                        help="Limit number of COMSOL files (default: all)")
    args = parser.parse_args()
    validate(args)
