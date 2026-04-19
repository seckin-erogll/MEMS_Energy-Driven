"""
plot_deflections.py — Deflection w(r) for 5 geometries, three data sources:
  • PINN  (this work, energy-driven)
  • COMSOL FEA (ground truth)
  • Atik analytical (Kirchhoff linear plate, Atik et al. 2020 eq. 24)

For each geometry the closest COMSOL file is chosen so all three curves
compare exactly the same geometry.

Usage:
    python plot_deflections.py --checkpoint checkpoints/ckpt_final.pt

Outputs:
    deflection_5geometries.png    — combined 5-panel overview
    deflection_<label>.pdf        — individual panel for each geometry
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model      import DiaphragmPINN, InputNormaliser
from lamination import compute_laminate_stiffness

# ── COMSOL data directory ─────────────────────────────────────────────────────
COMSOL_DIR = (
    "/home/seckin/Desktop/MEMS_Viscosity_Measurement/"
    "Pressure_Sensor_Model/COMSOL_Deflection_Outputs"
)

# pressures present in every COMSOL file (500 Pa increments); pick subset
COMPARE_PRESSURES = [500, 1000, 2000, 5000, 10000, 20000]  # Pa

N_POINTS = 300   # radial points for PINN / analytical curves


# ── 5 target geometries (label only; actual params chosen from COMSOL match) ─
TARGET_GEOS = [
    ("Small thin",       200,  1.0, 0.20, 1.0,  5.0),
    ("Medium balanced",  300,  2.0, 0.30, 2.0,  8.0),
    ("Large thick",      450,  4.0, 0.40, 4.0, 12.0),
    ("Small large gap",  180,  2.0, 0.25, 2.0, 14.0),
    ("Large thin",       480,  1.5, 0.20, 1.5,  6.0),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_comsol_geos(comsol_dir):
    """Return list of (filename, a_um, t1_um, t2_um, t3_um, ag_um)."""
    pat = re.compile(r"rad([\d.]+)_t1([\d.]+)_t2([\d.]+)_t3([\d.]+)_ag([\d.]+)\.csv")
    geos = []
    for f in os.listdir(comsol_dir):
        m = pat.search(f)
        if m:
            geos.append((f,) + tuple(float(m.group(i)) for i in range(1, 6)))
    return geos


def find_closest(geos, ta, tt1, tt2, tt3, tag):
    """Find the COMSOL file geometry closest to target (normalised L2)."""
    best_dist, best = 1e9, None
    for entry in geos:
        f, a, t1, t2, t3, ag = entry
        dist = (
            ((a  - ta ) / 350) ** 2
          + ((t1 - tt1) /   4) ** 2
          + ((t2 - tt2) / 0.3) ** 2
          + ((t3 - tt3) /   4) ** 2
          + ((ag - tag) /  10) ** 2
        )
        if dist < best_dist:
            best_dist, best = dist, entry
    return best   # (fname, a, t1, t2, t3, ag)


def load_comsol(comsol_dir, fname, pressures):
    """
    Load COMSOL r(um) and w(um) columns for the requested pressures.
    Returns dict  pressure_pa -> (r_um array, w_um array).
    """
    df = pd.read_csv(os.path.join(comsol_dir, fname))
    r_um = df["r_um"].values
    data = {}
    for P in pressures:
        col = f"w_{int(P)}Pa_um"
        if col in df.columns:
            data[P] = (r_um, df[col].values)
    return data


def kirchhoff_deflection(r_um, a_um, P_pa, t1_um, t2_um, t3_um):
    """
    Atik et al. 2020, eq. 24 (linear spring k1 = 192π D*/a²).
    w(r) = -P a⁴ / (64 D*) · (1 - (r/a)²)²   [µm]
    Valid for small deflections; ignores membrane stiffening and contact.
    """
    t1 = torch.tensor(t1_um * 1e-6)
    t2 = torch.tensor(t2_um * 1e-6)
    t3 = torch.tensor(t3_um * 1e-6)
    D_star, _, _, _ = compute_laminate_stiffness(t1, t2, t3)
    D = D_star.item()
    a  = a_um  * 1e-6    # m
    r  = r_um  * 1e-6    # m
    xi = r / a
    w_m = -P_pa * a ** 4 / (64.0 * D) * (1.0 - xi ** 2) ** 2
    return w_m * 1e6      # µm


def pinn_deflection(model, normaliser, r_um, a_um, P_pa,
                    t1_um, t2_um, t3_um, ag_um, device, dtype):
    """Run PINN forward pass; return w(r) in µm."""
    a_m  = a_um  * 1e-6
    t1_m = t1_um * 1e-6
    t2_m = t2_um * 1e-6
    t3_m = t3_um * 1e-6
    ag_m = ag_um * 1e-6

    r_t  = torch.tensor(r_um * 1e-6, dtype=dtype, device=device)
    xi   = r_t / a_m
    n    = xi.shape[0]

    x_in = normaliser.normalise(
        xi,
        torch.full((n,), P_pa,  dtype=dtype, device=device),
        torch.full((n,), t1_m,  dtype=dtype, device=device),
        torch.full((n,), t2_m,  dtype=dtype, device=device),
        torch.full((n,), t3_m,  dtype=dtype, device=device),
        torch.full((n,), a_m,   dtype=dtype, device=device),
        torch.full((n,), ag_m,  dtype=dtype, device=device),
    )
    with torch.no_grad():
        w, _ = model(x_in)
    return w.squeeze(-1).cpu().numpy() * 1e6   # µm


# ── plotting ──────────────────────────────────────────────────────────────────

def make_panel(ax, label, geo, model, normaliser, comsol_data, colors, device, dtype):
    """
    Draw one panel with PINN (solid), COMSOL (dashed), Atik (dotted) curves.

    geo  = (fname, a_um, t1_um, t2_um, t3_um, ag_um)
    """
    _, a_um, t1_um, t2_um, t3_um, ag_um = geo
    r_pinn = np.linspace(0.0, a_um, N_POINTS)

    legend_handles = []

    for color, P in zip(colors, COMPARE_PRESSURES):
        lbl = f"{P} Pa" if P < 1000 else f"{P // 1000} kPa"

        # ── PINN ──────────────────────────────────────────────────────────────
        w_pinn = pinn_deflection(model, normaliser, r_pinn, a_um, P,
                                 t1_um, t2_um, t3_um, ag_um, device, dtype)
        h_pinn, = ax.plot(r_pinn, w_pinn, color=color, linewidth=1.8,
                          linestyle="-", label=f"PINN {lbl}")

        # ── Atik (Kirchhoff) — clipped at contact floor ───────────────────────
        w_atik = kirchhoff_deflection(r_pinn, a_um, P, t1_um, t2_um, t3_um)
        w_atik_clipped = np.clip(w_atik, -ag_um, None)
        ax.plot(r_pinn, w_atik_clipped, color=color, linewidth=1.2,
                linestyle=":", alpha=0.85)

        # ── COMSOL ────────────────────────────────────────────────────────────
        if P in comsol_data:
            r_cs, w_cs = comsol_data[P]
            ax.plot(r_cs, w_cs, color=color, linewidth=1.2,
                    linestyle="--", alpha=0.9)

        legend_handles.append((h_pinn, color, lbl))

    # air gap floor
    ax.axhline(-ag_um, color="gray", linestyle="-.", linewidth=0.8, alpha=0.6)

    # y-axis: show from slightly above 0 down to ~10% below the air gap
    ax.set_ylim(-ag_um * 1.15, ag_um * 0.15)

    # labels & formatting
    ax.set_title(
        f"{label}\na={a_um:.0f} µm, ag={ag_um:.1f} µm\n"
        f"t1={t1_um:.2f} / t2={t2_um:.3f} / t3={t3_um:.2f} µm",
        fontsize=8, pad=4
    )
    ax.set_xlabel("r (µm)", fontsize=8)
    ax.set_ylabel("w (µm)", fontsize=8)
    ax.set_xlim(0, a_um)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7)

    return legend_handles


def build_legend_elements(colors):
    """Return Line2D proxies for the shared legend."""
    from matplotlib.lines import Line2D
    elements = []
    for color, P in zip(colors, COMPARE_PRESSURES):
        lbl = f"{P} Pa" if P < 1000 else f"{P // 1000} kPa"
        elements.append(Line2D([0], [0], color=color, linewidth=1.8, linestyle="-",  label=f"PINN — {lbl}"))
        elements.append(Line2D([0], [0], color=color, linewidth=1.2, linestyle="--", label=f"COMSOL — {lbl}"))
        elements.append(Line2D([0], [0], color=color, linewidth=1.2, linestyle=":",  label=f"Atik — {lbl}"))

    # style legend (collapsed)
    from matplotlib.lines import Line2D as L2D
    style_els = [
        L2D([0], [0], color="k", linewidth=1.8, linestyle="-",  label="PINN (this work)"),
        L2D([0], [0], color="k", linewidth=1.2, linestyle="--", label="COMSOL FEA"),
        L2D([0], [0], color="k", linewidth=1.2, linestyle=":",  label="Atik (Kirchhoff)"),
        L2D([0], [0], color="gray", linewidth=0.8, linestyle="-.", label="−air gap"),
    ]
    return style_els, elements


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    print(f"Device: {device}")

    # load model
    model      = DiaphragmPINN(n_hidden=5, hidden_dim=128).to(device).to(dtype)
    normaliser = InputNormaliser()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint (step {step})")

    # parse all COMSOL geometries
    all_geos = parse_comsol_geos(COMSOL_DIR)
    print(f"Found {len(all_geos)} COMSOL files")

    # find closest COMSOL match for each target geometry
    matched = []
    for label, ta, tt1, tt2, tt3, tag in TARGET_GEOS:
        geo = find_closest(all_geos, ta, tt1, tt2, tt3, tag)
        matched.append((label, geo))
        print(f"  {label}: {geo[0]}")

    cmap   = matplotlib.colormaps.get_cmap("plasma")
    colors = [cmap(i / (len(COMPARE_PRESSURES) - 1)) for i in range(len(COMPARE_PRESSURES))]

    # ── combined 5-panel figure ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5))
    fig.suptitle(
        f"MEMS Sensor Deflection w(r) — PINN vs COMSOL vs Atik (Kirchhoff)\n"
        f"(PINN checkpoint step {step})",
        fontsize=12, fontweight="bold"
    )

    for ax, (label, geo) in zip(axes, matched):
        _, a_um, t1_um, t2_um, t3_um, ag_um = geo
        comsol_data = load_comsol(COMSOL_DIR, geo[0], COMPARE_PRESSURES)
        make_panel(ax, label, geo, model, normaliser, comsol_data, colors, device, dtype)

    # legend: styles on left, pressures colour-coded on right
    style_els, pressure_els = build_legend_elements(colors)

    # style legend inside first axis
    axes[0].legend(handles=style_els, fontsize=7, loc="lower left",
                   framealpha=0.85, title="Line style", title_fontsize=7)

    # pressure colour legend below figure
    pressure_legend_els = [
        __import__("matplotlib.lines", fromlist=["Line2D"]).Line2D(
            [0], [0], color=c, linewidth=2.0,
            label=f"{P} Pa" if P < 1000 else f"{P // 1000} kPa"
        )
        for c, P in zip(colors, COMPARE_PRESSURES)
    ]
    fig.legend(handles=pressure_legend_els, loc="lower center",
               ncol=len(COMPARE_PRESSURES), fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=True, title="Pressure")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = "deflection_5geometries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # ── individual PDFs ───────────────────────────────────────────────────────
    for label, geo in matched:
        _, a_um, t1_um, t2_um, t3_um, ag_um = geo
        comsol_data = load_comsol(COMSOL_DIR, geo[0], COMPARE_PRESSURES)

        fig2, ax2 = plt.subplots(figsize=(7, 4.8))
        make_panel(ax2, label, geo, model, normaliser, comsol_data, colors, device, dtype)

        # combined legend
        from matplotlib.lines import Line2D
        handles = []
        for c, P in zip(colors, COMPARE_PRESSURES):
            lbl = f"{P} Pa" if P < 1000 else f"{P // 1000} kPa"
            handles += [
                Line2D([0], [0], color=c, lw=1.8, ls="-",  label=f"PINN {lbl}"),
                Line2D([0], [0], color=c, lw=1.2, ls="--", label=f"COMSOL {lbl}"),
                Line2D([0], [0], color=c, lw=1.2, ls=":",  label=f"Atik {lbl}"),
            ]
        handles += [Line2D([0], [0], color="gray", lw=0.8, ls="-.", label="−air gap")]

        ax2.legend(handles=handles, fontsize=6.5, ncol=3,
                   loc="lower left", framealpha=0.85)
        ax2.set_title(
            f"Deflection — {label}\n"
            f"a={a_um:.0f} µm  t1/t2/t3={t1_um:.2f}/{t2_um:.3f}/{t3_um:.2f} µm  "
            f"ag={ag_um:.1f} µm",
            fontsize=9
        )
        ax2.set_xlabel("Radial position r (µm)", fontsize=10)
        ax2.set_ylabel("Transverse deflection w (µm)", fontsize=10)

        plt.tight_layout()
        safe = label.replace(" ", "_")
        fname = f"deflection_{safe}.pdf"
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved → {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_final.pt")
    args = parser.parse_args()
    main(args)
