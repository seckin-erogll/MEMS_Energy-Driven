# MEMS Diaphragm PINN — Energy-Driven Model

Physics-Informed Neural Network for a clamped multilayer
**parylene-C / gold / parylene-C** circular diaphragm capacitive pressure sensor.

> **Important:** COMSOL data is used for **validation and optional warmup anchoring only**.
> The primary training signal is a physics energy functional (Föppl–von Kármán variational form).
> If the validation overlay only matched because COMSOL was in the loss, the result would be meaningless.

---

## File structure

| File | Purpose |
|------|---------|
| `lamination.py` | Analytical D\*, A₁₁, A₁₂ from (t1,t2,t3) via Classical Laminated Plate Theory |
| `model.py` | Two-head PINN (w, u) with hard boundary conditions |
| `energy.py` | Energy functional, Gauss–Legendre quadrature, autodiff derivatives |
| `train.py` | Training loop: κ continuation, warmup annealing, checkpointing |
| `validate.py` | COMSOL overlay plots, RMSE table, C(P) comparison |

---

## Physical problem

A clamped circular multilayer diaphragm (radius *a*, layers bottom→top: parylene-C *t₁* / gold *t₂* / parylene-C *t₃*) above an air gap *ag* to a fixed electrode. Uniform pressure *P* deflects it downward; at sufficient pressure it enters touch mode and the contact area grows.

Deflections are large compared to thickness → **Föppl–von Kármán** (not Kirchhoff bending alone).

### Material properties

| Material | E (GPa) | ν | εᵣ |
|----------|---------|---|-----|
| Parylene-C | 3.2 | 0.33 | 3.15 |
| Gold | 70 | 0.44 | — |

### Geometry training box

| Parameter | Range |
|-----------|-------|
| t₁ | [0.8, 1.2] µm |
| t₂ | [0.15, 0.25] µm |
| t₃ | [3.0, 5.0] µm |
| a | [280, 320] µm |
| ag | [4.0, 6.0] µm |
| P | [0, 20] kPa (log-uniform) |

---

## Lamination theory

Neutral axis (measured from bottom of stack):

```
z_NA = Σ Ēₖ tₖ z_{c,k} / Σ Ēₖ tₖ     where Ēₖ = Eₖ/(1−νₖ²)
```

Effective bending stiffness (D₁₁ of CLT, Atik Supp. eq. 24):

```
D* = (1/3) Σ Ēₖ (bₖ³ − b_{k-1}³)     bₖ = signed distance from NA to top of layer k
```

Membrane stiffness (A₁₁, A₁₂ of CLT):

```
A₁₁ = Σ Ēₖ tₖ
A₁₂ = Σ νₖ Ēₖ tₖ
```

---

## Energy functional

```
Π[w,u] = ∫₀ᵃ 2πr [ ½D*(w''+w'/r)²
                   + ½(A₁₁εᵣ² + 2A₁₂εᵣεθ + A₁₁εθ²)
                   + Pw
                   + κ[max(0,−(w+ag))]² ] dr
```

where εᵣ = u' + ½(w')²  and  εθ = u/r.

Sign convention: **w < 0 for downward deflection**.
The +Pw term (with w < 0) decreases Π as the plate deflects downward, driving equilibrium.
E-L check: D∇⁴w + P = 0 → w = −Pa⁴/(64D\*)·(1−r²/a²)² < 0 ✓

The obstacle term (κ·penetration²) enforces Signorini contact variationally.
κ starts at 10⁴ and ramps ×10 every 2000 steps up to 10¹⁰.

---

## How to reproduce

### Install dependencies

```bash
pip install torch numpy scipy pandas matplotlib
```

### Sanity check: lamination module

```bash
python lamination.py
```

Expected output (nominal t1=1, t2=0.2, t3=4 µm):
- z_NA ≈ 1.89 µm (below geometric midplane 2.6 µm)
- D\* ≈ 6.2×10⁻⁸ N·m
- A₁₁ ≈ 3.5×10⁴ N/m
- ν_eff ≈ 0.384

### Train — pure physics (no COMSOL, sanity check 1)

```bash
python train.py --no-warmup --steps 50000
```

Check: energy decreases monotonically; w(r) at mid-range geometry looks like
a clamped-plate profile; deep-touch case shows flat region at w=−ag.

### Train — with warmup anchor (recommended)

```bash
python train.py --steps 20000
```

COMSOL anchor is active for steps 0–2000 only, with λ annealing 1→0.
After step 2000 the loss is pure physics.

### Validate against COMSOL

```bash
python validate.py --checkpoint checkpoints/ckpt_final.pt
```

Outputs in `validation_outputs/`:
- `deflection_overlays.pdf` — w(r) overlay for each geometry × pressure
- `capacitance_CP.pdf` — C(P) overlay
- `rmse_table.csv` — per-geometry RMSE in µm

---

## References

1. Atik et al. 2020 *J. Micromech. Microeng.* **30** 115001 — multilayer diaphragm model, CLT spring constants.
2. Atik et al. 2020 Supplementary — explicit laminate stiffness derivations (eqs. 23–25).
3. Eroglu et al. 2025 ICCHMT — sensor geometry, capacitance formulation (eqs. 4–5).
