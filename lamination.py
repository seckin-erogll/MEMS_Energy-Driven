"""
lamination.py — Classical Laminated Plate Theory for the 3-layer
parylene-C / gold / parylene-C diaphragm.

Layer stack (bottom → top, indices 1-3):
  1  Parylene-C   t1   E_p=3.2 GPa  nu_p=0.33
  2  Gold         t2   E_Au=70 GPa  nu_Au=0.44
  3  Parylene-C   t3   (same as layer 1)

References
----------
Atik et al. 2020, Supplementary §2, eqs. (23)–(24):
  • Neutral axis  : d  (eq. 23)
  • Linear spring : k1 (eq. 24)  ↔  k1 = 192π D* / a²
  • Membrane      : A11, A12 (CLT A-matrix, isotropic layers)

All torch operations so gradients flow when called with tensor geometry.
"""

import torch

# ── material constants ────────────────────────────────────────────────────────
E_P   = 3.2e9          # Pa   parylene-C
NU_P  = 0.33
E_AU  = 70.0e9         # Pa   gold
NU_AU = 0.44

# reduced (plane-stress) moduli  Ē = E / (1 − ν²)
EBAR_P  = E_P  / (1.0 - NU_P**2)    # ≈ 3.591 GPa
EBAR_AU = E_AU / (1.0 - NU_AU**2)   # ≈ 86.80 GPa


def compute_laminate_stiffness(t1, t2, t3):
    """
    Compute effective laminate stiffness for a 3-layer plate.

    Parameters
    ----------
    t1, t2, t3 : torch.Tensor  (same shape, arbitrary batch dimensions)
        Layer thicknesses in metres.

    Returns
    -------
    D_star : Tensor  [N·m]
        Effective bending stiffness  D* = D₁₁ of CLT.
        Atik Supp. eq. 24:  k₁ = 192π D* / a²
        Energy density:  ½ D* (w'' + w'/r)²

    A11    : Tensor  [N/m]
        Membrane stiffness (diagonal of CLT A-matrix).
        A₁₁ = Σ Ēₖ tₖ

    A12    : Tensor  [N/m]
        Membrane Poisson coupling (off-diagonal of CLT A-matrix).
        A₁₂ = Σ νₖ Ēₖ tₖ

    z_NA   : Tensor  [m]
        Neutral-axis position measured from the bottom of the stack.

    Notes
    -----
    The full membrane strain energy density is:
        ½ [A11 εr² + 2 A12 εr εθ + A11 εθ²]
    with  εr = u' + ½(w')²   and   εθ = u/r.
    This is the correct biaxial CLT form; dropping εθ or the A12 coupling
    under-predicts membrane stiffening in the large-deflection regime.
    """
    # ── Step 1: reduced moduli (scalars broadcast with batch tensors) ─────────
    Ep_bar  = torch.as_tensor(EBAR_P,  dtype=t1.dtype, device=t1.device)
    Eau_bar = torch.as_tensor(EBAR_AU, dtype=t1.dtype, device=t1.device)
    nup     = torch.as_tensor(NU_P,    dtype=t1.dtype, device=t1.device)
    nuau    = torch.as_tensor(NU_AU,   dtype=t1.dtype, device=t1.device)

    # ── Step 2: neutral-axis position z_NA from the bottom of the stack ───────
    #   z_{c,1} = t1/2
    #   z_{c,2} = t1 + t2/2
    #   z_{c,3} = t1 + t2 + t3/2
    #
    #   z_NA = Σ Ēk tk z_{c,k} / Σ Ēk tk          (Atik Supp. eq. 23)
    #
    numerator = (Ep_bar  * t1 * (t1 / 2.0)
               + Eau_bar * t2 * (t1 + t2 / 2.0)
               + Ep_bar  * t3 * (t1 + t2 + t3 / 2.0))

    denom_membrane = Ep_bar * (t1 + t3) + Eau_bar * t2   # = A11

    z_NA = numerator / denom_membrane

    # ── Step 3: signed b-offsets from neutral axis to top of each layer ───────
    #   b0 = -z_NA            (bottom of stack)
    #   b1 = t1 - z_NA        (top of layer 1)
    #   b2 = t1+t2 - z_NA     (top of layer 2)
    #   b3 = t1+t2+t3 - z_NA  (top of layer 3 = top of stack)
    b0 = -z_NA
    b1 =  t1           - z_NA
    b2 =  t1 + t2      - z_NA
    b3 =  t1 + t2 + t3 - z_NA

    # ── Step 4: effective bending stiffness D* ────────────────────────────────
    #   D* = (1/3) Σ Ēk (bk³ − b_{k-1}³)          (Atik Supp. eq. 24 + 1/3)
    #
    # For a single homogeneous layer of thickness h:
    #   b0=-h/2, b1=h/2  →  D* = Ē h³/12 = E h³/[12(1-ν²)]  ✓
    D_star = (1.0 / 3.0) * (
        Ep_bar  * (b1**3 - b0**3)
      + Eau_bar * (b2**3 - b1**3)
      + Ep_bar  * (b3**3 - b2**3)
    )

    # ── Step 5: membrane stiffness coefficients ───────────────────────────────
    #   A11 = Σ Ēk tk          (already computed as denom_membrane)
    #   A12 = Σ νk Ēk tk
    A11 = denom_membrane
    A12 = nup * Ep_bar * (t1 + t3) + nuau * Eau_bar * t2

    return D_star, A11, A12, z_NA


# ── quick sanity check (run as script) ───────────────────────────────────────
if __name__ == "__main__":
    t1 = torch.tensor(1.0e-6)   # 1 µm
    t2 = torch.tensor(0.2e-6)   # 0.2 µm
    t3 = torch.tensor(4.0e-6)   # 4 µm

    D, A11, A12, zna = compute_laminate_stiffness(t1, t2, t3)

    print(f"z_NA  = {zna*1e6:.4f} µm  (geometric midplane = {(t1+t2+t3)*0.5e6:.4f} µm)")
    print(f"D*    = {D:.4e} N·m")
    print(f"A11   = {A11:.4e} N/m")
    print(f"A12   = {A12:.4e} N/m")
    print(f"ν_eff = A12/A11 = {(A12/A11):.4f}")

    # Compare D* against pure-parylene-C plate of same total thickness
    H = t1 + t2 + t3
    D_pary = EBAR_P * H**3 / 12.0
    print(f"\nD_parylene (same H) = {D_pary:.4e} N·m")
    print(f"Stiffness ratio D*/D_pary = {(D/D_pary):.3f}  (>1 expected, gold stiffens)")
