"""
energy.py — Variational energy functional for the clamped multilayer diaphragm.

Energy functional (Föppl–von Kármán, axisymmetric, variational form):

  Π[w,u] = ∫₀ᵃ 2πr [ ½D*(w''+w'/r)²
                     + ½(A11 εr² + 2A12 εr εθ + A11 εθ²)
                     + Pw
                     + κ [max(0,−(w+ag))]²  ] dr

  Sign convention: w < 0 for downward deflection.
  The +Pw term (with w<0) decreases Π as the plate deflects, driving equilibrium.
  E-L check: D∇⁴w + P = 0 → w = -Pa⁴/(64D*)·(1-r²/a²)²  (Kirchhoff, w<0) ✓

where
    εr = u' + ½(w')²     (von Kármán radial membrane strain)
    εθ = u/r              (tangential membrane strain)

Derivatives w', w'', u' are computed via torch.autograd.grad with respect to
the ξ-coordinate (ξ = r/a), then converted to r-derivatives via chain rule:
    ∂_r = (1/a) ∂_ξ

Integration: Gauss–Legendre quadrature on ξ ∈ (0,1) with N_QUAD=64 nodes.
ξ=0 is excluded (the integrand is finite there by symmetry, but autograd
can be numerically unstable at exactly ξ=0 for the 1/r term in εθ = u/r).
"""

import torch
import numpy as np


# ── Gauss–Legendre nodes & weights on (0,1) ─────────────────────────────────

def gauss_legendre_01(n: int, device=None, dtype=torch.float64):
    """
    Return (nodes, weights) for Gauss–Legendre quadrature on (0,1).
    Uses numpy to compute the rule, then moves to the requested device/dtype.
    Nodes are strictly interior: no endpoint at ξ=0 or ξ=1.
    """
    # GL nodes/weights on [-1,1]
    xi_ref, w_ref = np.polynomial.legendre.leggauss(n)
    # Map to (0,1):  ξ = (x+1)/2,  w = w_ref/2
    nodes   = torch.tensor((xi_ref + 1.0) / 2.0, dtype=dtype, device=device)
    weights = torch.tensor(w_ref / 2.0,           dtype=dtype, device=device)
    return nodes, weights


_QUAD_CACHE = {}

def get_quad(n: int, device, dtype):
    key = (n, str(device), dtype)
    if key not in _QUAD_CACHE:
        _QUAD_CACHE[key] = gauss_legendre_01(n, device=device, dtype=dtype)
    return _QUAD_CACHE[key]


# ── autodiff helpers ─────────────────────────────────────────────────────────

def _grad(output, inputs, create_graph=True, allow_unused=False):
    """Thin wrapper around torch.autograd.grad returning the gradient tensor."""
    grads = torch.autograd.grad(
        outputs=output.sum(),
        inputs=inputs,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=allow_unused,
    )
    return grads[0]


# ── main energy function ─────────────────────────────────────────────────────

def compute_energy(
    model,
    P,          # (B,)   pressure  [Pa]
    t1, t2, t3, # (B,)   layer thicknesses  [m]
    a,          # (B,)   diaphragm radius  [m]
    ag,         # (B,)   air gap  [m]
    D_star,     # (B,)   effective bending stiffness  [N·m]
    A11,        # (B,)   membrane stiffness  [N/m]
    A12,        # (B,)   membrane Poisson coupling  [N/m]
    normaliser,
    n_quad: int = 64,
    kappa: float = 1e3,
):
    """
    Evaluate the total potential energy Π for a batch of geometries and
    pressures by Gauss–Legendre quadrature on ξ ∈ (0,1).

    Parameters
    ----------
    model       : DiaphragmPINN
    P           : (B,)  pressure per sample  [Pa]
    t1,t2,t3    : (B,)  layer thicknesses  [m]
    a, ag       : (B,)  radius, air gap  [m]
    D_star      : (B,)  from lamination.compute_laminate_stiffness
    A11, A12    : (B,)  from lamination.compute_laminate_stiffness
    normaliser  : InputNormaliser
    n_quad      : number of GL quadrature nodes
    kappa       : obstacle penalty coefficient  [Pa/m²]  (ramped during training)

    Returns
    -------
    Pi : scalar  (mean over batch)  [J/m²·m²] = [J]
        Total potential energy, differentiable w.r.t. model parameters.
    """
    B = P.shape[0]
    device = P.device
    dtype  = P.dtype

    # ── quadrature nodes on (0,1) ─────────────────────────────────────────────
    xi_nodes, xi_weights = get_quad(n_quad, device, dtype)   # (Q,)

    # ── build full input tensor  (B × Q, 7) ──────────────────────────────────
    # Broadcast: for each geometry sample and each quadrature point
    # xi : (1, Q) → (B, Q)
    xi_bq = xi_nodes.unsqueeze(0).expand(B, n_quad)         # (B, Q)

    # geometry broadcast to (B, Q)
    def _bq(v):
        return v.unsqueeze(1).expand(B, n_quad)

    xi_bq_flat  = xi_bq.reshape(-1)          # (B*Q,)
    P_flat      = _bq(P).reshape(-1)
    t1_flat     = _bq(t1).reshape(-1)
    t2_flat     = _bq(t2).reshape(-1)
    t3_flat     = _bq(t3).reshape(-1)
    a_flat      = _bq(a).reshape(-1)
    ag_flat     = _bq(ag).reshape(-1)

    # ξ must have requires_grad for autograd derivatives
    xi_input = xi_bq_flat.detach().requires_grad_(True)

    # normalise and stack → (B*Q, 7)
    x_input = normaliser.normalise(
        xi_input, P_flat, t1_flat, t2_flat, t3_flat, a_flat, ag_flat
    )

    # ── network forward ───────────────────────────────────────────────────────
    w, u = model(x_input)   # each (B*Q, 1)
    w = w.squeeze(-1)       # (B*Q,)
    u = u.squeeze(-1)       # (B*Q,)

    # ── ξ-derivatives via autograd ────────────────────────────────────────────
    # w_xi  = ∂w/∂ξ  (first)
    w_xi  = _grad(w, xi_input)                # (B*Q,)
    # w_xi2 = ∂²w/∂ξ²  (second)
    w_xi2 = _grad(w_xi, xi_input)             # (B*Q,)
    # u_xi  = ∂u/∂ξ  (first)
    u_xi  = _grad(u, xi_input)                # (B*Q,)

    # ── chain rule:  ∂_r = (1/a) ∂_ξ ─────────────────────────────────────────
    a_flat_detach = a_flat.detach()
    w_r   = w_xi  / a_flat_detach             # ∂w/∂r
    w_rr  = w_xi2 / (a_flat_detach**2)        # ∂²w/∂r²
    u_r   = u_xi  / a_flat_detach             # ∂u/∂r

    # radial coordinate  r = ξ·a
    r = xi_input * a_flat_detach              # (B*Q,)

    # ── strains ───────────────────────────────────────────────────────────────
    eps_r     = u_r + 0.5 * w_r**2           # radial membrane strain
    eps_theta = u   / r                       # tangential membrane strain  u/r

    # ── integrand terms ───────────────────────────────────────────────────────
    # broadcast laminate coefficients to (B*Q,)
    Dstar_bq  = _bq(D_star).reshape(-1).detach()
    A11_bq    = _bq(A11).reshape(-1).detach()
    A12_bq    = _bq(A12).reshape(-1).detach()
    P_bq      = P_flat.detach()
    ag_bq     = ag_flat.detach()

    # 1) Bending energy density  ½ D* (w'' + w'/r)²
    bending = 0.5 * Dstar_bq * (w_rr + w_r / r)**2

    # 2) Full biaxial membrane energy density
    #    ½ [A11 εr² + 2 A12 εr εθ + A11 εθ²]
    membrane = 0.5 * (A11_bq * eps_r**2
                    + 2.0 * A12_bq * eps_r * eps_theta
                    + A11_bq * eps_theta**2)

    # 3) Pressure work density  +P w
    #    Convention: w < 0 for downward deflection.
    #    E-L equation from +P*w: D∇⁴w + P = 0  →  D∇⁴w = -P  (correct sign)
    #    E-L equation from -P*w: D∇⁴w - P = 0  →  D∇⁴w = +P  (gives upward deflection!)
    #    Verified: Kirchhoff solution w = -Pa⁴/64D*(1-r²/a²)² satisfies +P*w form. ✓
    pressure = P_bq * w

    # 4) Obstacle / Signorini penalty  κ [max(0, −(w+ag))]²
    #    Penalises penetration below the bottom electrode.
    penetration = torch.clamp(-(w + ag_bq), min=0.0)
    obstacle    = kappa * penetration**2

    # total integrand density × 2πr
    integrand = 2.0 * torch.pi * r * (bending + membrane + pressure + obstacle)

    # ── Gauss–Legendre quadrature ─────────────────────────────────────────────
    # integrand shape: (B*Q,) → (B, Q)
    integrand_bq = integrand.reshape(B, n_quad)

    # ∫₀ᵃ f(ξ·a) a dξ  — factor of a from the change of variables dr = a dξ
    a_factor = a.detach().unsqueeze(1)                       # (B, 1)
    Pi_per_sample = (integrand_bq * xi_weights.unsqueeze(0) * a_factor).sum(dim=1)

    return Pi_per_sample.mean()
