"""
model.py — Two-head PINN for the multilayer diaphragm.

Architecture
------------
Inputs  (7, normalised to [0,1]):
    [ξ = r/a,  P̂,  t̂1,  t̂2,  t̂3,  â,  âg]

Trunk:
    Linear(7→128) → tanh
    × 5 hidden layers: Linear(128→128) → tanh

Two heads (raw, before hard BCs):
    w_raw : Linear(128→1)
    u_raw : Linear(128→1)

Hard boundary + contact conditions (applied post-network):
    Profile:   w_profile(ξ) = (1 − ξ²)² · w_raw · W_ref
               enforces w(1)=0, w'(1)=0, symmetry at ξ=0.

    Contact:   w(ξ) = smooth_max(w_profile, −ag)
               using smooth_max(a,b) = (a+b+√((a−b)²+ε²))/2  with ε=50 nm.
               This is a hard constraint: w ≥ −ag everywhere, with a smooth
               transition of width ε at the contact line.  Gradient is always
               nonzero (unlike torch.clamp), so learning continues in the
               contact region.

    u(ξ) = ξ(1 − ξ) · u_raw · U_ref      → u(0)=0, u(1)=0

    W_ref = ag_phys  (deflections normalised by air gap)
    U_ref = 0.1 · ag_phys

Activations: tanh throughout — required for clean second derivatives of w
in the bending term.  ReLU/GELU would produce zero or undefined w''.
"""

import torch
import torch.nn as nn


class DiaphragmPINN(nn.Module):
    """
    Physics-Informed Neural Network for the clamped multilayer diaphragm.

    Parameters
    ----------
    n_hidden : int
        Number of hidden layers (default 5 → total depth 6 linear layers).
    hidden_dim : int
        Width of each hidden layer (default 128).
    """

    def __init__(self, n_hidden: int = 5, hidden_dim: int = 128):
        super().__init__()

        # ── trunk ──────────────────────────────────────────────────────────────
        layers = [nn.Linear(7, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)

        # ── two output heads ───────────────────────────────────────────────────
        self.head_w = nn.Linear(hidden_dim, 1)   # transverse deflection
        self.head_u = nn.Linear(hidden_dim, 1)   # radial in-plane displacement

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for linear layers, zero bias for heads."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor  shape (..., 7)
            Normalised inputs [ξ, P̂, t̂1, t̂2, t̂3, â, âg].
            ξ = x[..., 0]  must have requires_grad=True for energy computation.

        Returns
        -------
        w : Tensor  shape (..., 1)   [metres]  transverse deflection (≤0 downward)
        u : Tensor  shape (..., 1)   [metres]  radial in-plane displacement
        """
        xi = x[..., 0:1]   # normalised radial coordinate  ξ ∈ (0,1)

        # ── physical scales carried in the input ───────────────────────────────
        # x[..., 6] = âg = ag / ag_ref  (we pass normalised ag, recover physical)
        # We need W_ref and U_ref in physical metres.
        # The input vector carries âg as the 7th feature (index 6).
        # Recover ag_physical by multiplying by the reference gap (5 µm = midpoint).
        # Denormalize ag: âg = (ag - ag_lo)/(ag_hi - ag_lo), so ag = âg*(ag_hi-ag_lo)+ag_lo
        ag_lo  = 5.0e-6    # metres — lower bound of training range
        ag_hi  = 15.0e-6   # metres — upper bound of training range
        ag_hat = x[..., 6:7]
        ag_phys = ag_hat * (ag_hi - ag_lo) + ag_lo   # exact physical ag [m]

        W_ref = ag_phys            # deflection scale ≈ ag
        U_ref = 0.1 * ag_phys      # in-plane scale  ≈ 0.1 ag

        # ── trunk ──────────────────────────────────────────────────────────────
        h = self.trunk(x)

        # ── raw heads ──────────────────────────────────────────────────────────
        w_raw = self.head_w(h)     # unbounded scalar
        u_raw = self.head_u(h)     # unbounded scalar

        # ── hard boundary conditions ───────────────────────────────────────────
        # Profile: (1-ξ²)² enforces w(1)=0, w'(1)=0, and dw/dξ(0)=0 by symmetry.
        bc_w      = (1.0 - xi**2) ** 2
        w_profile = bc_w * w_raw * W_ref      # unconstrained (may exceed -ag)

        # Hard contact constraint via smooth-max: w ≥ -ag_phys everywhere.
        # smooth_max(a, b) = (a + b + √((a-b)² + ε²)) / 2  →  max(a,b) as ε→0
        # ε = 50 nm gives a ~50 nm transition band at the contact line.
        eps_c = 5e-8                          # 50 nm smoothing width [m]
        contact_floor = -ag_phys
        diff  = w_profile - contact_floor
        w = 0.5 * (w_profile + contact_floor
                   + torch.sqrt(diff**2 + eps_c**2))   # ≥ -ag, smooth

        # u: vanishes at centre and rim
        bc_u = xi * (1.0 - xi)
        u = bc_u * u_raw * U_ref

        return w, u


class InputNormaliser:
    """
    Utility to normalise raw physical inputs to [0,1].

    Geometry ranges (metres):
        a   ∈ [280, 320] µm
        t1  ∈ [0.8, 1.2] µm
        t2  ∈ [0.15, 0.25] µm
        t3  ∈ [3.0, 5.0] µm
        ag  ∈ [4.0, 6.0] µm
        P   ∈ [0, 20] kPa   (log-uniform sampling; normalise log(P+1))
    """

    # (lo, hi) in SI units — matched to the COMSOL dataset range
    RANGES = {
        "t1": (1.0e-6,   5.0e-6),
        "t2": (0.2e-6,   0.5e-6),
        "t3": (1.0e-6,   5.0e-6),
        "a":  (150e-6,  500e-6),
        "ag": (5.0e-6,  15.0e-6),
    }
    P_MAX = 20e3   # Pa

    @staticmethod
    def normalise(xi, P, t1, t2, t3, a, ag):
        """
        Map physical quantities to normalised inputs in [0,1].

        All inputs are Tensors (or broadcastable).  Returns stacked tensor
        of shape (..., 7): [ξ̂, P̂, t̂1, t̂2, t̂3, â, âg].
        """
        def _norm(val, lo, hi):
            return (val - lo) / (hi - lo)

        xi_n  = xi   # already in [0,1]
        P_n   = torch.log1p(P / 1e3) / torch.log1p(torch.tensor(20.0))  # log-scale
        t1_n  = _norm(t1,  *InputNormaliser.RANGES["t1"])
        t2_n  = _norm(t2,  *InputNormaliser.RANGES["t2"])
        t3_n  = _norm(t3,  *InputNormaliser.RANGES["t3"])
        a_n   = _norm(a,   *InputNormaliser.RANGES["a"])
        ag_n  = _norm(ag,  *InputNormaliser.RANGES["ag"])

        return torch.stack([xi_n, P_n, t1_n, t2_n, t3_n, a_n, ag_n], dim=-1)
