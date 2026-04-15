"""
Uncertainty Transform Modules for Norm-Ball Parameterizations.

These nn.Module transforms map free parameters (w̃) into disturbances (w)
that satisfy norm-ball constraints ||w|| ≤ γ_Δ ||v||.

All modules are differentiable and compatible with auto_LiRPA / α,β-CROWN.
"""

import torch
import torch.nn as nn
from typing import Optional


class TanhNormBallTransform(nn.Module):
    """
    Tanh norm-ball parameterization (smooth, tight).

    Maps free parameter w̃ ∈ [-c̄, c̄]^{n_w} to constrained disturbance w
    satisfying ||w|| < γ ||v|| strictly:

        w = γ ||v|| tanh(||w̃||) / (||w̃|| + ε) · w̃

    Properties:
      (i)   ||w|| < γ ||v|| for all w̃
      (ii)  w̃ → w is smooth (ε > 0 regularizes origin)
      (iii) Covers ≈99.5% of the norm ball for c̄ = 3

    For n_w = 1 (scalar uncertainty), the inflation factor is 1 (exact).

    Reference: Eq. (5) in the paper, Proposition 5.5.
    """

    def __init__(
        self,
        gamma_delta: float,
        n_w: int = 1,
        c_bar: float = 3.0,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            gamma_delta: Uncertainty gain bound (||Δ|| ≤ γ_Δ).
            n_w: Disturbance dimension.
            c_bar: Box bound for free parameter w̃ ∈ [-c̄, c̄]^{n_w}.
            epsilon: Regularization constant for numerical stability.
        """
        super().__init__()
        self.gamma_delta = gamma_delta
        self.n_w = n_w
        self.c_bar = c_bar
        self.epsilon = epsilon

    def forward(
        self,
        v: torch.Tensor,
        w_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform free parameter to norm-ball-constrained disturbance.

        Args:
            v: Uncertainty output (batch, n_v).  For LFT: v = u (control).
            w_tilde: Free parameter (batch, n_w), in [-c̄, c̄]^{n_w}.

        Returns:
            w: Constrained disturbance (batch, n_w), ||w|| < γ_Δ ||v||.
        """
        # ||v|| and ||w̃|| decomposed into Mul + ReduceSum + Sqrt
        # (auto_LiRPA does not support onnx::ReduceL2 directly)
        _eps_sq = self.epsilon * self.epsilon
        v_norm = torch.sqrt((v * v).sum(dim=-1, keepdim=True) + _eps_sq)

        if self.n_w == 1:
            # Scalar shortcut: tanh(|w̃|)·w̃/(|w̃|+ε) ≈ tanh(w̃) for n_w=1
            # Avoids sqrt, division, extra multiply — much tighter CROWN bounds.
            # Difference is O(ε) ≈ 1e-6 (negligible).
            w = self.gamma_delta * v_norm * torch.tanh(w_tilde)
            return w

        w_tilde_norm = torch.sqrt(
            (w_tilde * w_tilde).sum(dim=-1, keepdim=True) + _eps_sq
        )

        # Radial compression: tanh(||w̃||) / (||w̃|| + ε)
        # At w̃ = 0: tanh(r)/r → 1, so ratio → 1/(0 + ε) ≈ 1/ε → large.
        # But tanh(r) · r / (r + ε) → 0, so the product is well-defined.
        radial = torch.tanh(w_tilde_norm) / (w_tilde_norm + self.epsilon)

        # w = γ_Δ ||v|| · radial · w̃
        w = self.gamma_delta * v_norm * radial * w_tilde

        return w

    @property
    def w_tilde_bound(self) -> float:
        """Box bound for free parameter: w̃ ∈ [-c̄, c̄]^{n_w}."""
        return self.c_bar

    def coverage_fraction(self) -> float:
        """Fraction of the norm ball covered by w̃ ∈ [-c̄, c̄]^{n_w}.

        h(c̄) = c̄ · tanh(c̄) / (c̄ + ε) ≈ tanh(c̄) for large c̄.
        """
        r = self.c_bar
        return r * float(torch.tanh(torch.tensor(r))) / (r + self.epsilon)


class SectorBoundTransform(nn.Module):
    """
    Bilinear sector-bound uncertainty: wp = α · u_tilde · w_tilde.

    Models input-multiplicative uncertainty |δ(u)| ≤ α|u| via simple
    bilinear parameterization with w_tilde ∈ [-1, 1]:

        wp = α · u_tilde · w_tilde

    This covers 100% of the sector {|wp| ≤ α|u|} without tanh or abs,
    making it highly CROWN-friendly (BoundMul with McCormick envelopes).
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    @property
    def w_tilde_bound(self) -> float:
        return 1.0

    def forward(
        self,
        u_tilde: torch.Tensor,
        w_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map free parameter w̃ ∈ [-1, 1] to sector-bounded disturbance.

        Args:
            u_tilde: Controller output ũ (batch, nu).
            w_tilde: Free parameter w̃ (batch, nw), in [-1, 1].

        Returns:
            wp: Disturbance (batch, nw), satisfying |wp| ≤ α|ũ|.
        """
        return self.alpha * u_tilde * w_tilde

    def coverage_fraction(self) -> float:
        return 1.0


class DiskMarginTransform(nn.Module):
    """
    Disk margin uncertainty transform D(α, σ).

    Models input-multiplicative uncertainty at the plant input (Fig. 1 of
    Junnarkar, Arcak & Seiler 2024).  With controller output ũ:

        vp = ũ + ((1+σ)/2)·wp       (input to uncertainty block)
        wp = Δp(vp),  ‖Δp‖ < α      (uncertainty output)
        u_actual = ũ + wp            (actual plant input)

    The IQC constraint |wp| ≤ α|vp| = α|ũ + β·wp| (β = (1+σ)/2) yields
    a state-dependent interval for wp:

        wp ∈ [c(ũ) − r(|ũ|),  c(ũ) + r(|ũ|)]

    where (D = 1 − (αβ)²):
        c(ũ)  = (α²β / D) · ũ           centre (signed, shifts with ũ)
        r(|ũ|) = (α / D) · |ũ|            radius (always ≥ 0)

    Smooth parametrization (CROWN-friendly):
        wp = c(ũ) + r(|ũ|) · tanh(w̃)     w̃ ∈ [−c̄, c̄]

    For σ = 0 (default, centred disk):
        c = α²·ũ / (2 − α²/2)
        r = α·|ũ| / (1 − α²/4)

    The disk margin D(α, 0) with α = 0.353 implies ≥3 dB gain margin
    and ≥20° phase margin (simultaneous).
    """

    def __init__(self, alpha: float, sigma: float = 0.0, c_bar: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.c_bar = c_bar

        beta = (1.0 + sigma) / 2.0
        ab = alpha * beta
        if ab >= 1.0:
            raise ValueError(
                f"Need α·(1+σ)/2 < 1 for well-posedness, got {ab:.4f}"
            )

        D = 1.0 - ab * ab
        self._c_coeff = alpha * alpha * beta / D
        self._r_coeff = alpha / D

    @property
    def w_tilde_bound(self) -> float:
        return self.c_bar

    def forward(
        self,
        u_tilde: torch.Tensor,
        w_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map free parameter w̃ to disk-margin-feasible disturbance wp.

        Args:
            u_tilde: Controller output ũ (batch, nu).
            w_tilde: Free parameter w̃ (batch, nw), in [−c̄, c̄].

        Returns:
            wp: Disk margin disturbance (batch, nw), satisfying the IQC.
        """
        c = self._c_coeff * u_tilde
        r = self._r_coeff * torch.abs(u_tilde)
        return c + r * torch.tanh(w_tilde)

    def coverage_fraction(self) -> float:
        """Fraction of the feasible interval covered by w̃ ∈ [−c̄, c̄]."""
        import math
        return math.tanh(self.c_bar)


def disk_margin_iqc_z_fn(sigma: float = 0.0):
    """Return an IQC z-function for the disk margin D(α, σ).

    z = (v, w) where v = ũ + β·w,  β = (1+σ)/2.
    For σ = 0: v = ũ + w/2.

    The quadratic constraint is z^T M z ≥ 0 with M = [[α², 0], [0, -1]].
    """
    beta = (1.0 + sigma) / 2.0

    def z_fn(x: torch.Tensor, u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        v = u + beta * w  # IQC input: ũ + β·w
        return torch.cat([v, w], dim=-1)

    return z_fn


def disk_margin_iqc_M(alpha: float) -> torch.Tensor:
    """IQC multiplier matrix M for disk margin D(α, σ).

    M = diag(α², -1) for scalar (1D) uncertainty.
    The IQC is: z^T M z = α² v² - w² ≥ 0  ⟺  |w| ≤ α|v|.

    For multi-dimensional: M = diag(α² I_{nu}, -I_{nw}).
    """
    return torch.tensor([[alpha**2, 0.0], [0.0, -1.0]], dtype=torch.float32)


class BoxDisturbanceTransform(nn.Module):
    """
    Simple box-bounded disturbance (no state-dependent scaling).

    w = w̃  where w̃ ∈ [-w_max, w_max]^{n_w}.

    This is the identity transform — used when uncertainty is modeled as
    a fixed bounded disturbance rather than a norm-ball operator.
    """

    def __init__(self, w_max: float, n_w: int = 1):
        super().__init__()
        self.w_max_val = w_max
        self.n_w = n_w

    def forward(
        self,
        v: Optional[torch.Tensor],
        w_tilde: torch.Tensor,
    ) -> torch.Tensor:
        """Identity: w = w̃ (already bounded by PGD box constraints)."""
        return w_tilde

    @property
    def w_tilde_bound(self) -> float:
        return self.w_max_val
