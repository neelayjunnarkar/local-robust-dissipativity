import math
import typing
from typing import Optional, Union

import torch.nn as nn
import torch
import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.supply_rate as supply_rate_module


def soft_max(x: torch.Tensor, beta: float = 100):
    x_max = torch.max(x, dim=-1, keepdim=True).values
    eq = torch.exp(beta * (x - x_max))
    if torch.any(torch.isnan(eq)):
        raise Exception("soft_max contains NAN, consider to reduce beta.")
    ret = torch.sum(eq / torch.sum(eq, dim=-1, keepdim=True) * x, dim=-1, keepdim=True)
    return ret



def soft_min(x: torch.Tensor, beta: float = 100):
    return -soft_max(-x, beta)


class NeuralNetworkLyapunov(nn.Module):
    """
    V(x) = V_nominal(x) + network_output(x) + V_psd_output(x)
    V_nominal(x) contains NO optimizable parameters.
    network_output =
    ϕ(x) − ϕ(x*) if absolute_output = False
    |ϕ(x) − ϕ(x*)| if absolute_output = True
    V_psd_output =
    |(εI+RᵀR)(x-x*)|₁ if V_psd_form = "L1"
    (x-x*)ᵀ(εI+RᵀR)(x-x*) if V_psd_form = "quadratic".
    |R(x-x*)|₁ if V_psd_form = "L1_R_free"

    Pure-NN forms (no quadratic base):
    sigmoid(NN(x))              if V_psd_form = "nn_sigmoid"
    α · |NN(x) - NN(x*)|       if V_psd_form = "nn_abs"
    α · (NN(x) - NN(x*))²      if V_psd_form = "nn_sq"

    The optimizable parameters are the network ϕ and R.
    """

    # All V_psd_form values that are "pure NN" (no quadratic base).
    _PURE_NN_FORMS = {"nn_sigmoid", "nn_abs", "nn_sq", "nn_sigmoid_c1", "nn_sigmoid_abs"}

    def __init__(
        self,
        goal_state: torch.Tensor,
        hidden_widths: list,
        x_dim: int,
        R_frozen: torch.Tensor = None,
        R_trainable: torch.Tensor = None,
        absolute_output: bool = True,
        eps: float = 0.01,
        activation: nn.Module = nn.LeakyReLU,
        nominal: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None,
        V_psd_form: str = "L1",
        use_nonlinear: bool = True,
        nn_scale: float = 0.5,
        learnable_nn_scale: bool = False,
        *args,
        **kwargs
    ):
        """
        Args:
          goal_state: The target equilibrium state.
          hidden_widths: Widths of hidden layers for NN component.
          x_dim: Dimension of state.
          R_frozen: Frozen matrix for quadratic term (buffer).
          R_trainable: Additional trainable matrix for quadratic term (parameter).
          absolute_output: V(x) = |phi| + quadratic if True, else phi + quadratic.
          eps: Minimum eigenvalue for PSD term.
          activation: Activation function for NN.
          nominal: Fixed nominal Lyapunov component.
          V_psd_form: "quadratic", "L1", "L1_R_free",
            "quadratic_times_tanh" (V = x'Px · (1+α·tanh(NN)), α=nn_scale), or
            "quadratic_times_exp"  (V = x'Px · exp(NN)).
            Any "quadratic_times_*" form: NN zero-init ⇒ multiplier=1 at init.
            "quadratic_plus_sq"    (V = x'Px + α·(NN(x)−NN(0))²).  CROWN-friendly:
              additive (BoundAdd), no cross-product term.  Use tanh hidden layers
              with a linear output layer for smooth + tight CROWN bounds.
            "quadratic_plus_abs"   (V = x'Px + α·|NN(x)−NN(0)|).  CROWN-friendly:
              uses torch.abs() → auto_LiRPA BoundAbs (purpose-built, tight bounds).
              Allows signed NN contribution (linear growth in φ, not squared).
            "nn_sigmoid"  (V = sigmoid(NN(x))).  Pure-NN, no quadratic base.
              Inspired by Li et al. (2025).  V ∈ (0,1), V(x*) ≈ 0 (minimized).
              CROWN: BoundSigmoid (subclass of BoundTanh).
            "nn_sigmoid_c1"  (V = sigmoid(NN(x))).  Same as nn_sigmoid but
              designed for c₁-exclusion training (two-stage paper approach).
              Training excludes region V(x) < c₁ from decrease condition.
            "nn_sigmoid_abs"  (V = α·|σ(NN(x))−σ(NN(x*))|).  Pure-NN.
              Structural V(x*)=0 via abs of sigmoid difference.
              Bounded in [0, α], smooth sigmoid + abs.
            "nn_abs"  (V = α·|NN(x)−NN(x*)|).  Pure-NN, no quadratic base.
              V(x*)=0 exactly, V≥0.  CROWN: BoundAbs (tight, purpose-built).
            "nn_sq"   (V = α·(NN(x)−NN(x*))²).  Pure-NN, no quadratic base.
              V(x*)=0 exactly, V≥0, smooth.  CROWN: BoundSqr (optimizable).
            "quadratic_plus_sigmoid"  (V = x'Px + α·|σ(NN(x))−σ(NN(x*))|).
              Quadratic base + bounded sigmoid NN term.
              V(x*)=0 structurally.  Quadratic handles near-origin,
              sigmoid adds bounded flexibility far from origin.
          use_nonlinear: Whether to include the NN component.
          nn_scale: α for "quadratic_times_tanh" — multiplier ∈ (1-α, 1+α).
            Must be in (0,1) for that form. Also used as α in "quadratic_plus_sq",
            "quadratic_plus_abs", "nn_abs", "nn_sq".
            Ignored for "quadratic_times_exp" and "nn_sigmoid".
        """
        super().__init__(*args, **kwargs)
        self.goal_state = goal_state
        self.x_dim = x_dim
        self.absolute_output = absolute_output
        self.V_psd_form = V_psd_form
        self.eps = eps
        self.use_nonlinear = use_nonlinear
        if V_psd_form == "quadratic_times_tanh" and use_nonlinear:
            assert 0 < nn_scale < 1, f"nn_scale must be in (0,1) for quadratic_times_tanh with nonlinear, got {nn_scale}"
        self._learnable_nn_scale = learnable_nn_scale
        if learnable_nn_scale:
            self._log_nn_scale = nn.Parameter(torch.log(torch.tensor(float(nn_scale))))
        else:
            self._nn_scale_fixed = float(nn_scale)
        self._is_pure_nn = V_psd_form in self._PURE_NN_FORMS
        # quadratic_plus_sigmoid has R matrices despite using sigmoid NN
        self._has_quadratic_base = (not self._is_pure_nn) or V_psd_form == "quadratic_plus_sigmoid"
        
        # Linear/Quadratic components (R matrices) — skipped for pure-NN forms (except quadratic_plus_sigmoid)
        if self._has_quadratic_base:
            if R_frozen is not None:
                self.register_buffer('R_frozen', R_frozen.clone())
            else:
                self.R_frozen = None
                
            if R_trainable is not None:
                self.register_parameter('R_trainable', nn.Parameter(R_trainable.clone()))
            else:
                self.R_trainable = None
        else:
            self.R_frozen = None
            self.R_trainable = None
            
        # Neural Network component
        if use_nonlinear and hidden_widths is not None:
            layers = [nn.Linear(self.x_dim, 1 if len(hidden_widths) == 0 else hidden_widths[0])]
            for i, width in enumerate(hidden_widths):
                layers.append(activation())
                layers.append(nn.Linear(width, hidden_widths[i+1] if i != len(hidden_widths)-1 else 1))
            
            for l in layers:
                if isinstance(l, nn.Linear):
                    torch.nn.init.kaiming_uniform_(l.weight, nonlinearity="relu")
            
            # Zero-init the last linear layer for forms that require V = x^T P x at init.
            # quadratic_times_*: NN(x)=0 → multiplier=1 → V = x^T P x exactly.
            # quadratic_plus_sq:  NN(x)=0 → residual=0  → V = x^T P x exactly.
            if (V_psd_form.startswith("quadratic_times_")
                    or V_psd_form in ("quadratic_plus_sq", "quadratic_plus_abs")):
                last_linear = layers[-1]  # last element is always Linear
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

            # nn_sigmoid / nn_sigmoid_c1: initialise so NN(x) ≈ −5 everywhere
            # → V ≈ sigmoid(−5) ≈ 0.007.  Training shapes the level sets.
            if V_psd_form in ("nn_sigmoid", "nn_sigmoid_c1"):
                last_linear = layers[-1]
                nn.init.zeros_(last_linear.weight)
                last_linear.bias.data.fill_(-5.0)

            # nn_sigmoid_abs & quadratic_plus_sigmoid: keep small random init
            # for last layer so |σ(NN(x)) - σ(NN(x*))| ≈ small nonzero
            # → gradient flows through abs().  Zero-init gives dead gradient
            # since ∂|z|/∂z = sign(0) = 0 when z = |σ(0) - σ(0)| = 0.
            # std=0.01 keeps V ≈ pure quadratic base at init while allowing learning.
            if V_psd_form == "quadratic_plus_sigmoid":
                last_linear = layers[-1]
                nn.init.normal_(last_linear.weight, std=0.01)
                nn.init.normal_(last_linear.bias, std=0.01)

            self.net = nn.Sequential(*layers)
        else:
            self.net = None
            
        self.nominal = nominal

        # Cholesky factor of P = eps*I + R_frozen^T R_frozen + R_trainable^T R_trainable.
        # When set, forward() uses the CROWN-friendly form: y = x @ L; V = sum(y*y).
        # Precomputed via precompute_cholesky() before verification.
        self.register_buffer('L_chol', None)

    @property
    def nn_scale(self):
        if self._learnable_nn_scale:
            raw = torch.exp(self._log_nn_scale)
            # For quadratic_times_tanh, α must stay in (0,1) for positivity:
            # V = x'Px · (1 + α·tanh(NN)) > 0  requires α < 1.
            if self.V_psd_form == "quadratic_times_tanh":
                raw = torch.clamp(raw, max=0.999)
            return raw
        return self._nn_scale_fixed

    def get_nn_scale_value(self) -> float:
        if self._learnable_nn_scale:
            val = float(self._log_nn_scale.exp().item())
            if self.V_psd_form == "quadratic_times_tanh":
                val = min(val, 0.999)
            return val
        return self._nn_scale_fixed

    def _network_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_nonlinear and self.net is not None and len(self.net) > 0:
            phi = self.net(x)
            phi_star = self.net(self.goal_state)
            return phi - phi_star
        else:
            return torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)

    def _V_psd_output(self, x: torch.Tensor):
        # Pure-NN forms have no quadratic/L1 base term (except quadratic_plus_sigmoid).
        if self._is_pure_nn and self.V_psd_form != "quadratic_plus_sigmoid":
            return torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)

        x_diff = x - self.goal_state

        _is_quad = (self.V_psd_form == "quadratic"
                    or self.V_psd_form.startswith("quadratic_times_")
                    or self.V_psd_form in ("quadratic_plus_sq", "quadratic_plus_abs",
                                           "quadratic_plus_sigmoid"))

        # CROWN-friendly Cholesky form: V_quad = ||x @ L||^2 = sum(y * y)
        # Uses a single BoundLinear + BoundMul chain → tighter CROWN bounds
        # than the split R_frozen / R_trainable / eps*I computation.
        if _is_quad and self.L_chol is not None:
            y = x_diff @ self.L_chol.to(x.device)
            return torch.sum(torch.mul(y, y), dim=-1, keepdim=True)

        # Fallback: separate R-based computation (used during training
        # before precompute_cholesky() is called).
        v_psd = torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)

        if self.eps > 0 and _is_quad:
            v_psd = v_psd + self.eps * torch.sum(x_diff ** 2, dim=-1, keepdim=True)

        def compute_term(R_mat, form):
            if R_mat is None: return 0
            if (form == "quadratic" or form.startswith("quadratic_times_")
                    or form in ("quadratic_plus_sq", "quadratic_plus_abs",
                                "quadratic_plus_sigmoid")):
                return torch.sum((x_diff @ R_mat.T) ** 2, dim=-1, keepdim=True)
            elif form == "L1":
                Rx = x_diff @ R_mat.T
                return (torch.nn.functional.relu(Rx) + torch.nn.functional.relu(-Rx)).sum(dim=-1, keepdim=True)
            return 0

        v_psd = v_psd + compute_term(self.R_frozen, self.V_psd_form)
        v_psd = v_psd + compute_term(self.R_trainable, self.V_psd_form)

        return v_psd

    def forward(self, x):
        V_nominal = 0 if self.nominal is None else self.nominal(x)

        # --- Pure-NN forms: the NN alone defines V(x) ---
        if self.V_psd_form == "nn_sigmoid":
            # V(x) = sigmoid(NN(x)).  Always in (0,1).
            # V(x*) ≈ 0 (learned during training; init bias = −5).
            raw = self.net(x)
            return V_nominal + torch.sigmoid(raw)

        if self.V_psd_form == "nn_sigmoid_c1":
            # Same as nn_sigmoid but designed for c₁-exclusion training.
            raw = self.net(x)
            return V_nominal + torch.sigmoid(raw)

        if self.V_psd_form == "nn_sigmoid_abs":
            # V(x) = α · |σ(NN(x)) − σ(NN(x*))|
            # Structural V(x*)=0 via absolute sigmoid difference.
            raw = self.net(x)
            raw_star = self.net(self.goal_state)
            return V_nominal + self.nn_scale * torch.abs(
                torch.sigmoid(raw) - torch.sigmoid(raw_star)
            )

        if self.V_psd_form == "nn_abs":
            # V(x) = α · |NN(x) − NN(x*)|.  V(x*)=0 exactly, V≥0.
            network_output = self._network_output(x)
            return V_nominal + self.nn_scale * torch.abs(network_output)

        if self.V_psd_form == "nn_sq":
            # V(x) = α · (NN(x) − NN(x*))².  V(x*)=0 exactly, V≥0, smooth.
            network_output = self._network_output(x)
            return V_nominal + self.nn_scale * network_output ** 2

        # --- Quadratic-based forms ---
        network_output = self._network_output(x)
        V_psd_output = self._V_psd_output(x)

        if self.V_psd_form.startswith("quadratic_times_"):
            if self.net is None:
                # use_nonlinear=False: pure quadratic, no NN in the graph
                return V_nominal + V_psd_output
            else:
                return V_nominal + V_psd_output * self._compute_multiplier(network_output)
        elif self.V_psd_form == "quadratic_plus_sq":
            # V = x^T P x + α * (φ(x) − φ(0))²
            # network_output is already φ(x) − φ(0) (zero-mean at init).
            # Squaring is always ≥ 0, so V ≥ x^T P x.  No BoundMul cross-term
            # between the quadratic and the NN → CROWN bounds stay tight.
            sq_residual = network_output ** 2  # [batch, 1]
            return V_nominal + V_psd_output + self.nn_scale * sq_residual
        elif self.V_psd_form == "quadratic_plus_abs":
            # V = x^T P x + α·|φ(x) − φ(0)|
            # Uses torch.abs() so auto_LiRPA maps to BoundAbs for tight CROWN bounds.
            # Always ≥ 0, so V ≥ x^T P x.
            abs_residual = torch.abs(network_output)
            return V_nominal + V_psd_output + self.nn_scale * abs_residual
        elif self.V_psd_form == "quadratic_plus_sigmoid":
            # V = x'Px + α·|σ(NN(x)) − σ(NN(x*))|
            # Quadratic base handles near-origin; sigmoid NN adds bounded flexibility.
            # V(x*)=0 structurally.  σ difference bounded in (-1,1), abs in [0,1).
            if self.net is None:
                # use_nonlinear=False: pure quadratic, no NN in the graph
                return V_nominal + V_psd_output
            raw = self.net(x)
            raw_star = self.net(self.goal_state)
            sig_diff = torch.abs(torch.sigmoid(raw) - torch.sigmoid(raw_star))
            return V_nominal + V_psd_output + self.nn_scale * sig_diff
        elif self.absolute_output:
            return (
                V_nominal
                + torch.nn.functional.relu(network_output)
                + torch.nn.functional.relu(-network_output)
                + V_psd_output
            )
        else:
            return V_nominal + network_output + V_psd_output

    def _compute_multiplier(self, network_output: torch.Tensor) -> torch.Tensor:
        """Multiplier applied to the quadratic base V = x'Px.

        Dispatches on the suffix of V_psd_form:
          quadratic_times_tanh  → 1 + α·tanh(NN(x) - NN(0))  ∈ (1-α, 1+α) > 0
          quadratic_times_exp   → exp(NN(x) - NN(0))          always > 0
        Both equal 1 at initialisation (zero-init last layer → NN(x)=0).
        """
        suffix = self.V_psd_form[len("quadratic_times_"):]
        if suffix == "tanh":
            return 1.0 + self.nn_scale * torch.tanh(network_output)
        elif suffix == "exp":
            return torch.exp(network_output)
        else:
            raise ValueError(f"Unknown quadratic_times_* suffix: '{suffix}'")

    def precompute_cholesky(self):
        """Precompute Cholesky factor L such that x^T P x = ||Lx||^2 = sum(y*y).

        Assembles P = eps*I + R_frozen^T R_frozen + R_trainable^T R_trainable,
        verifies symmetry, and stores L = cholesky(P).  After calling this,
        forward() uses the compact form ``y = x @ L; sum(y * y)`` which gives
        tighter CROWN bounds (single BoundLinear + BoundMul chain instead of
        multiple separate quadratic terms).

        Call this before formal verification (after training is done).
        """
        if not self._has_quadratic_base:
            return

        device = self.goal_state.device
        P = self.eps * torch.eye(self.x_dim, device=device)
        if self.R_frozen is not None:
            P = P + self.R_frozen.T @ self.R_frozen
        if self.R_trainable is not None:
            P = P + self.R_trainable.T @ self.R_trainable

        # Symmetrise (numerical safety)
        P = (P + P.T) / 2

        if not torch.allclose(P, P.T, atol=1e-6):
            raise ValueError("Assembled P matrix is not symmetric")

        try:
            self.L_chol = torch.linalg.cholesky(P).detach()
        except torch.linalg.LinAlgError as e:
            raise ValueError(f"Cholesky decomposition of P failed: {e}")

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.goal_state = fn(self.goal_state)
        return self


class NeuralNetworkQuadraticLyapunov(nn.Module):
    """
    A quadratic Lyapunov function.
    This neural network output is
    V(x) = (x-x*)^T(εI+RᵀR)(x-x*),
    R is the parameters to be optimized.
    """

    def __init__(
        self,
        goal_state: torch.Tensor,
        x_dim: int,
        R_rows: int,
        eps: float,
        R: typing.Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """
        Args:
          x_dim: The dimension of state
          R_rows: The number of rows in matrix R.
          V(x) = (x-x*)^T(εI+RᵀR)(x-x*)
        """
        super().__init__(*args, **kwargs)
        self.goal_state = goal_state
        self.x_dim = x_dim
        assert self.goal_state.shape == (self.x_dim,)
        assert isinstance(eps, float)
        self.R_rows = R_rows
        assert eps >= 0
        self.eps = eps
        # Rt is the transpose of R
        if R is None:
            R = torch.rand((R_rows, self.x_dim)) - 0.5

        self.register_parameter(name="R", param=torch.nn.Parameter(R))

    def forward(self, x):
        x0 = x - self.goal_state
        Q = self.eps * torch.eye(self.x_dim, device=x.device) + (
            self.R.transpose(0, 1) @ self.R
        )
        return torch.sum(x0 * (x0 @ Q), axis=-1, keepdim=True)

    def dVdx(self, x):
        Q = self.eps * torch.eye(self.x_dim, device=x.device) + (
            self.R.transpose(0, 1) @ self.R
        )
        dVdx = 2 * x @ Q
        return dVdx

    def diff(self, x, x_next, kappa, lyapunov_x):
        # V(x) = (x_t - x_*)^T Q (x_t - x_*)
        # V(x_next) = (x_next - x_*)^T Q (x_next - x_*)
        # dV = (x_next - x_*)^T Q (x_next - x_*) - (1-kappa) (x_t - x_*)^T Q (x_t - x_*)
        #    = x_next^T Q x_next
        #        - (1-kappa) x_t^T Q x_t
        #        - 2 (x_next - (1-kappa) x_t)^T Q x_*
        #        + kappa * x_*^T Q x_*
        #    = (x_next - sqrt(1-kappa) x_t)^T Q (x_next + sqrt(1-kappa) x_t)
        #        - 2 (x_next - (1-kappa)x_t)^T Q x_*
        #        + kappa * x_*^T Q x_*
        sqrt_1_minus_kappa = math.sqrt(1 - kappa)
        x_d1 = x_next - sqrt_1_minus_kappa * x
        if kappa == 0:
            x_d2 = x_d1
        else:
            x_d2 = x_next - (1 - kappa) * x
        x_s = x_next + sqrt_1_minus_kappa * x
        Q = (
            self.eps * torch.eye(self.x_dim, device=x.device)
            + (self.R.transpose(0, 1) @ self.R)
        )
        dV = (
            torch.sum(x_d1 * (x_s @ Q), axis=-1, keepdim=True)
            - 2 * torch.sum(x_d2 * (self.goal_state @ Q), axis=-1, keepdim=True)
            + kappa * torch.sum(self.goal_state * (self.goal_state @ Q), axis=-1, keepdim=True)
        )
        return dV

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.goal_state = fn(self.goal_state)
        return self


class LyapunovPositivityLoss(nn.Module):
    """
    Compute the loss V(x) - |N(x−x*)|₁
    where N is a given matrix.
    """

    def __init__(
        self, lyapunov: NeuralNetworkLyapunov, Nt: torch.Tensor, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.lyapunov = lyapunov
        assert isinstance(Nt, torch.Tensor)
        assert Nt.shape[0] == self.lyapunov.x_dim
        self.Nt = Nt

    def forward(self, x):
        Nx = (x - self.lyapunov.goal_state) @ self.Nt
        # l1_term = (torch.nn.functional.relu(Nx) + torch.nn.functional.relu(-Nx)).sum(dim=1, keepdim=True)
        l1_term = torch.abs((x - self.lyapunov.goal_state) @ self.Nt).sum(
            dim=-1, keepdim=True
        )
        V = self.lyapunov(x)
        return V - l1_term

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.Nt = fn(self.Nt)


class LyapunovDerivativeSimpleLoss(nn.Module):
    """
    Require the Lyapunov function to always decrease, namely
    V(x_next) - V(x) <= -κ * V(x).

    We want to minimize
    V(x_next) - (1-κ)V(x)                (1)

    Since alpha-beta-crown verifies the quantity being non-negative rather than minimizing the loss, we compute the negation of (1) as
    (1-κ)V(x) - V(x_next)
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkLyapunov,
        kappa: float = 0.1,
        fuse_dV: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyap_nn
        self.kappa = kappa
        self.fuse_dV = fuse_dV

    def forward(self, x, save_new_x=False):
        # Run the system by one step with dt.
        u = self.controller(x)
        new_x = self.dynamics.forward(x, u)
        if save_new_x:
            self.new_x = new_x
        lyapunov_x = self.lyapunov(x)
        self.last_lyapunov_x = lyapunov_x.detach()

        # The following two formulations are equivalent.
        # When self.fuse_dV is enabled, we fuse x and x_next (new_x)
        # before entering the quadratic term in the Lyapunov function to compute
        # tighter bounds on dV, compared to computing two Lyapunov function
        # values separately.
        if self.fuse_dV:
            assert isinstance(self.lyapunov, NeuralNetworkQuadraticLyapunov)
            dV = self.lyapunov.diff(x, new_x, self.kappa, lyapunov_x)
            loss = -dV
        else:
            loss = (1 - self.kappa) * lyapunov_x - self.lyapunov(new_x)

        return loss


class LyapunovDerivativeLoss(nn.Module):
    """
    For a box region B = [x_lo, x_up], we want enforce the condition that the
    set S={x in B | V(x)<=rho} is an invariant set, and V
    decreases within S.
    Namely we want the following conditions to hold:
    1. V(x)<= rho => x_next in B.
    2. V(x)<= rho => V(x_next) - V(x) <= -κ * V(x).
    This is equivalent to the following conditions:
    ((x_next in B) ∧ (V(x_next) - V(x) <= -κ * V(x))) ∨ (V(x) > rho)
    where "∨" is "logical or", and "∧" is "logical and".

    We can hence impose the loss
    min (weight[0]*(rho - V(x)),
         weight[1] * ReLU(V(x_next) - (1-κ)V(x))+
             weight[2] * (∑ᵢReLU((x_lo[i] - x_next[i]) + ReLU(x_next[i] - x_up[i]))))

    Since alpha-beta-crown wants to verify the loss being non-negative instead
    of minimizing the loss, we return the negation of the loss
    -min (weight[0] * (rho - V(x)),
         weight[1] * ReLU(V(x_next) - (1-κ)V(x)) +
             weight[2] * (∑ᵢReLU((x_lo[i] - x_next[i]) + ReLU(x_next[i] - x_up[i]))))
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkLyapunov,
        box_lo: torch.Tensor,
        box_up: torch.Tensor,
        rho_multiplier: float,
        kappa: float = 0.1,
        beta: float = 100,
        hard_max: bool = True,
        loss_weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """
        Args:
            rho_multiplier: We use rho = rho_multiplier * min V(x_boundary)
            beta: the coefficient in soft max exponential. beta -> infinity
              recovers hard max.
            loss_weights: weight in the documentation above. Should all be non-negative.
        """
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyap_nn
        self.rho_multiplier = rho_multiplier
        self.box_lo = box_lo
        self.box_up = box_up
        self.kappa = kappa
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max
        if loss_weights is None:
            self.loss_weights = torch.tensor([1.0, 1.0, 1.0])
        else:
            assert loss_weights.shape == (3,)
            assert torch.all(loss_weights > 0)
            self.loss_weights = loss_weights

    def get_rho(self):
        rho_boundary = self.lyapunov(self.x_boundary).min()
        rho = self.rho_multiplier * rho_boundary
        return rho

    def forward(self, x: torch.Tensor, save_new_x: bool = False):
        # Run the system by one step with dt.
        u = self.controller(x)
        new_x = self.dynamics.forward(x, u)
        if save_new_x:
            self.new_x = new_x
        lyapunov_x = self.lyapunov(x)
        rho = self.get_rho()
        loss1 = self.loss_weights[0] * (rho - lyapunov_x)
        loss2 = torch.nn.functional.relu(
            self.loss_weights[1]
            * (self.lyapunov(new_x) - (1 - self.kappa) * lyapunov_x)
        )
        loss3 = self.loss_weights[2] * (
            torch.nn.functional.relu(self.box_lo - new_x).sum(dim=1, keepdim=True)
            + torch.nn.functional.relu(new_x - self.box_up).sum(dim=1, keepdim=True)
        )
        loss23 = loss2 + loss3
        if self.hard_max:
            loss = torch.min(
                torch.cat((loss1, loss23), dim=-1),
                dim=-1,
                keepdim=True,
            ).values
            return -loss
        else:
            loss = soft_min(torch.cat((loss1, loss23), dim=-1), self.beta)
            return -loss


class LyapunovDerivativeSimpleLossWithV(LyapunovDerivativeSimpleLoss):
    """
    The same as LyapunovDerivativeSimpleLoss, but with V(x) as the second output.
    Used for verification with level set.
    TODO: make it a template class to also support LyapunovDerivativeDOFLoss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not hasattr(self, "x_boundary") or self.x_boundary is None

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        # Output should be [N, 2] where N is the batch size.
        return torch.cat((loss, self.last_lyapunov_x), dim=1)


class LyapunovDerivativeSimpleLossWithVBox(LyapunovDerivativeSimpleLossWithV):
    """
    Additionally, output x_next for checking if x_next is within the bounding box.
    """

    def forward(self, *args, **kwargs):
        loss_and_V = super().forward(*args, save_new_x=True, **kwargs)
        return torch.cat((loss_and_V, self.new_x), dim=1)


# =============================================================================
# Dissipativity-Based Loss Functions (Generalization of Lyapunov)
# =============================================================================

class DissipativityDerivativeLoss(nn.Module):
    """Dissipativity-based derivative loss with ROA constraint.

    Condition (must be ≥ 0 everywhere in the verification domain):
        max{V(x) − ρ,  min{V(x) − V(x⁺) + s·supply,  ρ − V(x⁺)}} ≥ 0

    With IQC multiplier (Theorem 1 of main_v3.tex):
        Robust forward invariance:  V(x⁺) − V(x) + z^T M_rfi z ≤ 0
        Supply rate condition:      V(x⁺) − V(x) + z^T M_sr  z ≤ s(d,e)
    where z = (v, w), v is the IQC input (e.g. ũ + w/2 for disk margin),
    and λ > 0 scales M.

    Verification modes (set via ``verification_mode``):
        'combined'      — max{A, min{B, C}}, used for training (default)
        'dissipativity' — max{A, B}, single output for CROWN IBP pruning
                          (with IQC: B = V−V⁺−λ zᵀMz + s_scale·s; use this for IQC CROWN)
        'invariance'    — [C, V(x)], 2-output level-set for CROWN VNNLIB
        'combined_tight'— max{A, clamp(V+s, ρ) − V⁺}, 3 ReLUs
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkLyapunov,
        supply_rate: supply_rate_module.SupplyRate,
        box_lo: torch.Tensor = None,
        box_up: torch.Tensor = None,
        rho_multiplier: float = 1.0,
        w_max: Optional[torch.Tensor] = None,
        beta: float = 100,
        hard_max: bool = True,
        loss_weights: Optional[torch.Tensor] = None,
        s_scale: float = 1.0,
        learnable_s_scale: bool = False,
        relu_minmax: bool = False,
        verification_mode: str = 'combined',
        c1_threshold: float = 0.0,
        *args,
        **kwargs
    ):
        # Pop legacy kwargs for backward compatibility
        relu_minmax = kwargs.pop('relu_minmax', relu_minmax)
        verification_mode = kwargs.pop('verification_mode', verification_mode)
        c1_multiplier = kwargs.pop('c1_multiplier', 0.0)
        # Uncertainty transform (e.g., tanh norm-ball): maps (v, w̃) → w
        uncertainty_transform = kwargs.pop('uncertainty_transform', None)
        # IQC multiplier matrix M and learnable scale λ (Theorem 1)
        iqc_M = kwargs.pop('iqc_M', None)
        iqc_lambda_init = kwargs.pop('iqc_lambda_init', 1.0)
        learnable_iqc_lambda = kwargs.pop('learnable_iqc_lambda', True)
        # Function mapping (x, u, w) → z for IQC evaluation z^T M z
        iqc_z_fn = kwargs.pop('iqc_z_fn', None)
        super().__init__(*args, **kwargs)

        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyap_nn
        self.supply_rate = supply_rate
        self.verification_mode = verification_mode
        self.box_lo = box_lo
        self.box_up = box_up
        self.rho_multiplier = rho_multiplier
        self.beta = beta
        self.hard_max = hard_max
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.nx = dynamics.nx
        self.nw = (getattr(dynamics.continuous_time_system, 'nw', 1)
                   if hasattr(dynamics, 'continuous_time_system') else 1)

        self.uncertainty_transform = uncertainty_transform

        if w_max is not None:
            self.w_max = torch.tensor([w_max]) if isinstance(w_max, (int, float)) else w_max
        else:
            self.w_max = None

        if loss_weights is None:
            self.loss_weights = torch.tensor([1.0, 1.0, 1.0])
        else:
            assert loss_weights.shape == (3,)
            self.loss_weights = loss_weights

        # Supply-rate scaling (log-parameterised if learnable)
        self._s_scale_learnable = learnable_s_scale
        if learnable_s_scale:
            self._log_s_scale = nn.Parameter(torch.log(torch.tensor(float(s_scale))))
        else:
            self._s_scale_fixed = float(s_scale)

        # ReLU-based min/max avoids BoundReduceMin/Max crash in auto_LiRPA
        self.relu_minmax = relu_minmax
        if self.relu_minmax:
            self._relu = nn.ReLU()

        if supply_rate.requires_disturbance:
            assert w_max is not None, "Supply rate requires disturbance bound w_max"

        # c₁ exclusion zone: points with V(x) < c₁ are trivially safe.
        # This allows V(x*) > 0 forms like nn_sigmoid_c1 to work.
        # If c1_threshold > 0, use it as a fixed value.
        # If c1_multiplier > 0, compute c₁ = c1_multiplier * V(x*) dynamically.
        self.c1_threshold = float(c1_threshold)
        self._c1_multiplier = float(c1_multiplier)

        # IQC multiplier: M matrix and learnable λ > 0
        self._iqc_enabled = iqc_M is not None
        if self._iqc_enabled:
            if isinstance(iqc_M, torch.Tensor):
                self.register_buffer('iqc_M', iqc_M.float())
            else:
                self.register_buffer('iqc_M', torch.tensor(iqc_M, dtype=torch.float32))
            self._learnable_iqc_lambda = learnable_iqc_lambda
            if learnable_iqc_lambda:
                self._log_iqc_lambda = nn.Parameter(
                    torch.log(torch.tensor(float(iqc_lambda_init)))
                )
            else:
                self._iqc_lambda_fixed = float(iqc_lambda_init)
            self._iqc_z_fn = iqc_z_fn

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def iqc_lambda(self) -> float:
        if not self._iqc_enabled:
            return 0.0
        if self._learnable_iqc_lambda:
            return torch.exp(self._log_iqc_lambda)
        return self._iqc_lambda_fixed

    def get_iqc_lambda_value(self) -> float:
        if not self._iqc_enabled:
            return 0.0
        if self._learnable_iqc_lambda:
            return float(self._log_iqc_lambda.exp().item())
        return self._iqc_lambda_fixed
    @property
    def s_scale(self):
        if self._s_scale_learnable:
            return torch.exp(self._log_s_scale)
        return self._s_scale_fixed

    def get_s_scale_value(self) -> float:
        if self._s_scale_learnable:
            return float(self._log_s_scale.exp().item())
        return self._s_scale_fixed

    def get_rho(self):
        if hasattr(self, '_fixed_rho') and self._fixed_rho is not None:
            return torch.tensor(self._fixed_rho, dtype=torch.float32)
        if self.x_boundary is None:
            return torch.tensor(0.0)
        return self.rho_multiplier * self.lyapunov(self.x_boundary).min()

    @property
    def kappa(self):
        if isinstance(self.supply_rate, supply_rate_module.LyapunovSupplyRate):
            return self.supply_rate.kappa
        return 0.0

    @property
    def _uses_uncertainty(self):
        """Whether this loss uses uncertainty (w) in dynamics, independent of supply rate."""
        return (self.uncertainty_transform is not None
                or (self.w_max is not None and self.w_max.abs().sum() > 0))

    @property
    def requires_disturbance(self):
        return self.supply_rate.requires_disturbance or self._uses_uncertainty

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _sample_disturbance(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """Sample random disturbance (or free parameter w̃ when using transform)."""
        if not (self.supply_rate.requires_disturbance or self._uses_uncertainty) or self.w_max is None:
            return None
        if self.uncertainty_transform is not None:
            # Sample w̃ in [-c̄, c̄]^nw (transform maps it to w later)
            c_bar = self.uncertainty_transform.w_tilde_bound
            return (torch.rand(batch_size, self.nw, device=device) - 0.5) * 2 * c_bar
        w_max = self.w_max.to(device)
        return (torch.rand(batch_size, self.nw, device=device) - 0.5) * 2 * w_max

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, save_new_x: bool = False):
        # Dynamic c₁ update: c₁ = multiplier × V(x*), clamped below ρ
        if self._c1_multiplier > 0:
            with torch.no_grad():
                gs = self.lyapunov.goal_state
                if gs.dim() == 1:
                    gs = gs.unsqueeze(0)
                V_star = self.lyapunov(gs).item()
                c1_raw = self._c1_multiplier * V_star
                # Clamp c₁ strictly below ρ to prevent vacuous verification.
                # When c₁ ≥ ρ, every point in the ROA trivially satisfies
                # c₁ - V > 0, giving zero gradient signal to training.
                rho = float(self.get_rho())
                if rho > V_star:
                    # Place c₁ at most 30% of the way from V(x*) to ρ
                    c1_max = V_star + 0.3 * (rho - V_star)
                    self.c1_threshold = min(c1_raw, c1_max)
                else:
                    # ρ ≤ V(x*) or ρ=0: keep c₁ to protect equilibrium.
                    # Verification is vacuous but V doesn't collapse.
                    self.c1_threshold = c1_raw

        # Split joint [x, w] input if needed
        if w is None and x.shape[1] == self.nx + self.nw:
            w, x = x[:, self.nx:], x[:, :self.nx]
        if w is None and (self.supply_rate.requires_disturbance or self._uses_uncertainty):
            w = self._sample_disturbance(x.shape[0], x.device)

        # Dynamics step
        u = self.controller(x)

        # Uncertainty transform: w̃ → w = T(v, w̃) where v = u
        if w is not None and self.uncertainty_transform is not None:
            w = self.uncertainty_transform(u, w)

        new_x = self.dynamics.forward(x, u, w)
        if save_new_x:
            self.new_x = new_x

        # Lyapunov values
        V_x = self.lyapunov(x)
        V_next = self.lyapunov(new_x)
        self.last_lyapunov_x = V_x.detach()

        # Supply rate
        z_out = self.dynamics.output(x, u) if self.supply_rate.requires_output else None
        supply = self.supply_rate(w, z_out, V_x)

        iqc_term = self._compute_iqc_term(x, u, w, V_x)
        term1, term2a, term2b = self._compute_core_terms(
            V_x=V_x,
            V_next=V_next,
            supply=supply,
            iqc_term=iqc_term,
        )

        return self._combine(term1, term2a, term2b, V_x)

    def _compute_iqc_term(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        w: Optional[torch.Tensor],
        V_x: torch.Tensor,
    ) -> torch.Tensor:
        """Return λ z^T M z when IQC mode is enabled, else zeros."""
        if not self._iqc_enabled or w is None:
            return torch.zeros_like(V_x)

        z_iqc = self._compute_iqc_z(x, u, w)
        M = self.iqc_M.to(device=z_iqc.device, dtype=z_iqc.dtype)
        Mz = z_iqc @ M
        zMz = (z_iqc * Mz).sum(dim=-1, keepdim=True)
        return self.iqc_lambda * zMz

    def _compute_core_terms(
        self,
        V_x: torch.Tensor,
        V_next: torch.Tensor,
        supply: torch.Tensor,
        iqc_term: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build verification terms for either legacy or IQC mode."""
        c = self.get_rho()
        term1 = V_x - c

        if self._iqc_enabled:
            # Theorem 1 IQC conditions:
            #   V(x+) - V(x) + z^T M z <= s(d, e)
            #   V(x+) - V(x) + z^T M z <= 0
            term2a = V_x - V_next - iqc_term + self.s_scale * supply
            term2b = V_x - V_next - iqc_term
        else:
            # Legacy conditions:
            #   dissipativity: V(x+) - V(x) <= s(d, e)
            #   invariance:    V(x+) <= rho
            term2a = V_x - V_next + self.s_scale * supply
            term2b = c - V_next

        return term1, term2a, term2b

    def _combine(self, term1, term2a, term2b, V_x):
        """Combine terms according to ``self.verification_mode``.

        With c₁ > 0, the verified condition becomes:
            max{V-ρ, min{decrease, invariance}, c₁-V} ≥ 0
        The c₁-V clause makes points with V(x) < c₁ trivially safe
        (the disjunction V ≤ c₁ holds).
        """
        mode = self.verification_mode

        # Optional c₁ exclusion term: c₁ - V > 0 when V < c₁ → safe
        term_c1 = None
        if self.c1_threshold > 0:
            term_c1 = self.c1_threshold - V_x

        if mode == 'invariance':
            return torch.cat([term2b, V_x], dim=-1)

        if mode == 'dissipativity':
            out = self._relu_max(term1, term2a)
            if term_c1 is not None:
                out = self._relu_max(out, term_c1)
            return out

        if mode == 'combined_tight':
            if self.relu_minmax:
                clamped = term2a - self._relu(term2a - term2b)  # min(term2a, term2b)
                out = self._relu_max(term1, clamped)
                if term_c1 is not None:
                    out = self._relu_max(out, term_c1)
                return out
            else:
                clamped = torch.min(torch.cat((term2a, term2b), dim=-1), dim=-1, keepdim=True).values
                terms_outer = [term1, clamped]
                if term_c1 is not None:
                    terms_outer.append(term_c1)
                return torch.max(torch.cat(terms_outer, dim=-1), dim=-1, keepdim=True).values

        # 'combined' mode (training default):
        # max{term1, min{term2a, term2b}[, term_c1]}
        if self.relu_minmax:
            d2 = term2a - term2b
            term2 = 0.5 * (term2a + term2b - self._relu(d2) - self._relu(-d2))
            d1 = term1 - term2
            out = 0.5 * (term1 + term2 + self._relu(d1) + self._relu(-d1))
            if term_c1 is not None:
                out = self._relu_max(out, term_c1)
            return out
        else:
            term2 = torch.min(torch.cat((term2a, term2b), dim=-1), dim=-1, keepdim=True).values
            terms_outer = [term1, term2]
            if term_c1 is not None:
                terms_outer.append(term_c1)
            if self.hard_max:
                return torch.max(torch.cat(terms_outer, dim=-1), dim=-1, keepdim=True).values
            return soft_max(torch.cat(terms_outer, dim=-1), self.beta)

    def _relu_max(self, a, b):
        """Element-wise max(a, b) using ReLU or torch.max."""
        if self.relu_minmax:
            d = a - b
            return 0.5 * (a + b + self._relu(d) + self._relu(-d))
        return torch.max(torch.cat((a, b), dim=-1), dim=-1, keepdim=True).values

    def _compute_iqc_z(self, x, u, w):
        """Compute IQC signal z = (v, w) for z^T M z evaluation.

        Default: v = u (controller output), z = [v; w].
        If a custom iqc_z_fn is provided, use that instead.
        """
        if self._iqc_z_fn is not None:
            return self._iqc_z_fn(x, u, w)
        return torch.cat([u+w/2, w], dim=-1)
        # return torch.cat([u, w], dim=-1)



# =============================================================================
# Verification Wrapper for L2-gain (treats disturbance as adversarial input)
# =============================================================================

class DissipativityVerificationWrapper(nn.Module):
    """
    Wrapper for formal verification of dissipativity conditions.
    
    Treats disturbance w as part of the input for verification:
        Input: [state, disturbance] = [xe, w]
        Output: dissipativity loss (should be ≥ 0)
    
    This allows α-β-CROWN to verify over ALL (state, disturbance) pairs.
    """
    
    def __init__(self, loss_fn, state_dim: int, w_dim: int):
        """
        Args:
            loss_fn: Dissipativity loss function (e.g., DissipativityDerivativeLoss)
            state_dim: Dimension of state (2*nx for output feedback)
            w_dim: Dimension of disturbance
        """
        super().__init__()
        self.loss_fn = loss_fn
        self.state_dim = state_dim
        self.w_dim = w_dim
    
    def forward(self, xew: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for verification.
        
        Args:
            xew: Combined input [state, disturbance], shape (batch, state_dim + w_dim)
            
        Returns:
            Dissipativity loss, shape (batch, 1) or (batch, n_outputs)
        """
        # Split input into state and disturbance
        xe = xew[:, :self.state_dim]
        w = xew[:, self.state_dim:self.state_dim + self.w_dim]
        
        # Call underlying loss with explicit disturbance
        return self.loss_fn(xe, w)
    
    # Delegate attribute access to underlying loss for compatibility
    def __getattr__(self, name):
        if name in ['loss_fn', 'state_dim', 'w_dim', '_modules', '_parameters', '_buffers']:
            return super().__getattr__(name)
        return getattr(self.loss_fn, name)


class LyapunovDerivativeDOFLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute (1-κ)*V(x, e) - V(x_next, e_next)
    V(x, e), e = x - z
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        observer,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkLyapunov,
        box_lo: torch.Tensor,
        box_up: torch.Tensor,
        rho_multiplier: float,
        kappa=0.1,
        beta: float = 100,
        hard_max: bool = True,
        loss_weights: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.observer = observer
        self.controller = controller
        self.lyapunov = lyap_nn
        self.rho_multiplier = rho_multiplier
        self.box_lo = box_lo
        self.box_up = box_up
        self.kappa = kappa
        self.nx = dynamics.continuous_time_system.nx
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max
        if loss_weights is None:
            self.loss_weights = torch.tensor([1.0, 1.0, 1.0])
        else:
            assert loss_weights.shape == (3,)
            assert torch.all(loss_weights > 0)
            self.loss_weights = loss_weights

    def get_rho(self):
        rho_boundary = self.lyapunov(self.x_boundary).min()
        rho = self.rho_multiplier * rho_boundary
        return rho

    def forward(self, xe):
        # Run the system by one step with dt.
        x = xe[:, : self.nx]
        e = xe[:, self.nx :]
        z = x - e
        y = self.observer.h(x)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        # u = self.controller(z)
        new_x = self.dynamics.forward(x, u)
        new_z = self.observer.forward(z, u, y)
        new_xe = torch.cat((new_x, new_x - new_z), dim=1)
        self.new_xe = new_xe
        lyapunov_x = self.lyapunov(xe)
        rho = self.get_rho()
        # Save the results for reference.
        self.last_lyapunov_x = lyapunov_x.detach()
        loss1 = self.loss_weights[0] * (rho - lyapunov_x)
        loss2 = self.loss_weights[1] * torch.nn.functional.relu(
            self.lyapunov(new_xe) - (1 - self.kappa) * lyapunov_x
        )
        loss3 = self.loss_weights[2] * (
            torch.nn.functional.relu(self.box_lo - new_xe).sum(dim=1, keepdim=True)
            + torch.nn.functional.relu(new_xe - self.box_up).sum(dim=1, keepdim=True)
        )
        loss23 = loss2 + loss3
        if self.hard_max:
            loss = torch.min(
                torch.cat((loss1, loss23), dim=-1),
                dim=-1,
                keepdim=True,
            ).values
            # if loss.sum() > 0:
            #     print(xe[(loss > 0).squeeze()]/self.box_up)
            #     print(loss2.sum(), loss3.sum())
            return -loss
        else:
            loss = soft_min(torch.cat((loss1, loss23), dim=-1), self.beta)
            return -loss

class LyapunovDerivativeDOFSimpleLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute (1-κ)*V(x, e) - V(x_next, e_next)
    V(x, e), e = x - z
    """

    def __init__(self,
                 dynamics: dynamical_system.DiscreteTimeSystem,
                 observer,
                 controller: controllers.NeuralNetworkController,
                 lyap_nn: NeuralNetworkLyapunov,
                 kappa=0.1,
                 beta: float = 100,
                 hard_max: bool = True,
                 fuse_dV: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.observer = observer
        self.controller = controller
        self.lyapunov = lyap_nn
        self.kappa = kappa
        self.nx = dynamics.continuous_time_system.nx
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max
        self.fuse_dV = fuse_dV

    def forward(self, xe):
        # Run the system by one step with dt.
        x = xe[:, :self.nx]
        e = xe[:, self.nx:]
        z = x - e
        y = self.observer.h(x)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        # u = self.controller(z)
        new_x = self.dynamics.forward(x, u)
        new_z = self.observer.forward(z, u, y)
        lyapunov_x = self.lyapunov(xe)
        # Save the results for reference.
        self.last_lyapunov_x = lyapunov_x.detach()
        self.new_xe = torch.cat((new_x, new_x - new_z), dim=1)

        # The following two formulations are equivalent.
        # When self.fuse_dV is enabled, we fuse x (xe) and x_next (self.new_xe)
        # before entering the quadratic term in the Lyapunov function to compute
        # tighter bounds on dV, compared to computing two Lyapunov function
        # values separately.
        if self.fuse_dV:
            assert isinstance(self.lyapunov, NeuralNetworkQuadraticLyapunov)
            dV = self.lyapunov.diff(xe, self.new_xe, self.kappa, lyapunov_x)
            loss = -dV
        else:
            loss = (1 - self.kappa) * lyapunov_x - self.lyapunov(self.new_xe)

        return loss


class LyapunovDerivativeDOFLossWithV(LyapunovDerivativeDOFSimpleLoss):
    """
    The same as LyapunovDerivativeLoss, but with V(x) as the second output.
    Used for verification with level set.
    TODO: make it a template class to also support LyapunovDerivativeDOFLoss.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.x_boundary is None

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        # Output should be [N, 2] where N is the batch size.
        return torch.cat((loss, self.last_lyapunov_x), dim=1)


class LyapunovDerivativeDOFLossWithVBox(LyapunovDerivativeDOFLossWithV):
    """
    Additionally, output x_next for checking if x_next is within the bounding box.
    """

    def forward(self, *args, **kwargs):
        loss_and_v = super().forward(*args, **kwargs)
        return torch.cat((loss_and_v, self.new_xe), dim=1)



class ObserverLoss(nn.Module):
    """
    |x_next - x_ref_next|
    """

    def __init__(
        self,
        dynamics: dynamical_system.DiscreteTimeSystem,
        observer,
        controller,
        ekf_observer,
        roll_out_steps=150,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.observer = observer
        self.controller = controller
        self.ekf_observer = ekf_observer
        self.nx = dynamics.continuous_time_system.nx
        self.roll_out_steps = roll_out_steps

    def forward(self, xe):
        # Run the system by one step with dt.
        x = xe[:, : self.nx]
        e = xe[:, self.nx :]
        z = x - e
        y = self.observer.h(x)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        x_next = self.dynamics.forward(x, u)
        z_next = self.observer.forward(z, u, y)
        loss = torch.norm(x_next - z_next, p=2, dim=1)
        return loss.unsqueeze(1)

