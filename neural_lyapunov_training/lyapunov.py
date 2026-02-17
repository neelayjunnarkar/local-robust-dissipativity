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


def logsumexp(x: torch.Tensor, beta: float = 100):
    x_max = torch.max(x, dim=-1, keepdim=True).values
    eq = torch.exp(beta * (x - x_max))
    if torch.any(torch.isnan(eq)):
        raise Exception("logsumexp contains NAN, consider to reduce beta.")
    return torch.log(torch.sum(eq, dim=-1, keepdim=True)) / beta


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

    The optimizable parameters are the network ϕ and R.
    """

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
          V_psd_form: "quadratic", "L1", or "L1_R_free".
          use_nonlinear: Whether to include the NN component.
        """
        super().__init__(*args, **kwargs)
        self.goal_state = goal_state
        self.x_dim = x_dim
        self.absolute_output = absolute_output
        self.V_psd_form = V_psd_form
        self.eps = eps
        self.use_nonlinear = use_nonlinear
        
        # Linear/Quadratic components (R matrices)
        if R_frozen is not None:
            self.register_buffer('R_frozen', R_frozen.clone())
        else:
            self.R_frozen = None
            
        if R_trainable is not None:
            self.register_parameter('R_trainable', nn.Parameter(R_trainable.clone()))
        else:
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
            
            self.net = nn.Sequential(*layers)
        else:
            self.net = None
            
        self.nominal = nominal

    def _network_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_nonlinear and self.net is not None and len(self.net) > 0:
            phi = self.net(x)
            phi_star = self.net(self.goal_state)
            return phi - phi_star
        else:
            return torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)

    def _V_psd_output(self, x: torch.Tensor):
        x_diff = x - self.goal_state
        
        # Construct the effective P matrix or L1 coefficients
        # P = eps*I + R_frozen^T R_frozen + R_trainable^T R_trainable
        
        # We handle this by summing the outputs for simplicity and correctness
        v_psd = torch.zeros((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)
        
        # Base epsilon term for strictly positive definiteness
        if self.eps > 0 and self.V_psd_form == "quadratic":
            v_psd += self.eps * torch.sum(x_diff ** 2, dim=-1, keepdim=True)
            
        def compute_term(R_mat, form):
            if R_mat is None: return 0
            if form == "quadratic":
                return torch.sum((x_diff @ R_mat.T) ** 2, dim=-1, keepdim=True)
            elif form == "L1":
                # (eps*I + R^TR)x handled differently here for L1
                # For this configurable version, we just do |Rx|_1 for extra components
                Rx = x_diff @ R_mat.T
                return (torch.nn.functional.relu(Rx) + torch.nn.functional.relu(-Rx)).sum(dim=-1, keepdim=True)
            return 0

        v_psd += compute_term(self.R_frozen, self.V_psd_form)
        v_psd += compute_term(self.R_trainable, self.V_psd_form)
        
        return v_psd

    def forward(self, x):
        V_nominal = 0 if self.nominal is None else self.nominal(x)

        network_output = self._network_output(x)
        V_psd_output = self._V_psd_output(x)
        if self.absolute_output:
            return (
                V_nominal
                + torch.nn.functional.relu(network_output)
                + torch.nn.functional.relu(-network_output)
                + V_psd_output
            )
        else:
            return V_nominal + network_output + V_psd_output

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
    """
    Dissipativity-based derivative loss for state feedback with ROA constraints.
    
    Verifies the LOCAL dissipativity condition for forward invariance:
        V(x_{t+1}) - V(x_t) ≤ min(0, s(w, z))
    
    This ensures:
        1. When s < 0: V decreases at rate bounded by s
        2. When s ≥ 0: V still decreases (forward invariance)
    
    Equivalently (for verification, we want this ≥ 0):
        V(x_t) - V(x_{t+1}) + min(0, s(w, z)) ≥ 0
    
    Supply rate types:
        - Lyapunov:  s = -κV(x)           → always ≤ 0, standard Lyapunov
        - L2-gain:   s = γ²‖w‖² - ‖z‖²   → bounded L2 gain with invariance
        - Passivity: s = wᵀz              → passive system with invariance
    
    Loss structure (same as LyapunovDerivativeLoss):
        min(loss1, loss2 + loss3)
        - loss1: Outside ROA penalty (ρ - V(x))
        - loss2: Dissipativity violation
        - loss3: Box constraint violation
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
        *args,
        **kwargs
    ):
        """
        Args:
            dynamics: Discrete-time dynamical system
            controller: Neural network controller
            lyap_nn: Neural network Lyapunov function
            supply_rate: Supply rate object (Lyapunov, L2-gain, or Passivity)
            box_lo, box_up: Bounding box for state invariance
            rho_multiplier: Multiplier for sublevel set ρ
            w_max: Disturbance bound |w| ≤ w_max. Required for L2-gain/Passivity.
            beta: Soft-max coefficient
            hard_max: Use hard max (True) or soft max (False)
            loss_weights: Weights for loss components [outside_roa, dissipation, box]
            s_scale: Scaling factor for supply rate s(w, z)
        """
        super().__init__(*args, **kwargs)
        self.dynamics = dynamics
        self.controller = controller
        self.lyapunov = lyap_nn
        self.supply_rate = supply_rate
        self.observer = kwargs.get('observer', None)
        self.box_lo = box_lo
        self.box_up = box_up
        self.rho_multiplier = rho_multiplier
        self.beta = beta
        self.hard_max = hard_max
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.nx = dynamics.nx
        self.nw = getattr(dynamics.continuous_time_system, 'nw', 1) if hasattr(dynamics, 'continuous_time_system') else 1
        
        # Store w_max as tensor
        if w_max is not None:
            if isinstance(w_max, (int, float)):
                self.w_max = torch.tensor([w_max])
            else:
                self.w_max = w_max
        else:
            self.w_max = None
        
        if loss_weights is None:
            self.loss_weights = torch.tensor([1.0, 1.0, 1.0])
        else:
            assert loss_weights.shape == (3,)
            assert torch.all(loss_weights > 0)
            self.loss_weights = loss_weights
        
        self.s_scale = s_scale
        
        # Validate configuration
        if supply_rate.requires_disturbance:
            assert w_max is not None, "Supply rate requires disturbance bound w_max"
    
    def get_rho(self):
        """Compute ROA sublevel set value ρ."""
        if self.x_boundary is None:
            return torch.tensor(0.0)
        rho_boundary = self.lyapunov(self.x_boundary).min()
        rho = self.rho_multiplier * rho_boundary
        return rho
    
    def _sample_disturbance(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """Sample disturbance uniformly in [-w_max, w_max]."""
        if not self.supply_rate.requires_disturbance or self.w_max is None:
            return None
        w_max = self.w_max.to(device)
        w = (torch.rand(batch_size, self.nw, device=device) - 0.5) * 2 * w_max
        return w
    
    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, save_new_x: bool = False):
        """
        Compute dissipativity loss based on the condition:
        
        max{V(x) - c - ε, min{-V(F(x,u)) + V(x) + s(w,z), -V(F(x,u)) + c}} ≥ 0
        """
        # Handle joint [x, w] input if w is not provided explicitly
        if w is None and x.shape[1] == self.nx + self.nw:
            w = x[:, self.nx:]
            x = x[:, :self.nx]
            
        batch_size = x.shape[0]
        device = x.device
        
        # Sample disturbance if required but not provided
        if w is None and self.supply_rate.requires_disturbance:
            w = self._sample_disturbance(batch_size, device)
        
        # Determine if we are doing output feedback (DOF) or state feedback
        is_dof = self.observer is not None
        
        if is_dof:
            # DOF case: x is actually augmented state [state, error]
            # Split augmented state
            # x input here is actually xe
            xe = x
            state_dim = self.dynamics.nx # Should be same as self.nx
            x_state = xe[:, :state_dim]
            e_error = xe[:, state_dim:]
            
            # Reconstruct estimated state z = x - e
            z_est = x_state - e_error
            
            # Controller acts on estimated state z
            u = self.controller(z_est)
            
            # True state update
            new_x_state = self.dynamics.forward(x_state, u, w)
            
            # Observer update
            # Get measurement y (from true x)
            y = self.dynamics.observation(x_state)
            
            # Observer estimates next z based on current z, u, and y
            new_z_est = self.observer(z_est, u, y)
            
            # Error update: new_e = new_x - new_z
            new_e_error = new_x_state - new_z_est
            
            # Combine into new augmented state
            new_x = torch.cat((new_x_state, new_e_error), dim=1)
            
            # For supply rate calculation, we need true state output
            x_for_output = x_state
            
        else:
            # State feedback case
            x_for_output = x
            
            # Control input
            u = self.controller(x)
        
            # Next state (with disturbance if provided)
            new_x = self.dynamics.forward(x, u, w)
        if save_new_x:
            self.new_x = new_x
        
        # Lyapunov values
        V_x = self.lyapunov(x)
        V_next = self.lyapunov(new_x)
        self.last_lyapunov_x = V_x.detach()
        
        # Performance output z (for L2-gain/Passivity)
        z = None
        if self.supply_rate.requires_output:
            if is_dof:
                 # Use observer output function if available, otherwise dynamics output
                 if hasattr(self.observer, 'h'):
                     z = self.observer.h(x_for_output)
                 else:
                     z = self.dynamics.output(x_for_output, u)
            else:
                z = self.dynamics.output(x_for_output, u)
        
        # Compute supply rate s(w, z)
        supply = self.supply_rate(w, z, V_x)
        
        # Get c = ρ (ROA sublevel set value)
        c = self.get_rho()
        
        # Implement: max{V(x) - c - ε, min{-V(x_next) + V(x) + s, -V(x_next) + c}} ≥ 0
        # Using ε = 0 (margin is implicit in rho_multiplier)
        
        # Term 1: V(x) - c (outside ROA condition)
        term1 = V_x - c
        
        # Term 2a: -V(x_next) + V(x) + alpha * s = V(x) - V(x_next) + alpha * s (dissipativity)
        term2a = V_x - V_next + self.s_scale * supply
        
        # Term 2b: -V(x_next) + c = c - V(x_next) (forward invariance into ROA)
        term2b = c - V_next
        
        # Term 2: min{term2a, term2b}
        term2 = torch.min(torch.cat((term2a, term2b), dim=-1), dim=-1, keepdim=True).values
        
        # Final: max{term1, term2}
        if self.hard_max:
            result = torch.max(torch.cat((term1, term2), dim=-1), dim=-1, keepdim=True).values
        else:
            result = soft_max(torch.cat((term1, term2), dim=-1), self.beta)
        
        # Return result (should be ≥ 0 for valid condition)
        # For training, we want to minimize violations, so return as-is
        # (negative values indicate violations)
        return result
    
    @property
    def kappa(self):
        """For compatibility: extract kappa if using Lyapunov supply rate."""
        if isinstance(self.supply_rate, supply_rate_module.LyapunovSupplyRate):
            return self.supply_rate.kappa
        return 0.0
    
    @property
    def requires_disturbance(self):
        """Whether this loss requires disturbance input w for training/verification."""
        return self.supply_rate.requires_disturbance


class DissipativityDerivativeLossWithV(DissipativityDerivativeLoss):
    """
    Dissipativity loss with V(x) as second output for level set verification.
    """
    
    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, *args, **kwargs):
        loss = super().forward(x, w, *args, **kwargs)
        return torch.cat((loss, self.last_lyapunov_x), dim=1)


class DissipativityDerivativeLossWithVBox(DissipativityDerivativeLossWithV):
    """
    Additionally output x_next for bounding box verification.
    """
    
    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, *args, **kwargs):
        loss_and_V = super().forward(x, w, save_new_x=True, *args, **kwargs)
        return torch.cat((loss_and_V, self.new_x), dim=1)
    



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


class LyapunovContinuousTimeDerivativeLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute -κ*V - ∂V/∂ ẋ >= 0
    """

    def __init__(
        self,
        continuous_time_system,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkQuadraticLyapunov,
        kappa=0.1,
        beta: float = 100,
        hard_max: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.continuous_time_system = continuous_time_system
        self.controller = controller
        self.lyapunov = lyap_nn
        self.kappa = kappa
        self.nx = continuous_time_system.nx
        self.nq = continuous_time_system.nq
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max

    def forward(self, x):
        # Run the system by one step with dt.
        u = self.controller.forward(x)
        v_dot = self.continuous_time_system.forward(x, u)
        x_dot = torch.cat((x[:, self.nq :], v_dot), dim=1)
        dVdx = self.lyapunov.dVdx(x)
        V_dot = torch.sum(dVdx * x_dot, dim=-1, keepdim=True)
        lyapunov_x = self.lyapunov(x)
        self.last_lyapunov_x = lyapunov_x.detach()
        loss = -self.kappa * lyapunov_x - V_dot
        if self.x_boundary is not None:
            q = torch.cat(
                (loss, lyapunov_x - self.lyapunov(self.x_boundary).min()), dim=-1
            )
            if self.hard_max:
                return torch.max(q, dim=-1, keepdim=True).values
            else:
                return soft_max(q, self.beta)

        else:
            return loss


class LyapunovContinuousTimeDerivativeLossWithV(LyapunovContinuousTimeDerivativeLoss):
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


class LyapunovContinuousTimeDerivativeDOFLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute -κ*V - ∂V/∂ ẋ >= 0
    """

    def __init__(
        self,
        continuous_time_system,
        observer,
        controller: controllers.NeuralNetworkController,
        lyap_nn: NeuralNetworkQuadraticLyapunov,
        kappa=0.1,
        beta: float = 100,
        hard_max: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.continuous_time_system = continuous_time_system
        self.observer = observer
        self.controller = controller
        self.lyapunov = lyap_nn
        self.kappa = kappa
        self.nx = continuous_time_system.nx
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max

    def forward(self, xe):
        # Run the system by one step with dt.
        x = xe[:, : self.nx]
        e = xe[:, self.nx :]
        z = x - e
        y = self.observer.h(x)
        # u = self.controller.forward(z)
        ey = y - self.observer.h(z)
        u = self.controller.forward(torch.cat((z, ey), dim=1))
        v_dot = self.continuous_time_system.forward(x, u)
        x_dot = torch.cat((x[:, self.nq :], v_dot), dim=1)
        z_dot = self.observer.forward(z, u, y)
        e_dot = x_dot - z_dot
        xi_dot = torch.cat((x_dot, e_dot), dim=1)
        dVdxi = self.lyapunov.dVdx(xe)
        V_dot = torch.sum(dVdxi * xi_dot, dim=-1, keepdim=True)
        lyapunov_x = self.lyapunov(xe)
        loss = -self.kappa * lyapunov_x - V_dot
        self.last_lyapunov_x = lyapunov_x.detach()
        if self.x_boundary is not None:
            q = torch.cat(
                (loss, lyapunov_x - self.lyapunov(self.x_boundary).min()), dim=-1
            )
            if self.hard_max:
                return torch.max(q, dim=-1, keepdim=True).values
            else:
                return soft_max(q, self.beta)

        else:
            return loss


class CLFDerivativeLoss(nn.Module):
    """
    Lyapunov derivative loss for dynamic output feedback.
    Compute -κ*V - ∂V/∂ ẋ >= 0
    """

    def __init__(
        self,
        continuous_time_system,
        lyap_nn,
        u_abs_box: torch.Tensor,
        kappa=0.1,
        beta: float = 100,
        hard_max: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.continuous_time_system = continuous_time_system
        self.lyapunov = lyap_nn
        self.u_abs_box = u_abs_box
        self.kappa = kappa
        self.nx = continuous_time_system.nx
        self.nq = continuous_time_system.nq
        self.x_boundary: typing.Optional[torch.Tensor] = None
        self.beta = beta
        self.hard_max = hard_max

    def forward(self, x):
        # Run the system by one step with dt.
        f1 = self.continuous_time_system.f1(x)
        f2 = self.continuous_time_system.f2(x)
        dVdx = self.lyapunov.dVdx(x)
        Lf1 = torch.sum(dVdx * f1, dim=-1, keepdim=True)
        Lf2 = torch.sum(
            torch.abs(dVdx.unsqueeze(1) @ f2).squeeze() * self.u_abs_box,
            dim=-1,
            keepdim=True,
        )
        V_dot = Lf1 - Lf2
        lyapunov_x = self.lyapunov(x)
        self.last_lyapunov_x = lyapunov_x.detach()
        loss = -self.kappa * lyapunov_x - V_dot
        if self.x_boundary is not None:
            q = torch.cat(
                (loss, lyapunov_x - self.lyapunov(self.x_boundary).min()), dim=-1
            )
            if self.hard_max:
                return torch.max(q, dim=-1, keepdim=True).values
            else:
                return soft_max(q, self.beta)

        else:
            return loss


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
        # z_ekf = self.ekf_observer.forward(z, u, y)
        # loss = torch.norm(z_ekf - z_next, p=2, dim=1)
        return loss.unsqueeze(1)


class LyapunovLowerBoundLoss:
    """
    We want the condition V(x) >= ρ
    We compute V(x) - ρ
    """

    def __init__(self, lyap: NeuralNetworkLyapunov, rho_roa: float):
        self.lyapunov = lyap
        self.rho_roa = rho_roa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lyapunov(x) - self.rho_roa
