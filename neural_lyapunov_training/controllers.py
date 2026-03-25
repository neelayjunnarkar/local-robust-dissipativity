import torch
import torch.nn as nn
import numpy as np
from scipy import linalg as la


class LinearController(nn.Module):
    """
    Simple linear controller.
    """

    def __init__(self, K, u_equilibrium, trainable=True, **kwargs):
        """
        Args:
          K: the coefficients of the linear controller.
          u_equilibrium: the controller output at equilibrium.
        """
        super().__init__()
        self.u_equilibrium = u_equilibrium
        self.trainable = trainable
        if trainable:
            self.register_parameter(
                name="K", param=torch.nn.Parameter(K.clone().detach())
            )
        else:
            self.K = K.clone().requires_grad_(False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.K, self.u_equilibrium)

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.u_equilibrium = fn(self.u_equilibrium)
        if not self.trainable:
            self.K = fn(self.K)
        return self


class NeuralNetworkController(nn.Module):
    def __init__(
        self,
        nlayer=3,
        in_dim=2,
        out_dim=1,
        hidden_dim=64,
        clip_output=None,
        u_lo=None,
        u_up=None,
        x_equilibrium=None,
        u_equilibrium=None,
        activation=nn.ReLU,
        *args,
        **kwargs
    ):
        """
        Simple neural network controller.

        The controller output is computed through following steps
        1. The neural network computes net(x)
        2. (a) If clip_output is "tanh", we truncate the network to within u_lo and
           u_up as f(x) = tanh(net(x)) * (u_up - u_lo)/2 + (u_up + u_lo)/2
           (b) If clip_output is "clamp", we truncate the network as
           f(x) = clamp(net(x), u_lo, u_up)
           (c) If clip_output is None, we set f(x) = net(x)
        3. If x_equilibrium and u_equilibrium are not None, we set
           u = f(x) - f(x*) + u*
           where x* is x_equilibrium, u* is u_equilibrium.
        This controller guarantees that the control action is within
        [u_lo, u_up] and at equilibrium state x_equilibrium, the control action
        is the equilibrium action u_equilibrium
        """
        super().__init__(*args, **kwargs)
        assert clip_output in (None, "tanh", "clamp")
        self.clip_output = clip_output
        if u_lo is not None:
            assert u_lo.shape == (out_dim,)
        self.u_lo = u_lo
        if u_up is not None:
            assert u_up.shape == (out_dim,)
        self.u_up = u_up
        if x_equilibrium is not None:
            assert x_equilibrium.shape == (in_dim,)
        self.x_equilibrium = x_equilibrium
        if u_equilibrium is not None:
            assert u_equilibrium.shape == (out_dim,)
            if self.u_lo is not None:
                assert torch.all(u_equilibrium >= self.u_lo)
            if self.u_up is not None:
                assert torch.all(u_equilibrium <= self.u_up)
        self.u_equilibrium = u_equilibrium
        layers = [nn.Linear(in_dim, out_dim if nlayer == 1 else hidden_dim)]
        for n in range(1, nlayer - 1):
            layers.append(activation())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        if nlayer != 1:
            layers.append(activation())
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.layers = layers
        # print(f'Controller function:')
        # print(self.net)

    def _unclipped_output(self, x: torch.Tensor) -> torch.Tensor:
        unclipped_output = self.net(x)
        if self.x_equilibrium is not None and self.u_equilibrium is not None:
            unclipped_output = (
                self.net(x) - self.net(self.x_equilibrium) + self.u_equilibrium
            )
        return unclipped_output

    def forward(self, x):
        unclipped_output = self._unclipped_output(x)

        if self.clip_output is None:
            return unclipped_output
        else:
            if self.clip_output == "tanh":
                # Apply tanh to make output between a certain bound.
                f = (
                    torch.tanh(self.net(x)) * (self.u_up - self.u_lo) / 2
                    + (self.u_lo + self.u_up) / 2
                )
                if self.x_equilibrium is not None and self.u_equilibrium is not None:
                    f_equilibrium = (
                        torch.tanh(self.net(self.x_equilibrium))
                        * (self.u_up - self.u_lo)
                        / 2
                        + (self.u_lo + self.u_up) / 2
                    )
                    return f - f_equilibrium + self.u_equilibrium
                else:
                    return f
            elif self.clip_output == "clamp":
                # Instead of calling clamp direct, we use relu twice. Currently auto_LIRPA doesn't handle clamp.
                f1 = torch.nn.functional.relu(unclipped_output - self.u_lo) + self.u_lo
                f = -(torch.nn.functional.relu(self.u_up - f1) - self.u_up)
                return f

    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        self.x_equilibrium = fn(self.x_equilibrium)
        self.u_equilibrium = fn(self.u_equilibrium)
        if self.u_lo is not None:
            self.u_lo = fn(self.u_lo)
        if self.u_up is not None:
            self.u_up = fn(self.u_up)
        return self


class LinearPlusNeuralNetworkController(nn.Module):
    """
    Controller with explicit linear + nonlinear decomposition:
        u = K_linear @ x + NN_nonlinear(x)
    
    This structure allows:
    1. Direct initialization of K_linear from LQR/SDP
    2. Small initialization of NN_nonlinear (starts as perturbation)
    3. Both components are trainable
    
    The linear term handles the nominal control, while the NN term
    learns corrections for nonlinearities and saturation.
    """
    
    def __init__(
        self,
        in_dim=2,
        out_dim=1,
        K_frozen=None,
        K_trainable=None,
        nlayer=3,
        hidden_dim=64,
        clip_output=None,
        u_lo=None,
        u_up=None,
        x_equilibrium=None,
        u_equilibrium=None,
        activation=nn.Tanh,
        nn_weight_scale=0.01,  # Scale for NN initialization
        use_nonlinear=True,
        *args,
        **kwargs
    ):
        """
        Args:
            in_dim: Input dimension (state)
            out_dim: Output dimension (control)
            K_frozen: Frozen linear gain matrix (buffer).
            K_trainable: Trainable linear gain matrix (parameter).
            nlayer: Number of layers in nonlinear NN
            hidden_dim: Hidden dimension for NN
            clip_output: 'clamp', 'tanh', or None
            u_lo: Lower control bound (out_dim,)
            u_up: Upper control bound (out_dim,)
            x_equilibrium: Equilibrium state (in_dim,)
            u_equilibrium: Equilibrium control (out_dim,)
            activation: Activation function for NN
            nn_weight_scale: Scale factor for NN weight initialization
            use_nonlinear: Whether to include the NN component
        """
        super().__init__(*args, **kwargs)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        assert clip_output in (None, "tanh", "clamp")
        self.clip_output = clip_output
        self.use_nonlinear = use_nonlinear
        
        # Bounds and Equilibrium
        self.register_buffer('u_lo', u_lo) if u_lo is not None else setattr(self, 'u_lo', None)
        self.register_buffer('u_up', u_up) if u_up is not None else setattr(self, 'u_up', None)
        self.register_buffer('x_equilibrium', x_equilibrium) if x_equilibrium is not None else setattr(self, 'x_equilibrium', None)
        self.register_buffer('u_equilibrium', u_equilibrium) if u_equilibrium is not None else setattr(self, 'u_equilibrium', None)
        
        # Linear components
        if K_frozen is not None:
            self.register_buffer('K_frozen', K_frozen.clone())
        else:
            self.K_frozen = None
            
        if K_trainable is not None:
            self.K_trainable = nn.Parameter(K_trainable.clone())
        else:
            self.K_trainable = None
        
        # Nonlinear component
        if use_nonlinear:
            layers = [nn.Linear(in_dim, out_dim if nlayer == 1 else hidden_dim)]
            for n in range(1, nlayer - 1):
                layers.append(activation())
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if nlayer != 1:
                layers.append(activation())
                layers.append(nn.Linear(hidden_dim, out_dim))
            self.nn_nonlinear = nn.Sequential(*layers)
            
            with torch.no_grad():
                for layer in self.nn_nonlinear:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data *= nn_weight_scale
                        layer.bias.data *= nn_weight_scale
        else:
            self.nn_nonlinear = None

    def _linear_output(self, x: torch.Tensor) -> torch.Tensor:
        x_shifted = x - self.x_equilibrium if self.x_equilibrium is not None else x
        u_linear = torch.zeros((x.shape[0], self.out_dim), device=x.device, dtype=x.dtype)
        
        if self.K_frozen is not None:
            u_linear += torch.nn.functional.linear(x_shifted, self.K_frozen)
        if self.K_trainable is not None:
            u_linear += torch.nn.functional.linear(x_shifted, self.K_trainable)
            
        if self.u_equilibrium is not None:
            u_linear += self.u_equilibrium
        return u_linear

    def _nonlinear_output(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_nonlinear or self.nn_nonlinear is None:
            return torch.zeros((x.shape[0], self.out_dim), device=x.device, dtype=x.dtype)
        
        nn_output = self.nn_nonlinear(x)
        if self.x_equilibrium is not None:
            nn_eq = self.nn_nonlinear(self.x_equilibrium.reshape(1, -1))
            nn_output -= nn_eq
        return nn_output

    def _unclipped_output(self, x: torch.Tensor) -> torch.Tensor:
        return self._linear_output(x) + self._nonlinear_output(x)

    def forward(self, x):
        u = self._unclipped_output(x)
        if self.clip_output == "clamp":
            u = torch.max(torch.min(u, self.u_up), self.u_lo)
        elif self.clip_output == "tanh":
            u_mid = (self.u_up + self.u_lo) / 2
            u_range = (self.u_up - self.u_lo) / 2
            u = torch.tanh(u) * u_range + u_mid
        return u

    def _apply(self, fn):
        super()._apply(fn)
        # Parameters and buffers are handled automatically by nn.Module
        return self
    def _apply(self, fn):
        """Handles CPU/GPU transfer and type conversion."""
        super()._apply(fn)
        if self.x_equilibrium is not None:
            self.x_equilibrium = fn(self.x_equilibrium)
        if self.u_equilibrium is not None:
            self.u_equilibrium = fn(self.u_equilibrium)
        if self.u_lo is not None:
            self.u_lo = fn(self.u_lo)
        if self.u_up is not None:
            self.u_up = fn(self.u_up)
        return self


class LTIDynamicController(nn.Module):
    """
    Linear Time-Invariant Dynamic Output-Feedback Controller.

    Continuous-time formulation::

        ẋ_k = A_k x_k + B_k y
        u   = C_k x_k + D_k y

    Stored internally in discrete-time (Euler discretisation)::

        x_k[t+1] = Ā_k x_k[t] + B̄_k y[t]    (Ā_k = I + dt·A_k,  B̄_k = dt·B_k)
        u[t]     = C_k x_k[t] + D_k y[t]

    The controller operates on augmented state ξ = [x_p, x_k] where x_p is
    the plant state (dim n_p) and x_k is the controller internal state (dim n_k).

    * ``forward(ξ)`` → u   (called by the loss's ``self.controller(x)``).
    * ``evolve_state(x_k, y)`` → x_k_next  (called by the augmented dynamics).

    Output equation:  y = output_fn(x_p).  Default (None) means y = x_p.

    When ``trainable=False`` matrices are stored as buffers (frozen).
    When ``trainable=True`` matrices are ``nn.Parameter`` s (optimised).
    """

    def __init__(
        self,
        A_k: torch.Tensor,
        B_k: torch.Tensor,
        C_k: torch.Tensor,
        D_k: torch.Tensor,
        n_plant: int,
        dt: float,
        output_fn: nn.Module = None,
        trainable: bool = False,
        clip_output: str = None,
        u_lo: torch.Tensor = None,
        u_up: torch.Tensor = None,
    ):
        """
        Args:
            A_k: (n_k, n_k) continuous-time controller state matrix.
            B_k: (n_k, n_y) continuous-time controller input matrix.
            C_k: (n_u, n_k) controller output matrix.
            D_k: (n_u, n_y) controller feed-through matrix.
            n_plant: dimension of the plant state x_p.
            dt: discretisation time step (same as plant dynamics).
            output_fn: nn.Module mapping x_p → y.  None means y = x_p.
            trainable: if True all matrices become nn.Parameters.
            clip_output: ``'clamp'`` or ``None``.
            u_lo, u_up: control bounds (used when ``clip_output='clamp'``).
        """
        super().__init__()
        n_k = A_k.shape[0]
        self.n_k = n_k
        self.n_plant = n_plant
        self.dt = dt
        self.output_fn = output_fn
        self.clip_output = clip_output

        # Discretise: Ā = I + dt·A,  B̄ = dt·B
        A_kd = torch.eye(n_k, dtype=A_k.dtype) + dt * A_k
        B_kd = dt * B_k

        def _store(name, tensor):
            if trainable:
                self.register_parameter(name, nn.Parameter(tensor.clone()))
            else:
                self.register_buffer(name, tensor.clone())

        _store('A_kd', A_kd)
        _store('B_kd', B_kd)
        _store('C_k', C_k)
        _store('D_k', D_k)

        if u_lo is not None:
            self.register_buffer('u_lo', u_lo.clone())
        else:
            self.u_lo = None
        if u_up is not None:
            self.register_buffer('u_up', u_up.clone())
        else:
            self.u_up = None

    # -- helpers -------------------------------------------------------

    def _get_y(self, x_p: torch.Tensor) -> torch.Tensor:
        """Plant output from plant state."""
        if self.output_fn is not None:
            return self.output_fn(x_p)
        return x_p

    # -- public API ----------------------------------------------------

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """Compute control u from augmented state ξ = [x_p, x_k]."""
        x_p = xi[:, :self.n_plant]
        x_k = xi[:, self.n_plant:]
        y = self._get_y(x_p)
        u = x_k @ self.C_k.T + y @ self.D_k.T
        if self.clip_output == 'clamp' and self.u_lo is not None:
            u = torch.max(torch.min(u, self.u_up), self.u_lo)
        return u

    def evolve_state(self, x_k: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Discrete-time controller state update: x_k+ = Ā_k x_k + B̄_k y."""
        return x_k @ self.A_kd.T + y @ self.B_kd.T


class RINNController(nn.Module):
    """
    Recurrent Implicit Neural Network (RINN) Dynamic Controller.

    Continuous-time formulation::

        ẋ_k = A x_k + Bw w + By y
        v    = Cv x_k + Dvw w + Dvy y      (implicit layer equation)
        w    = φ(v)                         (element-wise activation)
        u    = Cu x_k + Duw w + Duy y

    ``Dvw`` is **strictly upper triangular**, so the implicit equation
    ``v = Cv x_k + Dvw φ(v) + Dvy y`` is solved exactly by a single
    backward sweep (back-substitution from the last element to the first).

    Stored internally in discrete-time (Euler discretisation)::

        x_k[t+1] = x_k[t] + dt·(A x_k + Bw w + By y)

    The controller operates on augmented state ξ = [x_p, x_k].

    * ``forward(ξ)`` → u   (called by the loss's ``self.controller(x)``).
    * ``evolve_state(x_k, y)`` → x_k_next  (called by augmented dynamics).
    * ``solve_implicit(x_k, y)`` → w  (back-substitution solve).

    When ``trainable=False`` matrices are stored as buffers (frozen).
    When ``trainable=True`` matrices are ``nn.Parameter`` s (optimised).
    """

    def __init__(
        self,
        A: torch.Tensor,
        Bw: torch.Tensor,
        By: torch.Tensor,
        Cv: torch.Tensor,
        Dvw: torch.Tensor,
        Dvy: torch.Tensor,
        Cu: torch.Tensor,
        Duw: torch.Tensor,
        Duy: torch.Tensor,
        n_plant: int,
        dt: float,
        output_fn: nn.Module = None,
        trainable: bool = False,
        freeze_dvw_lower_tri: bool = False,
        activation: str = 'relu',
        clip_output: str = None,
        u_lo: torch.Tensor = None,
        u_up: torch.Tensor = None,
    ):
        """
        Args:
            A:   (n_k, n_k) controller state matrix.
            Bw:  (n_k, n_w) controller implicit-layer input matrix.
            By:  (n_k, n_y) controller plant-output input matrix.
            Cv:  (n_w, n_k) implicit-layer state-coupling matrix.
            Dvw: (n_w, n_w) strictly upper triangular feed-through.
            Dvy: (n_w, n_y) implicit-layer plant-output coupling.
            Cu:  (n_u, n_k) output state matrix.
            Duw: (n_u, n_w) output implicit-layer coupling.
            Duy: (n_u, n_y) output feed-through.
            n_plant: dimension of the plant state x_p.
            dt: discretisation time step.
            output_fn: nn.Module mapping x_p → y.  None means y = x_p.
            trainable: if True all matrices become nn.Parameters.
            freeze_dvw_lower_tri: if True (and trainable), register a gradient
                hook that zeroes the lower-triangle grad of Dvw after each
                backward pass so the optimizer never updates those entries.
            activation: 'relu' or 'leaky_relu'.
            clip_output: 'clamp' or None.
            u_lo, u_up: control bounds.
        """
        super().__init__()
        n_k = A.shape[0]
        n_w = Dvw.shape[0]
        self.n_k = n_k
        self.n_w = n_w
        self.n_plant = n_plant
        self.dt = dt
        self.output_fn = output_fn
        self.clip_output = clip_output

        # Activation (must be nn.Module for auto_LiRPA)
        if activation == 'leaky_relu':
            self.phi = nn.LeakyReLU(negative_slope=0.01)
        else:
            self.phi = nn.ReLU()

        # Number of back-substitution sweeps = n_w (one per row)
        self.n_sweeps = n_w

        # Store matrices
        def _store(name, tensor):
            if trainable:
                self.register_parameter(name, nn.Parameter(tensor.clone()))
            else:
                self.register_buffer(name, tensor.clone())

        # State equation (continuous): ẋ_k = A x_k + Bw w + By y
        # Discretised: x_k+ = (I + dt·A) x_k + dt·Bw w + dt·By y
        A_kd = torch.eye(n_k, dtype=A.dtype, device=A.device) + dt * A
        Bw_kd = dt * Bw
        By_kd = dt * By

        _store('A_kd', A_kd)
        _store('Bw_kd', Bw_kd)
        _store('By_kd', By_kd)

        # Implicit-layer matrices (no discretisation needed)
        _store('Cv', Cv)
        _store('Dvw', Dvw)
        _store('Dvy', Dvy)

        # Optionally zero lower-triangle gradients of Dvw after every backward
        if trainable and freeze_dvw_lower_tri:
            mask = torch.triu(torch.ones(n_w, n_w, dtype=torch.bool), diagonal=1)
            self.register_buffer('_dvw_upper_mask', mask)

            def _zero_lower_tri_grad(grad: torch.Tensor) -> torch.Tensor:
                return grad * self._dvw_upper_mask

            self.Dvw.register_hook(_zero_lower_tri_grad)

        # Output equation matrices (no discretisation needed)
        _store('Cu', Cu)
        _store('Duw', Duw)
        _store('Duy', Duy)

        if u_lo is not None:
            self.register_buffer('u_lo', u_lo.clone())
        else:
            self.u_lo = None
        if u_up is not None:
            self.register_buffer('u_up', u_up.clone())
        else:
            self.u_up = None

    # -- helpers -------------------------------------------------------

    def _get_y(self, x_p: torch.Tensor) -> torch.Tensor:
        """Plant output from plant state."""
        if self.output_fn is not None:
            return self.output_fn(x_p)
        return x_p

    def solve_implicit(self, x_k: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Solve the implicit layer  v = Cv x_k + Dvw φ(v) + Dvy y  for w = φ(v).

        Because Dvw is strictly upper triangular, a single backward sweep
        (from row n_w-1 down to 0) gives the exact solution.

        At row i, v[i] depends only on w[j] for j > i (already solved).

        Uses a list of per-column tensors (no in-place ops) so autograd
        can differentiate through the solve when matrices are trainable.
        """
        # Base terms that don't depend on w
        base = x_k @ self.Cv.T + y @ self.Dvy.T  # (batch, n_w)

        # Collect solved columns; avoid in-place writes for autograd safety
        w_cols: list[torch.Tensor] = [None] * self.n_w  # type: ignore[assignment]

        for i in range(self.n_w - 1, -1, -1):
            # v[i] = base[i] + sum_{j > i} Dvw[i, j] * w[j]
            v_i = base[:, i]  # (batch,)
            for j in range(i + 1, self.n_w):
                v_i = v_i + self.Dvw[i, j] * w_cols[j]
            w_cols[i] = self.phi(v_i)  # (batch,)

        return torch.stack(w_cols, dim=1)  # (batch, n_w)

    # -- public API ----------------------------------------------------

    def forward(self, xi: torch.Tensor) -> torch.Tensor:
        """Compute control u from augmented state ξ = [x_p, x_k]."""
        x_p = xi[:, :self.n_plant]
        x_k = xi[:, self.n_plant:]
        y = self._get_y(x_p)

        w = self.solve_implicit(x_k, y)
        u = x_k @ self.Cu.T + w @ self.Duw.T + y @ self.Duy.T

        if self.clip_output == 'clamp' and self.u_lo is not None:
            u = torch.max(torch.min(u, self.u_up), self.u_lo)
        return u

    def evolve_state(self, x_k: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Discrete-time controller state update with implicit-layer solve."""
        w = self.solve_implicit(x_k, y)
        return x_k @ self.A_kd.T + w @ self.Bw_kd.T + y @ self.By_kd.T


class NeuralNetworkLuenbergerObserver(nn.Module):
    """
    Neural network observer that takes vectors as observations.
    The observer dynamics is z[t+1] = f(z[t], u[t]) + nn(z[t], y[t]-h(z[t]))
    """

    def __init__(
        self,
        z_dim,
        y_dim,
        dynamics,
        h,
        zero_obs_error,
        fc_hidden_dim=[16, 16, 8, 8],
        activation=nn.LeakyReLU,
    ):
        """
        z_dim: state estimate dimension, same as state dimension.
        y_dim: output dimension.
        h: observation function.
        zero_obs_error: y[t] - h(x[t]). when y[t] = h(z[t]), nn(z[t], y[t]-h(z[t])) = 0.
        """
        super().__init__()
        self.z_dim = z_dim
        self.dynamics = dynamics
        self.h = h
        self.zero_obs_error = zero_obs_error
        fc_layers = [nn.Linear(z_dim + y_dim, fc_hidden_dim[0])]
        n_fc_layers = len(fc_hidden_dim)
        for n in range(n_fc_layers - 1):
            fc_layers.append(activation())
            fc_layers.append(
                nn.Linear(
                    fc_hidden_dim[n],
                    fc_hidden_dim[n + 1] if n + 1 < n_fc_layers - 1 else z_dim,
                )
            )
        self.fc_net = nn.Sequential(*fc_layers)
        # print(f'Observer fully connected function:')
        # print(self.fc_net)

    def forward(self, z, u, y):
        batch_size = z.shape[0]
        K = torch.ones((batch_size, 1), device=self.zero_obs_error.device)
        z_nominal = self.dynamics(z, u)
        obs_error = y - self.h(z)
        Le = self.fc_net(torch.cat((z, obs_error), 1))
        L0 = self.fc_net(torch.cat((z, (K * self.zero_obs_error).to(z.device)), 1))
        unclipped_z = z_nominal + Le - L0
        return unclipped_z


class EKFObserver(nn.Module):
    def __init__(self, dynamics, h, gamma=0, delta=1e-3, lam=0, alpha=1.1):
        super().__init__()
        self.dynamics = dynamics
        self.nx = dynamics.continuous_time_system.nx
        self.nu = dynamics.nu
        self.ny = dynamics.continuous_time_system.ny
        self.h = h
        self.gamma = gamma
        self.delta = delta
        self.lam = lam
        self.Ix = torch.eye(self.nx)
        self.Iy = torch.eye(self.ny)
        self.alpha = alpha
        self.Q = self.delta * self.Ix
        self.R = self.delta * self.Iy
        self.P0, self.K0 = self.dare()

    def forward_varying_p(self, z0, Pk, u, y):
        batch_size = z0.shape[0]
        device = z0.device
        z1_prior = self.dynamics(z0, u)
        if not z1_prior.requires_grad:
            z1_prior.requires_grad = True
        Fk, _ = self.dynamics.linearized_dynamics(z0, u)

        h1_z = self.h(z1_prior)
        e1 = (y - h1_z).unsqueeze(-1)
        H1 = self.dynamics.continuous_time_system.linearized_observation(z1_prior)

        # Qk should actually depend on e0
        Qk = self.gamma * torch.kron(
            (e1.transpose(1, 2) @ e1), self.Ix.to(device)
        ) + self.delta * self.Ix.to(device)
        P1_prior = self.alpha**2 * Fk @ Pk @ Fk.transpose(1, 2) + Qk
        R1 = self.lam * H1 @ P1_prior @ H1.transpose(1, 2) + self.delta * self.Iy.to(
            device
        )
        K1 = (
            P1_prior
            @ H1.transpose(1, 2)
            @ torch.inverse(H1 @ P1_prior @ H1.transpose(1, 2) + R1)
        )
        z1 = z1_prior + (K1 @ e1).squeeze(-1)

        P1 = (self.Ix.to(device) - K1 @ H1) @ P1_prior @ (
            self.Ix.to(device) - K1 @ H1
        ).transpose(1, 2) + K1 @ R1 @ K1.transpose(1, 2)
        P_posterior = (self.Ix.to(device) - K1 @ H1) @ P1_prior
        return z1, P_posterior

    def forward(self, z0, u, y):
        batch_size = z0.shape[0]
        device = z0.device
        Pk = (torch.ones(batch_size, self.nx, self.nx) * self.P0).to(device)
        z1_prior = self.dynamics(z0, u)
        if not z1_prior.requires_grad:
            z1_prior.requires_grad = True
        Fk, _ = self.dynamics.linearized_dynamics(z0, u)

        h1_z = self.h(z1_prior)
        e1 = (y - h1_z).unsqueeze(-1)
        H1 = self.dynamics.continuous_time_system.linearized_observation(z1_prior)

        # Qk should actually depend on e0
        Qk = self.delta * self.Ix.to(device)
        P1_prior = self.alpha**2 * Fk @ Pk @ Fk.transpose(1, 2) + Qk
        R1 = self.lam * H1 @ P1_prior @ H1.transpose(1, 2) + self.delta * self.Iy.to(
            device
        )
        K1 = (
            P1_prior
            @ H1.transpose(1, 2)
            @ torch.linalg.inv(H1 @ P1_prior @ H1.transpose(1, 2) + R1)
        )
        z1 = z1_prior + (K1 @ e1).squeeze(-1)

        return z1

    def forward_constant_K(self, z0, u, y):
        K = (
            (torch.ones(z0.shape[0], self.K0.shape[0], self.K0.shape[1]) * self.K0)
            .to(y.device)
            .to(y.dtype)
        )
        z_prior = self.dynamics(z0, u)
        innovation = K @ (y - self.dynamics.continuous_time_system.h(z0)).unsqueeze(-1)
        z = z_prior + innovation.squeeze(-1)
        return z

    def dare(self):
        x0 = self.dynamics.x_equilibrium.unsqueeze(0)
        dtype = x0.dtype
        Ad, _ = self.dynamics.linearized_dynamics(
            x0, self.dynamics.u_equilibrium.unsqueeze(0)
        )
        Ad = Ad.squeeze().detach().numpy()
        Cd = (
            self.dynamics.continuous_time_system.linearized_observation(x0)
            .squeeze(0)
            .detach()
            .numpy()
        )
        Q = self.Q.cpu().detach().numpy()
        R = self.R.cpu().detach().numpy()
        P_prior = la.solve_discrete_are(Ad.T, Cd.T, Q, R)
        K = P_prior @ Cd.T @ np.linalg.inv(Cd @ P_prior @ Cd.T + R)
        return torch.tensor(P_prior, dtype=dtype), torch.tensor(K, dtype=dtype)
