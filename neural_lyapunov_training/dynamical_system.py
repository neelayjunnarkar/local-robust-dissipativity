import torch
import torch.nn as nn
from enum import Enum
import numpy as np
import control


class DiscreteTimeSystem(nn.Module):
    """
    Defines the interface for the discrete dynamical system.
    """

    def __init__(self, nx, nu, *args, **kwargs):
        super(DiscreteTimeSystem, self).__init__(*args, **kwargs)
        self.nx = nx
        self.nu = nu
        pass

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def x_equilibrium(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def u_equilibrium(self) -> torch.Tensor:
        raise NotImplementedError


class IntegrationMethod(Enum):
    ExplicitEuler = 1
    MidPoint = 2


class FirstOrderDiscreteTimeSystem(DiscreteTimeSystem):
    """
    This discrete-time system is constructed by discretizing a continuous time
    first-order dynamical system in time.
    """

    def __init__(
        self,
        continuous_time_system,
        dt: float,
        integration: IntegrationMethod = IntegrationMethod.ExplicitEuler,
    ):
        """
        Args:
          continuous_time_system: This system has to define a function
          xdot = f(x, u)
        """
        super(FirstOrderDiscreteTimeSystem, self).__init__(
            continuous_time_system.nx, continuous_time_system.nu
        )
        assert callable(getattr(continuous_time_system, "forward"))
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.dt = dt
        self.integration = integration
        self.continuous_time_system = continuous_time_system
        self.Ix = torch.eye(self.nx)

    def forward(self, x, u):
        """
        Compute x_next for a batch of x and u
        """
        assert x.shape[0] == u.shape[0]
        xdot = self.continuous_time_system.forward(x, u)
        if self.integration == IntegrationMethod.ExplicitEuler:
            x_next = x + xdot * self.dt
        else:
            raise NotImplementedError
        return x_next

    @property
    def x_equilibrium(self):
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self):
        return self.continuous_time_system.u_equilibrium


class SecondOrderDiscreteTimeSystem(DiscreteTimeSystem):
    """
    This discrete-time system is constructed by discretizing a continuous time
    second-order dynamical system in time.
    
    Supports optional disturbance input w for dissipativity analysis.
    """

    def __init__(
        self,
        continuous_time_system,
        dt: float,
        position_integration: IntegrationMethod = IntegrationMethod.MidPoint,
        velocity_integration: IntegrationMethod = IntegrationMethod.ExplicitEuler,
    ):
        """
        Args:
          continuous_time_system: This system has to define a function
          qddot = f(x, u) where x = [q, qdot].
        """
        super(SecondOrderDiscreteTimeSystem, self).__init__(
            continuous_time_system.nx, continuous_time_system.nu
        )
        assert callable(getattr(continuous_time_system, "forward"))
        self.nx = continuous_time_system.nx
        self.nu = continuous_time_system.nu
        self.nq = int(self.nx / 2)
        self.dt = dt
        self.velocity_integration = velocity_integration
        self.position_integration = position_integration
        self.continuous_time_system = continuous_time_system
        self.Ix = torch.eye(self.nx)
        # Disturbance dimension (if supported by continuous system)
        self.nw = getattr(continuous_time_system, 'nw', 0)

    def forward(self, x, u, w=None):
        """
        Compute x_next for a batch of x and u, with optional disturbance w.
        
        Args:
            x: state (batch, nx)
            u: control input (batch, nu)
            w: disturbance input (batch, nw), optional
            
        Returns:
            x_next: next state (batch, nx)
        """
        assert x.shape[0] == u.shape[0]
        # Pass disturbance to continuous-time system if supported
        if w is not None and hasattr(self.continuous_time_system, 'nw'):
            qddot = self.continuous_time_system.forward(x, u, w)
        else:
            qddot = self.continuous_time_system.forward(x, u)
        if self.velocity_integration == IntegrationMethod.ExplicitEuler:
            qdot_next = x[:, self.nq :] + qddot * self.dt
        else:
            raise NotImplementedError
        if self.position_integration == IntegrationMethod.MidPoint:
            q_next = x[:, : self.nq] + (qdot_next + x[:, self.nq :]) / 2 * self.dt
        elif self.position_integration == IntegrationMethod.ExplicitEuler:
            q_next = x[:, : self.nq] + x[:, self.nq :] * self.dt
        else:
            raise NotImplementedError
        return torch.cat((q_next, qdot_next), dim=1)
    
    def output(self, x, u):
        """
        Performance output z for dissipativity analysis.
        Delegates to continuous-time system if available.
        """
        if hasattr(self.continuous_time_system, 'output'):
            return self.continuous_time_system.output(x, u)
        else:
            return self.continuous_time_system.h(x)

    def linearized_dynamics(self, x, u):
        Ac, Bc = self.continuous_time_system.linearized_dynamics(x, u)
        Ad = self.dt * Ac + self.Ix.to(x.device)
        Bd = self.dt * Bc
        return Ad, Bd

    def output_feedback_linearized_lyapunov(self, K, L):
        """
        Given the control gain K and observer gain L, solve the discrete-time Lyapunov equation
        for the closed-loop system with the states and controls at equilibrium.
        The linearized dynamics are computed from the continuous-time system with Explicit Euler method.
        """
        x0 = self.x_equilibrium.unsqueeze(0)
        Ad, Bd = self.continuous_time_system.linearized_dynamics(
            x0, self.u_equilibrium.unsqueeze(0)
        )
        Ad = Ad.squeeze().detach().numpy()
        Bd = Bd.squeeze().detach().numpy()
        C = (
            self.continuous_time_system.linearized_observation(x0)
            .squeeze()
            .detach()
            .numpy()
        )
        Acl = np.vstack(
            (
                np.hstack((Ad + Bd @ K, -Bd @ K)),
                np.hstack((np.zeros([self.nx, self.nx]), Ad - L @ C)),
            )
        )
        Acl[np.abs(Acl) <= 1e-6] = 0
        S = control.dlyap(Acl, np.eye(2 * self.nx))
        return S

    @property
    def x_equilibrium(self):
        return self.continuous_time_system.x_equilibrium

    @property
    def u_equilibrium(self):
        return self.continuous_time_system.u_equilibrium


class NormalizedDynamicsWrapper(DiscreteTimeSystem):
    """
    Wraps a DiscreteTimeSystem in normalized coordinates z = S·x.
    
    Given a positive-definite matrix P (e.g., from LQR), S = chol(P),
    the coordinate transform z = S·x makes the quadratic form
    x^T P x = z^T z (a unit sphere in z-space).
    
    All inputs/outputs are in z-space. The wrapper handles z↔x conversion
    internally when calling the underlying dynamics.
    """

    def __init__(self, dynamics: DiscreteTimeSystem, S: torch.Tensor):
        """
        Args:
            dynamics: The underlying discrete-time system in x-space.
            S: (nx, nx) normalization matrix, typically chol(P).
        """
        super().__init__(dynamics.nx, dynamics.nu)
        self.inner = dynamics
        self.register_buffer('S', S.clone())
        self.register_buffer('S_inv', torch.linalg.inv(S))
        # Forward attributes from inner dynamics
        self.continuous_time_system = dynamics.continuous_time_system
        self.nw = getattr(dynamics, 'nw', 0)
        self.dt = dynamics.dt

    def _z_to_x(self, z: torch.Tensor) -> torch.Tensor:
        """Convert z-space coordinates to x-space: x = x_eq + S⁻¹·z"""
        return self.inner.x_equilibrium.to(z.device) + z @ self.S_inv.T

    def _x_to_z(self, x: torch.Tensor) -> torch.Tensor:
        """Convert x-space coordinates to z-space: z = S·(x - x_eq)"""
        x_diff = x - self.inner.x_equilibrium.to(x.device)
        return x_diff @ self.S.T

    def forward(self, z, u, w=None):
        """Dynamics in normalized error coordinates: z_next = S·(f(x_eq + S⁻¹·z, u, w) - x_eq)"""
        x = self._z_to_x(z)
        x_next = self.inner.forward(x, u, w)
        return self._x_to_z(x_next)

    def output(self, z, u):
        """Performance output z_perf (in x-space, not normalized)."""
        x = self._z_to_x(z)
        return self.inner.output(x, u)

    def linearized_dynamics(self, z, u):
        x = self._z_to_x(z)
        Ad, Bd = self.inner.linearized_dynamics(x, u)
        # In z-space (for error z): Ad_z = S Ad S⁻¹, Bd_z = S Bd
        Ad_z = self.S @ Ad @ self.S_inv
        Bd_z = self.S @ Bd
        return Ad_z, Bd_z

    @property
    def x_equilibrium(self):
        """Equilibrium in z-space (error coordinates) is ALWAYS 0."""
        return torch.zeros(self.nx, device=self.S.device, dtype=self.S.dtype)

    @property
    def u_equilibrium(self):
        return self.inner.u_equilibrium


class AugmentedLTICDynamics(DiscreteTimeSystem):
    """
    Augmented dynamics that evolves both plant and LTI controller states
    as a single higher-dimensional system.

    Augmented state  ξ = [x_p, x_k]  (dim = n_p + n_k).

    Given control u (computed externally by ``LTIDynamicController.forward``):

    * **Plant**:      x_p⁺ = plant_dynamics(x_p, u, w)
    * **Controller**: x_k⁺ = Ā_k x_k + B̄_k y ,  y = output_fn(x_p)

    The controller matrices (Ā_k, B̄_k) live inside the ``LTIDynamicController``
    instance and are shared with the controller forward pass so gradients flow
    through both paths when the matrices are trainable.

    Drop-in ``DiscreteTimeSystem`` — existing loss functions, PGD, sampling,
    domain expansion, and verification all work unchanged on ξ.
    """

    def __init__(self, plant_dynamics: DiscreteTimeSystem, ltic_controller):
        """
        Args:
            plant_dynamics: original plant ``DiscreteTimeSystem``.
            ltic_controller: ``LTIDynamicController`` instance.
        """
        super().__init__(
            nx=plant_dynamics.nx + ltic_controller.n_k,
            nu=plant_dynamics.nu,
        )
        self.plant = plant_dynamics
        self.ltic = ltic_controller
        self.n_plant = plant_dynamics.nx
        self.n_k = ltic_controller.n_k
        self.dt = plant_dynamics.dt
        self.nw = getattr(plant_dynamics, 'nw', 0)
        if hasattr(plant_dynamics, 'continuous_time_system'):
            self.continuous_time_system = plant_dynamics.continuous_time_system

    def forward(self, xi, u, w=None):
        """
        Evolve augmented state one time-step.

        Args:
            xi: (batch, n_plant + n_k) augmented state.
            u:  (batch, n_u) control.
            w:  (batch, n_w) or None disturbance.
        """
        x_p = xi[:, :self.n_plant]
        x_k = xi[:, self.n_plant:]
        y = self.ltic._get_y(x_p)

        if w is not None:
            x_p_next = self.plant.forward(x_p, u, w)
        else:
            x_p_next = self.plant.forward(x_p, u)

        x_k_next = self.ltic.evolve_state(x_k, y)

        return torch.cat([x_p_next, x_k_next], dim=1)

    def output(self, xi, u):
        """Performance output — delegates to plant dynamics."""
        x_p = xi[:, :self.n_plant]
        return self.plant.output(x_p, u)

    @property
    def x_equilibrium(self):
        """Equilibrium of augmented system: [x_p*, 0_k]."""
        x_p_eq = self.plant.x_equilibrium
        x_k_eq = torch.zeros(self.n_k, device=x_p_eq.device, dtype=x_p_eq.dtype)
        return torch.cat([x_p_eq, x_k_eq])

    @property
    def u_equilibrium(self):
        return self.plant.u_equilibrium


class AugmentedRINNDynamics(DiscreteTimeSystem):
    """
    Augmented dynamics that evolves both plant and RINN controller states
    as a single higher-dimensional system.

    Augmented state  ξ = [x_p, x_k]  (dim = n_p + n_k).

    Given control u (computed externally by ``RINNController.forward``):

    * **Plant**:      x_p⁺ = plant_dynamics(x_p, u, w_dist)
    * **Controller**: x_k⁺ = Ā x_k + B̄w w + B̄y y ,  with w from implicit solve

    The controller matrices live inside the ``RINNController`` instance
    and are shared with the controller forward pass so gradients flow
    through both paths when the matrices are trainable.

    Drop-in ``DiscreteTimeSystem`` — existing loss functions, PGD, sampling,
    domain expansion, and verification all work unchanged on ξ.
    """

    def __init__(self, plant_dynamics: DiscreteTimeSystem, rinn_controller):
        """
        Args:
            plant_dynamics: original plant ``DiscreteTimeSystem``.
            rinn_controller: ``RINNController`` instance.
        """
        super().__init__(
            nx=plant_dynamics.nx + rinn_controller.n_k,
            nu=plant_dynamics.nu,
        )
        self.plant = plant_dynamics
        self.rinn = rinn_controller
        self.n_plant = plant_dynamics.nx
        self.n_k = rinn_controller.n_k
        self.dt = plant_dynamics.dt
        self.nw = getattr(plant_dynamics, 'nw', 0)
        if hasattr(plant_dynamics, 'continuous_time_system'):
            self.continuous_time_system = plant_dynamics.continuous_time_system

    def forward(self, xi, u, w=None):
        """
        Evolve augmented state one time-step.

        Args:
            xi: (batch, n_plant + n_k) augmented state.
            u:  (batch, n_u) control.
            w:  (batch, n_w) or None disturbance (plant disturbance, not RINN w).
        """
        x_p = xi[:, :self.n_plant]
        x_k = xi[:, self.n_plant:]
        y = self.rinn._get_y(x_p)

        if w is not None:
            x_p_next = self.plant.forward(x_p, u, w)
        else:
            x_p_next = self.plant.forward(x_p, u)

        x_k_next = self.rinn.evolve_state(x_k, y)

        return torch.cat([x_p_next, x_k_next], dim=1)

    def output(self, xi, u):
        """Performance output — delegates to plant dynamics."""
        x_p = xi[:, :self.n_plant]
        return self.plant.output(x_p, u)

    @property
    def x_equilibrium(self):
        """Equilibrium of augmented system: [x_p*, 0_k]."""
        x_p_eq = self.plant.x_equilibrium
        x_k_eq = torch.zeros(self.n_k, device=x_p_eq.device, dtype=x_p_eq.dtype)
        return torch.cat([x_p_eq, x_k_eq])

    @property
    def u_equilibrium(self):
        return self.plant.u_equilibrium
