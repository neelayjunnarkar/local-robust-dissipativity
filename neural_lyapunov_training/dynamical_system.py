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
