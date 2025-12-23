import torch
from typing import Optional


class PendulumDynamics:
    """
    The inverted pendulum, with the upright equilibrium as the state origin.
    
    Supports optional additive disturbance w on the control input:
        τ_effective = u + w
    
    For dissipativity analysis:
        - w: disturbance input (additive to torque)
        - z: performance output (observation y = θ)
    """

    def __init__(self, m: float = 1, l: float = 1, beta: float = 1, g: float = 9.81):
        self.nx = 2
        self.nu = 1
        self.nw = 1  # Disturbance dimension (additive to u)
        self.ny = 1
        self.nz = 1  # Performance output dimension (same as observation)
        self.m = m  # Mass
        self.l = l  # Length
        self.g = g  # Gravity
        self.beta = beta  # Damping
        self.inertia = self.m * self.l**2

    def forward(self, x: torch.Tensor, u: torch.Tensor, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Dynamics with optional disturbance.
        
        Args:
            x: state (batch, 2) = [θ, θ̇]
            u: controller input (batch, 1)
            w: disturbance input (batch, 1), additive to u. Default None.
            
        Returns:
            θ̈: angular acceleration (batch, 1)
        """
        # States (theta, theta_dot)
        theta, theta_dot = x[:, 0], x[:, 1]
        theta = theta.unsqueeze(-1)
        theta_dot = theta_dot.unsqueeze(-1)
        
        # Effective input: u + w (disturbance is additive)
        u_eff = u if w is None else u + w
        
        # Dynamics according to http://underactuated.mit.edu/pend.html
        ml2 = self.m * self.l * self.l
        d_theta_dot = (
            (-self.beta / ml2) * theta_dot
            + (self.g / self.l) * torch.sin(theta)
            + u_eff / ml2
        )
        return d_theta_dot
    
    def output(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Performance output z for dissipativity analysis.
        
        For L2-gain analysis, z = y (observation) represents what we want to regulate.
        
        Args:
            x: state (batch, 2)
            u: control input (batch, 1)
            
        Returns:
            z: performance output (batch, nz)
        """
        return self.h(x)

    def linearized_dynamics(self, x, u):
        device = x.device
        batch_size = x.shape[0]
        A = torch.zeros((batch_size, self.nx, self.nx))
        B = torch.zeros((batch_size, self.nx, self.nu))
        A[:, 0, 1] = 1
        A[:, 1, 0] = self.g / self.l * torch.sin(x[:, 0])
        A[:, 1, 1] = -self.beta / (self.inertia)
        B[:, 1, 0] = 1 / self.inertia
        return A.to(device), B.to(device)

    def h(self, x):
        return x[:, : self.ny]

    def linearized_observation(self, x):
        batch_size = x.shape[0]
        C = torch.zeros(batch_size, self.ny, self.nx, device=x.device)
        C[:, 0] = 1
        return C

    @property
    def x_equilibrium(self):
        return torch.zeros((2,))

    @property
    def u_equilibrium(self):
        return torch.zeros((1,))
