"""
System Registry — Factory functions for creating dynamical systems.

Maps system names to factory functions that construct the continuous-time
dynamics, discrete-time wrapper, state labels, and measurement function.
"""

import torch
import torch.nn as nn
from typing import Optional

import neural_lyapunov_training.dynamical_system as dynamical_system


# ═══════════════════════════════════════════════════════════════════════
# Measurement output modules (nn.Module for auto_LiRPA compatibility)
# ═══════════════════════════════════════════════════════════════════════

class LinearMeasurement(nn.Module):
    """
    Linear measurement y = C x.

    nn.Module so auto_LiRPA can trace it.
    """

    def __init__(self, C: torch.Tensor):
        super().__init__()
        self.register_buffer('C', C.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.C.T


class FullStateMeasurement(nn.Module):
    """Identity measurement y = x (full-state feedback)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ═══════════════════════════════════════════════════════════════════════
# System factory
# ═══════════════════════════════════════════════════════════════════════

def create_system(cfg, device=None, dtype=torch.float32):
    """
    Create a dynamical system from Hydra config.

    Args:
        cfg: Full Hydra DictConfig.  Must contain ``system.type``.
        device: torch device.
        dtype: torch dtype.

    Returns:
        dynamics:        DiscreteTimeSystem instance.
        state_labels:    List of str for plot axes.
        output_fn:       nn.Module or None (measurement for RINN/LTIC).
        continuous_sys:  The continuous-time system (for LQR etc.).
    """
    sys_cfg = cfg.get('system', {})
    system_type = sys_cfg.get('type', 'pendulum')
    model_cfg = cfg.model

    dt = model_cfg.dt
    output_C = model_cfg.get('output_C', None)
    output_D = model_cfg.get('output_D', None)
    if output_C is not None:
        output_C = torch.tensor(output_C, dtype=dtype, device=device)
    if output_D is not None:
        output_D = torch.tensor(output_D, dtype=dtype, device=device)

    if system_type == 'pendulum':
        return _create_pendulum(model_cfg, dt, output_C, output_D, device, dtype)
    elif system_type == 'flexible_rod':
        return _create_flexible_rod(sys_cfg, model_cfg, dt, output_C, output_D, device, dtype)
    else:
        raise ValueError(f"Unknown system type: {system_type}")


def _create_pendulum(model_cfg, dt, output_C, output_D, device, dtype):
    """Create pendulum system (backward-compatible with existing configs)."""
    import neural_lyapunov_training.pendulum as pendulum_mod

    sys_params = {}
    # Allow system-specific overrides under system.params
    m = 0.15
    l = 0.5
    beta = 0.1

    continuous_sys = pendulum_mod.PendulumDynamics(
        m=m, l=l, beta=beta, output_C=output_C, output_D=output_D,
    )

    pos_int = dynamical_system.IntegrationMethod[
        model_cfg.get('position_integration', 'MidPoint')
    ]
    vel_int = dynamical_system.IntegrationMethod[
        model_cfg.get('velocity_integration', 'ExplicitEuler')
    ]

    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        continuous_sys, dt=dt,
        position_integration=pos_int,
        velocity_integration=vel_int,
    )

    state_labels = [r"$\theta$ (rad)", r"$\dot{\theta}$ (rad/s)"]
    output_fn = None  # full-state feedback

    return dynamics, state_labels, output_fn, continuous_sys


def _create_flexible_rod(sys_cfg, model_cfg, dt, output_C, output_D, device, dtype):
    """Create flexible rod on cart system."""
    import neural_lyapunov_training.flexible_rod as flexible_rod_mod

    params = sys_cfg.get('params', {})
    continuous_sys = flexible_rod_mod.FlexibleRodDynamics(
        m_b=params.get('m_b', 1.0),
        m_t=params.get('m_t', 0.1),
        L=params.get('L', 1.0),
        rho_l=params.get('rho_l', 0.1),
        output_C=output_C,
        output_D=output_D,
    )

    dynamics = dynamical_system.FirstOrderDiscreteTimeSystem(
        continuous_sys, dt=dt,
        integration=dynamical_system.IntegrationMethod.ExplicitEuler,
    )

    state_labels = [r"$p$ (m)", r"$\dot{p}$ (m/s)"]

    # Measurement: position only (n_y=1) for output-feedback controllers
    measurement = sys_cfg.get('measurement', 'position')
    if measurement == 'full_state':
        output_fn = None  # y = x_p
    else:
        # y = Cpy @ x = position only
        Cpy = continuous_sys.Cpy.to(device=device, dtype=dtype)
        output_fn = LinearMeasurement(Cpy)

    return dynamics, state_labels, output_fn, continuous_sys
