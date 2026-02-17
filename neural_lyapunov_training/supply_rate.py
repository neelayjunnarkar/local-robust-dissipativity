"""
Supply Rate Abstractions for Dissipativity-Based Stability Analysis.

Dissipativity condition: V(x_{t+1}) - V(x_t) ≤ s(w, z, V(x))

where:
    - V(x): storage function (Lyapunov function)
    - w: external disturbance input
    - z: performance output
    - s: supply rate

This module provides three supply rate types:
    1. Lyapunov:   s = -κ V(x)           → exponential stability
    2. L2-gain:    s = γ² ‖w‖² - ‖z‖²   → bounded L2 gain
    3. Passivity:  s = wᵀz               → passive systems
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn


class SupplyRateType(Enum):
    """Enumeration of supported supply rate types."""
    LYAPUNOV = "lyapunov"
    L2GAIN = "l2gain"
    PASSIVITY = "passivity"


class SupplyRate(ABC, nn.Module):
    """
    Abstract base class for supply rates in dissipativity analysis.
    
    The dissipativity inequality is:
        V(x_{t+1}) - V(x_t) ≤ s(w, z, V(x))
    
    For verification, we check:
        V(x_t) - V(x_{t+1}) + s(w, z, V(x)) ≥ 0
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(
        self,
        w: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        V_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the supply rate s(w, z, V(x)).
        
        Args:
            w: Disturbance input, shape (batch, w_dim) or None
            z: Performance output, shape (batch, z_dim) or None
            V_x: Lyapunov function value V(x), shape (batch, 1)
            
        Returns:
            Supply rate value, shape (batch, 1)
        """
        pass
    
    @property
    @abstractmethod
    def requires_disturbance(self) -> bool:
        """Whether this supply rate requires disturbance input w."""
        pass
    
    @property
    @abstractmethod
    def requires_output(self) -> bool:
        """Whether this supply rate requires performance output z."""
        pass


class LyapunovSupplyRate(SupplyRate):
    """
    Lyapunov supply rate: s = -κ V(x)
    
    This yields the standard Lyapunov decrease condition:
        V(x_{t+1}) - V(x_t) ≤ -κ V(x)
        ⟺ V(x_{t+1}) ≤ (1-κ) V(x)
    
    Guarantees exponential stability with rate κ.
    When κ = 0, this reduces to simple stability (V non-increasing).
    """
    
    def __init__(self, kappa: float = 0.001):
        """
        Args:
            kappa: Decay rate, must satisfy 0 ≤ κ < 1
                   κ = 0: simple stability (V non-increasing)
                   κ > 0: exponential stability
        """
        super().__init__()
        assert 0 <= kappa < 1, f"kappa must be in [0, 1), got {kappa}"
        self.kappa = kappa
    
    def forward(
        self,
        w: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        V_x: torch.Tensor
    ) -> torch.Tensor:
        """Returns s = -κ V(x)."""
        return -self.kappa * V_x
    
    @property
    def requires_disturbance(self) -> bool:
        return False
    
    @property
    def requires_output(self) -> bool:
        return False


class L2GainSupplyRate(SupplyRate):
    """
    L2-gain supply rate: s = γ² ‖w‖² - ‖z‖²
    
    This yields the L2-gain condition:
        V(x_{t+1}) - V(x_t) ≤ γ² ‖w‖² - ‖z‖²
    
    Summing over time and using V ≥ 0:
        Σ ‖z_t‖² ≤ γ² Σ ‖w_t‖² + V(x_0)
    
    Guarantees L2-gain from w to z is at most γ.
    """
    
    def __init__(self, gamma: float = 1.0):
        """
        Args:
            gamma: L2-gain bound, must be positive
        """
        super().__init__()
        assert gamma > 0, f"gamma must be positive, got {gamma}"
        self.gamma = gamma
    
    def forward(
        self,
        w: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        V_x: torch.Tensor
    ) -> torch.Tensor:
        """Returns s = γ² ‖w‖² - ‖z‖²."""
        assert w is not None, "L2-gain supply rate requires disturbance w"
        assert z is not None, "L2-gain supply rate requires output z"
        
        w_norm_sq = (w ** 2).sum(dim=-1, keepdim=True)
        z_norm_sq = (z ** 2).sum(dim=-1, keepdim=True)
        
        return  w_norm_sq - ((1/self.gamma) ** 2 ) * z_norm_sq
    
    @property
    def requires_disturbance(self) -> bool:
        return True
    
    @property
    def requires_output(self) -> bool:
        return True


class PassivitySupplyRate(SupplyRate):
    """
    Passivity supply rate: s = wᵀz
    
    This yields the passivity condition:
        V(x_{t+1}) - V(x_t) ≤ wᵀz
    
    The system is passive: it can only store energy supplied from outside.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        w: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        V_x: torch.Tensor
    ) -> torch.Tensor:
        """Returns s = wᵀz."""
        assert w is not None, "Passivity supply rate requires disturbance w"
        assert z is not None, "Passivity supply rate requires output z"
        assert w.shape == z.shape, f"w and z must have same shape, got {w.shape} and {z.shape}"
        
        return (w * z).sum(dim=-1, keepdim=True)
    
    @property
    def requires_disturbance(self) -> bool:
        return True
    
    @property
    def requires_output(self) -> bool:
        return True


def create_supply_rate(config) -> SupplyRate:
    """
    Factory function to create supply rate from config.
    
    Args:
        config: Configuration object with 'type' and type-specific parameters
        
    Returns:
        SupplyRate instance
        
    Example config:
        supply_rate:
          type: lyapunov
          kappa: 0.001
        
        supply_rate:
          type: l2gain
          gamma: 1.0
    """
    supply_type = config.get("type", "lyapunov").lower()
    
    if supply_type == "lyapunov":
        return LyapunovSupplyRate(kappa=config.get("kappa", 0.001))
    elif supply_type == "l2gain":
        return L2GainSupplyRate(gamma=config.get("gamma", 1.0))
    elif supply_type == "passivity":
        return PassivitySupplyRate()
    else:
        raise ValueError(f"Unknown supply rate type: {supply_type}")

