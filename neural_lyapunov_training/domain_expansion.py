"""
Dynamic domain expansion via closed-loop trajectory simulation (discrete-time).

Adapted from the Two-Stage paper (Li et al., 2025) `update_box_from_trajectories`,
but using discrete-time dynamics x_{k+1} = f(x_k, u_k) directly instead of
continuous-time ODE integration.

The idea: simulate trajectories from states OUTSIDE the currently verified ROA
(but still stabilizable), and expand the training box to cover the reachable
envelopes of those that converge. This grows the domain beyond what the
current levelset V < rho covers.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def simulate_discrete_trajectories(
    dynamics,
    controller,
    x0: torch.Tensor,
    steps: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate closed-loop discrete-time trajectories.

    Args:
        dynamics: DiscreteTimeSystem with forward(x, u) -> x_next.
        controller: nn.Module mapping x -> u.
        x0: [B, nx] initial conditions.
        steps: Number of discrete steps to simulate.

    Returns:
        (x_final, traj_min, traj_max): final states and per-trajectory
        coordinate-wise min/max envelopes (INCLUDING initial conditions).
    """
    x = x0.clone()
    traj_min = x.clone()
    traj_max = x.clone()

    with torch.no_grad():
        for _ in range(steps):
            u = controller(x)
            x = dynamics.forward(x, u)
            traj_min = torch.min(traj_min, x)
            traj_max = torch.max(traj_max, x)

    return x, traj_min, traj_max


def cap_box_growth(
    curr_lower: torch.Tensor,
    curr_upper: torch.Tensor,
    prop_lower: torch.Tensor,
    prop_upper: torch.Tensor,
    max_growth: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cap proposed box expansion to at most max_growth * current box size.

    Unlike simple multiplicative capping, this correctly handles boxes
    that aren't centered at the origin by capping the per-axis half-width
    growth factor.
    """
    curr_half = (curr_upper - curr_lower) / 2.0
    curr_center = (curr_upper + curr_lower) / 2.0
    max_half = curr_half * max_growth

    new_lower = torch.maximum(prop_lower, curr_center - max_half)
    new_upper = torch.minimum(prop_upper, curr_center + max_half)
    return new_lower, new_upper


def _sample_exploration_ics(
    lyapunov_fn,
    rho: float,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    num_trajectories: int,
    hard_lower: Optional[torch.Tensor] = None,
    hard_upper: Optional[torch.Tensor] = None,
    bdry_ratio: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample ICs for domain expansion from a WIDER probe box than the current training box.

    Key insight: our V is a PSD quadratic (unbounded), so V < k*rho maps to
    a tiny spatial ball near the origin. Filtering ICs by V gives points too
    close to origin; their trajectories barely escape the current box.

    Two-Stage avoids this because their sigmoid V maps V=0.9 directly to
    spatial ROA boundaries. For us, we must sample SPATIALLY at the frontier
    and let the convergence check in `update_domain_from_trajectories` decide
    which ICs actually stabilize — no V-based pre-filtering.

    Strategy: sample uniformly in probe box = bdry_ratio × current box,
    preferring the OUTER SHELL (states beyond the current training box)
    where new domain territory is found.

    Args:
        lyapunov_fn: Callable x -> V(x) [B,1]. (unused, kept for API compat)
        rho: Current certified ROA level. (unused, kept for API compat)
        lower_limit, upper_limit: Current training box.
        num_trajectories: Number of ICs to return.
        hard_lower, hard_upper: Absolute limits for probe box clamping.
        bdry_ratio: How far beyond the current box to probe (2.0 = 2× wider).

    Returns:
        (x0, V_vals): ICs and their V values (V_vals may be zeros)
    """
    device = lower_limit.device
    dtype = lower_limit.dtype
    nx = lower_limit.shape[0]

    # Probe box: bdry_ratio × current box, clamped to hard limits
    center = (upper_limit + lower_limit) / 2.0
    half = (upper_limit - lower_limit) / 2.0
    probe_lower = center - bdry_ratio * half
    probe_upper = center + bdry_ratio * half
    if hard_lower is not None:
        probe_lower = torch.maximum(probe_lower, hard_lower.to(device))
    if hard_upper is not None:
        probe_upper = torch.minimum(probe_upper, hard_upper.to(device))

    # Sample uniformly in the probe box — no V-based filtering.
    # The convergence simulation naturally selects stabilizable ICs.
    # Bias 2/3 of samples toward the OUTER SHELL (beyond current box)
    # to maximally probe new territory.
    n_inner = num_trajectories // 3
    n_outer = num_trajectories - n_inner

    # Inner: uniform in current training box
    x_inner = (
        torch.rand(n_inner, nx, device=device, dtype=dtype)
        * (upper_limit - lower_limit) + lower_limit
    )

    # Outer: uniform in probe box, reject if inside current box
    x_probe = (
        torch.rand(n_outer * 3, nx, device=device, dtype=dtype)
        * (probe_upper - probe_lower) + probe_lower
    )
    outside_mask = (
        (x_probe < lower_limit).any(dim=1) | (x_probe > upper_limit).any(dim=1)
    )
    x_outer_candidates = x_probe[outside_mask]
    if x_outer_candidates.shape[0] >= n_outer:
        x_outer = x_outer_candidates[:n_outer]
    else:
        # Not enough outside the box (box fills most of probe) — take all
        x_outer = x_probe[:n_outer]

    x0 = torch.cat([x_inner, x_outer], dim=0)
    V_vals = torch.zeros(x0.shape[0], device=device, dtype=dtype)

    return x0, V_vals


def update_domain_from_trajectories(
    dynamics,
    controller,
    lyapunov_fn,
    rho: float,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    goal_state: torch.Tensor,
    num_trajectories: int = 2000,
    traj_steps: int = 200,
    convergence_threshold: float = 0.01,
    expansion_factor: float = 1.0,
    max_growth: float = 2.0,
    hard_lower: Optional[torch.Tensor] = None,
    hard_upper: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Expand training domain based on converging closed-loop trajectories.

    Key insight: sample ICs near/beyond the ROA boundary (V ≈ rho), not deep
    inside it. Trajectories from V ≈ rho travel further before converging,
    pushing the bounding box outward.

    Process:
        1. Sample ICs from a probe box wider than the current training box
        2. Simulate discrete-time trajectories
        3. Keep only those that converge to the equilibrium
        4. Compute bounding box of their full reachable envelopes
        5. Expand with factor, cap growth, clamp to hard limits

    Returns:
        (new_lower, new_upper, num_converged)
    """
    device = lower_limit.device
    nx = lower_limit.shape[0]

    # 1. Sample ICs in a probe box wider than the current training box
    # (hard_lower/hard_upper as the outer limit, bdry_ratio=2 like Two-Stage)
    x0, V_x0 = _sample_exploration_ics(
        lyapunov_fn, rho, lower_limit, upper_limit,
        num_trajectories=num_trajectories,
        hard_lower=hard_lower,
        hard_upper=hard_upper,
        bdry_ratio=2.0,
    )

    if x0.shape[0] == 0:
        logger.info("Domain expansion: no candidates near ROA boundary")
        return lower_limit, upper_limit, 0

    # Compute probe box extents for logging
    center = (upper_limit + lower_limit) / 2.0
    half = (upper_limit - lower_limit) / 2.0
    probe_lower = center - 2.0 * half
    probe_upper = center + 2.0 * half
    if hard_lower is not None:
        probe_lower = torch.maximum(probe_lower, hard_lower.to(device))
    if hard_upper is not None:
        probe_upper = torch.minimum(probe_upper, hard_upper.to(device))
    probe_lower_str = ", ".join(f"{v:.3f}" for v in probe_lower.tolist())
    probe_upper_str = ", ".join(f"{v:.3f}" for v in probe_upper.tolist())

    logger.info(
        f"Domain expansion: simulating {x0.shape[0]} trajectories "
        f"(probe box: [{probe_lower_str}] to [{probe_upper_str}], rho={rho:.4f})"
    )

    # 2. Simulate
    x_final, traj_min, traj_max = simulate_discrete_trajectories(
        dynamics, controller, x0, steps=traj_steps
    )

    # 3. Filter converging
    goal = goal_state.to(device)
    converged = (x_final - goal).abs().max(dim=1)[0] < convergence_threshold
    num_converged = converged.sum().item()

    if num_converged == 0 or num_converged <= 0.01 * x0.shape[0]:
        logger.info(f"Domain expansion: only {num_converged}/{x0.shape[0]} converged (need >1%)")
        return lower_limit, upper_limit, 0

    conv_ratio = num_converged / x0.shape[0]

    # 4. Bounding box of converging trajectory envelopes (INCLUDING ICs)
    conv_x0 = x0[converged]
    global_min = torch.min(traj_min[converged].min(dim=0)[0], conv_x0.min(dim=0)[0])
    global_max = torch.max(traj_max[converged].max(dim=0)[0], conv_x0.max(dim=0)[0])

    # 5. Expand, cap, clamp
    new_lower = global_min * expansion_factor
    new_upper = global_max * expansion_factor

    new_lower, new_upper = cap_box_growth(
        lower_limit, upper_limit, new_lower, new_upper, max_growth
    )

    if hard_lower is not None:
        new_lower = torch.maximum(new_lower, hard_lower.to(device))
    if hard_upper is not None:
        new_upper = torch.minimum(new_upper, hard_upper.to(device))

    # Never shrink
    new_lower = torch.minimum(new_lower, lower_limit)
    new_upper = torch.maximum(new_upper, upper_limit)

    grew = not (torch.allclose(new_lower, lower_limit) and torch.allclose(new_upper, upper_limit))
    logger.info(
        f"Domain expansion: {num_converged}/{x0.shape[0]} converged ({conv_ratio:.1%}), "
        f"{'GREW' if grew else 'no change'} → "
        f"[{new_lower.tolist()}] to [{new_upper.tolist()}]"
    )

    return new_lower, new_upper, num_converged
