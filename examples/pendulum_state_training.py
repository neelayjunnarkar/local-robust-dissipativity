import os
import pdb

# Set wandb API key
os.environ["WANDB_API_KEY"] = "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"

import argparse
import copy
import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
import scipy.linalg
import torch
import torch.nn as nn
import wandb

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.supply_rate as supply_rate_module
import neural_lyapunov_training.train_utils as train_utils

device = torch.device("cpu")
dtype = torch.float


# =============================================================================
# Supply Rate Configuration (Dissipativity Framework)
# =============================================================================

def create_supply_rate(cfg, kappa: float) -> supply_rate_module.SupplyRate:
    """Create supply rate from config. Defaults to Lyapunov for backward compatibility."""
    if not hasattr(cfg, 'supply_rate') or cfg.supply_rate is None:
        return supply_rate_module.LyapunovSupplyRate(kappa=kappa)
    
    supply_type = cfg.supply_rate.get('type', 'lyapunov').lower()
    
    if supply_type == 'lyapunov':
        return supply_rate_module.LyapunovSupplyRate(kappa=kappa)
    elif supply_type == 'l2gain':
        return supply_rate_module.L2GainSupplyRate(gamma=cfg.supply_rate.get('gamma', 1.0))
    elif supply_type == 'passivity':
        return supply_rate_module.PassivitySupplyRate()
    else:
        raise ValueError(f"Unknown supply rate type: {supply_type}")


def get_val(val, n):
    if isinstance(val, (list, ListConfig)):
        return val[n]
    return val


def get_w_max(cfg):
    """Get disturbance bound from config."""
    if hasattr(cfg, 'supply_rate') and cfg.supply_rate is not None:
        w_max = cfg.supply_rate.get('w_max', None)
        if w_max is not None:
            return torch.tensor([w_max])
    return None


@torch.no_grad()
def diagnose_adversarial_examples(
    examples,          # list of lists/tensors, each [x1, ..., xn, w1, ..., wm]
    lyapunov_nn,
    dynamics,
    controller,
    supply_rate_fn,
    s_scale: float,
    rho: float,        # verified rho (= levelset c)
    nx: int,
    logger,
    label: str = "",
):
    """Print detailed diagnostics for user-specified adversarial [x, w] examples."""
    if not examples:
        return
    c = rho
    is_l2 = isinstance(supply_rate_fn, supply_rate_module.L2GainSupplyRate)
    header = f"ADVERSARIAL EXAMPLE DIAGNOSTICS ({label})" if label else "ADVERSARIAL EXAMPLE DIAGNOSTICS"
    logger.info("=" * 70)
    logger.info(header)
    logger.info(f"  rho (c) = {c}")
    logger.info("=" * 70)

    for i, ex in enumerate(examples):
        ex_t = torch.tensor(ex, dtype=torch.float).unsqueeze(0)  # (1, nx+nw)
        x = ex_t[:, :nx]
        w = ex_t[:, nx:] if ex_t.shape[1] > nx else None

        # V(x)
        V_x = lyapunov_nn(x)

        # Controller output
        u = controller(x)

        # Next state
        x_next = dynamics.forward(x, u, w)

        # V(x_next)
        V_next = lyapunov_nn(x_next)

        # Performance output z
        z = dynamics.output(x, u)

        # Supply rate
        if w is not None:
            s = supply_rate_fn(w, z, V_x)
        else:
            s = supply_rate_fn(None, z, V_x)

        logger.info(f"--- Example {i} ---")
        logger.info(f"  x     = {x.squeeze(0).tolist()}")
        if w is not None:
            logger.info(f"  w     = {w.squeeze(0).tolist()}")
        logger.info(f"  u     = {u.squeeze(0).tolist()}")
        logger.info(f"  x_next= {x_next.squeeze(0).tolist()}")

        in_set = -V_x + c
        logger.info(f"  in set:    -V(x) + c = {in_set.item():.6e}  "
                     f"(V(x)={V_x.item():.6e}, c={c:.6e})")

        still_in = -V_next + c
        logger.info(f"  still in:  -V(F(x,d)) + c = {still_in.item():.6e}  "
                     f"(V(F)={V_next.item():.6e}, c={c:.6e})")

        V_decrease = V_x - V_next
        dissip = V_x - V_next + s_scale * s
        logger.info(f"  dissipativ: V(x)-V(F)+s_scale*s = {dissip.item():.6e}  "
                     f"(V(x)={V_x.item():.6e}, V(F)={V_next.item():.6e}, "
                     f"s_scale*s={( s_scale * s).item():.6e})")

        if is_l2:
            gamma = supply_rate_fn.gamma
            w_norm_sq = (w ** 2).sum(dim=-1, keepdim=True) if w is not None else torch.tensor([[0.0]])
            z_norm_sq = (z ** 2).sum(dim=-1, keepdim=True)
            logger.info(f"  perf_out z = {z.squeeze(0).tolist()}, 1/gamma^2 = {(1/gamma)**2:.6e}")
            logger.info(f"  supply: ||w||^2 - 1/gamma^2 ||z||^2 = {s.item():.6e}  "
                         f"(||w||^2={w_norm_sq.item():.6e}, "
                         f"1/g^2*||z||^2={((1/gamma)**2 * z_norm_sq).item():.6e})")

        logger.info(f"  V decrease: V(x)-V(F) = {V_decrease.item():.6e}")

        # Verdict
        violated = []
        if in_set.item() < 0:
            violated.append("NOT in sublevel set")
        if still_in.item() < 0:
            violated.append("x_next NOT in sublevel set")
        if dissip.item() < 0:
            violated.append("DISSIPATIVITY VIOLATED")
        if violated:
            logger.warning(f"  >> VIOLATIONS: {', '.join(violated)}")
        else:
            logger.info(f"  >> ALL CONDITIONS SATISFIED")
    logger.info("=" * 70)


@torch.no_grad()
def sample_init_roa_anchors(
    lyapunov_nn,
    init_rho: float,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    nx: int,
    num_anchors: int = 256,
    expand_factor: float = 1.5,
    num_candidates: int = 500_000,
    logger=None,
) -> torch.Tensor:
    """
    Sample anchor points from the initial verified ROA for growth incentive.

    Returns points in two bands:
      - Boundary band:  V₀(x) ∈ [0.7·ρ₀, ρ₀]       (anchors — prevent shrinking)
      - Beyond band:    V₀(x) ∈ [ρ₀, α·ρ₀]          (stretch goals — push growth)

    Half the budget goes to each band; shortfall is filled from the other.
    """
    device = lower_limit.device
    n_bdry = num_anchors // 2
    n_beyond = num_anchors - n_bdry

    # Uniform samples in the box
    x_cand = (
        torch.rand(num_candidates, nx, device=device)
        * (upper_limit - lower_limit)
        + lower_limit
    )
    V_vals = lyapunov_nn(x_cand).squeeze(-1)

    # Boundary band: V ∈ [0.7ρ, ρ]
    bdry_mask = (V_vals >= 0.7 * init_rho) & (V_vals <= init_rho)
    bdry_pts = x_cand[bdry_mask]

    # Beyond band: V ∈ [ρ, α·ρ]
    beyond_mask = (V_vals > init_rho) & (V_vals <= expand_factor * init_rho)
    beyond_pts = x_cand[beyond_mask]

    # Fill each band up to budget; overflow goes to the other band
    if bdry_pts.shape[0] > n_bdry:
        idx = torch.randperm(bdry_pts.shape[0])[:n_bdry]
        bdry_pts = bdry_pts[idx]
    if beyond_pts.shape[0] > n_beyond:
        idx = torch.randperm(beyond_pts.shape[0])[:n_beyond]
        beyond_pts = beyond_pts[idx]

    anchors = torch.cat([bdry_pts, beyond_pts], dim=0)

    # If we got fewer than requested, that's fine — use what we have
    if logger:
        logger.info(
            f"Init ROA anchors: {bdry_pts.shape[0]} boundary + "
            f"{beyond_pts.shape[0]} beyond-boundary = {anchors.shape[0]} total "
            f"(requested {num_anchors}, expand_factor={expand_factor})"
        )
    return anchors


@torch.no_grad()
def roa_bounding_box(
    lyapunov_nn,
    rho: float,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    nx: int,
    margin: float = 1.2,
    num_samples: int = 500_000,
) -> torch.Tensor:
    """
    Compute an axis-aligned bounding box of {x : V(x) ≤ ρ} by sampling.

    Returns half-widths per dimension (positive), already multiplied by *margin*.
    The resulting box is ±half_widths.  Clamped to [lower_limit, upper_limit].
    """
    device = lower_limit.device
    x = (torch.rand(num_samples, nx, device=device)
         * (upper_limit - lower_limit) + lower_limit)
    V = lyapunov_nn(x).squeeze(-1)
    inside = x[V <= rho]                       # (M, nx)
    if inside.shape[0] == 0:
        # No points found — fall back to tiny box
        return torch.full((nx,), 0.01, device=device)
    half = inside.abs().max(dim=0).values      # per-axis max absolute coord
    half = half * margin                       # add margin
    # Clamp to hard limits
    hard_half = torch.max(upper_limit.abs(), lower_limit.abs())
    half = torch.min(half, hard_half)
    return half


def linearize_pendulum(pendulum_continuous: pendulum.PendulumDynamics):
    x = torch.tensor([[0.0, 0.0]])
    x.requires_grad = True
    u = torch.tensor([[0.0]])
    u.requires_grad = True
    qddot = pendulum_continuous.forward(x, u)
    A = torch.empty((2, 2))
    B = torch.empty((2, 1))
    A[0, 0] = 0
    A[0, 1] = 1
    B[0, 0] = 0
    A[1], B[1] = torch.autograd.grad(qddot[0, 0], [x, u])
    return A, B


def compute_lqr(pendulum_continuous: pendulum.PendulumDynamics):
    A, B = linearize_pendulum(pendulum_continuous)
    A_np, B_np = A.detach().numpy(), B.detach().numpy()
    Q = np.eye(2)
    R = np.eye(1) * 100
    S = scipy.linalg.solve_continuous_are(A_np, B_np, Q, R)
    K = -np.linalg.solve(R, B_np.T @ S)
    return K, S


def approximate_lqr(
    pendulum_continuous: pendulum.PendulumDynamics,
    controller: controllers.NeuralNetworkController,
    lyapunov_nn: lyapunov.NeuralNetworkLyapunov,
    upper_limit: torch.Tensor,
    logger,
):
    K, S = compute_lqr(pendulum_continuous)
    K_torch = torch.from_numpy(K).type(dtype).to(device)
    S_torch = torch.from_numpy(S).type(dtype).to(device)
    x = (torch.rand((100000, 2), dtype=dtype, device=device) - 0.5) * 2 * upper_limit
    V = torch.sum(x * (x @ S_torch), axis=1, keepdim=True)
    u = x @ K_torch.T

    def approximate(system, system_input, target, lr, max_iter):
        optimizer = torch.optim.Adam(system.parameters(), lr=lr)
        for i in range(max_iter):
            optimizer.zero_grad()
            output = torch.nn.MSELoss()(system.forward(system_input), target)
            logger.info(f"iter {i}, loss {output.item()}")
            output.backward()
            optimizer.step()

    approximate(controller, x, u, lr=0.01, max_iter=500)
    approximate(lyapunov_nn, x, V, lr=0.01, max_iter=1000)


def plot_V(V, lower_limit, upper_limit, S_norm=None):
    # If S_norm is provided, lower/upper_limit are in z-space.
    x_ticks = torch.linspace(lower_limit[0], upper_limit[0], 50, device=device)
    y_ticks = torch.linspace(lower_limit[1], upper_limit[1], 50, device=device)
    grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks, indexing='ij')
    with torch.no_grad():
        V_val = V(torch.stack((grid_x, grid_y), dim=2)).squeeze(2)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(grid_x.cpu().numpy(), grid_y.cpu().numpy(), V_val.cpu().numpy())
    if S_norm is not None:
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
    else:
        ax.set_xlabel(r"$\theta (rad)$")
        ax.set_ylabel(r"$\dot{\theta} (rad/s)$")
    ax.set_zlabel("V")
    return fig, ax


def plot_V_heatmap(V, lower_limit, upper_limit, rho, dynamics=None, trajectories=None):
    # If dynamics is NOT provided, it's just the old physical space plotting.
    # If dynamics IS provided, lower_limit/upper_limit are in z-space.
    
    if dynamics is None:
        # Standard physical plotting
        x_ticks = torch.linspace(lower_limit[0], upper_limit[0], 1000, device=device)
        y_ticks = torch.linspace(lower_limit[1], upper_limit[1], 1000, device=device)
        grid_x, grid_y = torch.meshgrid(x_ticks, y_ticks, indexing='ij')
        with torch.no_grad():
            V_val = V.forward(torch.stack((grid_x, grid_y), dim=2)).squeeze(2)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        im = ax.pcolormesh(grid_x.cpu(), grid_y.cpu(), V_val.cpu(), shading='auto')
        ax.contour(grid_x.cpu(), grid_y.cpu(), V_val.cpu(), [rho], colors="red", linewidths=2)
        
        if trajectories is not None:
            traj_data = trajectories.cpu().detach().numpy()
            if traj_data.ndim == 2:
                ax.plot(traj_data[:, 0], traj_data[:, 1], color='white', alpha=0.8, linewidth=1)
            else:
                for i in range(traj_data.shape[1]):
                     ax.plot(traj_data[:, i, 0], traj_data[:, i, 1], alpha=0.5, linewidth=0.5)

        ax.set_xlabel(r"$\theta$ (rad)")
        ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
        ax.set_title(f"Lyapunov V (Physical Coordinates), ρ={rho:.4f}")
        ax.grid(True, alpha=0.3)
        fig.colorbar(im, ax=ax)
        return fig, ax, None

    # Dual Plotting Mode (Normalized Error Coordinates)
    fig, (ax_z, ax_x) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. z-space plot (Normalized)
    z_ticks_1 = torch.linspace(lower_limit[0], upper_limit[0], 500, device=device)
    z_ticks_2 = torch.linspace(lower_limit[1], upper_limit[1], 500, device=device)
    grid_z1, grid_z2 = torch.meshgrid(z_ticks_1, z_ticks_2, indexing='ij')
    z_pts = torch.stack((grid_z1, grid_z2), dim=2)
    with torch.no_grad():
        V_z = V.forward(z_pts).squeeze(-1) 
    
    im_z = ax_z.pcolormesh(grid_z1.cpu(), grid_z2.cpu(), V_z.cpu(), shading='auto')
    ax_z.contour(grid_z1.cpu(), grid_z2.cpu(), V_z.cpu(), [rho], colors="red", linewidths=2)
    
    if trajectories is not None:
        # trajectories in z-space
        z_traj_data = trajectories.cpu().detach().numpy()
        N_steps = z_traj_data.shape[0]
        if z_traj_data.ndim == 2:
            for t in range(N_steps - 1):
                ax_z.plot(z_traj_data[t:t+2, 0], z_traj_data[t:t+2, 1], alpha=0.8)
        else:
            for i in range(z_traj_data.shape[1]):
                for t in range(N_steps - 1):
                    ax_z.plot(z_traj_data[t:t+2, i, 0], z_traj_data[t:t+2, i, 1], alpha=0.5)

    ax_z.set_xlabel(r"$z_1$ (Norm. Error Pos)")
    ax_z.set_ylabel(r"$z_2$ (Norm. Error Vel)")
    ax_z.set_title(f"Normalized z-space (Energy Units), ρ={rho:.6f}")
    ax_z.set_aspect('equal')
    ax_z.grid(True, alpha=0.3)
    
    # Draw z-box boundary (Rectangle)
    from matplotlib.patches import Rectangle
    rect_z = Rectangle((lower_limit[0].cpu(), lower_limit[1].cpu()), 
                       (upper_limit[0]-lower_limit[0]).cpu(), 
                       (upper_limit[1]-lower_limit[1]).cpu(),
                       linewidth=2, edgecolor='white', facecolor='none', linestyle='--', label='Training Box')
    ax_z.add_patch(rect_z)
    fig.colorbar(im_z, ax=ax_z)

    # 2. x-space plot (Physical)
    # Map physical limits based on z-box corners (Ordered for Polygon)
    corners_z = torch.tensor([
        [lower_limit[0], lower_limit[1]], # bl
        [upper_limit[0], lower_limit[1]], # br
        [upper_limit[0], upper_limit[1]], # tr
        [lower_limit[0], upper_limit[1]]  # tl
    ], device=device)
    
    corners_x = dynamics._z_to_x(corners_z)
    
    x_min_box, _ = corners_x.min(dim=0)
    x_max_box, _ = corners_x.max(dim=0)
    
    # Tight bounding box with 10% margin
    x_range_box = x_max_box - x_min_box
    plot_x_min = x_min_box - 0.1 * x_range_box
    plot_x_max = x_max_box + 0.1 * x_range_box
    
    x_ticks_1 = torch.linspace(plot_x_min[0].item(), plot_x_max[0].item(), 500, device=device)
    x_ticks_2 = torch.linspace(plot_x_min[1].item(), plot_x_max[1].item(), 500, device=device)
    grid_x1, grid_x2 = torch.meshgrid(x_ticks_1, x_ticks_2, indexing='ij')
    x_pts_phys = torch.stack((grid_x1, grid_x2), dim=2)
    
    # x -> z -> V: 
    z_pts_from_x = dynamics._x_to_z(x_pts_phys)
    with torch.no_grad():
        V_x = V.forward(z_pts_from_x).squeeze(-1)
    
    im_x = ax_x.pcolormesh(grid_x1.cpu(), grid_x2.cpu(), V_x.cpu(), shading='auto')
    ax_x.contour(grid_x1.cpu(), grid_x2.cpu(), V_x.cpu(), [rho], colors="red", linewidths=2)
    
    if trajectories is not None:
        # trajectories in x-space via wrapper mapping
        x_traj_data = dynamics._z_to_x(trajectories).cpu().detach().numpy()
        if x_traj_data.ndim == 2:
            for t in range(N_steps - 1):
                ax_x.plot(x_traj_data[t:t+2, 0], x_traj_data[t:t+2, 1], alpha=0.8)
        else:
            for i in range(x_traj_data.shape[1]):
                for t in range(N_steps - 1):
                    ax_x.plot(x_traj_data[t:t+2, i, 0], x_traj_data[t:t+2, i, 1], alpha=0.5)

    # Draw rotated training box in x-space (Rotated Rectangle via Eigen-Norm)
    from matplotlib.patches import Polygon
    poly_x = Polygon(corners_x.cpu().numpy(), linewidth=2, edgecolor='white', facecolor='none', linestyle='--', label='Training Box')
    ax_x.add_patch(poly_x)
    
    ax_x.set_xlabel(r"$\theta$ (rad)")
    ax_x.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax_x.set_title(f"Physical x-space (Centered at Upright), ρ={rho:.6f}")
    ax_x.set_xlim(plot_x_min[0].item(), plot_x_max[0].item())
    ax_x.set_ylim(plot_x_min[1].item(), plot_x_max[1].item())
    # ax_x.set_aspect('equal') # Essential for "rotated box" shape visualization
    ax_x.grid(True, alpha=0.3)
    fig.colorbar(im_x, ax=ax_x)
    
    plt.tight_layout()
    return fig, (ax_z, ax_x), None


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


# =============================================================================
# PGD Verification with Bisection
# =============================================================================

def pgd_find_verified_rho(
    lyapunov_nn,
    dynamics,
    controller,
    supply_rate,
    lower_limit,
    upper_limit,
    w_max,
    s_scale,
    V_decrease_within_roa=True,
    pgd_steps=300,
    num_seeds=5,
    num_samples=50000,
    num_samples_per_boundary=1024,
    rho_bisect_tol=0.0001,
    max_bisect_iters=20,
    logger=None,
):
    """Find the largest PGD-verified rho for the current model state.

    Uses bisection on ``rho_multiplier`` to find the largest sublevel set
    {x : V(x) <= rho} within the box where PGD cannot find counter-examples
    to the dissipativity condition.

    Args:
        lyapunov_nn: NeuralNetworkLyapunov instance.
        dynamics: DiscreteTimeSystem (possibly augmented for LTIC).
        controller: Controller module.
        supply_rate: SupplyRate instance.
        lower_limit, upper_limit: Box limits for the verification domain.
        w_max: Disturbance bound tensor or None.
        s_scale: Supply-rate scaling factor.
        V_decrease_within_roa: Whether to enforce V decrease only inside ROA.
        pgd_steps: PGD attack steps per seed.
        num_seeds: Number of random restarts.
        num_samples: Number of random starting points per seed.
        num_samples_per_boundary: Samples per face for boundary-V computation.
        rho_bisect_tol: Absolute tolerance on rho for bisection termination.
        max_bisect_iters: Maximum bisection iterations.
        logger: Optional logger.

    Returns:
        verified_rho (float): Largest rho passing PGD verification.
        max_rho (float): Theoretical maximum rho = min V(x) on box boundary.
        pgd_clean_at_max (bool): Whether full max_rho passed verification.
    """
    nx = lyapunov_nn.x_dim
    limit = upper_limit  # symmetric box assumed

    # ---- 1. Compute max possible rho = min V on box boundary ----
    x_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
        lyapunov_nn, lower_limit, upper_limit,
        num_samples_per_boundary=num_samples_per_boundary,
        eps=(upper_limit - lower_limit) / 2,
        steps=100, direction="minimize",
    ).detach()

    with torch.no_grad():
        max_rho = lyapunov_nn(x_boundary).min().item()

    if max_rho <= 0:
        if logger:
            logger.info("pgd_find_verified_rho: max_rho <= 0, no valid ROA.")
        return 0.0, max_rho, False

    # ---- helper: check whether a given rho_multiplier verifies ----
    def _check_rho_mult(rho_mult):
        loss_fn = lyapunov.DissipativityDerivativeLoss(
            dynamics, controller, lyapunov_nn,
            supply_rate=supply_rate,
            box_lo=lower_limit, box_up=upper_limit,
            rho_multiplier=rho_mult,
            w_max=w_max, hard_max=True, s_scale=s_scale,
        )
        loss_fn.x_boundary = x_boundary

        use_adv_w = supply_rate.requires_disturbance and w_max is not None
        if use_adv_w:
            nw = getattr(dynamics.continuous_time_system, 'nw', 1) if hasattr(dynamics, 'continuous_time_system') else 1
            ver_loss = lyapunov.DissipativityVerificationWrapper(loss_fn, nx, nw)
            limit_w = w_max
            limit_joint = torch.cat([limit, limit_w])
            lower_joint = torch.cat([lower_limit, -limit_w])
            upper_joint = torch.cat([upper_limit, limit_w])
            in_dim = nx + nw
        else:
            ver_loss = loss_fn
            limit_joint = limit
            lower_joint = lower_limit
            upper_joint = upper_limit
            in_dim = nx

        for seed in range(num_seeds):
            train_utils.set_seed(seed)
            if V_decrease_within_roa:
                x_min_bdry = train_utils.calc_V_extreme_on_boundary_pgd(
                    lyapunov_nn, lower_limit, upper_limit,
                    num_samples_per_boundary=num_samples_per_boundary,
                    eps=(upper_limit - lower_limit) / 2,
                    steps=100, direction="minimize",
                ).detach()
                loss_fn.x_boundary = torch.cat([x_min_bdry, x_boundary], dim=0)

            x_start = (torch.rand((num_samples, in_dim), device=device) - 0.5) * limit_joint * 2
            adv_x = train_utils.pgd_attack(
                x_start, ver_loss,
                eps=limit_joint, steps=pgd_steps,
                lower_boundary=lower_joint, upper_boundary=upper_joint,
                direction="minimize",
            ).detach()
            adv_out = torch.clamp(-ver_loss(adv_x), min=0.0)
            # Threshold scales with max_rho so normalised P (small V values)
            # doesn't hide violations behind a fixed 1e-4 cutoff.
            violation_tol = 0
            # max(1e-9, max_rho * 1e-3)
            if adv_out.max().item() > violation_tol:
                return False
        return True

    # ---- 2. Check full rho first ----
    full_clean = _check_rho_mult(1.0)
    if full_clean:
        if logger:
            logger.info(f"pgd_find_verified_rho: full max_rho={max_rho:.6f} verified clean.")
        return max_rho, max_rho, True

    # ---- 3. Bisect rho_multiplier in [0, 1] ----
    lo, hi = 0.0, 1.0
    for i in range(max_bisect_iters):
        mid = (lo + hi) / 2
        if _check_rho_mult(mid):
            lo = mid
        else:
            hi = mid
        current_rho = lo * max_rho
        if logger:
            logger.info(
                f"  bisect iter {i}: rho_mult=[{lo:.4f}, {hi:.4f}], "
                f"verified_rho={current_rho:.6f}"
            )
        # Terminate when the interval width (in rho_mult space) is small enough.
        # Use a relative tolerance so the bisection isn't fooled by a tiny max_rho.
        if (hi - lo) < rho_bisect_tol:
            break

    verified_rho = lo * max_rho
    if logger:
        logger.info(
            f"pgd_find_verified_rho: verified_rho={verified_rho:.6f} "
            f"(max_rho={max_rho:.6f}, rho_mult={lo:.4f})"
        )
    return verified_rho, max_rho, False




# =============================================================================
# Ellipsoid Projection Utilities
# =============================================================================

def project_ellipsoid(P: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Project the ellipsoid {x : x^T P x ≤ 1} onto the range of A.

    Ports the MATLAB function:
        function Phat = project(P, A)
            [U, S, V] = svd(A);  Sm = S(1:m,1:m);  V1 = V(:,1:m);  V2 = V(:,m+1:end);
            W = [U*Sm*V1'; V2'];
            Ptilde = inv(W') * P * inv(W);
            Phat = Ptilde11 - Ptilde12 * inv(Ptilde22) * Ptilde12';

    Returns Phat (m×m) such that the projected ellipsoid is {y : y^T Phat y ≤ 1}.
    """
    m = A.shape[0]
    assert m == np.linalg.matrix_rank(A), "A must have full row rank"
    U, s, Vh = np.linalg.svd(A, full_matrices=True)
    V = Vh.T          # n×n
    Sm = np.diag(s[:m])
    V1 = V[:, :m]     # n×m
    V2 = V[:, m:]     # n×(n-m)
    # W is n×n:  W = [U Sm V1^T ;  V2^T]
    W = np.vstack([U @ Sm @ V1.T, V2.T])
    # Ptilde = W^{-T} P W^{-1}   (MATLAB: W'\P * inv(W))
    Winv = np.linalg.inv(W)
    Ptilde = Winv.T @ P @ Winv
    Ptilde11 = Ptilde[:m, :m]
    Ptilde12 = Ptilde[:m, m:]
    Ptilde22 = Ptilde[m:, m:]
    Phat = Ptilde11 - Ptilde12 @ np.linalg.solve(Ptilde22, Ptilde12.T)
    return Phat


def ellipse_from_Phat(Phat: np.ndarray, rho: float, n_pts: int = 500):
    """Return (x, y) arrays for the ellipse {y : y^T Phat y = rho}.

    Uses the Cholesky factorisation Phat = L L^T / rho, then maps a unit circle
    through L^{-T}.
    """
    # Eigendecomposition: Phat = Q diag(lam) Q^T
    lam, Q = np.linalg.eigh(Phat)
    lam = np.maximum(lam, 1e-12)       # numerical safety
    t = np.linspace(0, 2 * np.pi, n_pts)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)   # 2×n
    # Ellipse axes scaled by sqrt(rho / lam_i)
    axes = np.sqrt(rho / lam)          # 2-vector
    xy = Q @ (axes[:, None] * circle)  # 2×n
    return xy[0], xy[1]


def simulate_closed_loop(
    dynamics,
    controller,
    x0_batch: torch.Tensor,
    max_steps: int = 300,
) -> torch.Tensor:
    """Simulate closed-loop trajectories.

    Args:
        x0_batch: (N, nx) initial conditions.
        max_steps: number of time steps.
    Returns:
        traj: (max_steps+1, N, nx) tensor.
    """
    x = x0_batch.clone()
    traj = [x]
    with torch.no_grad():
        for _ in range(max_steps):
            u = controller(x)
            x = dynamics.forward(x, u)
            traj.append(x)
    return torch.stack(traj, dim=0)   # (T+1, N, nx)


def compute_projected_levelset(
    lyapunov_nn: nn.Module,
    rho: float,
    dim_i: int,
    dim_j: int,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    grid_size: int = 80,
    n_opt_steps: int = 200,
    lr: float = 0.05,
) -> tuple:
    """Exact projection of {V(x) ≤ ρ} onto the (dim_i, dim_j) plane.

    For each 2D grid point (xi, xj), minimises V over the remaining
    state dimensions via batched Adam.  The minimum V value at each grid
    point is returned; drawing the contour at level ρ gives the exact
    geometric projection boundary.

    Returns:
        xi_vals:   1-D array of dim_i grid values.
        xj_vals:   1-D array of dim_j grid values.
        V_proj:    (grid_size, grid_size) array of min_V values.
    """
    device = lower_limit.device
    nx = lower_limit.shape[0]
    lo_np = lower_limit.cpu().numpy()
    hi_np = upper_limit.cpu().numpy()

    xi_vals = np.linspace(lo_np[dim_i], hi_np[dim_i], grid_size)
    xj_vals = np.linspace(lo_np[dim_j], hi_np[dim_j], grid_size)
    XI, XJ = np.meshgrid(xi_vals, xj_vals)          # each (G, G)
    G = grid_size
    N = G * G

    other_dims = [d for d in range(nx) if d not in (dim_i, dim_j)]
    n_other = len(other_dims)

    # Fixed block: fill in the (i,j) coordinates
    x_fixed = torch.zeros(N, nx, device=device)
    x_fixed[:, dim_i] = torch.tensor(XI.flatten(), dtype=torch.float32, device=device)
    x_fixed[:, dim_j] = torch.tensor(XJ.flatten(), dtype=torch.float32, device=device)

    if n_other == 0:
        with torch.no_grad():
            V_proj = lyapunov_nn(x_fixed).squeeze(-1).cpu().numpy().reshape(G, G)
        return xi_vals, xj_vals, V_proj

    lo_other = lower_limit[other_dims]               # (n_other,)
    hi_other = upper_limit[other_dims]               # (n_other,)

    # Optimisable block: initialise at equilibrium (zero)
    x_other = nn.Parameter(torch.zeros(N, n_other, device=device))
    opt = torch.optim.Adam([x_other], lr=lr)

    lyapunov_nn.eval()
    for _ in range(n_opt_steps):
        opt.zero_grad()
        x_oth_clamped = torch.clamp(x_other,
                                    lo_other.unsqueeze(0),
                                    hi_other.unsqueeze(0))
        x_full = x_fixed.detach().clone()
        for k, d in enumerate(other_dims):
            x_full[:, d] = x_oth_clamped[:, k]
        loss = lyapunov_nn(x_full).squeeze(-1).sum()
        loss.backward()
        opt.step()

    with torch.no_grad():
        x_oth_clamped = torch.clamp(x_other,
                                    lo_other.unsqueeze(0),
                                    hi_other.unsqueeze(0))
        x_full = x_fixed.clone()
        for k, d in enumerate(other_dims):
            x_full[:, d] = x_oth_clamped[:, k]
        V_proj = lyapunov_nn(x_full).squeeze(-1).cpu().numpy().reshape(G, G)

    return xi_vals, xj_vals, V_proj


def plot_ellipsoid_projections(
    P_init: np.ndarray,
    rho_init: float,
    P_final: np.ndarray,
    rho_final: float,
    nx: int,
    state_labels: list,
    trajectories: torch.Tensor = None,
    title: str = "ROA Ellipsoid Projections",
    lyapunov_final: nn.Module = None,
    lower_limit_tensor: torch.Tensor = None,
    upper_limit_tensor: torch.Tensor = None,
    n_levelset_samples: int = 400_000,  # kept for API compat, unused
    proj_grid_size: int = 80,
    proj_opt_steps: int = 200,
) -> plt.Figure:
    """Plot all C(nx,2) projected ROA ellipses on a grid of 2-D panels.

    For each panel the *quadratic* ellipsoid {x : x^T P x ≤ ρ} is drawn via
    `project_ellipsoid`. When ``lyapunov_final`` is provided, the exact
    nonlinear ROA projection boundary is overlaid: for each 2D grid point the
    remaining dims are optimized away (batched Adam), giving the true
    projection {(xi,xj) : min_{others} V(x) ≤ ρ}.

    Args:
        P_init:              nx×nx Lyapunov matrix for initial model.
        rho_init:            Verified ρ for initial model.
        P_final:             nx×nx Lyapunov matrix for trained model.
        rho_final:           Verified ρ for trained model.
        nx:                  State dimension.
        state_labels:        List of nx axis labels.
        trajectories:        Optional (T, N, nx) tensor of trajectories.
        title:               Figure suptitle.
        lyapunov_final:      Trained NeuralNetworkLyapunov for nonlinear overlay.
        lower_limit_tensor:  Lower box bound as torch.Tensor (nx,).
        upper_limit_tensor:  Upper box bound as torch.Tensor (nx,).
        proj_grid_size:      Grid resolution for exact projection.
        proj_opt_steps:      Adam steps per grid point optimisation.
    Returns:
        matplotlib Figure.
    """
    import itertools

    pairs = list(itertools.combinations(range(nx), 2))
    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

    traj_np = (trajectories.cpu().numpy() if isinstance(trajectories, torch.Tensor)
              else trajectories) if trajectories is not None else None  # (T, N, nx)

    # --- Pre-compute exact projected levelsets for the nonlinear V(x)=ρ ---
    # Each entry: (xi_vals, xj_vals, V_proj) for the corresponding pair.
    proj_grids = {}
    if (lyapunov_final is not None and rho_final > 0
            and lower_limit_tensor is not None and upper_limit_tensor is not None):
        for (pi, pj) in pairs:
            proj_grids[(pi, pj)] = compute_projected_levelset(
                lyapunov_final, rho_final, pi, pj,
                lower_limit_tensor, upper_limit_tensor,
                grid_size=proj_grid_size,
                n_opt_steps=proj_opt_steps,
            )

    for idx, (i, j) in enumerate(pairs):
        ax = axes_flat[idx]

        # Selection matrix A: A x = [x_i; x_j]
        A = np.zeros((2, nx))
        A[0, i] = 1.0
        A[1, j] = 1.0

        # ── Trajectories (drawn first so ellipses are on top) ────────────
        if traj_np is not None:
            n_traj = traj_np.shape[1]
            colors = plt.cm.cool(np.linspace(0, 1, n_traj))
            for k in range(n_traj):
                ax.plot(traj_np[:, k, i], traj_np[:, k, j],
                        color=colors[k], alpha=0.4, linewidth=0.8)
                ax.plot(traj_np[0, k, i], traj_np[0, k, j],
                        'o', color=colors[k], markersize=3, alpha=0.7)

        # ── Initial ROA projection ───────────────────────────────────────
        if rho_init > 0 and P_init is not None:
            try:
                Phat_init = project_ellipsoid(P_init, A)
                ex, ey = ellipse_from_Phat(Phat_init, rho_init)
                ax.plot(ex, ey, color='cyan', linewidth=2, linestyle='--',
                        label=f'Quadratic base (ρ₀={rho_init:.4f})', zorder=5)
                ax.fill(ex, ey, alpha=0.08, color='cyan')
            except Exception as e:
                if idx == 0:
                    print(f"[warn] project_ellipsoid(init) failed: {e}")

        # ── Final ROA projection (quadratic ellipse) ────────────────────
        if rho_final > 0 and P_final is not None:
            try:
                Phat_final = project_ellipsoid(P_final, A)
                ex, ey = ellipse_from_Phat(Phat_final, rho_final)
                ax.plot(ex, ey, color='red', linewidth=2, linestyle='-',
                        label=f'Quadratic approx (ρ={rho_final:.4f})', zorder=6)
                ax.fill(ex, ey, alpha=0.06, color='red')
            except Exception as e:
                if idx == 0:
                    print(f"[warn] project_ellipsoid(final) failed: {e}")

        # ── Nonlinear V(x)=ρ exact projection (batched Adam minimisation) ─
        if (i, j) in proj_grids:
            xi_vals, xj_vals, V_proj = proj_grids[(i, j)]
            try:
                cs = ax.contour(xi_vals, xj_vals, V_proj, levels=[rho_final],
                                colors=['orange'], linewidths=[2.5], zorder=8)
                # Filled region
                ax.contourf(xi_vals, xj_vals, V_proj, levels=[-np.inf, rho_final],
                            colors=['orange'], alpha=0.12, zorder=7)
                # Proxy for legend
                if idx == 0:
                    ax.plot([], [], color='orange', linewidth=2.5,
                            label='V(x)=ρ exact projection')
            except Exception as e:
                if idx == 0:
                    print(f"[warn] projection contour failed: {e}")

        ax.axhline(0, color='white', linewidth=0.4, alpha=0.4)
        ax.axvline(0, color='white', linewidth=0.4, alpha=0.4)
        ax.set_xlabel(state_labels[i], fontsize=11)
        ax.set_ylabel(state_labels[j], fontsize=11)
        ax.set_facecolor('#1a1a2e')
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', adjustable='datalim')
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    # Hide unused subplots
    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def plot_V_heatmap_with_trajectories(
    lyapunov_nn,
    rho: float,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    trajectories: torch.Tensor,
    title: str = "Lyapunov V Heatmap",
) -> plt.Figure:
    """Plot V heatmap for 2-D plant state (x1, x2) with trajectory overlays.

    Other dimensions are sliced at 0.  This is the 'old-style' 2-D heatmap —
    kept alongside the projection plots to show the Lyapunov landscape.
    """
    lo0, lo1 = lower_limit[0].item(), lower_limit[1].item()
    hi0, hi1 = upper_limit[0].item(), upper_limit[1].item()
    n_grid = 400
    t0 = torch.linspace(lo0, hi0, n_grid, device=device)
    t1 = torch.linspace(lo1, hi1, n_grid, device=device)
    g0, g1 = torch.meshgrid(t0, t1, indexing='ij')
    nx = lyapunov_nn.x_dim
    if nx > 2:
        pad = torch.zeros(*g0.shape, nx - 2, device=device)
        pts = torch.cat([torch.stack((g0, g1), dim=2), pad], dim=2)
    else:
        pts = torch.stack((g0, g1), dim=2)
    with torch.no_grad():
        V_vals = lyapunov_nn(pts).squeeze(-1).cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.set_facecolor('#1a1a2e')
    im = ax.pcolormesh(g0.cpu().numpy(), g1.cpu().numpy(), V_vals,
                       shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='V(x)')

    # ROA contour
    legend_handles = []
    cs = ax.contour(g0.cpu().numpy(), g1.cpu().numpy(), V_vals, [rho],
                    colors='red', linewidths=2)
    if any(len(s) > 0 for s in cs.allsegs[0]):
        legend_handles.append(
            plt.Line2D([0], [0], color='red', linewidth=2,
                       label=f'ROA (ρ={rho:.4f})'))

    # Trajectories
    if trajectories is not None:
        traj_np = (trajectories.cpu().numpy() if isinstance(trajectories, torch.Tensor)
                  else trajectories)  # (T, N, nx)
        n_traj = traj_np.shape[1]
        colors = plt.cm.cool(np.linspace(0, 1, n_traj))
        for k in range(n_traj):
            ax.plot(traj_np[:, k, 0], traj_np[:, k, 1],
                    color=colors[k], alpha=0.5, linewidth=0.9)
            ax.plot(traj_np[0, k, 0], traj_np[0, k, 1],
                    'o', color=colors[k], markersize=4, alpha=0.8)

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=9)
    slice_note = "" if nx <= 2 else f" (x₃=x₄=0 slice, {nx}D state)"
    ax.set_title(title + slice_note, fontsize=12)
    ax.set_xlabel(r"$x_1$  (θ, rad)", fontsize=11)
    ax.set_ylabel(r"$x_2$  (θ̇, rad/s)", fontsize=11)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_roa_comparison(
    V_init_fn,
    rho_init,
    V_final_fn,
    rho_final,
    lower_limit,
    upper_limit,
    nx,
    logger=None,
):
    """Plot initial vs trained ROA comparison (x1-x2 slice heatmap).

    Args:
        V_init_fn: Callable(x) -> V values for the initial model.
        rho_init: Verified rho before training.
        V_final_fn: Callable(x) -> V values for the trained model.
        rho_final: Verified rho after training.
        lower_limit, upper_limit: Box limits (full augmented dim).
        nx: State dimension.
        logger: Optional logger.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Use first two state dimensions for the grid
    lo0, lo1 = lower_limit[0].item(), lower_limit[1].item()
    hi0, hi1 = upper_limit[0].item(), upper_limit[1].item()

    n_grid = 500
    ticks_0 = torch.linspace(lo0, hi0, n_grid, device=device)
    ticks_1 = torch.linspace(lo1, hi1, n_grid, device=device)
    grid_0, grid_1 = torch.meshgrid(ticks_0, ticks_1, indexing='ij')

    if nx <= 2:
        pts = torch.stack((grid_0, grid_1), dim=2)
    else:
        # Pad remaining dims with zeros (slice at x_k = 0)
        pad = torch.zeros(*grid_0.shape, nx - 2, device=device)
        pts = torch.cat([torch.stack((grid_0, grid_1), dim=2), pad], dim=2)

    with torch.no_grad():
        V_init_vals = V_init_fn(pts).squeeze(-1)
        V_final_vals = V_final_fn(pts).squeeze(-1)

    g0 = grid_0.cpu().numpy()
    g1 = grid_1.cpu().numpy()

    # Heatmap of final V
    im = ax.pcolormesh(g0, g1, V_final_vals.cpu().numpy(), shading='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='V (trained)')

    # Build legend with proxy Line2D artists to avoid deprecated .collections API
    # and to handle empty contours (contour level outside plot domain) gracefully.
    legend_handles = []

    # Initial ROA contour
    if rho_init > 0:
        cs_init = ax.contour(
            g0, g1, V_init_vals.cpu().numpy(), [rho_init],
            colors='cyan', linewidths=2, linestyles='--',
        )
        # allsegs[0] is the list of path arrays for the first (only) level
        if any(len(seg) > 0 for seg in cs_init.allsegs[0]):
            legend_handles.append(
                plt.Line2D([0], [0], color='cyan', linewidth=2, linestyle='--',
                           label=f'Initial ROA ($\\rho_0$={rho_init:.4f})')
            )

    # Final ROA contour
    if rho_final > 0:
        cs_final = ax.contour(
            g0, g1, V_final_vals.cpu().numpy(), [rho_final],
            colors='red', linewidths=2, linestyles='-',
        )
        if any(len(seg) > 0 for seg in cs_final.allsegs[0]):
            legend_handles.append(
                plt.Line2D([0], [0], color='red', linewidth=2, linestyle='-',
                           label=f'Trained ROA ($\\rho$={rho_final:.4f})')
            )

    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
    slice_note = "" if nx <= 2 else f" (slice at $x_{{k}}=0$, {nx}D state)"
    ax.set_title(f"ROA Comparison: Initial vs Trained{slice_note}")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _np(t):
    """Detach / convert a tensor (or None) to a float32 numpy array."""
    if t is None:
        return None
    return t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=float)


def log_parameter_tables(
    controller,
    lyapunov_nn,
    controller_type: str,
    nx: int,
    logger,
    use_wandb: bool = False,
    normalize_coordinates: bool = False,
    S_norm=None,
):
    """
    Log all controller and Lyapunov parameter matrices as formatted tables.

    Each matrix is printed as a right-aligned grid to the logger.  When
    ``use_wandb=True`` a ``wandb.Table`` is created for every matrix and
    logged in a single ``wandb.log`` call so they appear in the Run summary.

    Coverage
    --------
    LTIC controller  →  A_k, B_k, C_k, D_k  (continuous-time)
                        Ā_k, B̄_k             (discrete-time stored internally)
    Linear / L+NN   →  K_frozen, K_trainable, K_effective
                        + x-space transform when normalize_coordinates=True
    Lyapunov        →  R_frozen, R_trainable
                        P_frozen, P_trainable, P_effective
    """

    def _make_table(label: str, M: np.ndarray):
        r, c = M.shape
        col_w = max(13, max(len(f"{v: .6f}") for v in M.ravel()) + 2)
        header = "      " + "".join(f"  {'col_'+str(j):<{col_w-1}}" for j in range(c))
        rows_str = "\n".join(
            "  [{:2d}]  ".format(i)
            + "  ".join(f"{M[i, j]:>{col_w}.6f}" for j in range(c))
            for i in range(r)
        )
        logger.info(f"  {label}  ({r}\u00d7{c}):\n{header}\n{rows_str}")
        if use_wandb:
            import wandb
            cols = ["row"] + [f"col_{j}" for j in range(c)]
            data = [[f"row_{i}"] + [float(M[i, j]) for j in range(c)] for i in range(r)]
            return wandb.Table(columns=cols, data=data)
        return None

    tables = {}
    line = "=" * 68

    # ── CONTROLLER ──────────────────────────────────────────────────────────
    logger.info("\n" + line)
    logger.info("CONTROLLER PARAMETERS")
    logger.info(line)

    with torch.no_grad():
        if controller_type == "ltic":
            dt   = controller.dt
            A_kd = _np(controller.A_kd)
            B_kd = _np(controller.B_kd)
            C_k  = _np(controller.C_k)
            D_k  = _np(controller.D_k)
            n_k  = A_kd.shape[0]
            A_k  = (A_kd - np.eye(n_k)) / dt   # recover continuous-time
            B_k  = B_kd / dt
            trainable = isinstance(
                getattr(controller, "A_kd", None), torch.nn.Parameter
            )
            logger.info(
                f"  Type: LTI Dynamic Controller  "
                f"(n_k={n_k}, dt={dt}, {'trainable' if trainable else 'frozen'})"
            )
            logger.info(
                "  Formulation (continuous-time):  "
                "ẋ_k = A_k x_k + B_k y     u = C_k x_k + D_k y"
            )
            logger.info(
                "  Formulation (discrete-time):    "
                "x_k[t+1] = Ā_k x_k[t] + B̄_k y[t]   (Ā=I+dt·A, B̄=dt·B)"
            )
            for lbl, mat in [
                ("A_k  [continuous-time state matrix]", A_k),
                ("B_k  [continuous-time input matrix]", B_k),
                ("C_k  [output matrix]", C_k),
                ("D_k  [feedthrough matrix]", D_k),
                ("Ā_k  [discrete state matrix  = I + dt·A_k]", A_kd),
                ("B̄_k  [discrete input matrix   = dt·B_k]", B_kd),
            ]:
                t = _make_table(lbl, mat)
                key = lbl.split("[")[0].strip().replace(" ", "_")
                if t: tables[f"controller/{key}"] = t

        elif controller_type == "rinn":
            dt    = controller.dt
            A_kd  = _np(controller.A_kd)
            Bw_kd = _np(controller.Bw_kd)
            By_kd = _np(controller.By_kd)
            Cv    = _np(controller.Cv)
            Dvw   = _np(controller.Dvw)
            Dvy   = _np(controller.Dvy)
            Cu    = _np(controller.Cu)
            Duw   = _np(controller.Duw)
            Duy   = _np(controller.Duy)
            n_k   = A_kd.shape[0]
            n_w   = Dvw.shape[0]
            # Recover continuous-time matrices
            A_ct  = (A_kd - np.eye(n_k)) / dt
            Bw_ct = Bw_kd / dt
            By_ct = By_kd / dt
            trainable = isinstance(
                getattr(controller, "A_kd", None), torch.nn.Parameter
            )
            phi_name = controller.phi.__class__.__name__
            logger.info(
                f"  Type: RINN Controller  "
                f"(n_k={n_k}, n_w={n_w}, dt={dt}, φ={phi_name}, "
                f"{'trainable' if trainable else 'frozen'})"
            )
            logger.info(
                "  State:   ẋ_k = A x_k + Bw w + By y"
            )
            logger.info(
                "  Implicit: v = Cv x_k + Dvw w + Dvy y,  w = φ(v)"
            )
            logger.info(
                "  Output:  u = Cu x_k + Duw w + Duy y"
            )
            for lbl, mat in [
                ("A   [continuous-time state matrix]", A_ct),
                ("Bw  [continuous-time implicit-layer input]", Bw_ct),
                ("By  [continuous-time plant-output input]", By_ct),
                ("Cv  [implicit-layer state coupling]", Cv),
                ("Dvw [implicit-layer feed-through, strictly upper tri]", Dvw),
                ("Dvy [implicit-layer plant-output coupling]", Dvy),
                ("Cu  [output state matrix]", Cu),
                ("Duw [output implicit-layer coupling]", Duw),
                ("Duy [output feed-through]", Duy),
                ("Ā   [discrete state matrix  = I + dt·A]", A_kd),
                ("B̄w  [discrete implicit input = dt·Bw]", Bw_kd),
                ("B̄y  [discrete plant input    = dt·By]", By_kd),
            ]:
                t = _make_table(lbl, mat)
                key = lbl.split("[")[0].strip().replace(" ", "_")
                if t: tables[f"controller/{key}"] = t

        else:
            K_frozen    = _np(getattr(controller, "K_frozen",    None))
            K_trainable = _np(getattr(controller, "K_trainable", None))
            logger.info(
                f"  Type: LinearPlusNN  "
                f"(K_frozen={'yes' if K_frozen is not None else 'no'}, "
                f"K_trainable={'yes' if K_trainable is not None else 'no'})"
            )
            if K_frozen is not None:
                t = _make_table("K_frozen  [frozen linear gain]", K_frozen)
                if t: tables["controller/K_frozen"] = t
            if K_trainable is not None:
                t = _make_table("K_trainable  [trainable linear gain]", K_trainable)
                if t: tables["controller/K_trainable"] = t
            parts = [m for m in [K_frozen, K_trainable] if m is not None]
            if parts:
                K_eff = sum(parts)
                if K_frozen is not None and K_trainable is not None:
                    lbl = "K_effective = K_frozen + K_trainable"
                elif K_frozen is not None:
                    lbl = "K_effective = K_frozen  (no trainable component)"
                else:
                    lbl = "K_effective = K_trainable  (no frozen component)"
                t = _make_table(lbl, K_eff)
                if t: tables["controller/K_effective"] = t
                if normalize_coordinates and S_norm is not None:
                    S_np = _np(S_norm) if isinstance(S_norm, torch.Tensor) else np.asarray(S_norm)
                    K_eff_x = K_eff @ np.linalg.inv(S_np)
                    t = _make_table("K_effective  [x-space, = K_z · S⁻¹]", K_eff_x)
                    if t: tables["controller/K_effective_xspace"] = t

    # ── LYAPUNOV ─────────────────────────────────────────────────────────────
    logger.info(line)
    logger.info("LYAPUNOV PARAMETERS")
    logger.info(line)
    logger.info(
        "  V(x) = ε‖x‖² + x\u1d40 (R_frozen\u1d40 R_frozen "
        "+ R_trainable\u1d40 R_trainable) x"
    )

    with torch.no_grad():
        eps         = float(lyapunov_nn.eps) if hasattr(lyapunov_nn, "eps") else 0.0
        R_frozen    = _np(getattr(lyapunov_nn, "R_frozen",    None))
        R_trainable = _np(getattr(lyapunov_nn, "R_trainable", None))
        logger.info(f"  ε = {eps:.6e}")

        if R_frozen is not None:
            t = _make_table("R_frozen  [upper-triangular, frozen]", R_frozen)
            if t: tables["lyapunov/R_frozen"] = t
            t = _make_table("P_frozen = R_frozen\u1d40 R_frozen", R_frozen.T @ R_frozen)
            if t: tables["lyapunov/P_frozen"] = t

        if R_trainable is not None:
            t = _make_table("R_trainable  [upper-triangular, trainable]", R_trainable)
            if t: tables["lyapunov/R_trainable"] = t
            t = _make_table(
                "P_trainable = R_trainable\u1d40 R_trainable",
                R_trainable.T @ R_trainable,
            )
            if t: tables["lyapunov/P_trainable"] = t

        P_eff = eps * np.eye(nx)
        if R_frozen    is not None: P_eff += R_frozen.T    @ R_frozen
        if R_trainable is not None: P_eff += R_trainable.T @ R_trainable
        t = _make_table(
            "P_effective = ε·I + \u03a3R\u1d40R  [full Lyapunov matrix, used in V(x)]",
            P_eff,
        )
        if t: tables["lyapunov/P_effective"] = t

    logger.info(line + "\n")

    if use_wandb and tables:
        import wandb
        wandb.log(tables)


@hydra.main(config_path="./config", config_name="pendulum_state_training")
def main(cfg: DictConfig):
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    train_utils.set_seed(cfg.seed)

    dt = cfg.model.dt
    output_C = cfg.model.get('output_C', None)
    output_D = cfg.model.get('output_D', None)
    if output_C is not None:
        output_C = torch.tensor(output_C, dtype=dtype, device=device)
    if output_D is not None:
        output_D = torch.tensor(output_D, dtype=dtype, device=device)

    pendulum_continuous = pendulum.PendulumDynamics(
        m=0.15, l=0.5, beta=0.1, output_C=output_C, output_D=output_D
    )
    dynamics = dynamical_system.SecondOrderDiscreteTimeSystem(
        pendulum_continuous,
        dt=dt,
        position_integration=dynamical_system.IntegrationMethod[
            cfg.model.position_integration
        ],
        velocity_integration=dynamical_system.IntegrationMethod[
            cfg.model.velocity_integration
        ],
    )

    logger = logging.getLogger(__name__)

    # --- CONFIGURABLE HYBRID INITIALIZATION ---
    manual_K = cfg.model.get('K_init', None)
    manual_P = cfg.model.get('P_init', None)
    
    use_crown_sdp_init = False
    gamma_target = None
    if getattr(cfg, 'use_sdp_init', False):
        use_crown_sdp_init = True
        gamma_target = cfg.supply_rate.gamma
        logger.info(f"SDP initialization requested with γ = {gamma_target}")

    con_cfg = cfg.model.get('controller', {})
    lya_cfg = cfg.model.get('lyapunov', {})
    
    use_frozen_con = con_cfg.get('use_frozen_linear', True) if (manual_K or use_crown_sdp_init) else False
    use_trainable_con = con_cfg.get('use_trainable_linear', False)
    use_nonlinear_con = con_cfg.get('use_nonlinear', True)
    
    use_frozen_lya = lya_cfg.get('use_frozen_quadratic', True) if (manual_P or use_crown_sdp_init) else False
    use_trainable_lya = lya_cfg.get('use_trainable_quadratic', False)
    use_nonlinear_lya = lya_cfg.get('use_nonlinear', True)

    # --- INITIAL SOLVES / READS ---
    K_init_base = None
    P_init_base = None
    
    if manual_K is not None:
        K_init_base = torch.tensor(manual_K, dtype=dtype, device=device)
        logger.info("✓ Using Manual K initialization")
    if manual_P is not None:
        P_init_base = torch.tensor(manual_P, dtype=dtype, device=device)
        logger.info("✓ Using Manual P initialization")
    if manual_K is not None and manual_P is not None:
        logger.info("  (Both K and P provided)")
    if use_crown_sdp_init:
        sdp_dt = 0.001
        logger.info(f"SDP solver using dt={sdp_dt}")
        K_sdp, P_sdp, _ = compute_l2gain_init_pendulum(
            pendulum_continuous, gamma=gamma_target, dt=sdp_dt, use_discrete=True, lambda_min_weight=10.0
        )
        if K_sdp is not None:
            K_init_base = K_sdp.to(dtype).to(device)
            P_init_base = P_sdp.to(dtype).to(device)
            logger.info("✓ Using SDP K and P initialization")
        else:
            raise RuntimeError("SDP initialization failed")

    # --- COORDINATE NORMALIZATION ---
    normalize_coordinates = cfg.model.get('normalize_coordinates', False)
    S_norm = None  # normalization matrix (stored for logging/inverse transforms)
    if normalize_coordinates and P_init_base is not None:
        # P = L * L^T (Cholesky), so S = L^T gives z^T z = (x-x_eq)^T P (x-x_eq)
        L = torch.linalg.cholesky(P_init_base)
        S_norm = L.T
        S_inv = torch.linalg.inv(S_norm)
        logger.info(f"✓ Coordinate normalization enabled: z = S·(x - x_eq)")
        logger.info(f"  Normalization method: Cholesky (S = L^T)")
        logger.info(f"  S = {S_norm.cpu().numpy()}")
        
        # Wrap dynamics: all downstream operates in z-space (error coordinates)
        dynamics = dynamical_system.NormalizedDynamicsWrapper(dynamics, S_norm)
        dynamics.to(device)
        
        # Transform K to z-space: 
        # u = K_x · (x - x_eq) = K_x · S⁻¹ · z = K_z · z
        if K_init_base is not None:
            K_init_base = K_init_base @ S_inv
            logger.info(f"  K transformed to z-space: {K_init_base.cpu().numpy()}")
        
        # In z-space, P = I (by construction). R_frozen/trainable will be set accordingly below.
        P_init_base = torch.eye(2, dtype=dtype, device=device)
        logger.info("  P in z-space = I (identity)")
    elif normalize_coordinates:
        logger.warning("normalize_coordinates=True but no P_init available. Skipping normalization.")

    # Resolve K components
    K_frozen = None
    K_trainable = None
    if use_frozen_con:
        if K_init_base is not None:
            K_frozen = K_init_base
        else:
            logger.warning("use_frozen_linear=True but no init matrix found. Disabling.")
            use_frozen_con = False
    
    if use_trainable_con:
        K_trainable = torch.zeros((1, 2), dtype=dtype, device=device)
        if not use_frozen_con and K_init_base is not None:
            K_trainable = K_init_base.clone()
            logger.info("✓ Initializing trainable K with base matrix")

    con_activation_name = con_cfg.get('activation', 'tanh')
    con_activation = ACTIVATION_MAP.get(con_activation_name, nn.Tanh)
    logger.info(f"Controller NN activation: {con_activation_name}")

    # --- CONTROLLER INSTANTIATION ---
    controller_type = con_cfg.get('type', 'linear_plus_nn')
    n_plant = dynamics.nx  # plant state dimension (before any augmentation)

    if controller_type == 'ltic':
        # ---- LTI Dynamic Controller ----
        ltic_cfg = con_cfg.get('ltic', {})
        A_k = torch.tensor(ltic_cfg['A_k'], dtype=dtype, device=device)
        B_k = torch.tensor(ltic_cfg['B_k'], dtype=dtype, device=device)
        C_k = torch.tensor(ltic_cfg['C_k'], dtype=dtype, device=device)
        D_k = torch.tensor(ltic_cfg['D_k'], dtype=dtype, device=device)
        n_k = A_k.shape[0]
        ltic_trainable = ltic_cfg.get('trainable', False)

        controller = controllers.LTIDynamicController(
            A_k=A_k, B_k=B_k, C_k=C_k, D_k=D_k,
            n_plant=n_plant,
            dt=dt,
            output_fn=None,  # full-state: y = x_p
            trainable=ltic_trainable,
            clip_output='clamp' if cfg.model.u_max else None,
            u_lo=torch.tensor([-cfg.model.u_max], dtype=dtype, device=device) if cfg.model.u_max else None,
            u_up=torch.tensor([cfg.model.u_max], dtype=dtype, device=device) if cfg.model.u_max else None,
        )
        # Wrap dynamics with augmented state [x_p, x_k]
        dynamics = dynamical_system.AugmentedLTICDynamics(dynamics, controller)
        logger.info(
            f"LTIC controller: n_k={n_k}, trainable={ltic_trainable}, "
            f"augmented state dim={dynamics.nx}"
        )
    elif controller_type == 'rinn':
        # ---- Recurrent Implicit Neural Network Controller ----
        rinn_cfg = con_cfg.get('rinn', {})
        A_r   = torch.tensor(rinn_cfg['A'],   dtype=dtype, device=device)
        Bw_r  = torch.tensor(rinn_cfg['Bw'],  dtype=dtype, device=device)
        By_r  = torch.tensor(rinn_cfg['By'],  dtype=dtype, device=device)
        Cv_r  = torch.tensor(rinn_cfg['Cv'],  dtype=dtype, device=device)
        Dvw_r = torch.tensor(rinn_cfg['Dvw'], dtype=dtype, device=device)
        Dvy_r = torch.tensor(rinn_cfg['Dvy'], dtype=dtype, device=device)
        Cu_r  = torch.tensor(rinn_cfg['Cu'],  dtype=dtype, device=device)
        Duw_r = torch.tensor(rinn_cfg['Duw'], dtype=dtype, device=device)
        Duy_r = torch.tensor(rinn_cfg['Duy'], dtype=dtype, device=device)
        n_k = A_r.shape[0]
        rinn_trainable = rinn_cfg.get('trainable', False)
        rinn_activation = rinn_cfg.get('activation', 'relu')
        rinn_freeze_dvw_lower = rinn_cfg.get('freeze_dvw_lower_tri', False)

        controller = controllers.RINNController(
            A=A_r, Bw=Bw_r, By=By_r,
            Cv=Cv_r, Dvw=Dvw_r, Dvy=Dvy_r,
            Cu=Cu_r, Duw=Duw_r, Duy=Duy_r,
            n_plant=n_plant,
            dt=dt,
            output_fn=None,  # full-state: y = x_p
            trainable=rinn_trainable,
            freeze_dvw_lower_tri=rinn_freeze_dvw_lower,
            activation=rinn_activation,
            clip_output='clamp' if cfg.model.u_max else None,
            u_lo=torch.tensor([-cfg.model.u_max], dtype=dtype, device=device) if cfg.model.u_max else None,
            u_up=torch.tensor([cfg.model.u_max], dtype=dtype, device=device) if cfg.model.u_max else None,
        )
        # Wrap dynamics with augmented state [x_p, x_k]
        dynamics = dynamical_system.AugmentedRINNDynamics(dynamics, controller)
        logger.info(
            f"RINN controller: n_k={n_k}, n_w={controller.n_w}, "
            f"activation={rinn_activation}, trainable={rinn_trainable}, "
            f"augmented state dim={dynamics.nx}"
        )
    else:
        # ---- Default: Linear + NN Controller ----
        controller = controllers.LinearPlusNeuralNetworkController(
            in_dim=n_plant, out_dim=1,
            K_frozen=K_frozen,
            K_trainable=K_trainable,
            nlayer=cfg.model.get('controller_nlayer', 3),
            hidden_dim=cfg.model.get('controller_hidden_dim', 64),
            clip_output="clamp",
            u_lo=torch.tensor([-cfg.model.u_max], dtype=dtype, device=device),
            u_up=torch.tensor([cfg.model.u_max], dtype=dtype, device=device),
            x_equilibrium=dynamics.x_equilibrium.to(dtype).to(device),
            u_equilibrium=dynamics.u_equilibrium.to(dtype).to(device),
            activation=con_activation,
            nn_weight_scale=0.01,
            use_nonlinear=use_nonlinear_con,
        )
    controller.to(device)
    controller.eval()
    if controller_type == 'ltic':
        _con_trainable = ltic_cfg.get('trainable', False)
    elif controller_type == 'rinn':
        _con_trainable = rinn_cfg.get('trainable', False)
    else:
        _con_trainable = use_trainable_con
    logger.info(f"Controller type={controller_type}, "
                f"Frozen={use_frozen_con if controller_type not in ('ltic', 'rinn') else 'N/A'}, "
                f"Trainable={_con_trainable}, "
                f"NN={use_nonlinear_con if controller_type not in ('ltic', 'rinn') else False}")

    # Resolve Lyapunov components
    nx = dynamics.nx  # augmented state dim (n_p + n_k for LTIC/RINN, n_p otherwise)

    # Build augmented limit vector (includes controller state limits for LTIC/RINN).
    plant_limit = torch.tensor(cfg.model.limit, dtype=dtype, device=device)
    if controller_type == 'ltic':
        controller_limit = torch.tensor(
            ltic_cfg.get('controller_limit', [1.0] * n_k),
            dtype=dtype, device=device,
        )
        model_limit = torch.cat([plant_limit, controller_limit])
        logger.info(f"Augmented limit: plant={plant_limit.tolist()}, "
                     f"controller={controller_limit.tolist()}")
    elif controller_type == 'rinn':
        controller_limit = torch.tensor(
            rinn_cfg.get('controller_limit', [1.0] * n_k),
            dtype=dtype, device=device,
        )
        model_limit = torch.cat([plant_limit, controller_limit])
        logger.info(f"Augmented limit: plant={plant_limit.tolist()}, "
                     f"controller={controller_limit.tolist()}")
    else:
        model_limit = plant_limit

    R_frozen = None
    R_trainable = None
    P_val = None
    p_norm = None
    lya_eps_from_P = 0.0

    if use_frozen_lya or use_trainable_lya:
        if P_init_base is not None:
            # For LTIC, P_init is n_p×n_p. Embed into (n_p+n_k)×(n_p+n_k)
            # with identity block for controller states.
            if P_init_base.shape[0] < nx:
                P_aug = torch.eye(nx, dtype=dtype, device=device)
                np_ = P_init_base.shape[0]
                P_aug[:np_, :np_] = P_init_base
                P_val = P_aug
            else:
                P_val = P_init_base
        elif not use_frozen_lya and use_trainable_lya:
             P_val = torch.eye(nx, dtype=dtype, device=device) * 0.1
        
        if P_val is not None:
            # Normalise P to unit Frobenius norm before factorisation so that
            # the initial R weights live in a numerically well-scaled range,
            # regardless of how the user provided P_init (rho adjusts accordingly).
            p_norm = P_val.norm()
            P_val = P_val / p_norm
            logger.info(f"  P_val normalised (\u2016P\u2016_F = {p_norm:.4e} \u2192 1.0)")

            # Factorise P_val so R^T R = P_val exactly.
            # torch.linalg.cholesky returns L with P = L @ L^T (lower triangular).
            # _V_psd_output computes ||x @ R^T||² = x^T R^T R x, so we need R = L^T
            # (upper triangular) so that R^T R = L L^T = P.
            try:
                R_val = torch.linalg.cholesky(P_val).T
            except torch.linalg.LinAlgError:
                min_ev = torch.linalg.eigvalsh(P_val).min().item()
                jitter = max(-min_ev + 1e-6, 1e-8)
                logger.warning(f"P_init Cholesky failed (min eigval={min_ev:.3e}); adding jitter {jitter:.2e}")
                R_val = torch.linalg.cholesky(P_val + jitter * torch.eye(nx, dtype=dtype, device=device)).T
            lya_eps_from_P = 0.0   # eps*I already absorbed into R^T R = P_val
            
            if use_frozen_lya:
                R_frozen = R_val
            
            if use_trainable_lya:
                if not use_frozen_lya:
                    R_trainable = R_val
                else:
                    # Start with zero for trainable if frozen is already using the init matrix
                    R_trainable = torch.zeros((nx, nx), dtype=dtype, device=device)
                    logger.info("✓ Initializing trainable R with zeros (Frozen R active)")

    # Instantiate Lyapunov
    absolute_output = lya_cfg.get('absolute_output', True)
    lya_activation_name = lya_cfg.get('activation', 'leaky_relu')
    lya_activation = ACTIVATION_MAP.get(lya_activation_name, nn.LeakyReLU)
    logger.info(f"Lyapunov NN activation: {lya_activation_name}")

    # Always keep a tiny positive eps floor so R_trainable cannot collapse to zero
    # during optimisation
    _eps_floor = 1e-6
    lya_eps = max(float(lya_eps_from_P), _eps_floor) if P_val is not None else 0.01

    # For any quadratic_times_* mode, honour the user's use_nonlinear flag.
    # quadratic_times_<fn> + use_nonlinear=False → pure quadratic (no NN in graph).
    # quadratic_times_<fn> + use_nonlinear=True  → V = x'Px · multiplier(NN(x)).
    #   tanh: multiplier = 1 + α·tanh(NN),  α = nn_scale ∈ (0,1)
    #   exp:  multiplier = exp(NN),           always > 0, no α constraint needed
    v_psd_form = cfg.model.V_psd_form
    nn_scale = lya_cfg.get('nn_scale', 0.5)
    if v_psd_form.startswith("quadratic_times_"):
        suffix = v_psd_form[len("quadratic_times_"):]
        if use_nonlinear_lya:
            lya_activation = nn.Tanh
            logger.info(
                f"quadratic_times_{suffix} mode: NN enabled (use_nonlinear=True), "
                f"tanh activation, nn_scale={nn_scale}, zero-init last layer"
            )
        else:
            logger.info(
                f"quadratic_times_{suffix} mode: NN disabled (use_nonlinear=False) "
                f"→ pure trainable quadratic"
            )

    lyapunov_nn = lyapunov.NeuralNetworkLyapunov(
        goal_state=dynamics.x_equilibrium.to(dtype).to(device),
        hidden_widths=cfg.model.lyapunov.hidden_widths if use_nonlinear_lya else None,
        x_dim=nx,
        R_frozen=R_frozen,
        R_trainable=R_trainable,
        absolute_output=absolute_output,
        eps=lya_eps,
        activation=lya_activation,
        V_psd_form=v_psd_form,
        use_nonlinear=use_nonlinear_lya,
        nn_scale=nn_scale,
    )
    lyapunov_nn.to(device)
    lyapunov_nn.eval()
    logger.info(f"Lyapunov instantiated: Frozen={use_frozen_lya}, Trainable={use_trainable_lya}, NN={use_nonlinear_lya}")

    # Logging parameter summary
    trainable_params = sum(p.numel() for p in list(controller.parameters()) + list(lyapunov_nn.parameters()) if p.requires_grad)
    frozen_buffers = sum(b.numel() for b in list(controller.buffers()) + list(lyapunov_nn.buffers()))
    logger.info(f"Summary: {trainable_params} trainable params, {frozen_buffers} frozen elements in buffers")

    kappa = cfg.model.kappa
    rho_multiplier = cfg.model.rho_multiplier
    supply_rate = create_supply_rate(cfg, kappa)
    w_max = get_w_max(cfg)
    
    logger.info(f"Using supply rate: {type(supply_rate).__name__}")
    if w_max is not None:
        logger.info(f"Disturbance bound w_max: {w_max.item():.4f}")
    
    # Unified dissipativity framework (backward compatible with pure Lyapunov)
    # Note: This initial derivative_lyaloss is just a placeholder and will be recreated in the training loop
    # Handle ListConfig from OmegaConf
    rho_mult_init = rho_multiplier[0] if hasattr(rho_multiplier, '__getitem__') and not isinstance(rho_multiplier, (int, float)) else rho_multiplier
    # Supply-rate scaling: V is built from P/\u2016P\u2016_F, so the dissipation
    # inequality must be scaled by 1/\u2016P\u2016_F to stay dimensionally consistent.
    # When s_scale is left at its default (1.0) and P was normalised, auto-set it.
    s_scale = cfg.loss.get('s_scale', 1.0)
    if s_scale == 1.0 and p_norm is not None and p_norm != 1.0:
        s_scale = 1.0 / float(p_norm)
        logger.info(f"Auto s_scale = 1/\u2016P\u2016_F = {s_scale:.6e}  (P was normalised)")
    elif s_scale != 1.0:
        logger.info(f"Using supply rate scaling s_scale = {s_scale}")

    derivative_lyaloss = lyapunov.DissipativityDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        supply_rate=supply_rate,
        box_lo=torch.zeros(nx),
        box_up=torch.zeros(nx),
        rho_multiplier=rho_mult_init,
        w_max=w_max,
        hard_max=cfg.train.hard_max,
        s_scale=s_scale,
    )

    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    grid_size = torch.tensor([50] * nx, device=device)
    if cfg.approximate_lqr:
        rho_mult_init = get_val(rho_multiplier, 0)
        limit_scale_init = get_val(cfg.model.limit_scale, 0)
        limit = limit_scale_init * model_limit
        upper_limit = limit
        approximate_lqr(
            pendulum_continuous, controller, lyapunov_nn, upper_limit, logger
        )
        torch.save(
            {"state_dict": derivative_lyaloss.state_dict()},
            os.path.join(os.getcwd(), "lyaloss_lqr.pth"),
        )

    if not cfg.approximate_lqr and cfg.model.load_lyaloss is not None:
        load_lyaloss = os.path.join(
            os.path.dirname(__file__), "../", cfg.model.load_lyaloss
        )
        derivative_lyaloss.load_state_dict(torch.load(load_lyaloss, map_location=device)["state_dict"])

    if absolute_output:
        positivity_lyaloss = None
    else:
        positivity_lyaloss = lyapunov.LyapunovPositivityLoss(
            lyapunov_nn, 0.01 * torch.eye(2, device=device)
        )

    if cfg.train.wandb.enabled:
        wandb.init(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=cfg.train.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    save_lyaloss = cfg.model.save_lyaloss
    V_decrease_within_roa = cfg.model.V_decrease_within_roa

    # ── State labels (needed by both pre-training and post-training plots) ─
    if controller_type == 'ltic':
        _nk_sl = ltic_cfg.get('n_k', 0)
        _plant_labels = [r"$\theta$ (rad)", r"$\dot{\theta}$ (rad/s)"]
        _ctrl_labels  = [rf"$x_{{k{i+1}}}$" for i in range(_nk_sl)]
        state_labels  = (_plant_labels + _ctrl_labels)[:nx]
    elif controller_type == 'rinn':
        _nk_sl = cfg.model.controller.rinn.get('n_k', 0)
        _plant_labels = [r"$\theta$ (rad)", r"$\dot{\theta}$ (rad/s)"]
        _ctrl_labels  = [rf"$x_{{k{i+1}}}$" for i in range(_nk_sl)]
        state_labels  = (_plant_labels + _ctrl_labels)[:nx]
    else:
        state_labels = [r"$\theta$ (rad)", r"$\dot{\theta}$ (rad/s)"][:nx]

    # =====================================================================
    # Pre-training verification: find largest verified rho with initial model
    # =====================================================================
    verify_init = cfg.get('verify_init', False)
    init_rho = 0.0
    init_lyapunov_state = None

    if verify_init:
        logger.info("="*60)
        logger.info("PRE-TRAINING VERIFICATION (verify_init=True)")
        logger.info("="*60)

        # Use full hard limits for initial verification
        init_lower = -model_limit
        init_upper = model_limit

        init_rho, init_max_rho, init_clean = pgd_find_verified_rho(
            lyapunov_nn, dynamics, controller, supply_rate,
            init_lower, init_upper, w_max, s_scale,
            V_decrease_within_roa=V_decrease_within_roa,
            pgd_steps=cfg.get('pgd_verifier_steps', 300),
            num_seeds=cfg.get('pgd_verifier_num_seeds', 5),
            num_samples=50000,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            rho_bisect_tol=cfg.get('verify_init_rho_tol', 0.005),
            max_bisect_iters=cfg.get('verify_init_max_bisect', 20),
            logger=logger,
        )

        logger.info(f"Initial verified rho: {init_rho:.6f}  (max_rho={init_max_rho:.6f})")
        logger.info(f"Clean at full rho: {init_clean}")

        # Snapshot initial model state for comparison plotting later
        init_lyapunov_state = {
            k: v.clone() for k, v in lyapunov_nn.state_dict().items()
        }

        # Diagnose user-specified adversarial examples (pre-training)
        adv_examples = cfg.get('test_adversarial_examples', None)
        if adv_examples and init_rho > 0:
            diagnose_adversarial_examples(
                examples=adv_examples,
                lyapunov_nn=lyapunov_nn,
                dynamics=dynamics,
                controller=controller,
                supply_rate_fn=supply_rate,
                s_scale=s_scale,
                rho=init_rho,
                nx=nx,
                logger=logger,
                label="PRE-TRAINING",
            )

        # Plot initial ROA (2D slice for >2D)
        if init_rho > 0:
            lo0 = init_lower[0].item()
            lo1 = init_lower[1].item()
            hi0 = init_upper[0].item()
            hi1 = init_upper[1].item()
            n_grid = 500
            t0 = torch.linspace(lo0, hi0, n_grid, device=device)
            t1 = torch.linspace(lo1, hi1, n_grid, device=device)
            g0, g1 = torch.meshgrid(t0, t1, indexing='ij')
            if nx <= 2:
                pts = torch.stack((g0, g1), dim=2)
            else:
                pad = torch.zeros(*g0.shape, nx - 2, device=device)
                pts = torch.cat([torch.stack((g0, g1), dim=2), pad], dim=2)
            with torch.no_grad():
                V_vals = lyapunov_nn(pts).squeeze(-1)

            fig_init, ax_init = plt.subplots(1, 1, figsize=(8, 7))
            im_init = ax_init.pcolormesh(
                g0.cpu().numpy(), g1.cpu().numpy(), V_vals.cpu().numpy(),
                shading='auto', cmap='viridis',
            )
            ax_init.contour(
                g0.cpu().numpy(), g1.cpu().numpy(), V_vals.cpu().numpy(),
                [init_rho], colors='cyan', linewidths=2,
            )
            slice_note = "" if nx <= 2 else f" (slice $x_k=0$, {nx}D)"
            ax_init.set_title(
                f"Initial Verified ROA: $\\rho_0$={init_rho:.4f}{slice_note}"
            )
            ax_init.set_xlabel(r"$x_1$")
            ax_init.set_ylabel(r"$x_2$")
            ax_init.grid(True, alpha=0.3)
            fig_init.colorbar(im_init, ax=ax_init, label='V(x)')
            fig_init.tight_layout()
            init_plot_path = os.path.join(os.getcwd(), "V_init_roa.png")
            fig_init.savefig(init_plot_path, dpi=150)
            plt.close(fig_init)
            logger.info(f"Initial ROA plot saved: {init_plot_path}")

            # Ellipsoid projections for initial model
            with torch.no_grad():
                P_init_eff_pre = lyapunov_nn.eps * torch.eye(nx, device=device)
                if hasattr(lyapunov_nn, 'R_frozen') and lyapunov_nn.R_frozen is not None:
                    P_init_eff_pre = P_init_eff_pre + lyapunov_nn.R_frozen.T @ lyapunov_nn.R_frozen
                if hasattr(lyapunov_nn, 'R_trainable') and lyapunov_nn.R_trainable is not None:
                    P_init_eff_pre = P_init_eff_pre + lyapunov_nn.R_trainable.T @ lyapunov_nn.R_trainable
            fig_proj_pre = plot_ellipsoid_projections(
                P_init=P_init_eff_pre.cpu().numpy(),
                rho_init=init_rho,
                P_final=None,
                rho_final=0.0,
                nx=nx,
                state_labels=state_labels,
                trajectories=None,
                title=f"Initial ROA Projections  (ρ₀={init_rho:.5f})",
            )
            proj_pre_path = os.path.join(os.getcwd(), "V_init_roa_proj.png")
            fig_proj_pre.savefig(proj_pre_path, dpi=150)
            plt.close(fig_proj_pre)
            logger.info(f"Initial ROA projections saved: {proj_pre_path}")
            if cfg.train.wandb.enabled:
                wandb.log({"V_init_roa_proj": wandb.Image(proj_pre_path)})

        else:
            logger.warning("Initial verified rho is 0; skipping initial ROA plot.")

        logger.info("="*60)

    if cfg.train.train_lyaloss:
        for n in range(len(cfg.model.limit_scale)):
            limit_scale = cfg.model.limit_scale[n]
            limit = limit_scale * model_limit
            lower_limit = -limit
            upper_limit = limit

            # Unified dissipativity loss with current limits
            derivative_lyaloss = lyapunov.DissipativityDerivativeLoss(
                dynamics,
                controller,
                lyapunov_nn,
                supply_rate=supply_rate,
                box_lo=lower_limit,
                box_up=upper_limit,
                rho_multiplier=get_val(rho_multiplier, n),
                w_max=w_max,
                hard_max=cfg.train.hard_max,
                s_scale=s_scale,
            )

            # When domain_expansion is active, optionally override the initial
            # training box to a small fraction of the hard limits, then grow it.
            # domain_init_scale controls where verification starts:
            #   - numeric (e.g. 0.15): use that fraction of model_limit
            #   - "auto": derive from the initial verified ROA bounding box
            domain_expansion_flag = cfg.train.get('domain_expansion', False)
            domain_init_scale = cfg.model.get('domain_init_scale', None)
            domain_init_margin = cfg.model.get('domain_init_margin', 1.2)

            if domain_expansion_flag and domain_init_scale is not None:
                if str(domain_init_scale).lower() == 'auto':
                    # --- Auto-domain from verified ROA ---
                    if init_rho > 0 and init_lyapunov_state is not None:
                        # Use initial V₀ to compute bounding box of {x: V₀(x) ≤ ρ₀}
                        current_state = {k: v.clone() for k, v in lyapunov_nn.state_dict().items()}
                        lyapunov_nn.load_state_dict(init_lyapunov_state)
                        init_half = roa_bounding_box(
                            lyapunov_nn, init_rho,
                            -model_limit, model_limit, nx,
                            margin=domain_init_margin,
                        )
                        lyapunov_nn.load_state_dict(current_state)
                        lower_limit = -init_half
                        upper_limit = init_half
                        derivative_lyaloss.box_lo = lower_limit
                        derivative_lyaloss.box_up = upper_limit
                        logger.info(
                            f"Stage {n}: domain_init_scale=auto (from init ROA), "
                            f"margin={domain_init_margin}, "
                            f"starting box ±{init_half.tolist()}, "
                            f"hard limits ±{model_limit.tolist()}"
                        )
                    else:
                        logger.warning(
                            "domain_init_scale='auto' but init_rho=0 or verify_init disabled; "
                            "using full model_limit as fallback."
                        )
                else:
                    # --- Fixed scale ---
                    init_limit = float(domain_init_scale) * model_limit
                    lower_limit = -init_limit
                    upper_limit = init_limit
                    derivative_lyaloss.box_lo = lower_limit
                    derivative_lyaloss.box_up = upper_limit
                    logger.info(
                        f"Stage {n}: domain_init_scale={domain_init_scale}, "
                        f"starting box ±{init_limit.tolist()}, "
                        f"hard limits ±{model_limit.tolist()}"
                    )

            if save_lyaloss:
                save_lyaloss_path = os.path.join(
                    os.getcwd(), f"lyaloss_{limit_scale}.pth"
                )
            else:
                save_lyaloss_path = None

            candidate_roa_states = limit_scale * torch.tensor(
                cfg.loss.candidate_roa_states,
                device=device,
                dtype=torch.float32,
            )
            # Pad with zeros for controller states in LTIC/RINN mode
            if controller_type in ('ltic', 'rinn') and candidate_roa_states.shape[-1] < nx:
                pad_width = nx - candidate_roa_states.shape[-1]
                pad = torch.zeros(*candidate_roa_states.shape[:-1], pad_width,
                                  device=device, dtype=torch.float32)
                candidate_roa_states = torch.cat([candidate_roa_states, pad], dim=-1)

            # ── Auto-populate anchors from initial verified ROA ───────────
            use_anchors = cfg.get('use_init_roa_anchors', False)
            anchor_weight = cfg.get('init_roa_anchor_weight', 0.1)
            anchor_expand = cfg.get('init_roa_anchor_expand', 1.5)
            anchor_count  = cfg.get('init_roa_num_anchors', 256)
            cand_weight = get_val(cfg.loss.candidate_roa_states_weight, n)
            always_cand = cfg.loss.always_candidate_roa_regulizer

            if use_anchors and init_rho > 0 and init_lyapunov_state is not None:
                # Temporarily reload initial V to sample from its landscape
                current_state = {k: v.clone() for k, v in lyapunov_nn.state_dict().items()}
                lyapunov_nn.load_state_dict(init_lyapunov_state)

                anchors = sample_init_roa_anchors(
                    lyapunov_nn, init_rho,
                    -model_limit, model_limit, nx,
                    num_anchors=anchor_count,
                    expand_factor=anchor_expand,
                    logger=logger,
                )

                # Restore current V
                lyapunov_nn.load_state_dict(current_state)

                if anchors.shape[0] > 0:
                    candidate_roa_states = torch.cat([candidate_roa_states, anchors], dim=0)
                    cand_weight = max(cand_weight, anchor_weight)
                    always_cand = True
                    logger.info(
                        f"Init ROA anchors active: "
                        f"candidate_roa_states={candidate_roa_states.shape[0]}, "
                        f"weight={cand_weight}, always_on=True"
                    )

            _train_ret = train_utils.train_lyapunov_with_buffer(
                derivative_lyaloss=derivative_lyaloss,
                positivity_lyaloss=positivity_lyaloss,
                observer_loss=None,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                grid_size=grid_size,
                learning_rate=cfg.train.learning_rate,
                weight_decay=0.0,
                max_iter=get_val(cfg.train.max_iter, n),
                enable_wandb=cfg.train.wandb.enabled,
                derivative_ibp_ratio=cfg.loss.ibp_ratio_derivative,
                derivative_sample_ratio=cfg.loss.sample_ratio_derivative,
                positivity_ibp_ratio=cfg.loss.ibp_ratio_positivity,
                positivity_sample_ratio=cfg.loss.sample_ratio_positivity,
                save_best_model=save_lyaloss_path,
                pgd_steps=cfg.train.pgd_steps,
                buffer_size=cfg.train.buffer_size,
                batch_size=cfg.train.batch_size,
                epochs=cfg.train.epochs,
                samples_per_iter=cfg.train.samples_per_iter,
                l1_reg=get_val(cfg.loss.l1_reg, n),
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                Vmin_x_pgd_buffer_size=cfg.train.get('Vmin_x_pgd_buffer_size', 500000),
                V_decrease_within_roa=V_decrease_within_roa,
                Vmin_x_boundary_weight=cfg.loss.get('Vmin_x_boundary_weight', 0.0),
                Vmax_x_boundary_weight=cfg.loss.get('Vmax_x_boundary_weight', 0.0),
                candidate_roa_states=candidate_roa_states,
                candidate_roa_states_weight=cand_weight,
                logger=logger,
                always_candidate_roa_regulizer=always_cand,
                # Zubov sampling
                zubov_sampling=cfg.train.get('zubov_sampling', False),
                zubov_num_bands=cfg.train.get('zubov_num_bands', 5),
                zubov_pgd_steps=cfg.train.get('zubov_pgd_steps', 30),
                zubov_step_size=cfg.train.get('zubov_step_size', 1e-2),
                # Domain expansion
                domain_expansion=cfg.train.get('domain_expansion', False),
                domain_update_interval=cfg.train.get('domain_update_interval', 10),
                domain_traj_steps=cfg.train.get('domain_traj_steps', 200),
                domain_num_trajectories=cfg.train.get('domain_num_trajectories', 2000),
                domain_convergence_thresh=cfg.train.get('domain_convergence_thresh', 0.01),
                domain_max_growth=cfg.train.get('domain_max_growth', 2.0),
                domain_hard_lower=-model_limit,
                domain_hard_upper=model_limit,
            )
            # Capture the final domain limits (may have grown via domain expansion)
            if _train_ret.lower_limit is not None:
                lower_limit = _train_ret.lower_limit
                upper_limit = _train_ret.upper_limit

        torch.save(
            {
                "state_dict": lyapunov_nn.state_dict(),
                "rho": derivative_lyaloss.get_rho() if derivative_lyaloss.x_boundary is not None else 0.0,
            },
            os.path.join(os.getcwd(), "lyapunov_nn.pth"),
        )
    else:
        limit = cfg.model.limit_scale[-1] * model_limit.to(device=device)
        lower_limit = -limit
        upper_limit = limit
        derivative_lyaloss.x_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
            lyapunov_nn,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            eps=limit,
            steps=100,
            direction="minimize",
        )

    # Check with pgd attack.
    # Handle ListConfig from OmegaConf
    rho_mult_final = rho_multiplier[-1] if hasattr(rho_multiplier, '__getitem__') and not isinstance(rho_multiplier, (int, float)) else rho_multiplier
    derivative_lyaloss_check = lyapunov.DissipativityDerivativeLoss(
        dynamics,
        controller,
        lyapunov_nn,
        supply_rate=supply_rate,
        box_lo=lower_limit,
        box_up=upper_limit,
        rho_multiplier=rho_mult_final,
        w_max=w_max,
        hard_max=True,
        s_scale=s_scale,
    )
    
    use_adversarial_w = supply_rate.requires_disturbance and w_max is not None
    if use_adversarial_w:
        nw = dynamics.continuous_time_system.nw
        verification_loss = lyapunov.DissipativityVerificationWrapper(
            derivative_lyaloss_check, nx, nw
        )
        limit_w = w_max
        limit_joint = torch.cat([limit, limit_w])
        lower_limit_joint = torch.cat([lower_limit, -limit_w])
        upper_limit_joint = torch.cat([upper_limit, limit_w])
        input_dim = nx + nw
    else:
        verification_loss = derivative_lyaloss_check
        limit_joint = limit
        lower_limit_joint = lower_limit
        upper_limit_joint = upper_limit
        input_dim = nx

    pgd_verifier_find_counterexamples = False
    # Configuration for verifier seeds and early exit
    verifier_seeds = cfg.get('pgd_verifier_num_seeds', 10)
    stop_early = cfg.get('pgd_verifier_stop_early', True)
    
    for seed in range(verifier_seeds):
        train_utils.set_seed(seed)
        if V_decrease_within_roa:
            x_min_boundary = train_utils.calc_V_extreme_on_boundary_pgd(
                lyapunov_nn,
                lower_limit,
                upper_limit,
                num_samples_per_boundary=cfg.train.num_samples_per_boundary,
                eps=limit,
                steps=100,
                direction="minimize",
            )
            if derivative_lyaloss.x_boundary is not None:
                derivative_lyaloss_check.x_boundary = torch.cat(
                    (x_min_boundary, derivative_lyaloss.x_boundary), dim=0
                )
        x_check_start = (
            (
                torch.rand((50000, input_dim), device=device)
                - 0.5
            )
            * limit_joint
            * 2
        )
        adv_x = train_utils.pgd_attack(
            x_check_start,
            verification_loss,
            eps=limit_joint,
            steps=cfg.pgd_verifier_steps,
            lower_boundary=lower_limit_joint,
            upper_boundary=upper_limit_joint,
            direction="minimize",
        ).detach()
        adv_lya = verification_loss(adv_x)
        adv_output = torch.clamp(-adv_lya, min=0.0)
        max_adv_violation = adv_output.max().item()
        verification_tol = 1e-7  # Below this, violations are numerical noise
        pgd_has_violations = (adv_output > verification_tol).any().item()
        msg = f"pgd attack max violation {max_adv_violation}, total violation {adv_output.sum().item()}"
        if max_adv_violation > verification_tol:
            pgd_verifier_find_counterexamples = True
        logger.info(msg)
        if cfg.train.wandb.enabled:
            wandb.log({
                "pgd_counterexamples": int(pgd_has_violations),
                "num_violations": int((adv_output > 0).sum().item()),
                "max_violation": float(max_adv_violation),
                "total_violation": float(adv_output.sum().item()),
            })
            
        if pgd_verifier_find_counterexamples and stop_early:
            logger.info("PGD verifier found counter examples. Stopping early.")
            break
    logger.info(
        f"PGD verifier finds counter examples? {pgd_verifier_find_counterexamples}"
    )

    x0 = (torch.rand((40, nx), device=device) - 0.5) * 2 * limit
    x_traj, V_traj = models.simulate(derivative_lyaloss, 500, x0)
    plt.plot(torch.stack(V_traj).cpu().detach().squeeze().numpy())
    vtraj_path = os.path.join(os.getcwd(), "Vtraj_roa.png")
    plt.savefig(vtraj_path)
    if cfg.train.wandb.enabled:
        wandb.log({"V_trajectory": wandb.Image(vtraj_path)})
    plt.close()

    # pdb.set_trace()
    # If x_boundary was flushed (e.g. by a forced domain expansion on the last
    # iteration), repopulate it before reading rho so we don't get 0.
    if derivative_lyaloss.x_boundary is None:
        x_bdry = train_utils.calc_V_extreme_on_boundary_pgd(
            derivative_lyaloss.lyapunov,
            lower_limit,
            upper_limit,
            num_samples_per_boundary=cfg.train.get('num_samples_per_boundary', 500),
            eps=(upper_limit - lower_limit) / 2,
            steps=100,
            direction="minimize",
        ).detach()
        derivative_lyaloss.x_boundary = x_bdry
    rho = derivative_lyaloss.get_rho().item()
    
    # Calculate and log P matrix
    with torch.no_grad():
        eps = lyapunov_nn.eps
        P = eps * torch.eye(lyapunov_nn.x_dim, device=device)
        if hasattr(lyapunov_nn, 'R_frozen') and lyapunov_nn.R_frozen is not None:
            P += lyapunov_nn.R_frozen.T @ lyapunov_nn.R_frozen
        if hasattr(lyapunov_nn, 'R_trainable') and lyapunov_nn.R_trainable is not None:
            P += lyapunov_nn.R_trainable.T @ lyapunov_nn.R_trainable
        P_np = P.cpu().numpy()
    
    # --- FINAL PARAMETER TABLES ---
    log_parameter_tables(
        controller=controller,
        lyapunov_nn=lyapunov_nn,
        controller_type=controller_type,
        nx=nx,
        logger=logger,
        use_wandb=cfg.train.wandb.enabled,
        normalize_coordinates=normalize_coordinates,
        S_norm=S_norm,
    )

    # =====================================================================
    # Visualisation: heatmap + ellipsoid projections
    # =====================================================================
    # (state_labels already built above, before pre-training block)

    # ── Simulate closed-loop trajectories ────────────────────────────────
    n_vis_traj = 24
    # Sample ICs spread across the plant state box; controller state starts at 0
    torch.manual_seed(42)
    ic_plant = (torch.rand(n_vis_traj, 2, device=device) - 0.5) * 2 * limit[:2] * 0.9
    if nx > 2:
        ic_ctrl = torch.zeros(n_vis_traj, nx - 2, device=device)
        ic_all  = torch.cat([ic_plant, ic_ctrl], dim=1)
    else:
        ic_all = ic_plant
    traj_tensor = simulate_closed_loop(dynamics, controller, ic_all, max_steps=400)
    # traj_tensor: (T+1, N, nx)

    # ── Old-style V heatmap (x1-x2 slice) ───────────────────────────────
    fig_heat = plot_V_heatmap_with_trajectories(
        lyapunov_nn, rho, lower_limit, upper_limit, traj_tensor,
        title="Trained Lyapunov V  (x₃=x₄=0 slice)"
    )
    heatmap_path = os.path.join(os.getcwd(), "V_roa.png")
    fig_heat.savefig(heatmap_path, dpi=150)
    plt.close(fig_heat)
    logger.info(f"V heatmap saved: {heatmap_path}")

    # ── Effective P matrix for final model ───────────────────────────────
    with torch.no_grad():
        P_final_eff = lyapunov_nn.eps * torch.eye(nx, device=device)
        if hasattr(lyapunov_nn, 'R_frozen') and lyapunov_nn.R_frozen is not None:
            P_final_eff += lyapunov_nn.R_frozen.T @ lyapunov_nn.R_frozen
        if hasattr(lyapunov_nn, 'R_trainable') and lyapunov_nn.R_trainable is not None:
            P_final_eff += lyapunov_nn.R_trainable.T @ lyapunov_nn.R_trainable
        P_final_np = P_final_eff.cpu().numpy()

    # =====================================================================
    # Post-training comparison: initial vs trained ROA
    # =====================================================================
    comparison_path     = None
    proj_path_init      = None
    proj_path_final     = None
    proj_path_cmp       = None
    final_verified_rho  = rho
    improvement         = 0.0

    if verify_init and init_lyapunov_state is not None:
        logger.info("="*60)
        logger.info("POST-TRAINING ROA COMPARISON")
        logger.info("="*60)

        # ── Reconstruct initial model ─────────────────────────────────────
        lyapunov_init = copy.deepcopy(lyapunov_nn)
        lyapunov_init.load_state_dict(init_lyapunov_state)
        lyapunov_init.eval()

        # Effective P for initial model
        with torch.no_grad():
            P_init_eff = lyapunov_init.eps * torch.eye(nx, device=device)
            if hasattr(lyapunov_init, 'R_frozen') and lyapunov_init.R_frozen is not None:
                P_init_eff += lyapunov_init.R_frozen.T @ lyapunov_init.R_frozen
            if hasattr(lyapunov_init, 'R_trainable') and lyapunov_init.R_trainable is not None:
                P_init_eff += lyapunov_init.R_trainable.T @ lyapunov_init.R_trainable
            P_init_np = P_init_eff.cpu().numpy()

        # ── PGD bisection on trained model ────────────────────────────────
        final_verified_rho, final_max_rho, final_clean = pgd_find_verified_rho(
            lyapunov_nn, dynamics, controller, supply_rate,
            lower_limit, upper_limit, w_max, s_scale,
            V_decrease_within_roa=V_decrease_within_roa,
            pgd_steps=cfg.get('pgd_verifier_steps', 300),
            num_seeds=cfg.get('pgd_verifier_num_seeds', 5),
            num_samples=50000,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            rho_bisect_tol=cfg.get('verify_init_rho_tol', 0.005),
            max_bisect_iters=cfg.get('verify_init_max_bisect', 20),
            logger=logger,
        )

        logger.info(f"Initial verified rho:  {init_rho:.6f}")
        logger.info(f"Final verified rho:    {final_verified_rho:.6f}")
        if init_rho > 0:
            improvement = (final_verified_rho / init_rho - 1) * 100
            logger.info(f"Improvement:           {improvement:+.1f}%")
        elif final_verified_rho > 0:
            improvement = float('inf')
            logger.info(f"Improvement:           from 0 -> {final_verified_rho:.6f}")
        else:
            improvement = 0.0
            logger.info("Improvement:           none (both rho = 0)")

        # Diagnose user-specified adversarial examples (post-training)
        adv_examples = cfg.get('test_adversarial_examples', None)
        if adv_examples and final_verified_rho > 0:
            diagnose_adversarial_examples(
                examples=adv_examples,
                lyapunov_nn=lyapunov_nn,
                dynamics=dynamics,
                controller=controller,
                supply_rate_fn=supply_rate,
                s_scale=s_scale,
                rho=final_verified_rho,
                nx=nx,
                logger=logger,
                label="POST-TRAINING",
            )

        # ── Old-style V heatmap comparison (x1-x2 slice) ─────────────────
        fig_cmp = plot_roa_comparison(
            V_init_fn=lyapunov_init,
            rho_init=init_rho,
            V_final_fn=lyapunov_nn,
            rho_final=final_verified_rho,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            nx=nx,
            logger=logger,
        )
        comparison_path = os.path.join(os.getcwd(), "V_roa_comparison.png")
        fig_cmp.savefig(comparison_path, dpi=150)
        plt.close(fig_cmp)
        logger.info(f"Comparison heatmap saved:    {comparison_path}")

        # ── Ellipsoid projection: initial model only ──────────────────────
        if init_rho > 0:
            fig_proj_init = plot_ellipsoid_projections(
                P_init   = P_init_np,
                rho_init = init_rho,
                P_final  = None,
                rho_final= 0.0,
                nx       = nx,
                state_labels = state_labels,
                trajectories = None,
                title    = f"Initial ROA Projections  (ρ₀={init_rho:.5f})",
            )
            proj_path_init = os.path.join(os.getcwd(), "roa_proj_init.png")
            fig_proj_init.savefig(proj_path_init, dpi=150)
            plt.close(fig_proj_init)
            logger.info(f"Initial ROA projections saved: {proj_path_init}")

        # ── Ellipsoid projection: trained model only (+ nonlinear overlay) ─
        if final_verified_rho > 0:
            fig_proj_final = plot_ellipsoid_projections(
                P_init   = None,
                rho_init = 0.0,
                P_final  = P_final_np,
                rho_final= final_verified_rho,
                nx       = nx,
                state_labels = state_labels,
                trajectories = traj_tensor,
                title    = f"Trained ROA Projections  (ρ={final_verified_rho:.5f})",
                lyapunov_final      = lyapunov_nn,
                lower_limit_tensor  = lower_limit,
                upper_limit_tensor  = upper_limit,
            )
            proj_path_final = os.path.join(os.getcwd(), "roa_proj_final.png")
            fig_proj_final.savefig(proj_path_final, dpi=150)
            plt.close(fig_proj_final)
            logger.info(f"Trained ROA projections saved: {proj_path_final}")

        # ── Ellipsoid projection: both overlaid (+ nonlinear overlay) ────
        if init_rho > 0 or final_verified_rho > 0:
            fig_proj_cmp = plot_ellipsoid_projections(
                P_init   = P_init_np   if init_rho > 0           else None,
                rho_init = init_rho,
                P_final  = P_final_np  if final_verified_rho > 0 else None,
                rho_final= final_verified_rho,
                nx       = nx,
                state_labels = state_labels,
                trajectories = traj_tensor,
                title    = (f"ROA Projection Comparison  "
                            f"ρ₀={init_rho:.5f} → ρ={final_verified_rho:.5f}  "
                            f"({improvement:+.1f}%)"),
                lyapunov_final      = lyapunov_nn if final_verified_rho > 0 else None,
                lower_limit_tensor  = lower_limit,
                upper_limit_tensor  = upper_limit,
            )
            proj_path_cmp = os.path.join(os.getcwd(), "roa_proj_comparison.png")
            fig_proj_cmp.savefig(proj_path_cmp, dpi=150)
            plt.close(fig_proj_cmp)
            logger.info(f"ROA projection comparison saved: {proj_path_cmp}")

        if cfg.train.wandb.enabled:
            if proj_path_cmp:
                wandb.log({"roa_proj_comparison": wandb.Image(proj_path_cmp)})
            wandb.run.summary["init_verified_rho"]   = init_rho
            wandb.run.summary["final_verified_rho"]  = final_verified_rho
            wandb.run.summary["rho_improvement_pct"] = improvement

        logger.info("="*60)

    else:
        # No verify_init: still generate projection plot for final model
        P_init_np = None
        if final_verified_rho > 0 or rho > 0:
            rho_plot = final_verified_rho if final_verified_rho > 0 else rho
            fig_proj_final = plot_ellipsoid_projections(
                P_init=None, rho_init=0.0,
                P_final=P_final_np, rho_final=rho_plot,
                nx=nx, state_labels=state_labels,
                trajectories=traj_tensor,
                title=f"Trained ROA Projections  (ρ={rho_plot:.5f})",
                lyapunov_final     = lyapunov_nn,
                lower_limit_tensor = lower_limit,
                upper_limit_tensor = upper_limit,
            )
            proj_path_final = os.path.join(os.getcwd(), "roa_proj_final.png")
            fig_proj_final.savefig(proj_path_final, dpi=150)
            plt.close(fig_proj_final)
            logger.info(f"ROA projections saved: {proj_path_final}")

    if cfg.train.wandb.enabled:
        if heatmap_path is not None:
            wandb.log({"V_heatmap": wandb.Image(heatmap_path)})
        wandb.run.summary["final_rho"] = rho
        wandb.run.summary["final_pgd_violations"] = int((adv_output > 0).sum().item())
        wandb.run.summary["final_max_violation"] = float(max_adv_violation)
        wandb.run.summary["final_verification_success"] = not pgd_verifier_find_counterexamples
        wandb.finish()

    pass


if __name__ == "__main__":
    main()