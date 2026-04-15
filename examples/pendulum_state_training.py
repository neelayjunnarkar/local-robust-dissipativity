import os

import argparse
import copy
import hydra
import logging
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf, ListConfig
import random
import torch
import torch.nn as nn
import wandb

import neural_lyapunov_training.controllers as controllers
import neural_lyapunov_training.dynamical_system as dynamical_system
import neural_lyapunov_training.lyapunov as lyapunov
import neural_lyapunov_training.models as models
import neural_lyapunov_training.pendulum as pendulum
import neural_lyapunov_training.supply_rate as supply_rate_module
import neural_lyapunov_training.systems as systems_module
import neural_lyapunov_training.train_utils as train_utils
import neural_lyapunov_training.uncertainty as uncertainty_module

device = torch.device("cpu")
dtype = torch.float


# =============================================================================
# Supply Rate Configuration (Dissipativity Framework)
# =============================================================================

def create_supply_rate(cfg, kappa: float) -> supply_rate_module.SupplyRate:
    """Create supply rate from config. Defaults to Lyapunov for backward compatibility."""
    if not hasattr(cfg, 'supply_rate') or cfg.supply_rate is None:
        return supply_rate_module.LyapunovSupplyRate(kappa=kappa)
    
    supply_type = cfg.supply_rate.get('type', 'lyapunov')
    if supply_type is None:
        supply_type = 'lyapunov'
    supply_type = str(supply_type).lower()
    
    if supply_type in ('lyapunov', 'none'):
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
    """Get disturbance/uncertainty bound from config.

    The bound comes from two independent sources:
      1. Exogenous disturbance (L2-gain): supply_rate.d_max or supply_rate.w_max
         — only when the supply rate *requires* disturbance (L2-gain, passivity)
      2. Uncertainty (IQC/disk margin): supply_rate.uncertainty.w_bound
         — only when an uncertainty block is configured

    Returns None when neither is active (pure Lyapunov, no uncertainty).
    """
    if not hasattr(cfg, 'supply_rate') or cfg.supply_rate is None:
        return None

    sr = cfg.supply_rate
    sr_type = sr.get('type', 'lyapunov').lower()

    # 1) Exogenous disturbance bound (for L2-gain / passivity supply rates)
    if sr_type in ('l2gain', 'passivity'):
        d_max = sr.get('d_max', sr.get('w_max', None))
        if d_max is not None:
            return torch.tensor([d_max])
        return None

    # 2) Uncertainty bound (IQC mode)
    unc_cfg = sr.get('uncertainty', None)
    if unc_cfg is not None:
        unc_type = unc_cfg.get('type', 'box').lower()

        if unc_type == 'iqc':
            w_bound = unc_cfg.get('w_bound', None)
            if w_bound is not None:
                return torch.tensor([w_bound])
            # Derive from α and controller limit: |w| ≤ α·u_max/(1−α·β)
            iqc_cfg = unc_cfg.get('iqc', {})
            alpha = iqc_cfg.get('alpha', 0.353)
            sigma = iqc_cfg.get('sigma', 0.0)
            beta = (1.0 + sigma) / 2.0
            ctrl_limit = cfg.model.get('controller_limit', None)
            u_max = cfg.model.get('u_max', None)
            if ctrl_limit is not None:
                u_max_val = max(ctrl_limit) if isinstance(ctrl_limit, (list, tuple)) else float(ctrl_limit)
            elif u_max is not None:
                u_max_val = float(u_max)
            else:
                return None
            w_bar = alpha * u_max_val / (1.0 - alpha * beta)
            return torch.tensor([w_bar])

        if unc_type == 'disk_margin':
            c_bar = unc_cfg.get('c_bar', 3.0)
            return torch.tensor([c_bar])

        if unc_type == 'tanh_norm_ball':
            c_bar = unc_cfg.get('c_bar', 3.0)
            return torch.tensor([c_bar])

        # box uncertainty
        box_w = sr.get('d_max', sr.get('w_max', None))
        if box_w is not None:
            return torch.tensor([box_w])

    # Pure Lyapunov without uncertainty — no disturbance
    return None


def create_uncertainty_transform(cfg, nw: int = 1):
    """Create uncertainty transform from config.

    Returns (transform, w_tilde_bound) where:
      - transform: nn.Module  (v, w̃) → w,  or None for box-bounded
      - w_tilde_bound: float  bound for the free parameter w̃
    """
    if not hasattr(cfg, 'supply_rate') or cfg.supply_rate is None:
        return None, None

    unc_cfg = cfg.supply_rate.get('uncertainty', None)
    if unc_cfg is None:
        return None, None

    unc_type = unc_cfg.get('type', 'box').lower()
    if unc_type in ('box', 'iqc'):
        return None, None  # IQC uses direct box bounds on w, no transform

    if unc_type == 'tanh_norm_ball':
        gamma_delta = unc_cfg.get('gamma_delta', 0.1)
        c_bar = unc_cfg.get('c_bar', 3.0)
        epsilon = unc_cfg.get('epsilon', 1e-6)
        transform = uncertainty_module.TanhNormBallTransform(
            gamma_delta=gamma_delta, n_w=nw, c_bar=c_bar, epsilon=epsilon,
        )
        return transform, c_bar

    if unc_type == 'disk_margin':
        alpha = unc_cfg.get('alpha', 0.353)
        sigma = unc_cfg.get('sigma', 0.0)
        c_bar = unc_cfg.get('c_bar', 3.0)
        transform = uncertainty_module.DiskMarginTransform(
            alpha=alpha, sigma=sigma, c_bar=c_bar,
        )
        return transform, c_bar

    if unc_type == 'sector_bound':
        alpha = unc_cfg.get('alpha', 0.353)
        transform = uncertainty_module.SectorBoundTransform(alpha=alpha)
        return transform, 1.0

    raise ValueError(f"Unknown uncertainty type: {unc_type}")


def create_iqc_params(cfg):
    """Create IQC multiplier parameters from config.

    Returns dict with keys: iqc_M, iqc_z_fn, iqc_lambda_init, learnable_iqc_lambda
    or empty dict if IQC is not configured.
    """
    if not hasattr(cfg, 'supply_rate') or cfg.supply_rate is None:
        return {}

    unc_cfg = cfg.supply_rate.get('uncertainty', None)
    if unc_cfg is None:
        return {}

    unc_type = unc_cfg.get('type', 'box').lower()
    if unc_type != 'iqc':
        return {}

    iqc_cfg = unc_cfg.get('iqc', {})
    iqc_type = iqc_cfg.get('type', 'disk_margin').lower()

    if iqc_type == 'disk_margin':
        alpha = iqc_cfg.get('alpha', 0.353)
        sigma = iqc_cfg.get('sigma', 0.0)
        M = uncertainty_module.disk_margin_iqc_M(alpha)
        z_fn = uncertainty_module.disk_margin_iqc_z_fn(sigma)
    elif iqc_type == 'custom':
        M = torch.tensor(iqc_cfg['M'], dtype=torch.float32)
        z_fn = None  # default z = (u, w)
    else:
        raise ValueError(f"Unknown IQC type: {iqc_type}")

    lambda_init = float(iqc_cfg.get('lambda_init', 1.0))
    learnable = bool(iqc_cfg.get('learnable_lambda', True))

    return {
        'iqc_M': M,
        'iqc_z_fn': z_fn,
        'iqc_lambda_init': lambda_init,
        'learnable_iqc_lambda': learnable,
    }


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
    num_candidates: int = 5_000_000,
    n_plant: int = None,
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

    # 1. Apply the band condition on the full 4D Neural Network V(x) FIRST
    with torch.no_grad():
        V_vals_full = lyapunov_nn(x_cand).squeeze(-1)
        
    # Boundary band: V ∈ [0.75ρ, ρ]
    bdry_mask_full = (V_vals_full >= 0.75 * init_rho) & (V_vals_full <= init_rho)
    # Beyond band: V ∈ [ρ, α·ρ]
    beyond_mask_full = (V_vals_full > init_rho) & (V_vals_full <= expand_factor * init_rho)
    
    # Filter candidates to only those within the two spatial bands
    valid_band_mask = bdry_mask_full | beyond_mask_full
    x_cand = x_cand[valid_band_mask]
    V_vals_full = V_vals_full[valid_band_mask]

    if n_plant is not None and n_plant < nx and x_cand.shape[0] > 0:
        with torch.no_grad():
            # Calculate analytical projection of 4D P matrix onto plant state
            P = get_effective_P(lyapunov_nn, device=device)
            if P is None:
                # Pure-NN forms: skip P-based projection filtering
                pass
            else:
                # Exact projection onto plant states via Schur complement
                P11 = P[:n_plant, :n_plant]
                P12 = P[:n_plant, n_plant:]
                P21 = P[n_plant:, :n_plant]
                P22 = P[n_plant:, n_plant:]
                P_proj = P11 - P12 @ torch.linalg.inv(P22) @ P21

                # 2. Project to plant state (implicitly by evaluating only plant dims)
                x_plant = x_cand[:, :n_plant]
                
                # 3. Filter using projection: remove points safely inside the 0.75 boundary
                v_proj = (x_plant @ P_proj * x_plant).sum(dim=1)
                outside_mask = v_proj > 0.75 * init_rho
                
                # 4. Lift remaining points back (by keeping their old 4D coordinates)
                x_cand = x_cand[outside_mask]
                V_vals_full = V_vals_full[outside_mask]

                if logger is not None:
                    logger.info(f"\n[Anchor Logic] Filtered candidate anchors down to {x_cand.shape[0]} "
                                f"using the projected boundaries: removed points inside 0.75 * rho_0 (rho_0={init_rho:.5f}).")
                    
                    try:
                        import matplotlib.pyplot as plt
                        import os
                        fig, ax = plt.subplots(figsize=(7, 6))
                        
                        # Plot a subset to avoid huge image files (up to 5000 points)
                        plot_cand = x_cand[:5000].cpu().numpy()
                        if plot_cand.shape[0] > 0:
                            ax.scatter(plot_cand[:, 0], plot_cand[:, 1], s=2, alpha=0.3, color='blue',
                                       label='Remaining Samples (proj 2D)', zorder=1)
                        
                        # Grid for exact quadratic contour
                        lo_np = lower_limit.cpu().numpy()
                        hi_np = upper_limit.cpu().numpy()
                        x_grid = np.linspace(lo_np[0], hi_np[0], 200)
                        y_grid = np.linspace(lo_np[1], hi_np[1], 200)
                        XX, YY = np.meshgrid(x_grid, y_grid)
                        grid_pts = np.c_[XX.ravel(), YY.ravel()]
                        grid_tf = torch.tensor(grid_pts, dtype=torch.float32, device=device)
                        
                        if n_plant > 2:
                            padding = torch.zeros((grid_tf.shape[0], n_plant - 2), device=device)
                            grid_tf_eval = torch.cat([grid_tf, padding], dim=1)
                        else:
                            grid_tf_eval = grid_tf
                            
                        v_grid = (grid_tf_eval @ P_proj * grid_tf_eval).sum(dim=1).view(200, 200).cpu().numpy()
                        
                        ax.contour(XX, YY, v_grid, levels=[0.75 * init_rho], colors='red', linewidths=2, zorder=5)
                        ax.contour(XX, YY, v_grid, levels=[init_rho], colors='black', linestyles='dashed', linewidths=2, zorder=5)
                        
                        from matplotlib.lines import Line2D
                        custom_lines = [
                            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, alpha=0.5, label='Filtered Point Cloud'),
                            Line2D([0], [0], color='red', lw=2, label=r'Exclusion Cutoff: $x_p^T P_{proj} x_p = 0.75\rho_0$'),
                            Line2D([0], [0], color='black', lw=2, ls='dashed', label=r'Initial ROA: $x_p^T P_{proj} x_p = 1.0\rho_0$')
                        ]
                        ax.legend(handles=custom_lines, loc='upper right', fontsize=9)
                        ax.set_xlabel('Plant State 1')
                        ax.set_ylabel('Plant State 2')
                        ax.set_title('Anchor Point Sampling vs Analytical P-Matrix Projection')
                        
                        plot_path = os.path.join(os.getcwd(), "anchor_sampling_projection.png")
                        fig.savefig(plot_path, dpi=150)
                        plt.close(fig)
                        logger.info(f"Saved exact anchor bounds diagnostic to {plot_path}")
                    except Exception as e:
                        logger.warning(f"Failed to plot anchor projection: {e}")

    # Now split the resulting filtered candidates into our two targeted buckets
    bdry_mask = (V_vals_full >= 0.75 * init_rho) & (V_vals_full <= init_rho)
    bdry_pts = x_cand[bdry_mask]

    beyond_mask = (V_vals_full > init_rho) & (V_vals_full <= expand_factor * init_rho)
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


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
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
    c1_threshold=0.0,
    c1_multiplier=0.0,
    uncertainty_transform=None,
    **iqc_kwargs,
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
            c1_threshold=c1_threshold, c1_multiplier=c1_multiplier,
            uncertainty_transform=uncertainty_transform,
            **iqc_kwargs,
        )
        loss_fn.x_boundary = x_boundary

        use_adv_w = (supply_rate.requires_disturbance or uncertainty_transform is not None
                     or iqc_kwargs.get('iqc_M', None) is not None) and w_max is not None
        if use_adv_w:
            nw = getattr(dynamics.continuous_time_system, 'nw', 1) if hasattr(dynamics, 'continuous_time_system') else 1
            ver_loss = lyapunov.DissipativityVerificationWrapper(loss_fn, nx, nw)
            limit_w = w_max.to(limit.device)
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
            # Use a small absolute floor (1e-7) to absorb fp32 numerical noise
            # plus a relative term for larger rho values.
            violation_tol = max(1e-7, max_rho * 1e-4)
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


def get_effective_P(lyapunov_nn, device=None):
    """Compute effective P matrix from a Lyapunov network, or None for pure-NN forms."""
    if lyapunov_nn._is_pure_nn:
        return None
    if device is None:
        device = next(lyapunov_nn.parameters()).device
    nx = lyapunov_nn.x_dim
    with torch.no_grad():
        P = lyapunov_nn.eps * torch.eye(nx, device=device)
        if hasattr(lyapunov_nn, 'R_frozen') and lyapunov_nn.R_frozen is not None:
            P += lyapunov_nn.R_frozen.T @ lyapunov_nn.R_frozen
        if hasattr(lyapunov_nn, 'R_trainable') and lyapunov_nn.R_trainable is not None:
            P += lyapunov_nn.R_trainable.T @ lyapunov_nn.R_trainable
    return P


def pretrain_quadratic_imitation(
    lyapunov_nn,
    P_matrix: torch.Tensor,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    V_psd_form: str,
    n_samples: int = 100000,
    n_steps: int = 50000,
    lr: float = 1e-3,
    logger=None,
):
    """Pre-train pure-NN Lyapunov network to imitate V_quad(x) = (x-x*)^T P (x-x*).

    For nn_sigmoid: targets are normalized to (0, ~0.8) to fit the sigmoid range.
    For nn_abs / nn_sq: targets are the raw quadratic values.
    """
    device = P_matrix.device
    nx = lyapunov_nn.x_dim
    goal = lyapunov_nn.goal_state.unsqueeze(0)  # (1, nx)

    # Pre-compute V_max for normalization (evaluate on domain corners)
    corner_coords = [torch.tensor([lower_limit[i].item(), upper_limit[i].item()],
                                  device=device) for i in range(nx)]
    corners = torch.stack(torch.meshgrid(*corner_coords, indexing='ij'), dim=-1).reshape(-1, nx)
    V_corners = ((corners - goal) @ P_matrix * (corners - goal)).sum(dim=-1)
    V_max = V_corners.max().item()

    optimizer = torch.optim.Adam(lyapunov_nn.net.parameters(), lr=lr)
    lyapunov_nn.train()

    for step in range(n_steps):
        x = lower_limit + (upper_limit - lower_limit) * torch.rand(n_samples, nx, device=device)
        x_centered = x - goal
        V_quad = (x_centered @ P_matrix * x_centered).sum(dim=-1, keepdim=True)

        V_nn = lyapunov_nn(x)

        if V_psd_form == "nn_sigmoid":
            # Map quadratic values into sigmoid range (0.01, 0.81).
            V_target = V_quad / (V_max + 1e-8) * 0.8 + 0.01
        elif V_psd_form in ("nn_sigmoid_c1", "nn_sigmoid_abs"):
            # For c₁-exclusion forms, use raw V_quad targets.
            # This gives lower V(x*) ≈ 0.005 vs ρ ≈ 0.02, keeping
            # the V(x*)/ρ ratio small enough for c₁ < ρ feasibility.
            V_target = V_quad
        else:
            V_target = V_quad

        loss = torch.nn.functional.mse_loss(V_nn, V_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 and logger:
            logger.info(f"  Imitation step {step}/{n_steps}: loss={loss.item():.6f}")

    lyapunov_nn.eval()
    if logger:
        with torch.no_grad():
            V_origin = lyapunov_nn(goal).item()
            x_test = lower_limit + (upper_limit - lower_limit) * torch.rand(1000, nx, device=device)
            V_test = lyapunov_nn(x_test)
            logger.info(
                f"  Imitation done. V(x*)={V_origin:.6f}, "
                f"V range: [{V_test.min().item():.6f}, {V_test.max().item():.6f}]"
            )


def pretrain_decrease(
    lyapunov_nn, dynamics, controller,
    lower_limit, upper_limit,
    kappa=0.001,
    c1_frac=1.5,
    n_steps=3000,
    n_samples=50000,
    lr=5e-4,
    logger=None,
):
    """Pre-train Lyapunov NN for V decrease after imitation.

    The imitation phase teaches V shape (≈ quadratic) but does not
    guarantee the Lyapunov decrease V(f(x)) ≤ (1−κ)V(x).  For sigmoid-
    based V with V(x*) > 0, this gap makes the init verified ρ collapse
    to V(x*), leaving zero room for c₁ exclusion.

    This phase fine-tunes the NN so that decrease approximately holds
    in the band V ∈ [c₁, max_rho], while preserving V(x*).
    """
    device = next(lyapunov_nn.parameters()).device
    nx = lower_limit.shape[0]
    goal = lyapunov_nn.goal_state.unsqueeze(0)

    with torch.no_grad():
        V_star_init = lyapunov_nn(goal).item()
    c1_value = c1_frac * V_star_init

    optimizer = torch.optim.Adam(lyapunov_nn.net.parameters(), lr=lr)

    # Freeze controller during decrease pre-training
    ctrl_grads = {n: p.requires_grad for n, p in controller.named_parameters()}
    for p in controller.parameters():
        p.requires_grad_(False)

    lyapunov_nn.train()

    for step in range(n_steps):
        x = lower_limit + (upper_limit - lower_limit) * torch.rand(
            n_samples, nx, device=device
        )

        V_x = lyapunov_nn(x)

        with torch.no_grad():
            u = controller(x)
            x_next = dynamics.forward(x, u)

        V_next = lyapunov_nn(x_next.detach())

        # Only penalize decrease violations outside c₁ zone
        mask = (V_x.squeeze(-1) > c1_value).float().detach()
        decrease_viol = torch.relu(
            V_next.squeeze(-1) - (1 - kappa) * V_x.squeeze(-1)
        )
        decrease_loss = (decrease_viol * mask).sum() / (mask.sum() + 1)

        # Keep V(x*) close to initial value
        V_star = lyapunov_nn(goal)
        origin_loss = (V_star.squeeze() - V_star_init) ** 2

        loss = decrease_loss + 10.0 * origin_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 and logger:
            n_active = int(mask.sum().item())
            n_viol = int((decrease_viol * mask > 0).sum().item())
            logger.info(
                f"  Decrease step {step}/{n_steps}: loss={loss.item():.6f}, "
                f"violations={n_viol}/{n_active}, V(x*)={V_star.item():.6f}"
            )

    # Restore controller grad flags
    for n, p in controller.named_parameters():
        p.requires_grad_(ctrl_grads[n])

    lyapunov_nn.eval()
    if logger:
        with torch.no_grad():
            V_origin = lyapunov_nn(goal).item()
            x_test = lower_limit + (upper_limit - lower_limit) * torch.rand(
                1000, nx, device=device
            )
            V_test = lyapunov_nn(x_test)
            logger.info(
                f"  Decrease pre-training done. V(x*)={V_origin:.6f}, "
                f"V range: [{V_test.min().item():.6f}, {V_test.max().item():.6f}]"
            )


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


def compute_roa_volume_projected(
    lyapunov_nn: nn.Module,
    rho: float,
    dim_i: int,
    dim_j: int,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    grid_size: int = 200,
    n_opt_steps: int = 200,
    lr: float = 0.05,
) -> float:
    """Compute area of the ROA projection onto the (dim_i, dim_j) plane.

    Uses compute_projected_levelset and counts grid cells where V_proj <= rho.
    Returns area in the original coordinate units.
    """
    xi_vals, xj_vals, V_proj = compute_projected_levelset(
        lyapunov_nn, rho, dim_i, dim_j,
        lower_limit, upper_limit,
        grid_size=grid_size, n_opt_steps=n_opt_steps, lr=lr,
    )
    dx = xi_vals[1] - xi_vals[0]
    dy = xj_vals[1] - xj_vals[0]
    mask = V_proj <= rho
    return float(mask.sum()) * dx * dy


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
    proj_grid_size: int = 80,
    proj_opt_steps: int = 200,
    extra_rho_levels: list = None,
) -> plt.Figure:
    """Plot all C(nx,2) projected ROA ellipses on a grid of 2-D panels.

    For each panel the *quadratic* ellipsoid {x : x^T P x ≤ ρ} is drawn via
    `project_ellipsoid`. When ``lyapunov_final`` is provided, the exact
    nonlinear ROA projection boundary is overlaid via batched Adam.

    Args:
        extra_rho_levels: Optional list of (rho, color, linestyle, label) tuples
            for additional V(x)=ρ contours from ``lyapunov_final``.
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
    proj_grids = {}
    need_proj = rho_final > 0 or (extra_rho_levels and any(r > 0 for r, *_ in extra_rho_levels))
    if (lyapunov_final is not None and need_proj
            and lower_limit_tensor is not None and upper_limit_tensor is not None):
        # Use the largest ρ for grid sizing (rho is only used for docstring in compute_projected_levelset)
        max_rho = max(rho_final, max((r for r, *_ in (extra_rho_levels or []) if r > 0), default=0))
        for (pi, pj) in pairs:
            proj_grids[(pi, pj)] = compute_projected_levelset(
                lyapunov_final, max_rho, pi, pj,
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

        # ── Extra V(x)=ρ contour levels (e.g. PGD vs CROWN comparison) ──
        if extra_rho_levels and (i, j) in proj_grids:
            xi_vals, xj_vals, V_proj = proj_grids[(i, j)]
            for (extra_rho, color, ls, label_str) in extra_rho_levels:
                if extra_rho > 0:
                    try:
                        ax.contour(xi_vals, xj_vals, V_proj, levels=[extra_rho],
                                   colors=[color], linewidths=[2], linestyles=[ls], zorder=9)
                        if idx == 0:
                            ax.plot([], [], color=color, linewidth=2, linestyle=ls,
                                    label=label_str)
                    except Exception:
                        pass

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


def plot_flow_field(
    dynamics: nn.Module,
    controller: nn.Module,
    lyapunov_nn: nn.Module,
    rho: float,
    lower_limit: torch.Tensor,
    upper_limit: torch.Tensor,
    state_labels: list,
    trajectories: torch.Tensor = None,
    title: str = "Closed-Loop Flow Field",
    grid_density: int = 25,
) -> plt.Figure:
    """Plot quiver (vector field) of the closed-loop dynamics with V contours.

    For 2-D systems, plots a single panel. For higher-dim systems, plots all
    C(n_plant, 2) pairwise projections (other states held at 0) in a grid.

    Args:
        dynamics: Discrete-time dynamics f(x, u).
        controller: Controller u(x).
        lyapunov_nn: Lyapunov network V(x).
        rho: Verified ρ (ROA level).
        lower_limit, upper_limit: Domain bounds (nx,).
        state_labels: List of state label strings.
        trajectories: Optional (T+1, N, nx) tensor of closed-loop trajectories.
        title: Figure title.
        grid_density: Number of grid points per axis for quiver.
    Returns:
        matplotlib Figure.
    """
    import itertools

    nx = lyapunov_nn.x_dim
    n_plant = 2  # assume first 2 are plant states for projection
    device = next(lyapunov_nn.parameters()).device
    lo = lower_limit.cpu().numpy()
    hi = upper_limit.cpu().numpy()

    pairs = list(itertools.combinations(range(min(nx, 4)), 2))
    n_pairs = len(pairs)
    ncols = min(3, n_pairs)
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes_flat = np.array(axes).flatten() if n_pairs > 1 else [axes]

    traj_np = (trajectories.cpu().numpy() if isinstance(trajectories, torch.Tensor)
               else trajectories) if trajectories is not None else None

    for idx, (di, dj) in enumerate(pairs):
        ax = axes_flat[idx]

        # Build 2D grid for this projection
        xi = np.linspace(lo[di], hi[di], grid_density)
        xj = np.linspace(lo[dj], hi[dj], grid_density)
        XI, XJ = np.meshgrid(xi, xj)
        flat_i = XI.ravel()
        flat_j = XJ.ravel()
        n_pts = len(flat_i)

        # Full state: zero except dims di, dj
        x_np = np.zeros((n_pts, nx), dtype=np.float32)
        x_np[:, di] = flat_i
        x_np[:, dj] = flat_j
        x_t = torch.tensor(x_np, device=device)

        with torch.no_grad():
            u_t = controller(x_t)
            x_next = dynamics.forward(x_t, u_t)
            dx = (x_next - x_t).cpu().numpy()
            V_vals = lyapunov_nn(x_t).squeeze(-1).cpu().numpy()

        DI = dx[:, di].reshape(XI.shape)
        DJ = dx[:, dj].reshape(XI.shape)
        V_grid = V_vals.reshape(XI.shape)

        # Magnitude for color
        mag = np.sqrt(DI**2 + DJ**2)
        mag_norm = mag / (mag.max() + 1e-12)

        # Background: V heatmap
        im = ax.pcolormesh(XI, XJ, V_grid, cmap='Blues', alpha=0.35, shading='auto', zorder=0)

        # Quiver (vector field)
        ax.quiver(XI, XJ, DI, DJ, mag_norm,
                  cmap='coolwarm', scale=None, alpha=0.75,
                  width=0.003, headwidth=3.5, zorder=3)

        # V = rho contour (ROA boundary)
        if rho > 0:
            ax.contour(XI, XJ, V_grid, levels=[rho],
                       colors=['orange'], linewidths=2.5, zorder=5)
            # Additional contours at fractions of rho
            sub_levels = [rho * f for f in [0.25, 0.5, 0.75]]
            ax.contour(XI, XJ, V_grid, levels=sub_levels,
                       colors=['gray'], linewidths=0.8, linestyles='dotted', alpha=0.6, zorder=4)

        # Trajectories
        if traj_np is not None:
            n_traj = traj_np.shape[1]
            for k in range(n_traj):
                ax.plot(traj_np[:, k, di], traj_np[:, k, dj],
                        color='lime', alpha=0.3, linewidth=0.6, zorder=6)
                ax.plot(traj_np[0, k, di], traj_np[0, k, dj],
                        'o', color='lime', markersize=2, alpha=0.5, zorder=7)

        ax.plot(0, 0, '+', color='white', markersize=10, markeredgewidth=2, zorder=10)
        ax.set_xlabel(state_labels[di], fontsize=11)
        ax.set_ylabel(state_labels[dj], fontsize=11)
        ax.set_facecolor('#0d1117')
        ax.grid(True, alpha=0.15, color='gray')
        ax.set_xlim(lo[di], hi[di])
        ax.set_ylim(lo[dj], hi[dj])
        ax.set_aspect('equal', adjustable='datalim')
        if idx == 0 and rho > 0:
            ax.plot([], [], color='orange', linewidth=2.5, label=f'V(x)=ρ={rho:.4f}')
            ax.legend(fontsize=8, loc='upper right')

    for idx in range(n_pairs, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig


def _np(t):
    """Detach / convert a tensor (or None) to a float32 numpy array."""
    if t is None:
        return None
    return t.detach().cpu().float().numpy() if isinstance(t, torch.Tensor) else np.asarray(t, dtype=float)


def run_crown_verification(
    cfg, formal_cfg, derivative_lyaloss, init_rho, logger,
    label="", verify_subdir="formal_verification",
    model_pth_override=None, pure_quadratic=False,
):
    """Run α,β-CROWN formal verification with bisection.

    Args:
        cfg: Full Hydra config.
        formal_cfg: cfg.formal_verification sub-config.
        derivative_lyaloss: Loss module (provides s_scale and state_dict).
        init_rho: Starting ρ for bisection.
        label: Human-readable label for logging.
        verify_subdir: Subdirectory name for results.
        model_pth_override: Path to checkpoint (if not default).
        pure_quadratic: Disable Lyapunov NN for faster verification.

    Returns:
        Verified ρ (0.0 on failure).
    """
    try:
        from neural_lyapunov_training.verify_dissipativity import run_verification
    except ImportError:
        logger.warning("α,β-CROWN not available. Skipping formal verification.")
        return 0.0

    logger.info("=" * 60)
    logger.info(f"FORMAL VERIFICATION (α,β-CROWN) — {label}")
    logger.info("=" * 60)

    try:
        import copy as _copy
        training_cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        # Optional: reduce c_bar for verification (smaller → tighter CROWN bounds)
        _verify_c_bar = formal_cfg.get("verify_c_bar", None)
        if _verify_c_bar is not None:
            training_cfg_dict = _copy.deepcopy(training_cfg_dict)
            training_cfg_dict.setdefault("supply_rate", {}).setdefault("uncertainty", {})["c_bar"] = _verify_c_bar
            logger.info(f"  [verify] Overriding c_bar={_verify_c_bar} for CROWN verification")
        _verify_nominal = formal_cfg.get("verify_nominal", False)
        _formal_init_rho = formal_cfg.get("formal_init_rho", None)
        _effective_init_rho = float(_formal_init_rho) if _formal_init_rho is not None else init_rho
        if _formal_init_rho is not None:
            logger.info(f"  [verify] Using formal_init_rho={_effective_init_rho} (PGD rho={init_rho:.2f})")
        result = run_verification(
            training_dir=os.getcwd(),
            init_rho=_effective_init_rho,
            training_cfg=training_cfg_dict,
            logger=logger,
            timeout=formal_cfg.get("timeout", 200),
            rho_eps=formal_cfg.get("rho_eps", 0.001),
            rho_multiplier=formal_cfg.get("rho_multiplier", 1.2),
            max_bisect_iters=formal_cfg.get("max_bisect_iters", 30),
            hole_size=formal_cfg.get("hole_size", 0.001),
            model_pth_override=model_pth_override,
            verify_subdir=verify_subdir,
            device=formal_cfg.get("device", "auto"),
            batch_size=formal_cfg.get("batch_size", None),
            effective_s_scale=derivative_lyaloss.get_s_scale_value(),
            pure_quadratic=pure_quadratic,
            smart_bracket=formal_cfg.get("smart_bracket", False),
            bracket_shift=formal_cfg.get("bracket_shift", None),
            pgd_restarts=formal_cfg.get("pgd_restarts", 10000),
            domain_tightening=formal_cfg.get("domain_tightening", True),
            tightening_samples=formal_cfg.get("tightening_samples", 500000),
            tightening_margin=formal_cfg.get("tightening_margin", 1.05),
            verify_nominal=_verify_nominal,
            verify_c_bar=_verify_c_bar,
            enable_incomplete_verification=formal_cfg.get("enable_incomplete_verification", False),
            extra_input_splits=formal_cfg.get("extra_input_splits", 2),
        )

        if result["success"]:
            verified_rho = result["verified_rho"]
            logger.info(f"  CROWN-verified rho: {verified_rho:.6f}")
            return verified_rho
        else:
            logger.warning(f"  CROWN verification did not succeed ({label}).")
            return 0.0
    except Exception as e:
        logger.error(f"Formal verification failed ({label}): {e}")
        import traceback
        traceback.print_exc()
        return 0.0
    finally:
        logger.info("=" * 60)


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
    if lyapunov_nn._is_pure_nn:
        logger.info(f"  Pure-NN Lyapunov: V_psd_form={lyapunov_nn.V_psd_form}")
        logger.info(f"  nn_scale={lyapunov_nn.get_nn_scale_value():.4f}"
                     f"{' (learnable)' if lyapunov_nn._learnable_nn_scale else ''}")
    else:
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
    global device
    OmegaConf.save(cfg, os.path.join(os.getcwd(), "config.yaml"))

    # Allow config to override training device (default: cpu)
    _train_device = cfg.get("train", {}).get("device", "cpu") if hasattr(cfg, "train") else "cpu"
    if _train_device == "auto":
        _train_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(_train_device)

    train_utils.set_seed(cfg.seed)

    # ─── System construction (generic via system factory) ────────────────
    dynamics, state_labels_plant, system_output_fn, continuous_sys = \
        systems_module.create_system(cfg, device=device, dtype=dtype)
    # Backward compat: keep references used downstream
    pendulum_continuous = continuous_sys
    dt = cfg.model.dt

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

    # Pure-NN Lyapunov forms don't use R matrices at all
    v_psd_form = cfg.model.V_psd_form
    _is_pure_nn = v_psd_form in lyapunov.NeuralNetworkLyapunov._PURE_NN_FORMS
    if _is_pure_nn:
        use_frozen_lya = False
        use_trainable_lya = False
        use_nonlinear_lya = True  # NN is the only Lyapunov component
        logger.info(f"Pure-NN Lyapunov mode ({v_psd_form}): "
                     "R matrices disabled, NN provides all structure")

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
            output_fn=system_output_fn,
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

        # Load matrices from a .pth file or from inline YAML values.
        rinn_weights_path = rinn_cfg.get('weights_path', None)
        if rinn_weights_path is not None:
            # --- Load from .pth state_dict (keys stored transposed: X_T) ---
            rinn_weights_path = os.path.expanduser(rinn_weights_path)
            if not os.path.isabs(rinn_weights_path):
                rinn_weights_path = os.path.join(hydra.utils.get_original_cwd(), rinn_weights_path)
            logger.info(f"Loading RINN matrices from: {rinn_weights_path}")
            wd = torch.load(rinn_weights_path, map_location=device, weights_only=False)
            A_r   = wd['A_T'].t().contiguous().to(dtype)
            Bw_r  = wd['Bw_T'].t().contiguous().to(dtype)
            By_r  = wd['By_T'].t().contiguous().to(dtype)
            Cv_r  = wd['Cv_T'].t().contiguous().to(dtype)
            Dvw_r = wd['Dvw_T'].t().contiguous().to(dtype)
            Dvy_r = wd['Dvy_T'].t().contiguous().to(dtype)
            Cu_r  = wd['Cu_T'].t().contiguous().to(dtype)
            Duw_r = wd['Duw_T'].t().contiguous().to(dtype)
            Duy_r = wd['Duy_T'].t().contiguous().to(dtype)
            logger.info(f"  A_r =\n{A_r.detach().cpu()}")
            logger.info(f"  Bw_r =\n{Bw_r.detach().cpu()}")
            logger.info(f"  By_r =\n{By_r.detach().cpu()}")
            logger.info(f"  Cv_r =\n{Cv_r.detach().cpu()}")
            logger.info(f"  Dvw_r =\n{Dvw_r.detach().cpu()}")
            logger.info(f"  Dvy_r =\n{Dvy_r.detach().cpu()}")
            logger.info(f"  Cu_r =\n{Cu_r.detach().cpu()}")
            logger.info(f"  Duw_r =\n{Duw_r.detach().cpu()}")
            logger.info(f"  Duy_r =\n{Duy_r.detach().cpu()}")
            # Optionally override P_init from the same file
            if 'P' in wd and manual_P is None:
                P_from_pth = wd['P'].to(dtype).to(device)
                if isinstance(P_from_pth, torch.Tensor) and P_from_pth.ndim == 2:
                    P_init_base = P_from_pth
                    logger.info(f"  P_init loaded from .pth: shape={P_init_base.shape}")
                    logger.info(f"  P_init raw from .pth =\n{P_init_base.detach().cpu()}")
            logger.info(
                f"  RINN matrices loaded: n_k={A_r.shape[0]}, n_w={Bw_r.shape[1]}"
            )
        else:
            # --- Inline matrices from YAML ---
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
            output_fn=system_output_fn,
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
        logger.info(f"  controller.A_kd =\n{controller.A_kd.detach().cpu()}")
        logger.info(f"  controller.Bw_kd =\n{controller.Bw_kd.detach().cpu()}")
        logger.info(f"  controller.By_kd =\n{controller.By_kd.detach().cpu()}")
        logger.info(f"  controller.Cv =\n{controller.Cv.detach().cpu()}")
        logger.info(f"  controller.Dvw =\n{controller.Dvw.detach().cpu()}")
        logger.info(f"  controller.Dvy =\n{controller.Dvy.detach().cpu()}")
        logger.info(f"  controller.Cu =\n{controller.Cu.detach().cpu()}")
        logger.info(f"  controller.Duw =\n{controller.Duw.detach().cpu()}")
        logger.info(f"  controller.Duy =\n{controller.Duy.detach().cpu()}")
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
    if _is_pure_nn:
        lya_eps = 0.0  # pure-NN forms don't use eps·‖x‖² base
    else:
        lya_eps = max(float(lya_eps_from_P), _eps_floor) if P_val is not None else 0.01

    # For any quadratic_times_* mode, honour the user's use_nonlinear flag.
    # quadratic_times_<fn> + use_nonlinear=False → pure quadratic (no NN in graph).
    # quadratic_times_<fn> + use_nonlinear=True  → V = x'Px · multiplier(NN(x)).
    #   tanh: multiplier = 1 + α·tanh(NN),  α = nn_scale ∈ (0,1)
    #   exp:  multiplier = exp(NN),           always > 0, no α constraint needed
    # quadratic_plus_sq: V = x'Px + α·(NN(x)−NN(0))²  — CROWN-friendly additive form.
    v_psd_form = cfg.model.V_psd_form
    nn_scale = lya_cfg.get('nn_scale', 0.5)
    if v_psd_form.startswith("quadratic_times_"):
        suffix = v_psd_form[len("quadratic_times_"):]
        if use_nonlinear_lya:
            # Hidden-layer activation comes from config (leaky_relu/relu preferred for
            # CROWN compatibility — tighter piecewise-linear bounds). The outer tanh
            # multiplier in V_psd_form already provides the nonlinear structure.
            logger.info(
                f"quadratic_times_{suffix} mode: NN enabled (use_nonlinear=True), "
                f"hidden activation={lya_activation_name}, nn_scale={nn_scale}, zero-init last layer"
            )
        else:
            logger.info(
                f"quadratic_times_{suffix} mode: NN disabled (use_nonlinear=False) "
                f"→ pure trainable quadratic"
            )
    elif v_psd_form == "quadratic_plus_sq":
        logger.info(
            f"quadratic_plus_sq mode: V = x'Px + {nn_scale}·(NN(x)−NN(0))², "
            f"hidden activation={lya_activation_name} (smooth+CROWN-friendly), "
            f"zero-init last layer"
        )
    elif v_psd_form == "quadratic_plus_abs":
        logger.info(
            f"quadratic_plus_abs mode: V = x'Px + {nn_scale}·|NN(x)−NN(0)|, "
            f"hidden activation={lya_activation_name}, "
            f"zero-init last layer"
        )
    elif _is_pure_nn:
        logger.info(
            f"Pure-NN mode ({v_psd_form}): V defined entirely by NN, "
            f"no quadratic base, nn_scale={nn_scale}, "
            f"hidden activation={lya_activation_name}"
        )

    learnable_nn_scale = lya_cfg.get('learnable_nn_scale', False)

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
        learnable_nn_scale=learnable_nn_scale,
    )
    lyapunov_nn.to(device)
    lyapunov_nn.eval()
    logger.info(f"Lyapunov instantiated: Frozen={use_frozen_lya}, Trainable={use_trainable_lya}, NN={use_nonlinear_lya}")
    if learnable_nn_scale:
        logger.info(f"  nn_scale is LEARNABLE, initial value={nn_scale}")

    # Logging parameter summary
    trainable_params = sum(p.numel() for p in list(controller.parameters()) + list(lyapunov_nn.parameters()) if p.requires_grad)
    frozen_buffers = sum(b.numel() for b in list(controller.buffers()) + list(lyapunov_nn.buffers()))
    logger.info(f"Summary: {trainable_params} trainable params, {frozen_buffers} frozen elements in buffers")

    kappa = cfg.model.kappa
    rho_multiplier = cfg.model.rho_multiplier
    supply_rate = create_supply_rate(cfg, kappa)
    w_max = get_w_max(cfg)

    # Uncertainty transform (e.g. tanh norm-ball for IQC / L2-gain)
    nw = getattr(dynamics, 'nw', 0) or 1
    uncertainty_transform, w_tilde_bound = create_uncertainty_transform(cfg, nw=nw)
    # When using a state-dependent transform, PGD optimizes w̃ in [-c̄, c̄]
    # instead of w in [-w_max, w_max]; override w_max for PGD bounds.
    if uncertainty_transform is not None:
        uncertainty_transform = uncertainty_transform.to(device)
        if w_max is None:
            w_max = torch.tensor([w_tilde_bound])
        else:
            w_max = torch.tensor([w_tilde_bound])
        xform_name = type(uncertainty_transform).__name__
        if hasattr(uncertainty_transform, 'gamma_delta'):
            logger.info(
                f"Uncertainty transform: {xform_name} "
                f"(γ_Δ={uncertainty_transform.gamma_delta}, c̄={w_tilde_bound}, "
                f"coverage={uncertainty_transform.coverage_fraction():.3f})"
            )
        elif hasattr(uncertainty_transform, 'alpha'):
            sigma_str = f", σ={uncertainty_transform.sigma}" if hasattr(uncertainty_transform, 'sigma') else ""
            logger.info(
                f"Uncertainty transform: {xform_name} "
                f"(α={uncertainty_transform.alpha}{sigma_str}, "
                f"c̄={w_tilde_bound}, coverage={uncertainty_transform.coverage_fraction():.3f})"
            )
        else:
            logger.info(f"Uncertainty transform: {xform_name} (c̄={w_tilde_bound})")
    
    logger.info(f"Using supply rate: {type(supply_rate).__name__}")
    if w_max is not None:
        logger.info(f"Disturbance bound w_max: {w_max.item():.4f}")
    
    # Unified dissipativity framework (backward compatible with pure Lyapunov)
    # Note: This initial derivative_lyaloss is just a placeholder and will be recreated in the training loop
    if P_val is not None:
        logger.info(f"  P_val before normalization =\n{P_val.detach().cpu()}")
    # Handle ListConfig from OmegaConf
    rho_mult_init = rho_multiplier[0] if hasattr(rho_multiplier, '__getitem__') and not isinstance(rho_multiplier, (int, float)) else rho_multiplier
    if P_val is not None:
        logger.info(f"  P_val after normalization =\n{P_val.detach().cpu()}")
    # Supply-rate scaling: V is built from P/\u2016P\u2016_F, so the dissipation
    # inequality must be scaled by 1/\u2016P\u2016_F to stay dimensionally consistent.
    # When s_scale is left at its default (1.0) and P was normalised, auto-set it.
    s_scale = cfg.loss.get('s_scale', 1.0)
    if s_scale == 1.0 and p_norm is not None and p_norm != 1.0:
        s_scale = 1.0 / float(p_norm)
        logger.info(f"Auto s_scale = 1/\u2016P\u2016_F = {s_scale:.6e}  (P was normalised)")
    elif s_scale != 1.0:
        logger.info(f"Using supply rate scaling s_scale = {s_scale}")

    learnable_s_scale = cfg.loss.get('learnable_s_scale', False)
    if learnable_s_scale:
        logger.info(f"s_scale is LEARNABLE (log-parameterized, init={s_scale:.4e}); "
                    f"will be optimized jointly with V and controller at lr*0.1")

    c1_threshold = cfg.loss.get('c1_threshold', 0.0)
    c1_multiplier = cfg.loss.get('c1_multiplier', 0.0)

    # IQC multiplier parameters (Theorem 1)
    iqc_params = create_iqc_params(cfg)
    if iqc_params:
        logger.info(f"IQC mode enabled: M shape={iqc_params['iqc_M'].shape}, "
                     f"λ_init={iqc_params['iqc_lambda_init']:.4f}, "
                     f"learnable={iqc_params['learnable_iqc_lambda']}")

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
        learnable_s_scale=learnable_s_scale,
        c1_threshold=c1_threshold,
        c1_multiplier=c1_multiplier,
        uncertainty_transform=uncertainty_transform,
        **iqc_params,
    )

    dynamics.to(device)
    controller.to(device)
    lyapunov_nn.to(device)
    grid_size = torch.tensor([50] * nx, device=device)

    if cfg.model.load_lyaloss is not None:
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
        _ctrl_labels  = [rf"$x_{{k{i+1}}}$" for i in range(_nk_sl)]
        state_labels  = (state_labels_plant + _ctrl_labels)[:nx]
    elif controller_type == 'rinn':
        _nk_sl = cfg.model.controller.rinn.get('n_k', 0)
        _ctrl_labels  = [rf"$x_{{k{i+1}}}$" for i in range(_nk_sl)]
        state_labels  = (state_labels_plant + _ctrl_labels)[:nx]
    else:
        state_labels = state_labels_plant[:nx]

    # =====================================================================
    # Quadratic imitation pre-training for pure-NN Lyapunov forms
    # =====================================================================
    if _is_pure_nn and P_init_base is not None:
        logger.info("="*60)
        logger.info(f"QUADRATIC IMITATION PRE-TRAINING ({v_psd_form})")
        logger.info("="*60)
        # Use unnormalized P_init, embedded to full state dim if needed
        P_imitation = P_init_base.clone().to(device)
        if P_imitation.shape[0] < nx:
            P_aug = torch.zeros(nx, nx, dtype=P_imitation.dtype, device=device)
            np_ = P_imitation.shape[0]
            P_aug[:np_, :np_] = P_imitation
            # Small identity for controller states
            P_aug[np_:, np_:] = P_imitation.diagonal().mean() * torch.eye(
                nx - np_, dtype=P_imitation.dtype, device=device)
            P_imitation = P_aug
        pretrain_quadratic_imitation(
            lyapunov_nn, P_imitation,
            lower_limit=-model_limit, upper_limit=model_limit,
            V_psd_form=v_psd_form,
            n_samples=cfg.get('imitation_n_samples', 10000),
            n_steps=cfg.get('imitation_n_steps', 100000),
            lr=cfg.get('imitation_lr', 3e-4),
            logger=logger,
        )

        # For c₁-exclusion forms (nn_sigmoid_c1, nn_sigmoid_abs):
        # imitation only teaches V shape, not decrease.  Pre-train decrease
        # so that init verified ρ >> V(x*), enabling non-vacuous c₁.
        if v_psd_form in ("nn_sigmoid_c1", "nn_sigmoid_abs") and c1_multiplier > 0:
            logger.info("DECREASE PRE-TRAINING (c₁-exclusion form)")
            pretrain_decrease(
                lyapunov_nn, dynamics, controller,
                lower_limit=-model_limit, upper_limit=model_limit,
                kappa=supply_rate.kappa if hasattr(supply_rate, 'kappa') else 0.001,
                c1_frac=c1_multiplier,
                n_steps=cfg.get('decrease_pretrain_steps', 3000),
                n_samples=cfg.get('decrease_pretrain_samples', 50000),
                lr=cfg.get('decrease_pretrain_lr', 5e-4),
                logger=logger,
            )

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
            init_lower, init_upper, w_max, derivative_lyaloss.get_s_scale_value(),
            V_decrease_within_roa=V_decrease_within_roa,
            pgd_steps=cfg.get('pgd_verifier_steps', 300),
            num_seeds=cfg.get('pgd_verifier_num_seeds', 5),
            num_samples=50000,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            rho_bisect_tol=cfg.get('verify_init_rho_tol', 0.005),
            max_bisect_iters=cfg.get('verify_init_max_bisect', 20),
            logger=logger,
            c1_threshold=c1_threshold, c1_multiplier=c1_multiplier,
            uncertainty_transform=uncertainty_transform,
            **iqc_params,
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
                s_scale=derivative_lyaloss.get_s_scale_value(),
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
            P_init_eff_pre = get_effective_P(lyapunov_nn, device=device)
            P_init_pre_np = P_init_eff_pre.cpu().numpy() if P_init_eff_pre is not None else None
            fig_proj_pre = plot_ellipsoid_projections(
                P_init=P_init_pre_np,
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

    # =====================================================================
    # Initial Formal Verification (α,β-CROWN) — before training
    # =====================================================================
    init_crown_rho = 0.0
    formal_cfg = cfg.get("formal_verification", {})
    formal_verification_enabled = formal_cfg.get("enabled", False) if formal_cfg else False
    verify_initial_crown = formal_cfg.get("verify_initial", False) if formal_cfg else False
    verify_final_crown = formal_cfg.get("verify_final", False) if formal_cfg else False

    if verify_init and formal_verification_enabled and verify_initial_crown and init_rho > 0:
        # Save RNG state so CROWN verification doesn't perturb training
        _rng_state_torch = torch.random.get_rng_state()
        _rng_state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _rng_state_np = np.random.get_state()
        _rng_state_py = random.getstate()

        # Precompute Cholesky factor for CROWN-friendly quadratic form
        if hasattr(lyapunov_nn, 'precompute_cholesky'):
            lyapunov_nn.precompute_cholesky()

        # Save initial checkpoint for CROWN
        init_pth_path = os.path.join(os.getcwd(), "lyaloss_init.pth")
        torch.save(
            {"state_dict": derivative_lyaloss.state_dict(),
             "rho": derivative_lyaloss.get_rho() if derivative_lyaloss.x_boundary is not None else 0.0,
             "s_scale": derivative_lyaloss.get_s_scale_value()},
            init_pth_path,
        )

        # Reset L_chol so training uses the R-based path (gradients through R_trainable)
        if hasattr(lyapunov_nn, 'L_chol'):
            lyapunov_nn.L_chol = None
        init_crown_rho = run_crown_verification(
            cfg, formal_cfg, derivative_lyaloss, init_rho, logger,
            label="initial (pre-training)",
            verify_subdir="formal_verification_init",
            model_pth_override=init_pth_path,
            pure_quadratic=formal_cfg.get("init_pure_quadratic", False),
        )
        if cfg.train.wandb.enabled and init_crown_rho > 0:
            wandb.run.summary["init_crown_rho"] = init_crown_rho

        # Restore RNG state so training is deterministic regardless of CROWN
        torch.random.set_rng_state(_rng_state_torch)
        if _rng_state_cuda is not None:
            torch.cuda.set_rng_state_all(_rng_state_cuda)
        np.random.set_state(_rng_state_np)
        random.setstate(_rng_state_py)

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
                learnable_s_scale=learnable_s_scale,
                c1_threshold=c1_threshold,
                c1_multiplier=c1_multiplier,
                uncertainty_transform=uncertainty_transform,
                **iqc_params,
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
                    n_plant=n_plant,
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
                # Domain expansion
                domain_expansion=cfg.train.get('domain_expansion', False),
                domain_update_interval=cfg.train.get('domain_update_interval', 10),
                domain_traj_steps=cfg.train.get('domain_traj_steps', 200),
                domain_num_trajectories=cfg.train.get('domain_num_trajectories', 2000),
                domain_convergence_thresh=cfg.train.get('domain_convergence_thresh', 0.01),
                domain_max_growth=cfg.train.get('domain_max_growth', 2.0),
                domain_hard_lower=-model_limit,
                domain_hard_upper=model_limit,
                v_origin_weight=cfg.loss.get('v_origin_weight', 0.0),
            )
            # Capture the final domain limits (may have grown via domain expansion)
            if _train_ret.lower_limit is not None:
                lower_limit = _train_ret.lower_limit
                upper_limit = _train_ret.upper_limit

        # Precompute Cholesky factor for CROWN-friendly quadratic form
        if hasattr(lyapunov_nn, 'precompute_cholesky'):
            lyapunov_nn.precompute_cholesky()

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
        s_scale=derivative_lyaloss.get_s_scale_value(),
        c1_threshold=derivative_lyaloss.c1_threshold,
        c1_multiplier=c1_multiplier,
        uncertainty_transform=uncertainty_transform,
    )
    
    use_adversarial_w = supply_rate.requires_disturbance and w_max is not None
    if use_adversarial_w:
        nw = dynamics.continuous_time_system.nw
        verification_loss = lyapunov.DissipativityVerificationWrapper(
            derivative_lyaloss_check, nx, nw
        )
        limit_w = w_max.to(limit.device)
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

    # Trajectory steps scaled to dt (default ~5 seconds of sim time)
    dt_val = cfg.model.get('dt', 0.01)
    vis_traj_steps = cfg.train.get('vis_traj_steps', max(500, int(5.0 / dt_val)))

    x0 = (torch.rand((40, nx), device=device) - 0.5) * 2 * limit
    x_traj, V_traj = models.simulate(derivative_lyaloss, vis_traj_steps, x0)
    plt.plot(torch.stack(V_traj).cpu().detach().squeeze().numpy())
    vtraj_path = os.path.join(os.getcwd(), "Vtraj_roa.png")
    plt.savefig(vtraj_path)
    if cfg.train.wandb.enabled:
        wandb.log({"V_trajectory": wandb.Image(vtraj_path)})
    plt.close()

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
        P_eff = get_effective_P(lyapunov_nn, device=device)
        P_np = P_eff.cpu().numpy() if P_eff is not None else None
    
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
    n_vis_traj = 100
    # Sample ICs spanning the full plot domain (grid + random)
    torch.manual_seed(42)
    # Grid of ICs covering the plant state box uniformly
    n_grid_per_dim = 8  # 8×8 = 64 grid ICs
    g1 = torch.linspace(-limit[0].item(), limit[0].item(), n_grid_per_dim, device=device)
    g2 = torch.linspace(-limit[1].item(), limit[1].item(), n_grid_per_dim, device=device)
    grid1, grid2 = torch.meshgrid(g1, g2, indexing='ij')
    ic_grid = torch.stack([grid1.reshape(-1), grid2.reshape(-1)], dim=1)
    # Random ICs filling the remaining budget
    n_rand = max(n_vis_traj - ic_grid.shape[0], 0)
    ic_rand = (torch.rand(n_rand, 2, device=device) - 0.5) * 2 * limit[:2]
    ic_plant = torch.cat([ic_grid, ic_rand], dim=0)
    if nx > 2:
        ic_ctrl = torch.zeros(ic_plant.shape[0], nx - 2, device=device)
        ic_all  = torch.cat([ic_plant, ic_ctrl], dim=1)
    else:
        ic_all = ic_plant
    traj_tensor = simulate_closed_loop(dynamics, controller, ic_all, max_steps=vis_traj_steps)
    # traj_tensor: (T+1, N, nx)

    # ── Effective P matrix for final model ───────────────────────────────
    P_final_eff = get_effective_P(lyapunov_nn, device=device)
    P_final_np = P_final_eff.cpu().numpy() if P_final_eff is not None else None

    # ── Provisional trained-model plots (before final PGD verification) ──
    if rho > 0:
        fig_heat = plot_ellipsoid_projections(
            P_init=None, rho_init=0.0,
            P_final=P_final_np, rho_final=rho,
            nx=nx, state_labels=state_labels,
            trajectories=traj_tensor,
            title=f"Trained ROA Projections  (ρ={rho:.5f})",
            lyapunov_final=lyapunov_nn,
            lower_limit_tensor=lower_limit,
            upper_limit_tensor=upper_limit,
        )
        heatmap_path = os.path.join(os.getcwd(), "V_roa_training.png")
        fig_heat.savefig(heatmap_path, dpi=150)
        plt.close(fig_heat)
        logger.info(f"Provisional ROA projection plot saved: {heatmap_path}")
    else:
        heatmap_path = None

    # ── Provisional flow field (before final PGD verification) ───────────
    fig_flow = plot_flow_field(
        dynamics=dynamics,
        controller=controller,
        lyapunov_nn=lyapunov_nn,
        rho=rho,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        state_labels=state_labels,
        trajectories=traj_tensor,
        title=f"Closed-Loop Flow Field  (ρ={rho:.5f})",
    )
    flow_path = os.path.join(os.getcwd(), "flow_field_training.png")
    fig_flow.savefig(flow_path, dpi=150)
    plt.close(fig_flow)
    logger.info(f"Provisional flow field plot saved: {flow_path}")
    if cfg.train.wandb.enabled:
        wandb.log({"flow_field": wandb.Image(flow_path)})

    # =====================================================================
    # Post-training comparison: initial vs trained ROA
    # =====================================================================
    proj_path_cmp       = None
    final_verified_rho  = rho
    improvement         = 0.0

    if verify_init and init_lyapunov_state is not None:
        logger.info("="*60)
        logger.info("POST-TRAINING ROA COMPARISON")
        logger.info("="*60)

        # ── Reconstruct initial model ─────────────────────────────────────
        lyapunov_init = copy.deepcopy(lyapunov_nn)
        lyapunov_init.load_state_dict(init_lyapunov_state, strict=False)
        lyapunov_init.L_chol = None
        lyapunov_init.eval()

        # Effective P for initial model
        P_init_eff = get_effective_P(lyapunov_init, device=device)
        P_init_np = P_init_eff.cpu().numpy() if P_init_eff is not None else None

        # ── PGD bisection on trained model ────────────────────────────────
        final_verified_rho, final_max_rho, final_clean = pgd_find_verified_rho(
            lyapunov_nn, dynamics, controller, supply_rate,
            lower_limit, upper_limit, w_max, derivative_lyaloss.get_s_scale_value(),
            V_decrease_within_roa=V_decrease_within_roa,
            pgd_steps=cfg.get('pgd_verifier_steps', 300),
            num_seeds=cfg.get('pgd_verifier_num_seeds', 5),
            num_samples=50000,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            rho_bisect_tol=cfg.get('verify_init_rho_tol', 0.005),
            max_bisect_iters=cfg.get('verify_init_max_bisect', 20),
            logger=logger,
            c1_threshold=derivative_lyaloss.c1_threshold,
            c1_multiplier=c1_multiplier,
            uncertainty_transform=uncertainty_transform,
            **iqc_params,
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
                s_scale=derivative_lyaloss.get_s_scale_value(),
                rho=final_verified_rho,
                nx=nx,
                logger=logger,
                label="POST-TRAINING",
            )

        # ── ROA projection comparison (init vs trained + exact V(x)=ρ) ───
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
        # No verify_init: still run PGD bisection to get the true verified rho
        P_init_np = None
        logger.info("="*60)
        logger.info("POST-TRAINING PGD VERIFICATION (verify_init=False)")
        logger.info("="*60)
        final_verified_rho, final_max_rho, final_clean = pgd_find_verified_rho(
            lyapunov_nn, dynamics, controller, supply_rate,
            lower_limit, upper_limit, w_max, derivative_lyaloss.get_s_scale_value(),
            V_decrease_within_roa=V_decrease_within_roa,
            pgd_steps=cfg.get('pgd_verifier_steps', 300),
            num_seeds=cfg.get('pgd_verifier_num_seeds', 5),
            num_samples=50000,
            num_samples_per_boundary=cfg.train.num_samples_per_boundary,
            rho_bisect_tol=cfg.get('verify_init_rho_tol', 0.005),
            max_bisect_iters=cfg.get('verify_init_max_bisect', 20),
            logger=logger,
            c1_threshold=derivative_lyaloss.c1_threshold,
            c1_multiplier=c1_multiplier,
            uncertainty_transform=uncertainty_transform,
            **iqc_params,
        )
        logger.info(f"Final PGD-verified rho: {final_verified_rho:.6f}  (max possible: {final_max_rho:.6f})")
        logger.info("="*60)

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

    final_plot_rho = final_verified_rho if final_verified_rho > 0 else rho
    if final_plot_rho > 0:
        fig_final_heat = plot_ellipsoid_projections(
            P_init=None, rho_init=0.0,
            P_final=P_final_np, rho_final=final_plot_rho,
            nx=nx, state_labels=state_labels,
            trajectories=traj_tensor,
            title=(
                f"Final PGD-Verified ROA Projections  (ρ={final_plot_rho:.5f})"
                if final_verified_rho > 0
                else f"Trained ROA Projections  (ρ={final_plot_rho:.5f})"
            ),
            lyapunov_final=lyapunov_nn,
            lower_limit_tensor=lower_limit,
            upper_limit_tensor=upper_limit,
        )
        heatmap_path = os.path.join(os.getcwd(), "V_roa.png")
        fig_final_heat.savefig(heatmap_path, dpi=150)
        plt.close(fig_final_heat)
        logger.info(f"Final ROA projection plot saved: {heatmap_path}")

        fig_final_flow = plot_flow_field(
            dynamics=dynamics,
            controller=controller,
            lyapunov_nn=lyapunov_nn,
            rho=final_plot_rho,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            state_labels=state_labels,
            trajectories=traj_tensor,
            title=(
                f"Closed-Loop Flow Field  (PGD-verified ρ={final_plot_rho:.5f})"
                if final_verified_rho > 0
                else f"Closed-Loop Flow Field  (ρ={final_plot_rho:.5f})"
            ),
        )
        flow_path = os.path.join(os.getcwd(), "flow_field.png")
        fig_final_flow.savefig(flow_path, dpi=150)
        plt.close(fig_final_flow)
        logger.info(f"Final flow field plot saved: {flow_path}")

    if cfg.train.wandb.enabled:
        if heatmap_path is not None:
            wandb.log({"V_heatmap": wandb.Image(heatmap_path)})
        wandb.run.summary["final_rho"] = rho
        wandb.run.summary["final_pgd_violations"] = int((adv_output > 0).sum().item())
        wandb.run.summary["final_max_violation"] = float(max_adv_violation)
        wandb.run.summary["final_verification_success"] = not pgd_verifier_find_counterexamples

    # =====================================================================
    # Formal Verification (α,β-CROWN) — optional, triggered by config
    # =====================================================================
    final_crown_rho = 0.0

    if formal_verification_enabled and verify_final_crown and final_verified_rho > 0:
        init_rho_formal = final_verified_rho
        final_crown_rho = run_crown_verification(
            cfg, formal_cfg, derivative_lyaloss_check, init_rho_formal, logger,
            label="final (post-training)",
            verify_subdir="formal_verification_final",
        )
        if cfg.train.wandb.enabled:
            wandb.run.summary["formal_verified_rho"] = final_crown_rho
            wandb.run.summary["formal_verification_success"] = final_crown_rho > 0
    elif formal_verification_enabled and final_verified_rho <= 0:
        logger.info("Skipping formal verification: no valid PGD-verified rho found.")

    # =====================================================================
    # PGD vs CROWN Comparison (projection plot with extra ρ levels)
    # =====================================================================
    if formal_verification_enabled and (init_crown_rho > 0 or final_crown_rho > 0):
        try:
            extra_levels = []
            if final_verified_rho > 0:
                extra_levels.append((final_verified_rho, 'lime', '-', f'PGD ρ={final_verified_rho:.5f}'))
            fig_crown_cmp = plot_ellipsoid_projections(
                P_init=P_init_np if (verify_init and init_rho > 0) else None,
                rho_init=init_crown_rho if init_crown_rho > 0 else init_rho,
                P_final=P_final_np,
                rho_final=final_crown_rho if final_crown_rho > 0 else final_verified_rho,
                nx=nx, state_labels=state_labels,
                trajectories=traj_tensor if 'traj_tensor' in locals() else None,
                title=f"PGD vs CROWN  (ρ_CROWN={final_crown_rho:.5f})",
                lyapunov_final=lyapunov_nn,
                lower_limit_tensor=lower_limit,
                upper_limit_tensor=upper_limit,
                extra_rho_levels=extra_levels if extra_levels else None,
            )
            crown_cmp_path = os.path.join(os.getcwd(), "roa_pgd_vs_crown.png")
            fig_crown_cmp.savefig(crown_cmp_path, dpi=150)
            plt.close(fig_crown_cmp)
            logger.info(f"PGD vs CROWN comparison saved: {crown_cmp_path}")

            if cfg.train.wandb.enabled:
                wandb.log({"roa_pgd_vs_crown": wandb.Image(crown_cmp_path)})
                wandb.run.summary["init_crown_rho"] = init_crown_rho
                wandb.run.summary["final_crown_rho"] = final_crown_rho
        except Exception as e:
            logger.error(f"Failed to generate comparison plot: {e}")
            import traceback
            traceback.print_exc()

    # =====================================================================
    # 4-Region Comparison Plot: Init PGD / Init CROWN / Final PGD / Final CROWN
    # =====================================================================
    have_init = init_rho > 0
    have_final = (final_verified_rho > 0 or final_crown_rho > 0)
    if formal_verification_enabled and have_init and have_final:
        try:
            import itertools as _it
            _pairs = list(_it.combinations(range(nx), 2))
            _ncols = min(3, len(_pairs))
            _nrows = int(np.ceil(len(_pairs) / _ncols))
            fig4, axes4 = plt.subplots(_nrows, _ncols, figsize=(6 * _ncols, 5.5 * _nrows))
            axes4_flat = np.array(axes4).flatten() if len(_pairs) > 1 else [axes4]

            # Pre-compute nonlinear projection grid for final lyapunov
            _max_rho = max(final_verified_rho, final_crown_rho, init_rho, init_crown_rho)
            _proj4 = {}
            if lyapunov_nn is not None and lower_limit is not None and upper_limit is not None:
                for (pi, pj) in _pairs:
                    _proj4[(pi, pj)] = compute_projected_levelset(
                        lyapunov_nn, _max_rho, pi, pj,
                        lower_limit, upper_limit,
                        grid_size=80, n_opt_steps=200,
                    )

            # Region specs: (rho, color, linestyle, label, is_init)
            _regions = []
            if init_rho > 0:
                _regions.append((init_rho, '#00bfff', '--', f'Init PGD ρ={init_rho:.4f}', True))
            if init_crown_rho > 0:
                _regions.append((init_crown_rho, '#ff6b6b', '--', f'Init CROWN ρ={init_crown_rho:.4f}', True))
            if final_verified_rho > 0:
                _regions.append((final_verified_rho, 'lime', '-', f'Final PGD ρ={final_verified_rho:.4f}', False))
            if final_crown_rho > 0:
                _regions.append((final_crown_rho, 'orange', '-', f'Final CROWN ρ={final_crown_rho:.4f}', False))

            for idx, (i, j) in enumerate(_pairs):
                ax = axes4_flat[idx]
                A_sel = np.zeros((2, nx)); A_sel[0, i] = 1.0; A_sel[1, j] = 1.0

                for (rho_val, color, ls, label_str, is_init_region) in _regions:
                    if rho_val <= 0:
                        continue
                    if is_init_region and P_init_np is not None:
                        # Init regions are ellipsoidal (pure quadratic)
                        try:
                            Phat = project_ellipsoid(P_init_np, A_sel)
                            ex, ey = ellipse_from_Phat(Phat, rho_val)
                            ax.plot(ex, ey, color=color, linewidth=2, linestyle=ls, zorder=5)
                            ax.fill(ex, ey, alpha=0.06, color=color)
                            if idx == 0:
                                ax.plot([], [], color=color, linewidth=2, linestyle=ls, label=label_str)
                        except Exception:
                            pass
                    elif not is_init_region and (i, j) in _proj4:
                        # Final regions use nonlinear level sets
                        xi_v, xj_v, V_proj = _proj4[(i, j)]
                        try:
                            ax.contour(xi_v, xj_v, V_proj, levels=[rho_val],
                                       colors=[color], linewidths=[2.5], linestyles=[ls], zorder=8)
                            ax.contourf(xi_v, xj_v, V_proj, levels=[-np.inf, rho_val],
                                        colors=[color], alpha=0.08, zorder=7)
                            if idx == 0:
                                ax.plot([], [], color=color, linewidth=2.5, linestyle=ls, label=label_str)
                        except Exception:
                            pass

                ax.axhline(0, color='white', linewidth=0.4, alpha=0.4)
                ax.axvline(0, color='white', linewidth=0.4, alpha=0.4)
                ax.set_xlabel(state_labels[i], fontsize=11)
                ax.set_ylabel(state_labels[j], fontsize=11)
                ax.set_facecolor('#1a1a2e')
                ax.grid(True, alpha=0.25)
                ax.set_aspect('equal', adjustable='datalim')
                if idx == 0:
                    ax.legend(fontsize=8, loc='upper right')

            for idx in range(len(_pairs), len(axes4_flat)):
                axes4_flat[idx].set_visible(False)

            fig4.suptitle("4-Region ROA Comparison", fontsize=13, fontweight='bold')
            fig4.tight_layout()
            four_region_path = os.path.join(os.getcwd(), "roa_4region_comparison.png")
            fig4.savefig(four_region_path, dpi=150)
            plt.close(fig4)
            logger.info(f"4-region comparison saved: {four_region_path}")
        except Exception as e:
            logger.error(f"Failed to generate 4-region comparison: {e}")
            import traceback
            traceback.print_exc()

    # =====================================================================
    # Save results JSON for comparison scripts
    # =====================================================================
    import json as _json

    # Compute projected ROA area on plant states (dim 0, 1) for all 4 rho values
    def _compute_area(rho_val):
        if rho_val <= 1e-6:
            return 0.0
        try:
            return compute_roa_volume_projected(
                lyapunov_nn, rho_val, dim_i=0, dim_j=1,
                lower_limit=lower_limit, upper_limit=upper_limit,
                grid_size=200, n_opt_steps=200,
            )
        except Exception as e:
            logger.warning(f"ROA volume computation failed for rho={rho_val}: {e}")
            return 0.0

    roa_area_final_pgd = _compute_area(final_verified_rho)
    roa_area_final_crown = _compute_area(final_crown_rho)
    # Init areas use quadratic V = x'Px → ellipsoid area = π·√(ρ²/det(P_sub))
    def _ellipsoid_area(P_np, rho_val):
        if P_np is None or rho_val <= 1e-6:
            return 0.0
        P_sub = P_np[:2, :2]  # plant states (θ, θ̇)
        det_P = np.linalg.det(P_sub)
        if det_P <= 0:
            return 0.0
        return np.pi * rho_val / np.sqrt(det_P)
    roa_area_init_pgd = _ellipsoid_area(P_init_np, init_rho)
    roa_area_init_crown = _ellipsoid_area(P_init_np, init_crown_rho)

    if roa_area_final_pgd > 0:
        logger.info(f"ROA area (final PGD): {roa_area_final_pgd:.6f}")
    if roa_area_final_crown > 0:
        logger.info(f"ROA area (final CROWN): {roa_area_final_crown:.6f}")
    if roa_area_init_pgd > 0:
        logger.info(f"ROA area (init PGD, ellipsoid): {roa_area_init_pgd:.6f}")
    if roa_area_init_crown > 0:
        logger.info(f"ROA area (init CROWN, ellipsoid): {roa_area_init_crown:.6f}")

    # Sanitize NaN values
    _safe_rho = float(final_verified_rho) if not (isinstance(final_verified_rho, float) and final_verified_rho != final_verified_rho) else 0.0
    import math
    if math.isnan(_safe_rho):
        _safe_rho = 0.0

    results = {
        "V_psd_form": cfg.model.V_psd_form,
        "init_pgd_rho": float(init_rho) if not math.isnan(float(init_rho)) else 0.0,
        "init_crown_rho": float(init_crown_rho) if not math.isnan(float(init_crown_rho)) else 0.0,
        "final_pgd_rho": _safe_rho,
        "final_crown_rho": float(final_crown_rho) if not math.isnan(float(final_crown_rho)) else 0.0,
        "roa_area_init_pgd": float(roa_area_init_pgd),
        "roa_area_init_crown": float(roa_area_init_crown),
        "roa_area_final_pgd": float(roa_area_final_pgd),
        "roa_area_final_crown": float(roa_area_final_crown),
        "roa_area_plant": float(roa_area_final_pgd),  # backward compat
        "learnable_s_scale": bool(cfg.loss.get('learnable_s_scale', False)),
        "learnable_nn_scale": bool(lya_cfg.get('learnable_nn_scale', False)),
        "nn_scale_final": float(lyapunov_nn.get_nn_scale_value()),
        "use_trainable_quadratic": bool(lya_cfg.get('use_trainable_quadratic', False)),
        "hidden_widths": list(cfg.model.lyapunov.hidden_widths) if cfg.model.lyapunov.get('hidden_widths') else [],
        "trainable_controller": bool(cfg.model.controller.get(cfg.model.controller.type, {}).get('trainable', False)),
        "init_roa_anchor_expand": float(cfg.get('init_roa_anchor_expand', 1.2)),
        "run_dir": os.getcwd(),
        "pgd_bisect_verified": bool(final_verified_rho > 0 and final_max_rho > 0),
        "iqc_lambda_final": derivative_lyaloss.get_iqc_lambda_value() if hasattr(derivative_lyaloss, 'get_iqc_lambda_value') else None,
        "s_scale_final": derivative_lyaloss.get_s_scale_value() if hasattr(derivative_lyaloss, 'get_s_scale_value') else None,
    }
    results_path = os.path.join(os.getcwd(), "results.json")
    with open(results_path, "w") as _f:
        _json.dump(results, _f, indent=2)
    logger.info(f"Results JSON saved: {results_path}")

    if cfg.train.wandb.enabled:
        wandb.finish()

    pass


if __name__ == "__main__":
    main()