"""
Post-training formal verification of dissipativity via α,β-CROWN.

Hybrid approach verifies two conditions (both must hold ∀ (ξ,w)):
  1. Dissipativity: max(V(x)−ρ, V(x)−V(x⁺)+s·supply) ≥ 0  [single-output]
  2. Invariance: [ρ−V(x⁺), V(x)] with VNNLIB level-set V(x)≤ρ  [2-output]

**Domain tightening**: For each candidate ρ, we compute a tight bounding box
around the sublevel set {ξ : V(ξ) ≤ ρ} and use that as the CROWN input domain
instead of the full training limits.  This dramatically reduces the search space.

Usage:
    python -m neural_lyapunov_training.verify_dissipativity \\
        --training_dir output/.../11-27-45 --init_rho 0.069 --timeout 200
"""
import argparse
import copy
import os
import sys
import time
import types

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config extraction helpers
# ---------------------------------------------------------------------------

def _extract_rinn_params(mcfg):
    """Extract RINN controller parameters from model config."""
    rinn = mcfg["controller"]["rinn"]
    params = {k: rinn[k] for k in ("A", "Bw", "By", "Cv", "Dvw", "Dvy", "Cu", "Duw", "Duy")}
    params["activation"] = rinn.get("activation", "relu")
    u_max = mcfg.get("u_max", None)
    if u_max:
        params["u_lo"], params["u_up"] = [-u_max], [u_max]
    return params


def _extract_lyap_params(mcfg):
    """Extract Lyapunov function parameters from model config."""
    lyap = mcfg.get("lyapunov", {})
    hw = lyap.get("hidden_widths")
    trainable_P = lyap.get("use_trainable_quadratic", False) or lyap.get("use_frozen_quadratic", False)
    return {
        "hidden_widths": list(hw) if hw else None,
        "eps": 1e-6 if trainable_P else 0.01,
        "V_psd_form": mcfg.get("V_psd_form", "quadratic"),
        "use_nonlinear": lyap.get("use_nonlinear", True),
        "nn_scale": lyap.get("nn_scale", 0.5),
    }


def _get_full_state_limits(mcfg):
    """Return augmented state limits [plant, controller] as a list."""
    model_limit = list(mcfg.get("limit", [0.6, 2.0]))
    ctrl_type = mcfg.get("controller", {}).get("type", "linear_plus_nn")
    if ctrl_type in ("ltic", "rinn"):
        ctrl_cfg = mcfg["controller"].get(ctrl_type, {})
        n_k = ctrl_cfg.get("n_k", 2)
        ctrl_limit = list(ctrl_cfg.get("controller_limit", [1.0] * n_k))
        return model_limit + ctrl_limit
    return model_limit


# ---------------------------------------------------------------------------
# Level-set bounding box computation
# ---------------------------------------------------------------------------

def _build_lyapunov_evaluator(training_cfg, model_pth, device="cpu"):
    """Build a lightweight Lyapunov function evaluator from training config + checkpoint.

    Returns a callable ``V(x_batch) -> values`` and the state dimension ``nx``.
    """
    from neural_lyapunov_training import lyapunov, controllers, dynamical_system
    import neural_lyapunov_training.pendulum as pendulum

    mcfg = training_cfg["model"]
    lyap_cfg = mcfg.get("lyapunov", {})

    # Build plant to get dimensions
    pend_ct = pendulum.PendulumDynamics(m=0.15, l=0.5, beta=0.1)
    plant = dynamical_system.SecondOrderDiscreteTimeSystem(
        pend_ct, mcfg.get("dt", 0.01),
        position_integration=dynamical_system.IntegrationMethod.MidPoint,
        velocity_integration=dynamical_system.IntegrationMethod.ExplicitEuler,
    )
    n_plant = plant.nx

    # Controller state dimension
    ctrl_cfg = mcfg.get("controller", {})
    ctrl_type = ctrl_cfg.get("type", "linear_plus_nn")
    if ctrl_type in ("ltic", "rinn"):
        n_k = ctrl_cfg.get(ctrl_type, {}).get("n_k", 2)
    else:
        n_k = 0
    nx = n_plant + n_k

    # Build Lyapunov
    lp = _extract_lyap_params(mcfg)
    act_map = {"relu": torch.nn.ReLU, "leaky_relu": torch.nn.LeakyReLU,
               "tanh": torch.nn.Tanh, "elu": torch.nn.ELU}
    act = act_map.get(lyap_cfg.get("activation", "leaky_relu"), torch.nn.LeakyReLU)

    lyap_nn = lyapunov.NeuralNetworkLyapunov(
        goal_state=torch.zeros(nx),
        x_dim=nx,
        R_frozen=torch.zeros(nx, nx),
        R_trainable=torch.zeros(nx, nx),
        hidden_widths=lp.get("hidden_widths"),
        eps=lp.get("eps", 0.01),
        V_psd_form=lp.get("V_psd_form", "quadratic"),
        use_nonlinear=lp.get("use_nonlinear", True),
        nn_scale=lp.get("nn_scale", 0.5),
        activation=act,
    )

    # Load weights (only lyapunov.* keys)
    raw = torch.load(model_pth, map_location=device, weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    lyap_sd = {}
    for k, v in sd.items():
        if k.startswith("lyapunov."):
            lyap_sd[k[len("lyapunov."):]] = v
    lyap_nn.load_state_dict(lyap_sd, strict=False)
    lyap_nn.to(device).eval()

    @torch.no_grad()
    def evaluate_V(x_batch):
        return lyap_nn(x_batch.to(device)).squeeze(-1)

    return evaluate_V, nx


def compute_levelset_bbox(V_fn, nx, state_lower, state_upper, rho,
                          n_samples=500000, margin=1.05):
    """Compute tight bounding box for the sublevel set {ξ : V(ξ) ≤ ρ}.

    Samples points uniformly in the full domain, evaluates V(x), and computes
    per-dimension min/max of points where V(x) ≤ ρ.  Adds a safety margin.

    Args:
        V_fn: callable V(x_batch) -> values, shape (batch,)
        nx: state dimension
        state_lower: list of lower bounds for full domain
        state_upper: list of upper bounds for full domain
        rho: sublevel set value
        n_samples: number of random samples
        margin: multiplicative margin (1.05 = 5% expansion)

    Returns:
        (tight_lower, tight_upper): lists of per-dimension bounds,
        or (state_lower, state_upper) if level set fills the full domain.
    """
    lo = torch.tensor(state_lower, dtype=torch.float32)
    hi = torch.tensor(state_upper, dtype=torch.float32)

    # Sample uniformly in the box
    x = lo + (hi - lo) * torch.rand(n_samples, nx)

    # Evaluate V in batches to avoid OOM
    batch_size = 50000
    V_vals = []
    for i in range(0, n_samples, batch_size):
        V_vals.append(V_fn(x[i:i + batch_size]).cpu())
    V_vals = torch.cat(V_vals)

    # Points inside the level set
    mask = V_vals <= rho
    n_inside = mask.sum().item()
    if n_inside < 10:
        # Level set too small to estimate — return full domain (conservative)
        return list(state_lower), list(state_upper)

    x_inside = x[mask]

    # Per-dimension min/max
    x_min = x_inside.min(dim=0).values
    x_max = x_inside.max(dim=0).values

    # Apply margin: expand by margin factor around the center
    center = (x_min + x_max) / 2
    half_width = (x_max - x_min) / 2 * margin

    tight_lo = (center - half_width).tolist()
    tight_hi = (center + half_width).tolist()

    # Clamp to full domain (don't go outside training limits)
    tight_lo = [max(tl, sl) for tl, sl in zip(tight_lo, state_lower)]
    tight_hi = [min(th, sh) for th, sh in zip(tight_hi, state_upper)]

    return tight_lo, tight_hi


# ---------------------------------------------------------------------------
# Build the Customized(...) model string for ABCROWN config
# ---------------------------------------------------------------------------

_ACT_STR_MAP = {
    "relu": "torch.nn.ReLU", "leaky_relu": "torch.nn.LeakyReLU",
    "tanh": "torch.nn.Tanh", "elu": "torch.nn.ELU",
}


def build_customized_string(
    training_cfg, rho=0.0, models_py_path=None,
    s_scale_override=None, verification_mode='combined',
):
    """Build ``Customized(...)`` string for the α,β-CROWN YAML config."""
    if models_py_path is None:
        models_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models.py"))

    mcfg = training_cfg["model"]
    lyap = mcfg.get("lyapunov", {})
    supply = training_cfg.get("supply_rate", {})
    loss_cfg = training_cfg.get("loss", {})

    rinn_params = _extract_rinn_params(mcfg)
    lyap_params = _extract_lyap_params(mcfg)
    act_str = _ACT_STR_MAP.get(lyap.get("activation", "leaky_relu"), "torch.nn.LeakyReLU")

    s_scale = float(s_scale_override) if s_scale_override is not None else loss_cfg.get("s_scale", 1.0)

    parts = [
        f'"{models_py_path}"',
        f'"create_pendulum_rinn_state_feedback_model"',
        f"dt={mcfg.get('dt', 0.01)}",
        f"gamma={supply.get('gamma', 100.0)}",
        f"w_max={supply.get('w_max', 0.024525)}",
        f"rho={rho}",
        f"s_scale={s_scale}",
        f"rinn_parameters={repr(rinn_params)}",
        f'lyapunov_parameters={{**{repr(lyap_params)}, "activation": {act_str}}}',
        f"output_C={repr(mcfg.get('output_C', [[1.0, 0.0], [0.0, 1.0]]))}",
        f'verification_mode="{verification_mode}"',
    ]
    return "Customized(" + ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# ABCROWN config template generation
# ---------------------------------------------------------------------------

def generate_config_template(
    config_path, training_cfg, model_pth_path, csv_name,
    state_lower, state_upper, w_lower, w_upper,
    timeout=100000000, device="auto", batch_size=None,
    s_scale_override=None, verification_mode='combined',
    use_domain_placeholders=False,
):
    """Write an α,β-CROWN YAML config with ``__RHO__`` placeholder.

    If ``use_domain_placeholders`` is True, the data.dataset field uses
    ``__DOMAIN_LOWER__`` and ``__DOMAIN_UPPER__`` placeholders that will
    be replaced per-ρ by ``_check_rho`` with the tight level-set bbox.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if batch_size is None:
        if device == "cuda":
            # Adaptive GPU batch size based on Lyapunov NN width.
            # Larger NNs need smaller batches to avoid OOM.
            lyap_widths = training_cfg.get("model", {}).get("lyapunov", {}).get("hidden_widths", [32, 32])
            max_width = max(lyap_widths) if lyap_widths else 32
            if max_width >= 128:
                batch_size = 32768
            elif max_width >= 64:
                batch_size = 65536
            else:
                batch_size = 131072
        else:
            batch_size = 1000

    model_str = build_customized_string(
        training_cfg, rho="__RHO__",
        s_scale_override=s_scale_override, verification_mode=verification_mode,
    )
    total_lower = list(state_lower) + list(w_lower)
    total_upper = list(state_upper) + list(w_upper)
    ndim = len(total_lower)

    models_py_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), "models.py"))
    if use_domain_placeholders:
        dataset_str = (
            f'Customized("{models_py_abs}", '
            f'"box_data", lower_limit=__DOMAIN_LOWER__, upper_limit=__DOMAIN_UPPER__, '
            f"ndim={ndim}, scale=1.0, hole_size=0.0)"
        )
    else:
        dataset_str = (
            f'Customized("{models_py_abs}", '
            f'"box_data", lower_limit={total_lower}, upper_limit={total_upper}, '
            f"ndim={ndim}, scale=1.0, hole_size=0.0)"
        )

    config = {
        "general": {
            "device": device,
            "conv_mode": "matrix",
            "enable_incomplete_verification": (device == "cuda"),
            "root_path": os.path.dirname(os.path.abspath(config_path)),
            "csv_name": csv_name,
        },
        "model": {
            "name": model_str,
            "path": model_pth_path,
            "input_shape": [-1, ndim],
        },
        "data": {
            "dataset": dataset_str,
        },
        "attack": {"pgd_order": "before"},
        "solver": {
            "batch_size": batch_size,
            "min_batch_size_ratio": 0.0,
            "bound_prop_method": "crown",
        },
        "bab": {
            "override_timeout": timeout,
            "decision_thresh": -1e-6,
            "branching": {
                "method": "sb",
                "input_split": {
                    "enable": True,
                    "ibp_enhancement": True,
                    "compare_with_old_bounds": True,
                    "adv_check": 100,
                    "sb_coeff_thresh": 0.01,
                    "enable_clip_domains": True,
                },
            },
        },
    }
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


# ---------------------------------------------------------------------------
# Single-ρ verification
# ---------------------------------------------------------------------------

def _ensure_abcrown_on_path():
    """Add alpha-beta-CROWN to sys.path if needed."""
    try:
        import complete_verifier  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in [os.path.join(root, "alpha-beta-CROWN", "complete_verifier"),
              os.path.join(root, "alpha-beta-CROWN")]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def _check_rho(rho, check_args, config_template_path, timeout_override=None,
               pgd_restarts_override=None, tight_state_lower=None,
               tight_state_upper=None):
    """Verify single ρ: regenerate VNNLIB specs, rewrite config, run ABCROWN.

    Args:
        timeout_override: If set, use this timeout instead of check_args.timeout.
        pgd_restarts_override: If set, use this PGD restart count instead of default.
        tight_state_lower: If provided, use tight domain for VNNLIB + CROWN config.
        tight_state_upper: If provided, use tight domain for VNNLIB + CROWN config.
    """
    from contextlib import redirect_stdout, redirect_stderr
    _ensure_abcrown_on_path()
    from complete_verifier import ABCROWN

    mode = check_args.mode
    timeout = timeout_override if timeout_override is not None else check_args.timeout
    pgd_restarts = pgd_restarts_override if pgd_restarts_override is not None else 10000

    # Use tight domain if provided, else fall back to full domain
    vnnlib_lower = tight_state_lower if tight_state_lower is not None else check_args.lower_limit
    vnnlib_upper = tight_state_upper if tight_state_upper is not None else check_args.upper_limit

    t_start = time.time()
    if tight_state_lower is not None:
        vol_full = np.prod([u - l for l, u in zip(check_args.lower_limit, check_args.upper_limit)])
        vol_tight = np.prod([u - l for l, u in zip(vnnlib_lower, vnnlib_upper)])
        ratio = vol_tight / vol_full if vol_full > 0 else 1.0
        print(f"Checking rho={rho} ({mode}) timeout={timeout}s pgd={pgd_restarts} "
              f"domain={[f'{l:.2f}' for l in vnnlib_lower]}→{[f'{u:.2f}' for u in vnnlib_upper]} "
              f"({ratio:.1%} of full)")
    else:
        print(f"Checking rho={rho} ({mode}) timeout={timeout}s pgd={pgd_restarts} [full domain]")

    # 1. Regenerate VNNLIB specs with tight domain
    cmd_parts = [
        f"{sys.executable} -m neural_lyapunov_training.generate_vnnlib",
        f"--lower_limit {' '.join(map(str, vnnlib_lower))}",
        f"--upper_limit {' '.join(map(str, vnnlib_upper))}",
        f"--hole_size {check_args.hole_size}",
    ]
    if mode == 'invariance':
        cmd_parts.append(f"--value_levelset {rho}")
    cmd_parts.append("--no_check_x_next")
    if check_args.extra_input_lower:
        cmd_parts.append(f"--extra_input_lower {' '.join(map(str, check_args.extra_input_lower))}")
        cmd_parts.append(f"--extra_input_upper {' '.join(map(str, check_args.extra_input_upper))}")
    output_gen = os.path.join(check_args.output_folder, f"rho_{rho:.5f}_spec.txt")
    cmd_parts.append(f"-- {check_args.spec_prefix} >{output_gen} 2>&1")
    os.system(" ".join(cmd_parts))

    # 2. Rewrite config with actual ρ AND tight domain
    with open(config_template_path) as f:
        content = f.read()
    # Replace ρ placeholder
    content = content.replace("__RHO__", str(rho))
    # Replace domain placeholders if tight domain is provided
    if tight_state_lower is not None:
        total_tight_lower = list(vnnlib_lower) + list(check_args.extra_input_lower)
        total_tight_upper = list(vnnlib_upper) + list(check_args.extra_input_upper)
        content = content.replace("__DOMAIN_LOWER__", str(total_tight_lower))
        content = content.replace("__DOMAIN_UPPER__", str(total_tight_upper))
    config_path = os.path.join(check_args.output_folder, f"config_rho_{rho:.6f}.yaml")
    with open(config_path, "w") as f:
        f.write(content)

    # 3. Run ABCROWN
    output_path = os.path.join(check_args.output_folder, f"rho_{rho:.5f}.txt")
    try:
        with open(output_path, "w") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                verifier = ABCROWN(
                    args=[], csv_name=f'{check_args.spec_prefix}.csv',
                    config=config_path, override_timeout=timeout,
                    pgd_order="before", pgd_restarts=pgd_restarts,
                )
                ret = verifier.main()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            elapsed = time.time() - t_start
            print(f"  CUDA OOM → unknown ({elapsed:.1f}s)")
            return "unknown"
        raise

    # Parse result
    if any("unsafe" in k for k in ret):
        result = "unsafe"
    elif "unknown" in ret:
        result = "unknown"
    else:
        result = "safe"
    elapsed = time.time() - t_start
    print(f"  Result: {result} ({elapsed:.1f}s)")
    return result


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _strip_checkpoint_keys(ckpt_path, keys_to_remove):
    """Remove specified keys from a checkpoint's state_dict and re-save."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw["state_dict"] = {k: v for k, v in raw["state_dict"].items()
                             if not any(pat in k for pat in keys_to_remove)}
    else:
        raw = {k: v for k, v in raw.items()
               if not any(pat in k for pat in keys_to_remove)}
    torch.save(raw, ckpt_path)


def _patch_missing_r_frozen(ckpt_path, nx):
    """Add lyapunov.R_frozen to checkpoint if missing (older checkpoints).

    CROWN loads with strict=True, so the checkpoint must have all keys
    that the model registers as buffers/parameters.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    if "lyapunov.R_frozen" not in sd:
        sd["lyapunov.R_frozen"] = torch.zeros(nx, nx)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw["state_dict"] = sd
        else:
            raw = sd
        torch.save(raw, ckpt_path)
        return True
    return False


def _patch_missing_dvw_upper_mask(ckpt_path):
    """Add _dvw_upper_mask buffers if missing (fixed-controller checkpoints).

    The verification model factory always registers _dvw_upper_mask as a buffer,
    but fixed-controller checkpoints may not include it.  Infer the size from
    the Dvw matrix in the checkpoint and add an upper-triangular bool mask.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    patched = False
    for prefix in ["dynamics.rinn", "controller"]:
        mask_key = f"{prefix}._dvw_upper_mask"
        dvw_key = f"{prefix}.Dvw"
        if mask_key not in sd and dvw_key in sd:
            n = sd[dvw_key].shape[0]
            sd[mask_key] = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            patched = True
    if patched:
        if isinstance(raw, dict) and "state_dict" in raw:
            raw["state_dict"] = sd
        else:
            raw = sd
        torch.save(raw, ckpt_path)
    return patched


def _read_and_apply_learned_nn_scale(model_pth, training_cfg):
    """Read learned nn_scale from checkpoint and override the config value.

    When learnable_nn_scale=True, the checkpoint stores _log_nn_scale as an
    nn.Parameter.  The verification model uses a fixed nn_scale, so we need to
    convert the learned value and inject it into the config before building the
    Customized() model string.
    """
    try:
        ckpt = torch.load(model_pth, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        if "lyapunov._log_nn_scale" in sd:
            log_val = sd["lyapunov._log_nn_scale"]
            nn_scale = float(torch.exp(log_val).item())
            V_psd_form = training_cfg.get("model", {}).get("V_psd_form", "")
            if V_psd_form == "quadratic_times_tanh":
                nn_scale = min(nn_scale, 0.999)
            training_cfg.setdefault("model", {}).setdefault("lyapunov", {})["nn_scale"] = nn_scale
    except Exception:
        pass


def _read_s_scale_from_checkpoint(model_pth):
    """Read effective_s_scale from checkpoint if available."""
    try:
        ckpt = torch.load(model_pth, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "s_scale" in ckpt:
            return float(ckpt["s_scale"])
    except Exception:
        pass
    return None


def _find_model_checkpoint(training_dir, model_pth_override=None):
    """Locate model checkpoint, trying several candidates."""
    if model_pth_override:
        if os.path.exists(model_pth_override):
            return os.path.abspath(model_pth_override)
        return None
    for name in ["lyaloss_1.pth", "lyaloss_1.0.pth", "lyapunov_nn.pth"]:
        p = os.path.join(training_dir, name)
        if os.path.exists(p):
            return os.path.abspath(p)
    return None


# ---------------------------------------------------------------------------
# Main verification pipeline
# ---------------------------------------------------------------------------

def run_verification(
    training_dir, init_rho, training_cfg, logger=None,
    timeout=200, rho_eps=0.001, rho_multiplier=1.2, max_bisect_iters=30,
    hole_size=0.001, model_pth_override=None, verify_subdir="formal_verification",
    device="auto", batch_size=None, effective_s_scale=None, pure_quadratic=False,
    smart_bracket=False, bracket_shift=None, pgd_restarts=10000, rho_bisect_tol=None,
    domain_tightening=True, tightening_samples=500000, tightening_margin=1.05,
):
    """Run hybrid formal verification with bisection on ρ.

    Checks invariance first (faster per-spec, quicker safe/unknown signal)
    then dissipativity (single-output, reliable falsification).
    Both must pass for a candidate ρ to be accepted.

    Domain tightening (default ON):
        For each candidate ρ, computes a tight bounding box around the
        sublevel set {ξ : V(ξ) ≤ ρ} and uses it as the CROWN input domain.
        This dramatically reduces the volume CROWN must search.

    Speedup options:
        smart_bracket: Skip bracket-finding phase.  When combined with
            bracket_shift, uses the shifted-probe strategy (see run_bisect).
            Without bracket_shift, uses init_rho as midpoint with bracket
            [init_rho/multiplier, init_rho*multiplier].
        bracket_shift: Factor in (0,1) for the first probe when smart_bracket
            is True.  E.g. 0.95 probes at 0.95×init_rho first: if safe the
            bracket is [0.95×init_rho, mult×init_rho]; if unsafe the bracket
            is [init_rho/mult, 0.95×init_rho].  Default None (disabled).
        pgd_restarts: Number of PGD restarts for ABCROWN falsification.
        rho_bisect_tol: Alias for rho_eps (for backwards compatibility).

    Args:
        pure_quadratic: Disable Lyapunov NN (use V=x'Px only).
        domain_tightening: Compute tight level-set bbox per ρ (default True).
        tightening_samples: Number of random samples for bbox estimation.
        tightening_margin: Multiplicative margin for tight bbox (e.g. 1.05 = 5%).

    Returns:
        dict with ``verified_rho``, ``rho_upper``, ``success``.
    """
    if rho_bisect_tol is not None:
        rho_eps = rho_bisect_tol

    def _log(msg):
        if logger:
            logger.info(msg)
        print(msg)

    # Pure-quadratic mode: disable NN, strip checkpoint
    if pure_quadratic:
        training_cfg = copy.deepcopy(training_cfg)
        training_cfg["model"].setdefault("lyapunov", {})["use_nonlinear"] = False
        _log("[verify] pure_quadratic=True: NN component disabled")

    mcfg = training_cfg["model"]
    supply = training_cfg.get("supply_rate", {})
    full_limit = _get_full_state_limits(mcfg)
    w_max_val = supply.get("w_max", 0.024525)
    w_lower, w_upper = [-w_max_val], [w_max_val]

    # Find model checkpoint
    model_pth = _find_model_checkpoint(training_dir, model_pth_override)
    if model_pth is None:
        _log(f"[verify] ERROR: No model checkpoint found in {training_dir}")
        return {"verified_rho": 0.0, "rho_upper": 0.0, "success": False,
                "error": "No checkpoint found"}

    # Create a working copy to avoid modifying the original checkpoint
    import shutil
    verify_dir = os.path.join(training_dir, verify_subdir)
    os.makedirs(verify_dir, exist_ok=True)
    effective_model_pth = os.path.join(verify_dir, "model_verify.pth")
    shutil.copy2(model_pth, effective_model_pth)

    # For pure_quadratic, strip lyapunov.net keys from the copy
    if pure_quadratic:
        raw = torch.load(effective_model_pth, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "state_dict" in raw:
            raw["state_dict"] = {k: v for k, v in raw["state_dict"].items()
                                 if "lyapunov.net" not in k}
        else:
            raw = {k: v for k, v in raw.items() if "lyapunov.net" not in k}
        torch.save(raw, effective_model_pth)
        _log(f"[verify] Stripped checkpoint: {effective_model_pth}")

    # Read learned s_scale from checkpoint
    if effective_s_scale is None:
        effective_s_scale = _read_s_scale_from_checkpoint(model_pth)

    # Read learned nn_scale from checkpoint (learnable_nn_scale training param)
    # and inject into config so the verification model uses the trained value.
    _read_and_apply_learned_nn_scale(model_pth, training_cfg)

    # Strip training-only parameters so CROWN's strict load succeeds
    _strip_checkpoint_keys(effective_model_pth, ["_log_s_scale", "_log_nn_scale"])

    state_lower = [-v for v in full_limit]
    state_upper = list(full_limit)
    nx_state = len(state_lower)

    # Patch old checkpoints missing lyapunov.R_frozen
    if _patch_missing_r_frozen(effective_model_pth, nx_state):
        _log("[verify] Patched checkpoint: added missing lyapunov.R_frozen")

    # Patch checkpoints missing _dvw_upper_mask (fixed-controller checkpoints)
    _patch_missing_dvw_upper_mask(effective_model_pth)

    _log(f"[verify] Model: {model_pth}")
    _log(f"[verify] init_rho={init_rho}, s_scale={effective_s_scale}")
    _log(f"[verify] Full domain: {state_lower} → {state_upper}")

    # Build Lyapunov evaluator for domain tightening
    V_fn = None
    nx = len(state_lower)
    if domain_tightening:
        try:
            V_fn, nx = _build_lyapunov_evaluator(training_cfg, effective_model_pth)
            # Quick sanity check: V(0) should be ~0
            v0 = V_fn(torch.zeros(1, nx)).item()
            _log(f"[verify] Domain tightening ON (V(0)={v0:.6f}, "
                 f"samples={tightening_samples}, margin={tightening_margin})")
        except Exception as e:
            _log(f"[verify] WARNING: Could not build V(x) evaluator: {e}")
            _log("[verify] Falling back to full domain (no tightening)")
            V_fn = None
            domain_tightening = False

    # Invariance first (faster per-spec, quicker safe/unknown signal),
    # then dissipativity (single-output, heavier but reliable falsification).
    MODES = ['invariance', 'dissipativity']

    try:
        from neural_lyapunov_training.bisect import run_bisect

        # Set up per-mode directories and config templates
        mode_resources = {}
        for mode in MODES:
            mode_dir = os.path.join(training_dir, verify_subdir, mode)
            specs_dir = os.path.join(mode_dir, "specs")
            output_dir = os.path.join(mode_dir, "output")
            os.makedirs(specs_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            spec_prefix = os.path.abspath(os.path.join(specs_dir, "bisect"))
            config_template = os.path.abspath(os.path.join(mode_dir, "config_template.yaml"))

            generate_config_template(
                config_template, training_cfg, effective_model_pth,
                spec_prefix + ".csv", state_lower, state_upper, w_lower, w_upper,
                timeout=timeout, device=device, batch_size=batch_size,
                s_scale_override=effective_s_scale, verification_mode=mode,
                use_domain_placeholders=domain_tightening,
            )

            check_args = types.SimpleNamespace(
                spec_prefix=spec_prefix, output_folder=output_dir,
                lower_limit=state_lower, upper_limit=state_upper,
                hole_size=hole_size, timeout=timeout,
                extra_input_lower=w_lower, extra_input_upper=w_upper,
                mode=mode,
            )
            mode_resources[mode] = (check_args, config_template)
            _log(f"[verify] {mode}: ready (domain_placeholders={domain_tightening})")

        # Cache for tight bboxes: rho -> (tight_lower, tight_upper)
        _bbox_cache = {}

        def _get_tight_bbox(rho):
            """Compute (and cache) tight bbox for a given ρ."""
            if not domain_tightening or V_fn is None:
                return None, None
            if rho not in _bbox_cache:
                t_lo, t_hi = compute_levelset_bbox(
                    V_fn, nx, state_lower, state_upper, rho,
                    n_samples=tightening_samples, margin=tightening_margin,
                )
                _bbox_cache[rho] = (t_lo, t_hi)
            return _bbox_cache[rho]

        # Combined check: both modes must pass
        def combined_check_fn(rho):
            tight_lo, tight_hi = _get_tight_bbox(rho)
            for m in MODES:
                ca, tpl = mode_resources[m]
                res = _check_rho(rho, ca, tpl,
                                 pgd_restarts_override=pgd_restarts,
                                 tight_state_lower=tight_lo,
                                 tight_state_upper=tight_hi)
                _log(f"[verify] rho={rho:.6f} — {m}: {res}")
                if res != "safe":
                    return "unsafe" if res == "unsafe" else "unknown"
            return "safe"

        rho_l, rho_u = run_bisect(
            combined_check_fn, init_rho,
            rho_eps=rho_eps, rho_multiplier=rho_multiplier,
            max_iters=max_bisect_iters, smart_bracket=smart_bracket,
            bracket_shift=bracket_shift,
        )
        _log(f"[verify] Verified rho={rho_l:.6f} (upper={rho_u:.6f})")
        return {"verified_rho": rho_l, "rho_upper": rho_u, "success": rho_l > 0}

    except Exception as e:
        _log(f"[verify] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"verified_rho": 0.0, "rho_upper": 0.0, "success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Formal verification of dissipativity with bisection on rho.")
    parser.add_argument("--training_dir", type=str, required=True,
                        help="Training output directory (contains lyaloss_*.pth + config.yaml).")
    parser.add_argument("--init_rho", type=float, required=True, help="Initial rho from PGD.")
    parser.add_argument("--timeout", type=int, default=200, help="ABCROWN timeout per run (s).")
    parser.add_argument("--rho_eps", type=float, default=0.001, help="Bisection precision.")
    parser.add_argument("--max_bisect_iters", type=int, default=30)
    parser.add_argument("--hole_size", type=float, default=0.001, help="Hole size around origin.")
    parser.add_argument("--smart_bracket", action="store_true",
                        help="Skip bracket-finding; use init_rho as bracket midpoint.")
    parser.add_argument("--bracket_shift", type=float, default=None,
                        help="Shifted-probe factor for smart_bracket (e.g. 0.95). "
                             "Probes bracket_shift×init_rho first; if safe, bracket is "
                             "[probe, mult×init_rho]; if unsafe, [init_rho/mult, probe].")
    parser.add_argument("--pgd_restarts", type=int, default=10000,
                        help="PGD restart count for ABCROWN falsification.")
    parser.add_argument("--no_domain_tightening", action="store_true",
                        help="Disable per-rho level-set domain tightening.")
    parser.add_argument("--tightening_samples", type=int, default=500000,
                        help="Number of random samples for level-set bbox estimation.")
    parser.add_argument("--tightening_margin", type=float, default=1.05,
                        help="Multiplicative margin for tight bbox (1.05 = 5%% expansion).")
    args = parser.parse_args()

    config_path = os.path.join(args.training_dir, "config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(args.training_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        print(f"ERROR: Cannot find config.yaml in {args.training_dir}")
        sys.exit(1)

    with open(config_path, "r") as f:
        training_cfg = yaml.safe_load(f)

    result = run_verification(
        training_dir=args.training_dir, init_rho=args.init_rho,
        training_cfg=training_cfg, timeout=args.timeout,
        rho_eps=args.rho_eps, max_bisect_iters=args.max_bisect_iters,
        hole_size=args.hole_size, smart_bracket=args.smart_bracket,
        bracket_shift=args.bracket_shift,
        pgd_restarts=args.pgd_restarts,
        domain_tightening=not args.no_domain_tightening,
        tightening_samples=args.tightening_samples,
        tightening_margin=args.tightening_margin,
    )

    print(f"\n{'='*60}")
    print(f"  Verified rho:  {result['verified_rho']:.6f}")
    print(f"  Upper bound:   {result['rho_upper']:.6f}")
    print(f"  Success:       {result['success']}")
    if "error" in result:
        print(f"  Error:         {result['error']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
