"""Bisection on rho.

Supports two modes:
  1. Standard (default): ρ is varied as the VNNLIB level-set constraint.
  2. ``--rho_in_config``: ρ is baked into the model via the α,β-CROWN config
     file.  A placeholder string (default ``__RHO__``) in the config is
     replaced with the current ρ value for each verification step.
     VNNLIB specs are generated once upfront.
"""
import argparse
import os
import sys
from contextlib import redirect_stdout, redirect_stderr


def _ensure_abcrown_on_path():
    """Add alpha-beta-CROWN directories to sys.path if not already importable."""
    try:
        import complete_verifier  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.dirname(_here)  # project root
    for _p in [
        os.path.join(_root, "alpha-beta-CROWN", "complete_verifier"),
        os.path.join(_root, "alpha-beta-CROWN"),
        os.path.join(_root, "verification", "complete_verifier"),
        os.path.join(_root, "verification"),
    ]:
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)


def check_rho(rho, args, additional_args):
    """Verify a single ρ value.  Returns ``'safe'``, ``'unsafe'``, or ``'unknown'``."""
    _ensure_abcrown_on_path()
    from complete_verifier import ABCROWN

    print(f"Checking rho={rho}")
    output_gen_spec = os.path.join(
        args.output_folder, f"rho_{rho:.5f}_spec.txt")

    if not getattr(args, 'rho_in_config', False):
        # Standard mode: regenerate VNNLIB specs with this ρ as level-set value
        command = (
            "python -m neural_lyapunov_training.generate_vnnlib "
            f"--lower_limit {' '.join(map(str, args.lower_limit))} "
            f"--upper_limit {' '.join(map(str, args.upper_limit))} "
            f"--hole_size {args.hole_size} "
            f"--value_levelset {rho} "
        )
        if getattr(args, 'check_x_next_only', False):
            command += "--check_x_next_only "
        extra_lower = getattr(args, 'extra_input_lower', [])
        extra_upper = getattr(args, 'extra_input_upper', [])
        if extra_lower:
            command += f"--extra_input_lower {' '.join(map(str, extra_lower))} "
            command += f"--extra_input_upper {' '.join(map(str, extra_upper))} "
        command += f"-- {args.spec_prefix} >{output_gen_spec} 2>&1"
        os.system(command)
        config_to_use = args.config
    else:
        # rho_in_config mode: specs already generated; rewrite config with ρ
        rho_key = getattr(args, 'rho_config_key', '__RHO__')
        with open(args.config, "r") as f:
            config_content = f.read()
        config_content = config_content.replace(rho_key, f"{rho}")
        config_to_use = os.path.join(args.output_folder, f"config_rho_{rho:.6f}.yaml")
        with open(config_to_use, "w") as f:
            f.write(config_content)

    print("Start verification")
    output_path = os.path.join(args.output_folder, f"rho_{rho:.5f}.txt")
    print("Output path:", output_path)
    with open(output_path, "w") as file:
        with redirect_stdout(file), redirect_stderr(file):
            verifier = ABCROWN(
                args=additional_args,
                csv_name=f'{args.spec_prefix}.csv',
                config=config_to_use,
                override_timeout=args.timeout,
                pgd_order="before",
                pgd_restarts=10000
            )
            ret = verifier.main()
    print("Result:", ret)
    result = "safe"
    for k, v in ret.items():
        if "unsafe" in k:
            result = "unsafe"
    if result == "safe" and "unknown" in ret.keys():
        result = "unknown"
    print(result)
    print()
    return result


def run_bisect(check_fn, init_rho, rho_eps=0.001, rho_multiplier=1.2, max_iters=None,
               max_shrink_iters=10, smart_bracket=False, bracket_shift=None):
    """Run bisection to find the maximum verified ρ.

    Args:
        check_fn: callable(rho) -> ``'safe'``/``'unsafe'``/``'unknown'``
        init_rho: Initial ρ value (typically the PGD-verified ρ).
        rho_eps: Bisection precision.
        rho_multiplier: Growth / shrink factor.
        max_iters: Maximum bisection iterations (``None`` = unlimited).
        max_shrink_iters: Maximum shrink iterations when initial ρ is not safe.
        smart_bracket: If True, skip the bracket-finding phase and use a
            pre-computed bracket based on init_rho.
        bracket_shift: Only used when ``smart_bracket=True``.  A factor in
            (0, 1) that shifts the first probe below init_rho (e.g. 0.95).
            Strategy:
              1. Probe ``probe = bracket_shift * init_rho`` first.
              2. If safe   → bracket = ``[probe, init_rho * rho_multiplier]``.
              3. If unsafe → bracket = ``[init_rho / rho_multiplier, probe]``;
                             verify the lower bound; shrink further if needed.
            When None (default), falls back to the original smart_bracket
            behaviour: bracket = ``[init_rho / rho_multiplier,
            init_rho * rho_multiplier]`` with lower-bound verification.

    Returns:
        ``(rho_lower, rho_upper)`` bounding the maximum verified ρ.
    """
    import time
    t0 = time.time()
    total_checks = 0

    def _check(rho):
        nonlocal total_checks
        total_checks += 1
        t = time.time() - t0
        print(f"[bisect] check #{total_checks} | rho={rho:.6f} | elapsed={t:.1f}s")
        return check_fn(rho)

    if smart_bracket and bracket_shift is not None:
        # Shifted-probe strategy:
        #   First probe at (bracket_shift * init_rho) — cheaper than full pgd_rho.
        #   The CROWN-safe region is almost always slightly below the PGD region,
        #   so starting the probe at e.g. 0.95×pgd gives a high success rate on
        #   the first check.  Then adaptively set the bracket:
        #     safe   → [probe, mult × init_rho]  (search upward)
        #     unsafe → [init_rho / mult, probe]  (search downward, verify lower)
        if not (0.0 < bracket_shift < 1.0):
            raise ValueError(f"bracket_shift must be in (0, 1), got {bracket_shift}")
        probe = bracket_shift * init_rho
        print(f"[bisect] Shifted smart bracket: probe={probe:.6f} "
              f"({bracket_shift:.2%} of pgd_rho={init_rho:.6f})")
        ret = _check(probe)
        if ret == "safe":
            rho_l = probe
            rho_u = init_rho * rho_multiplier
            print(f"[bisect] Probe safe → bracket [{rho_l:.6f}, {rho_u:.6f}] (search upward)")
        else:
            # Probe failed: safe region is somewhere below probe.  Set upper
            # bound to probe and look for a safe lower bound.
            rho_u = probe
            rho_l = init_rho / rho_multiplier
            print(f"[bisect] Probe unsafe ({ret}) → bracket [{rho_l:.6f}, {rho_u:.6f}] "
                  f"(search downward; verifying lower bound...)")
            ret_l = _check(rho_l)
            if ret_l != "safe":
                print(f"[bisect] Lower bound {rho_l:.6f} also NOT safe ({ret_l}). "
                      f"Shrinking further...")
                rho = rho_l
                shrink_count = 0
                while ret_l != "safe":
                    rho /= rho_multiplier
                    shrink_count += 1
                    if rho < 1e-10 or shrink_count >= max_shrink_iters:
                        print(f"[bisect] Could not find any safe rho after "
                              f"{shrink_count} shrink iterations.")
                        return 0.0, 0.0
                    ret_l = _check(rho)
                rho_l = rho
                rho_u = rho * rho_multiplier  # tighten upper bound

    elif smart_bracket:
        # Original smart_bracket: no probe-shift — verify lower bound directly.
        rho_l = init_rho / rho_multiplier
        rho_u = init_rho * rho_multiplier
        print(f"[bisect] Smart bracket: [{rho_l:.6f}, {rho_u:.6f}]")
        print(f"[bisect] Verifying lower bound rho_l={rho_l:.6f}...")
        ret = _check(rho_l)
        if ret != "safe":
            print(f"[bisect] Smart bracket lower bound NOT safe ({ret}). "
                  f"Falling back to standard bracket-finding from rho={rho_l:.6f}.")
            # Fall back: shrink from rho_l until we find a safe value
            rho = rho_l
            shrink_count = 0
            while ret != "safe":
                rho /= rho_multiplier
                shrink_count += 1
                if rho < 1e-10 or shrink_count >= max_shrink_iters:
                    print(f"[bisect] Could not find any safe rho after {shrink_count} shrink iterations.")
                    return 0.0, 0.0
                ret = _check(rho)
            rho_l = rho
            rho_u = rho * rho_multiplier
    else:
        rho = init_rho
        ret = _check(rho)

        if ret == "safe":
            while ret == "safe":
                rho *= rho_multiplier
                ret = _check(rho)
            rho_l = rho / rho_multiplier
            rho_u = rho
        else:
            shrink_count = 0
            while ret != "safe":
                rho /= rho_multiplier
                shrink_count += 1
                if rho < 1e-10 or shrink_count >= max_shrink_iters:
                    print(f"[bisect] Could not find any safe rho after {shrink_count} shrink iterations.")
                    return 0.0, 0.0
                ret = _check(rho)
            rho_l = rho
            rho_u = rho * rho_multiplier

    print(f"[bisect] Bracket: [{rho_l:.6f}, {rho_u:.6f}] | "
          f"width={rho_u - rho_l:.6f} | rho_eps={rho_eps}")
    n_iter = 0
    while rho_u - rho_l > rho_eps:
        if max_iters is not None and n_iter >= max_iters:
            print(f"[bisect] Reached max_iters={max_iters}, stopping.")
            break
        rho_m = (rho_l + rho_u) / 2
        print(f"[bisect] iter {n_iter+1}: rho_l={rho_l:.6f}, rho_u={rho_u:.6f}, rho_m={rho_m:.6f}")
        if _check(rho_m) == "safe":
            rho_l = rho_m
        else:
            rho_u = rho_m
        n_iter += 1

    elapsed = time.time() - t0
    print(f"[bisect] Done: rho_l={rho_l:.6f} | {total_checks} checks | {elapsed:.1f}s total")
    return rho_l, rho_u


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec_prefix",
        type=str,
        default="specs/bisect",
        help="Filename prefix for the specs.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Folder for the output.",
    )
    parser.add_argument(
        "-l",
        "--lower_limit",
        type=float,
        nargs="+",
        help="Lower limit of state dimension. A list of state_dim numbers.",
    )
    parser.add_argument(
        "-u",
        "--upper_limit",
        type=float,
        nargs="+",
        help="Upper limit of state dimension. A list of state_dim numbers.",
    )
    parser.add_argument(
        "-o",
        "--hole_size",
        type=float,
        default=0.001,
        help="Relative size of the hole in the middle to skip verification (0.0 - 1.0).",
    )
    parser.add_argument(
        "--init_rho",
        type=float,
        default=None,
        required=True,
        help="Initial rho value."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="Configuration file for verification."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=200,
        help="Timeout for running verification and attack."
    )
    parser.add_argument(
        "--rho_eps",
        type=float,
        default=0.001,
        help="Precision of the rho bisection."
    )
    parser.add_argument(
        "--rho_multiplier",
        type=float,
        default=1.2,
        help="Multiplier for enlarging rho."
    )
    parser.add_argument(
        "--check_x_next_only",
        action="store_true",
        help="Only check the x_next condition but not the dV condition."
    )
    # --- New args for dissipativity / rho-in-config mode ---
    parser.add_argument(
        "--rho_in_config",
        action="store_true",
        help="Vary rho in the ABCROWN config file (model factory) "
             "instead of the VNNLIB level-set constraint.",
    )
    parser.add_argument(
        "--rho_config_key",
        type=str,
        default="__RHO__",
        help="Placeholder string in the config to replace with the rho value.",
    )
    parser.add_argument(
        "--extra_input_lower",
        type=float,
        nargs="*",
        default=[],
        help="Lower bounds for extra (non-state) input dimensions (e.g., disturbance).",
    )
    parser.add_argument(
        "--extra_input_upper",
        type=float,
        nargs="*",
        default=[],
        help="Upper bounds for extra (non-state) input dimensions.",
    )
    parser.add_argument(
        "--smart_bracket",
        action="store_true",
        help="Skip bracket-finding; use init_rho as bracket midpoint.",
    )
    parser.add_argument(
        "--max_bisect_iters",
        type=int,
        default=None,
        help="Maximum number of bisection iterations (default: unlimited).",
    )
    args, additional_args = parser.parse_known_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # For rho_in_config mode, generate VNNLIB specs once upfront
    if args.rho_in_config:
        command = (
            "python -m neural_lyapunov_training.generate_vnnlib "
            f"--lower_limit {' '.join(map(str, args.lower_limit))} "
            f"--upper_limit {' '.join(map(str, args.upper_limit))} "
            f"--hole_size {args.hole_size} "
            f"--value_levelset 0 --no_check_x_next "
        )
        if args.extra_input_lower:
            command += f"--extra_input_lower {' '.join(map(str, args.extra_input_lower))} "
            command += f"--extra_input_upper {' '.join(map(str, args.extra_input_upper))} "
        command += f"-- {args.spec_prefix}"
        print(f"[bisect] Generating fixed VNNLIB specs...")
        os.system(command)

    check_fn = lambda rho: check_rho(rho, args, additional_args)
    rho_l, rho_u = run_bisect(
        check_fn, args.init_rho,
        rho_eps=args.rho_eps,
        rho_multiplier=args.rho_multiplier,
        smart_bracket=args.smart_bracket,
        max_iters=args.max_bisect_iters,
    )
    print(f"Final: rho_l={rho_l}, rho_u={rho_u}")
    print(f"rho_u={rho_u}")
