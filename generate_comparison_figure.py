#!/usr/bin/env python3
"""
generate_comparison_figure.py  –  Publication-quality ROA comparison figure.

Produces ONE figure with TWO subplots comparing four certified regions
projected onto plant state (θ, θ̇):
  (a) ℓ₂-gain bound experiment
  (b) Sector-bound uncertainty

Each subplot shows:
  1. LMI / SDP baseline
  2. Before Training
  3. Fixed Controller (Alg. 1)
  4. Trained Controller (Alg. 1)

Generates 4 output files (8 with PDF copies):
  *_horiz.{png,pdf}       – side-by-side (1×2), no flow
  *_vert.{png,pdf}        – stacked (2×1), no flow
  *_horiz_flow.{png,pdf}  – side-by-side (1×2), with flow field
  *_vert_flow.{png,pdf}   – stacked (2×1), with flow field

Usage:
    python generate_comparison_figure.py \\
        --l2_trainable_dir  output/pendulum_state/00-01-51_final_results_pendulum \\
        --l2_fixed_dir      output/pendulum_state/09-29-33_final_results_no_train \\
        --unc_fixed_dir     output/pendulum_state/2026-03-29/14-59-23_fixed_contoller_uncertainty \\
        --unc_trainable_dir output/pendulum_state/2026-03-30/06-59-46_trainable_controller_uncertainty \\
        --unc_lmi_rho 0.0458 \\
        --output tex/figures/roa_comparison
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "alpha-beta-CROWN"))
sys.path.insert(0, os.path.join(ROOT, "alpha-beta-CROWN", "complete_verifier"))

from generate_paper_figure import (
    rebuild_system,
    build_lmi_quadratic_lyapunov,
    build_init_lyapunov,
    compute_projections,
    compute_flow_field,
    PanelSpec,
    ROARegion,
)

# ===========================================================================
# Style
# ===========================================================================
FS_LABEL  = 11
FS_TICK   = 9
FS_LEGEND = 9

C_LMI       = "#6B4E71"   # muted purple
C_BEFORE    = "#1B5EA6"   # dark blue
C_FIXED     = "#2CA02C"   # green
C_TRAINABLE = "#E8712A"   # dark orange
C_FLOW      = "#C0C0C0"   # light gray

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "cm",
    "axes.labelsize":     FS_LABEL,
    "axes.titlesize":     FS_LABEL,
    "xtick.labelsize":    FS_TICK,
    "ytick.labelsize":    FS_TICK,
    "legend.fontsize":    FS_LEGEND,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid":          True,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "grid.color":         "#CCCCCC",
    "grid.linestyle":     ":",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.8,
})

# Region specs: (label, color, linestyle, linewidth, fill_alpha)
REGION_STYLES = {
    "lmi":       ("LMI",                C_LMI,       ":",  2.0, 0.22),
    "before":    ("Before Training",    C_BEFORE,    "-",  2.0, 0.15),
    "fixed":     ("Fixed Controller",   C_FIXED,     "-",  2.2, 0.18),
    "trainable": ("Trained Controller", C_TRAINABLE, "-",  2.5, 0.20),
}


# ===========================================================================
# Region builders
# ===========================================================================
def make_region(key: str, rho: float, lyap_fn) -> ROARegion:
    label, color, ls, lw, fa = REGION_STYLES[key]
    return ROARegion(
        label=label, rho=rho, color=color, lyap_fn=lyap_fn,
        linestyle=ls, linewidth=lw, fill_alpha=fa,
        show_inner_contours=False,
    )


def build_subplot_regions(
    fixed_system: dict,
    trainable_system: dict,
    fixed_run_dir: str,
    trainable_run_dir: str,
    fixed_results: dict,
    trainable_results: dict,
    lmi_rho: float,
    device: str,
    use_paper_init: bool = False,
) -> List[ROARegion]:
    """Build the 4 regions for one subplot."""
    regions: List[ROARegion] = []

    # 1. LMI
    try:
        s_lmi = dict(fixed_system)
        s_lmi["lmi_rho"] = lmi_rho
        lmi_lya, rho_lmi = build_lmi_quadratic_lyapunov(s_lmi, device)
        regions.append(make_region("lmi", rho_lmi, lmi_lya))
    except Exception as e:
        print(f"  Warning: LMI baseline skipped: {e}")

    # 2. Before Training — use initial quadratic checkpoint
    lya0 = build_init_lyapunov(fixed_system, fixed_run_dir, device)
    if lya0 is not None:
        if use_paper_init:
            init_rho = float(trainable_results.get(
                "paper_init_crown_rho",
                fixed_results.get("init_crown_rho", 0)))
        else:
            init_rho = float(fixed_results.get("init_crown_rho", 0))
        if init_rho > 0:
            regions.append(make_region("before", init_rho, lya0))

    # 3. Fixed Controller after training
    rho_fixed = float(fixed_results.get("final_crown_rho", 0))
    if rho_fixed > 0:
        regions.append(make_region("fixed", rho_fixed,
                                   fixed_system["lyapunov_nn"]))

    # 4. Train Both after training
    rho_train = float(trainable_results.get("final_crown_rho", 0))
    if rho_train > 0:
        regions.append(make_region("trainable", rho_train,
                                   trainable_system["lyapunov_nn"]))

    return regions


# ===========================================================================
# Flow field (reuse from generate_paper_figure)
# ===========================================================================
def compute_flow_for_panel(
    system: dict, panel: PanelSpec,
    n_flow: int = 28, device: str = "cuda",
):
    return compute_flow_field(system, panel, n_flow=n_flow, device=device)


# ===========================================================================
# Determine common axis limits
# ===========================================================================
def auto_axis_limits(
    all_regions: List[List[ROARegion]],
    margin: float = 1.12,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Find tight axis limits that cover all regions across both subplots."""
    x_lo, x_hi = 1e6, -1e6
    y_lo, y_hi = 1e6, -1e6

    for regions in all_regions:
        for r in regions:
            if not r.panel_data:
                continue
            xi, xj, V = r.panel_data[0]  # first (only) panel
            mask = V <= r.rho
            if not mask.any():
                continue
            XI, XJ = np.meshgrid(xi, xj)
            x_lo = min(x_lo, float(XI[mask].min()))
            x_hi = max(x_hi, float(XI[mask].max()))
            y_lo = min(y_lo, float(XJ[mask].min()))
            y_hi = max(y_hi, float(XJ[mask].max()))

    # symmetric limits centered on 0
    xm = max(abs(x_lo), abs(x_hi)) * margin
    ym = max(abs(y_lo), abs(y_hi)) * margin
    return (-xm, xm), (-ym, ym)


# ===========================================================================
# Drawing
# ===========================================================================
def draw_subplot(
    ax: plt.Axes,
    regions: List[ROARegion],
    title: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    flow_data=None,
    show_flow: bool = False,
):
    ax.set_facecolor("white")

    # Flow field
    if show_flow and flow_data is not None:
        XI, XJ, DI, DJ = flow_data
        speed = np.sqrt(DI**2 + DJ**2)
        lw = 0.3 + 0.7 * speed / (speed.max() + 1e-10)
        ax.streamplot(XI, XJ, DI, DJ, color=C_FLOW, linewidth=lw,
                      density=1.8, arrowsize=0.55, arrowstyle="-|>", zorder=1)

    # ROA regions (back to front: LMI first, Train Both last)
    for region in regions:
        if not region.panel_data:
            continue
        xi, xj, V = region.panel_data[0]

        ax.contourf(xi, xj, V, levels=[-1e12, region.rho],
                    colors=[region.color], alpha=region.fill_alpha, zorder=3)
        ax.contour(xi, xj, V, levels=[region.rho],
                   colors=[region.color], linewidths=region.linewidth,
                   linestyles=region.linestyle, zorder=8)

    # Equilibrium
    ax.plot(0.0, 0.0, "ko", markersize=4, zorder=11)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot\theta$ (rad/s)")
    ax.set_title(title, pad=6)
    ax.tick_params(direction="in", which="both")
    ax.set_aspect("auto")


def render_figure(
    l2_regions: List[ROARegion],
    unc_regions: List[ROARegion],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: str,
    layout: str = "horiz",
    show_flow: bool = False,
    flow_data=None,
):
    if layout == "horiz":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    else:  # vert
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 6.0))
    fig.patch.set_facecolor("white")

    draw_subplot(ax1, l2_regions, r"(a) $\ell_2$-gain bound", xlim, ylim,
                 flow_data=flow_data, show_flow=show_flow)
    draw_subplot(ax2, unc_regions, r"(b) Sector-bound uncertainty", xlim, ylim,
                 flow_data=flow_data, show_flow=show_flow)

    # Common legend
    handles = []
    for key in ["lmi", "before", "fixed", "trainable"]:
        label, color, ls, lw, _ = REGION_STYLES[key]
        handles.append(Line2D([0], [0], color=color, linewidth=lw,
                               linestyle=ls, label=label))

    if layout == "horiz":
        fig.legend(
            handles=handles, loc="lower center", ncol=4,
            frameon=True, fancybox=False, edgecolor="#888888",
            fontsize=FS_LEGEND,
            bbox_to_anchor=(0.5, -0.01),
            columnspacing=0.9, handletextpad=0.4, borderpad=0.35,
        )
        fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=2.5)
    else:
        fig.legend(
            handles=handles, loc="lower center", ncol=4,
            frameon=True, fancybox=False, edgecolor="#888888",
            fontsize=FS_LEGEND,
            bbox_to_anchor=(0.5, -0.01),
            columnspacing=0.7, handletextpad=0.3, borderpad=0.35,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 1], h_pad=2.0)

    for ext in (".png", ".pdf"):
        p = output_path.replace(".png", ext)
        fig.savefig(p, facecolor="white", edgecolor="none")
        print(f"  Saved: {p}")
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Generate ROA comparison figure (L2-gain vs. uncertainty).")
    ap.add_argument("--l2_trainable_dir", required=True)
    ap.add_argument("--l2_fixed_dir", required=True)
    ap.add_argument("--unc_fixed_dir", required=True)
    ap.add_argument("--unc_trainable_dir", required=True)
    ap.add_argument("--unc_lmi_rho", type=float, default=0.0458)
    ap.add_argument("--output", default="tex/figures/roa_comparison",
                    help="Output basename (without extension)")
    ap.add_argument("--grid_sz", type=int, default=300)
    ap.add_argument("--n_flow", type=int, default=28)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # Absolute paths
    dirs = {
        "l2_train": os.path.abspath(args.l2_trainable_dir),
        "l2_fixed": os.path.abspath(args.l2_fixed_dir),
        "unc_fixed": os.path.abspath(args.unc_fixed_dir),
        "unc_train": os.path.abspath(args.unc_trainable_dir),
    }

    # Load results
    results = {}
    for key, d in dirs.items():
        rpath = os.path.join(d, "results.json")
        with open(rpath) as f:
            results[key] = json.load(f)
        print(f"  [{key}] crown_rho={results[key].get('final_crown_rho', 'N/A'):.4f}  "
              f"area={results[key].get('roa_area_final_crown', 'N/A'):.1f}")

    # Rebuild systems
    print("\nRebuilding systems ...")
    systems = {}
    for key, d in dirs.items():
        print(f"  {key} ...")
        systems[key] = rebuild_system(d, device=args.device)

    # L2 gain LMI rho from results
    l2_lmi_rho = float(results["l2_train"].get("lmi_rho", 0.00845))

    # Build regions for each subplot
    print("\nBuilding L2-gain regions ...")
    l2_regions = build_subplot_regions(
        systems["l2_fixed"], systems["l2_train"],
        dirs["l2_fixed"], dirs["l2_train"],
        results["l2_fixed"], results["l2_train"],
        lmi_rho=l2_lmi_rho, device=args.device,
        use_paper_init=True,
    )
    for r in l2_regions:
        print(f"    {r.label}: rho={r.rho:.4f}")

    print("\nBuilding uncertainty regions ...")
    unc_regions = build_subplot_regions(
        systems["unc_fixed"], systems["unc_train"],
        dirs["unc_fixed"], dirs["unc_train"],
        results["unc_fixed"], results["unc_train"],
        lmi_rho=args.unc_lmi_rho, device=args.device,
        use_paper_init=False,
    )
    for r in unc_regions:
        print(f"    {r.label}: rho={r.rho:.4f}")

    # Compute projections (plant state only: dims 0, 1)
    panel = PanelSpec(
        dim_i=0, dim_j=1,
        xlabel=r"$\theta$ (rad)", ylabel=r"$\dot\theta$ (rad/s)",
        title="", xlim=(-3.3, 3.3), ylim=(-9.9, 9.9),
    )

    # Bounds for projection optimization
    def get_limits(system: dict, device: str):
        lim = system["limit"]
        clim = system["controller_limit"]
        lo = torch.tensor([-lim[0], -lim[1]] + [-c for c in clim],
                          device=device, dtype=torch.float32)
        return lo, -lo

    print("\nComputing L2-gain projections ...")
    lo, hi = get_limits(systems["l2_fixed"], args.device)
    compute_projections(l2_regions, [panel], lo, hi,
                        grid_sz=args.grid_sz, n_opt_steps=300)

    print("\nComputing uncertainty projections ...")
    lo, hi = get_limits(systems["unc_fixed"], args.device)
    compute_projections(unc_regions, [panel], lo, hi,
                        grid_sz=args.grid_sz, n_opt_steps=300)

    # Axis limits
    xlim, ylim = auto_axis_limits([l2_regions, unc_regions], margin=1.15)
    print(f"\nAxis limits: x={xlim}, y={ylim}")

    # Flow field (using fixed controller run — same as initial controller)
    print("\nComputing flow field ...")
    flow_panel = PanelSpec(
        dim_i=0, dim_j=1,
        xlabel="", ylabel="", title="",
        xlim=xlim, ylim=ylim,
    )
    flow_data = compute_flow_for_panel(
        systems["l2_fixed"], flow_panel,
        n_flow=args.n_flow, device=args.device,
    )

    # Render 4 variants
    base = args.output.rstrip("/")
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    for layout in ("horiz", "vert"):
        for show_flow in (False, True):
            suffix = f"_{layout}" + ("_flow" if show_flow else "")
            path = f"{base}{suffix}.png"
            print(f"\nRendering {os.path.basename(path)} ...")
            render_figure(
                l2_regions, unc_regions,
                xlim, ylim, path,
                layout=layout,
                show_flow=show_flow,
                flow_data=flow_data,
            )

    # Copy to tex/figures/
    tex_fig_dir = os.path.join(ROOT, "tex", "figures")
    if os.path.isdir(os.path.join(ROOT, "tex")):
        os.makedirs(tex_fig_dir, exist_ok=True)
        for layout in ("horiz", "vert"):
            for show_flow in (False, True):
                suffix = f"_{layout}" + ("_flow" if show_flow else "")
                for ext in (".png", ".pdf"):
                    src = f"{base}{suffix}{ext}"
                    if os.path.exists(src):
                        dst = os.path.join(tex_fig_dir, os.path.basename(src))
                        if os.path.abspath(src) != os.path.abspath(dst):
                            shutil.copy(src, dst)
                            print(f"  Copied -> {dst}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for subplot_name, regions in [("L2-gain", l2_regions),
                                   ("Uncertainty", unc_regions)]:
        print(f"\n{subplot_name}:")
        for r in regions:
            if r.panel_data:
                xi, xj, V = r.panel_data[0]
                dx = float(xi[1] - xi[0])
                dy = float(xj[1] - xj[0])
                area = float((V <= r.rho).sum()) * dx * dy
                print(f"  {r.label:20s}  rho={r.rho:.4f}  area={area:.2f}")
            else:
                print(f"  {r.label:20s}  rho={r.rho:.4f}  (no projection)")


if __name__ == "__main__":
    main()
