#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt

from naff_quartic_spline import NAFFQuarticSpline


THIS_DIR = Path(__file__).resolve().parent
MOMENTUM_ACCEPTANCE_REPO = Path("/Users/lisepauwels/phd/code/sps-momentum-acceptance")

sys.path.insert(0, str(MOMENTUM_ACCEPTANCE_REPO / "helper_functions"))
sys.path.insert(0, str(MOMENTUM_ACCEPTANCE_REPO / "tune_scan_workflow"))

from tune_diagram import TuneMap
from TuneScan import _setup_cavities, install_errors, optimise_tune_chroma


DEFAULT_LINE_PATH = THIS_DIR / "sps_q20_inj.json"
DEFAULT_NAFF_MAP = Path(
    "/Users/lisepauwels/phd/data/sps-measurements/results_chroma/"
    "analysis_chroma_harmonics/tune_maps/npz/"
    "measured_tune_map_h0_fit_on_all_points.npz"
)
DEFAULT_QUARTIC_SPLINE = THIS_DIR / "naff_quartic_spline" / "naff_h0_quartic_spline_fit.npz"
DEFAULT_OUTPUT = THIS_DIR / "naff_h0_vs_twiss_with_errors_qx_qy_vs_delta.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fitted NAFF h0 tune map against an xsuite twiss scan with errors."
    )
    parser.add_argument("--line-path", type=Path, default=DEFAULT_LINE_PATH)
    parser.add_argument("--naff-map", type=Path, default=DEFAULT_NAFF_MAP)
    parser.add_argument("--quartic-spline", type=Path, default=DEFAULT_QUARTIC_SPLINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qx", type=float, default=20.13)
    parser.add_argument("--qy", type=float, default=20.18)
    parser.add_argument("--xi-x", type=float, default=0.154)
    parser.add_argument("--xi-y", type=float, default=0.3)
    parser.add_argument("--delta-min", type=float, default=-3e-3)
    parser.add_argument("--delta-max", type=float, default=3e-3)
    parser.add_argument("--n-points", type=int, default=121)
    parser.add_argument("--error-variant", default="all")
    return parser.parse_args()


def twiss_scan(line: xt.Line, delta_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tw0 = line.twiss4d(delta0=0.0)
    rows: list[tuple[float, float, float]] = [(0.0, float(tw0.qx), float(tw0.qy))]

    co_prev = tw0.particle_on_co
    neg_rows: list[tuple[float, float, float]] = []
    for delta in delta_grid[delta_grid < 0][::-1]:
        tw = line.twiss4d(delta0=float(delta), co_guess=co_prev)
        co_prev = tw.particle_on_co
        neg_rows.append((float(delta), float(tw.qx), float(tw.qy)))

    co_prev = tw0.particle_on_co
    pos_rows: list[tuple[float, float, float]] = []
    for delta in delta_grid[delta_grid > 0]:
        tw = line.twiss4d(delta0=float(delta), co_guess=co_prev)
        co_prev = tw.particle_on_co
        pos_rows.append((float(delta), float(tw.qx), float(tw.qy)))

    rows = neg_rows[::-1] + rows + pos_rows
    d = np.array([r[0] for r in rows], dtype=float)
    qx = np.array([r[1] for r in rows], dtype=float)
    qy = np.array([r[2] for r in rows], dtype=float)
    return d, qx, qy


def main() -> None:
    args = parse_args()

    if args.quartic_spline.exists():
        measured_model = NAFFQuarticSpline.load(args.quartic_spline)
        measured_label = "NAFF h0 quartic spline"
    else:
        measured_model = TuneMap.load(str(args.naff_map))
        measured_label = "NAFF h0 cubic TuneMap"

    line = xt.Line.from_json(args.line_path)
    _setup_cavities(line)
    install_errors(line, args.error_variant)
    optimise_tune_chroma(
        line,
        xi_x=args.xi_x,
        xi_y=args.xi_y,
        qx=args.qx,
        qy=args.qy,
    )

    delta_grid = np.linspace(args.delta_min, args.delta_max, args.n_points)
    qx_naff, qy_naff = measured_model(delta_grid, extrapolate=True)
    d_model, qx_model, qy_model = twiss_scan(line, delta_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, constrained_layout=True)

    axes[0].plot(delta_grid, qx_naff, lw=2.0, label=measured_label)
    axes[0].plot(d_model, qx_model, "--", lw=2.0, label=f"Twiss + errors ({args.error_variant})")
    axes[0].set_title("Qx(delta)")
    axes[0].set_ylabel("Qx")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(delta_grid, qy_naff, lw=2.0, label=measured_label)
    axes[1].plot(d_model, qy_model, "--", lw=2.0, label=f"Twiss + errors ({args.error_variant})")
    axes[1].set_title("Qy(delta)")
    axes[1].set_ylabel("Qy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("delta = dp/p")

    fig.suptitle(
        f"NAFF h0 vs Twiss with errors | Qx={args.qx:.3f}, Qy={args.qy:.3f}, "
        f"xi_x={args.xi_x:.3f}, xi_y={args.xi_y:.3f}"
    )
    fig.savefig(args.output, dpi=220)

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
