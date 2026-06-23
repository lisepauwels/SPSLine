#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MOMENTUM_ACCEPTANCE_REPO = Path("/Users/lisepauwels/phd/code/sps-momentum-acceptance")
sys.path.insert(0, str(MOMENTUM_ACCEPTANCE_REPO / "helper_functions"))

from naff_quartic_spline import NAFFQuarticSpline
from tune_diagram import TuneMap


DEFAULT_POINT_TABLE = Path(
    "/Users/lisepauwels/phd/data/sps-measurements/results_chroma/"
    "analysis_chroma_harmonics/tables/tune_sweep_map_by_harmonic.csv"
)
DEFAULT_CUBIC_MAP = Path(
    "/Users/lisepauwels/phd/data/sps-measurements/results_chroma/"
    "analysis_chroma_harmonics/tune_maps/npz/"
    "measured_tune_map_h0_fit_on_all_points.npz"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "naff_quartic_spline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a quartic spline from the NAFF h0 point cloud and store derivatives up to 4th order."
    )
    parser.add_argument("--point-table", type=Path, default=DEFAULT_POINT_TABLE)
    parser.add_argument("--cubic-map", type=Path, default=DEFAULT_CUBIC_MAP)
    parser.add_argument("--harmonic-index", type=int, default=0)
    parser.add_argument("--smoothing-scale", type=float, default=1.0)
    parser.add_argument("--delta0", type=float, default=0.0)
    parser.add_argument("--n-grid", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.point_table)
    branch = df[df["harmonic_index"] == args.harmonic_index].sort_values(["delta", "repetition"])
    if branch.empty:
        raise ValueError(f"No rows found for harmonic_index={args.harmonic_index}")

    delta_points = branch["delta"].to_numpy(dtype=float)
    qx_points = branch["qx_full"].to_numpy(dtype=float)
    qy_points = branch["qy_full"].to_numpy(dtype=float)

    quartic = NAFFQuarticSpline.from_points(
        delta_points,
        qx_points,
        qy_points,
        smoothing_scale=args.smoothing_scale,
        k=4,
    )
    quartic.save(args.output_dir / "naff_h0_quartic_spline_fit.npz")

    cubic = TuneMap.load(str(args.cubic_map))
    delta_grid = np.linspace(quartic.delta_min, quartic.delta_max, args.n_grid)
    qx_quartic, qy_quartic = quartic(delta_grid)
    qx_cubic, qy_cubic = cubic(delta_grid, extrapolate=True)
    qx_poly_coeff = np.polyfit(delta_points, qx_points, deg=4)
    qy_poly_coeff = np.polyfit(delta_points, qy_points, deg=4)
    qx_poly = np.poly1d(qx_poly_coeff)(delta_grid)
    qy_poly = np.poly1d(qy_poly_coeff)(delta_grid)

    deriv_payload = {
        "source_point_table": str(args.point_table),
        "source_cubic_map": str(args.cubic_map),
        "harmonic_index": int(args.harmonic_index),
        "delta_min": float(quartic.delta_min),
        "delta_max": float(quartic.delta_max),
        "delta0": float(args.delta0),
        "smoothing_scale": float(args.smoothing_scale),
        "derivatives_at_delta0": {},
    }
    for order in range(0, 5):
        qx_d0, qy_d0 = quartic.derivative(order, args.delta0)
        deriv_payload["derivatives_at_delta0"][f"order_{order}"] = {
            "qx": float(qx_d0),
            "qy": float(qy_d0),
        }

    (args.output_dir / "naff_h0_quartic_spline_derivatives.json").write_text(
        json.dumps(deriv_payload, indent=2)
    )

    sampled = {
        "delta": delta_grid,
        "qx_order_0": qx_quartic,
        "qy_order_0": qy_quartic,
        "qx_cubic_reference": qx_cubic,
        "qy_cubic_reference": qy_cubic,
    }
    for order in range(1, 5):
        qx_d, qy_d = quartic.derivative(order, delta_grid)
        sampled[f"qx_order_{order}"] = qx_d
        sampled[f"qy_order_{order}"] = qy_d
    np.savez(args.output_dir / "naff_h0_quartic_spline_samples_and_derivatives.npz", **sampled)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, constrained_layout=True)
    axes[0].scatter(delta_points, qx_points, s=14, alpha=0.22, label="NAFF h0 points")
    axes[0].plot(delta_grid, qx_cubic, color="black", lw=2.0, label="Stored cubic TuneMap")
    axes[0].plot(delta_grid, qx_quartic, color="C3", lw=2.0, ls="--", label="Quartic spline")
    axes[0].set_title("Qx(delta)")
    axes[0].set_ylabel("Qx")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(delta_points, qy_points, s=14, alpha=0.22, label="NAFF h0 points")
    axes[1].plot(delta_grid, qy_cubic, color="black", lw=2.0, label="Stored cubic TuneMap")
    axes[1].plot(delta_grid, qy_quartic, color="C3", lw=2.0, ls="--", label="Quartic spline")
    axes[1].set_title("Qy(delta)")
    axes[1].set_ylabel("Qy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("delta = dp/p")
    fig.suptitle("NAFF h0 quartic spline vs stored cubic TuneMap")
    fig.savefig(args.output_dir / "naff_h0_quartic_spline_vs_cubic_tunemap.png", dpi=220)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, constrained_layout=True)
    axes2[0].scatter(delta_points, qx_points, s=14, alpha=0.22, label="NAFF h0 points")
    axes2[0].plot(delta_grid, qx_cubic, color="black", lw=2.0, label="Cubic spline TuneMap")
    axes2[0].plot(delta_grid, qx_quartic, color="C3", lw=2.0, ls="--", label="Quartic spline")
    axes2[0].plot(delta_grid, qx_poly, color="C0", lw=1.8, ls="-.", label="Degree-4 polynomial")
    axes2[0].set_title("Qx(delta)")
    axes2[0].set_ylabel("Qx")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend()

    axes2[1].scatter(delta_points, qy_points, s=14, alpha=0.22, label="NAFF h0 points")
    axes2[1].plot(delta_grid, qy_cubic, color="black", lw=2.0, label="Cubic spline TuneMap")
    axes2[1].plot(delta_grid, qy_quartic, color="C3", lw=2.0, ls="--", label="Quartic spline")
    axes2[1].plot(delta_grid, qy_poly, color="C0", lw=1.8, ls="-.", label="Degree-4 polynomial")
    axes2[1].set_title("Qy(delta)")
    axes2[1].set_ylabel("Qy")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend()
    for ax in axes2:
        ax.set_xlabel("delta = dp/p")
    fig2.suptitle("NAFF h0 comparison: cubic spline, quartic spline, and degree-4 polynomial")
    fig2.savefig(args.output_dir / "naff_h0_cubic_vs_quartic_vs_degree4_polynomial.png", dpi=220)

    print(f"Saved quartic spline to {args.output_dir / 'naff_h0_quartic_spline_fit.npz'}")
    print(f"Saved derivative JSON to {args.output_dir / 'naff_h0_quartic_spline_derivatives.json'}")
    print(f"Saved derivative samples to {args.output_dir / 'naff_h0_quartic_spline_samples_and_derivatives.npz'}")


if __name__ == "__main__":
    main()
