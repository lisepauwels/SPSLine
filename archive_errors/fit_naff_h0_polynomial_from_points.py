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

from tune_diagram import TuneMap


DEFAULT_POINT_TABLE = Path(
    "/Users/lisepauwels/phd/data/sps-measurements/results_chroma/"
    "analysis_chroma_harmonics/tables/tune_sweep_map_by_harmonic.csv"
)
DEFAULT_SPLINE_MAP = Path(
    "/Users/lisepauwels/phd/data/sps-measurements/results_chroma/"
    "analysis_chroma_harmonics/tune_maps/npz/"
    "measured_tune_map_h0_fit_on_all_points.npz"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "naff_polynomial_fits"
DEFAULT_JSON = OUTPUT_DIR / "naff_h0_polynomial_fit_from_points_deg4.json"
DEFAULT_PLOT = OUTPUT_DIR / "naff_h0_polynomial_fit_from_points_deg4.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a 4th-order polynomial directly to the NAFF h0 point cloud and compare to the spline map."
    )
    parser.add_argument("--point-table", type=Path, default=DEFAULT_POINT_TABLE)
    parser.add_argument("--spline-map", type=Path, default=DEFAULT_SPLINE_MAP)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--degrees", type=int, nargs="+", default=None)
    parser.add_argument("--harmonic-index", type=int, default=0)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument(
        "--combined-json-output",
        type=Path,
        default=OUTPUT_DIR / "naff_h0_polynomial_fit_from_points_deg4_to_deg7.json",
    )
    parser.add_argument(
        "--combined-plot-output",
        type=Path,
        default=OUTPUT_DIR / "naff_h0_polynomial_fit_from_points_deg4_to_deg7.png",
    )
    parser.add_argument("--n-grid", type=int, default=400)
    return parser.parse_args()


def coeffs_descending_to_ascending(coeffs: np.ndarray) -> list[float]:
    return [float(x) for x in coeffs[::-1]]


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.point_table)
    branch = df[df["harmonic_index"] == args.harmonic_index].sort_values(["delta", "repetition"])
    if branch.empty:
        raise ValueError(f"No rows found for harmonic_index={args.harmonic_index} in {args.point_table}")

    delta_points = branch["delta"].to_numpy(dtype=float)
    qx_points = branch["qx_full"].to_numpy(dtype=float)
    qy_points = branch["qy_full"].to_numpy(dtype=float)

    tm = TuneMap.load(str(args.spline_map))
    delta_grid = np.linspace(tm.delta_min, tm.delta_max, args.n_grid)
    qx_spline, qy_spline = tm(delta_grid)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)
    degrees = args.degrees if args.degrees is not None else [args.degree]
    fit_payloads = []
    curves = []

    for degree in degrees:
        qx_coeff_desc = np.polyfit(delta_points, qx_points, deg=degree)
        qy_coeff_desc = np.polyfit(delta_points, qy_points, deg=degree)
        qx_poly = np.poly1d(qx_coeff_desc)
        qy_poly = np.poly1d(qy_coeff_desc)
        qx_poly_grid = qx_poly(delta_grid)
        qy_poly_grid = qy_poly(delta_grid)
        qx_poly_points = qx_poly(delta_points)
        qy_poly_points = qy_poly(delta_points)

        payload = {
            "source_point_table": str(args.point_table),
            "source_spline_map": str(args.spline_map),
            "harmonic_index": int(args.harmonic_index),
            "degree": int(degree),
            "coefficient_order": "ascending_in_power_of_delta",
            "qx_coefficients": coeffs_descending_to_ascending(qx_coeff_desc),
            "qy_coefficients": coeffs_descending_to_ascending(qy_coeff_desc),
            "fit_range": {
                "delta_min": float(delta_points.min()),
                "delta_max": float(delta_points.max()),
            },
            "qx_rmse_vs_points": float(np.sqrt(np.mean((qx_poly_points - qx_points) ** 2))),
            "qy_rmse_vs_points": float(np.sqrt(np.mean((qy_poly_points - qy_points) ** 2))),
            "qx_rmse_vs_spline": float(np.sqrt(np.mean((qx_poly_grid - qx_spline) ** 2))),
            "qy_rmse_vs_spline": float(np.sqrt(np.mean((qy_poly_grid - qy_spline) ** 2))),
        }
        fit_payloads.append(payload)
        curves.append((degree, qx_poly_grid, qy_poly_grid))

        if len(degrees) == 1:
            args.json_output.write_text(json.dumps(payload, indent=2))

    if len(degrees) > 1:
        args.combined_json_output.parent.mkdir(parents=True, exist_ok=True)
        args.combined_json_output.write_text(json.dumps({"fits": fit_payloads}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, constrained_layout=True)

    axes[0].scatter(delta_points, qx_points, s=14, alpha=0.22, label="NAFF h0 points")
    axes[0].plot(delta_grid, qx_spline, lw=2.2, color="black", label="Spline fit")
    for degree, qx_poly_grid, _ in curves:
        axes[0].plot(delta_grid, qx_poly_grid, "--", lw=1.8, label=f"Polynomial deg {degree}")
    axes[0].set_title("Qx(delta)")
    axes[0].set_ylabel("Qx")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(delta_points, qy_points, s=14, alpha=0.22, label="NAFF h0 points")
    axes[1].plot(delta_grid, qy_spline, lw=2.2, color="black", label="Spline fit")
    for degree, _, qy_poly_grid in curves:
        axes[1].plot(delta_grid, qy_poly_grid, "--", lw=1.8, label=f"Polynomial deg {degree}")
    axes[1].set_title("Qy(delta)")
    axes[1].set_ylabel("Qy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for ax in axes:
        ax.set_xlabel("delta = dp/p")

    if len(degrees) == 1:
        fig.suptitle(f"NAFF h{args.harmonic_index} polynomial fit from points vs spline")
        fig.savefig(args.plot_output, dpi=220)
        print(f"Saved coefficients to {args.json_output}")
        print(f"Saved comparison plot to {args.plot_output}")
        print(f"Qx coefficients (ascending powers): {fit_payloads[0]['qx_coefficients']}")
        print(f"Qy coefficients (ascending powers): {fit_payloads[0]['qy_coefficients']}")
    else:
        fig.suptitle(
            f"NAFF h{args.harmonic_index} polynomial fits from points vs spline "
            f"(degrees {', '.join(str(d) for d in degrees)})"
        )
        fig.savefig(args.combined_plot_output, dpi=220)
        print(f"Saved combined coefficients to {args.combined_json_output}")
        print(f"Saved combined comparison plot to {args.combined_plot_output}")


if __name__ == "__main__":
    main()
