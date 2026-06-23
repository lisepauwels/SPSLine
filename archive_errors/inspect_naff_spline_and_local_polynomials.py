#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-codex"))

import matplotlib.pyplot as plt
import numpy as np


MOMENTUM_ACCEPTANCE_REPO = Path("/Users/lisepauwels/phd/code/sps-momentum-acceptance")
sys.path.insert(0, str(MOMENTUM_ACCEPTANCE_REPO / "helper_functions"))

from tune_diagram import TuneMap


DEFAULT_MAP = Path(
    "/Users/lisepauwels/phd/data/sps-measurements/results_chroma/"
    "analysis_chroma_harmonics/tune_maps/npz/"
    "measured_tune_map_h0_fit_on_all_points.npz"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "naff_spline_inspection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a saved NAFF TuneMap, evaluate spline derivatives, and "
            "compare local polynomial approximations of orders 4, 5, 6, 7 to the spline."
        )
    )
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--delta0", type=float, default=0.0, help="Expansion / inspection point.")
    parser.add_argument("--orders", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument(
        "--window-half-width",
        type=float,
        default=1.5e-3,
        help="Half-width of the local fitting window around delta0.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        default=400,
        help="Number of evaluation points in the local comparison window.",
    )
    parser.add_argument(
        "--derivative-order",
        type=int,
        default=1,
        help="Spline derivative order to print/evaluate. Meaningful only up to 3 for a cubic spline.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=OUTPUT_DIR / "naff_h0_spline_derivatives_and_local_polyfits.json",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=OUTPUT_DIR / "naff_h0_local_polynomial_orders_4_5_6_7_vs_spline.png",
    )
    return parser.parse_args()


def eval_spline_derivative(tm: TuneMap, plane: str, order: int, delta):
    if order < 0:
        raise ValueError("Derivative order must be non-negative.")
    cs = tm._cs_qx if plane.upper() == "QX" else tm._cs_qy
    if order == 0:
        return cs(delta)
    if order > 3:
        raise ValueError(
            "TuneMap uses a cubic spline. Derivatives above 3rd order are not meaningful "
            "for this representation."
        )
    return cs.derivative(order)(delta)


def local_polyfit_against_spline(
    tm: TuneMap,
    plane: str,
    delta0: float,
    degree: int,
    window_half_width: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    delta_min = max(tm.delta_min, delta0 - window_half_width)
    delta_max = min(tm.delta_max, delta0 + window_half_width)
    delta_fit = np.linspace(delta_min, delta_max, n_grid)
    values = eval_spline_derivative(tm, plane, 0, delta_fit)
    coeff_desc = np.polyfit(delta_fit - delta0, values, deg=degree)
    poly = np.poly1d(coeff_desc)
    values_poly = poly(delta_fit - delta0)
    return delta_fit, values, values_poly, coeff_desc


def ascending_coeffs(coeff_desc: np.ndarray) -> list[float]:
    return [float(x) for x in coeff_desc[::-1]]


def main() -> None:
    args = parse_args()
    tm = TuneMap.load(str(args.map))

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.plot_output.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, object] = {
        "source_map": str(args.map),
        "delta0": float(args.delta0),
        "map_delta_min": float(tm.delta_min),
        "map_delta_max": float(tm.delta_max),
        "requested_derivative_order": int(args.derivative_order),
        "spline_representation_note": (
            "TuneMap uses a cubic spline. Derivatives above 3rd order are not meaningful "
            "for the spline itself."
        ),
        "spline_values": {},
        "local_polynomial_fits": {},
    }

    for plane in ["QX", "QY"]:
        plane_key = plane.lower()
        payload["spline_values"][plane_key] = {
            "value": float(eval_spline_derivative(tm, plane, 0, args.delta0)),
        }
        for order in [1, 2, 3]:
            payload["spline_values"][plane_key][f"derivative_order_{order}"] = float(
                eval_spline_derivative(tm, plane, order, args.delta0)
            )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True, constrained_layout=True)

    for ax, plane in zip(axes, ["QX", "QY"]):
        plane_key = plane.lower()
        delta_fit, values_spline, _, _ = local_polyfit_against_spline(
            tm,
            plane,
            args.delta0,
            degree=max(args.orders),
            window_half_width=args.window_half_width,
            n_grid=args.n_grid,
        )
        ax.plot(delta_fit, values_spline, color="black", lw=2.2, label="Spline")

        payload["local_polynomial_fits"][plane_key] = {}
        for degree in args.orders:
            delta_fit, values_spline, values_poly, coeff_desc = local_polyfit_against_spline(
                tm,
                plane,
                args.delta0,
                degree=degree,
                window_half_width=args.window_half_width,
                n_grid=args.n_grid,
            )
            rmse = float(np.sqrt(np.mean((values_poly - values_spline) ** 2)))
            payload["local_polynomial_fits"][plane_key][f"degree_{degree}"] = {
                "coefficient_order": "ascending_in_power_of_(delta-delta0)",
                "coefficients": ascending_coeffs(coeff_desc),
                "rmse_vs_spline": rmse,
            }
            ax.plot(delta_fit, values_poly, "--", lw=1.8, label=f"Local poly deg {degree}")

        ax.axvline(args.delta0, color="0.5", lw=1.0, ls=":")
        ax.set_title(f"{plane}(delta)")
        ax.set_xlabel("delta = dp/p")
        ax.set_ylabel(plane)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Local polynomial orders {', '.join(str(o) for o in args.orders)} vs spline "
        f"around delta0={args.delta0:+.3e}"
    )
    fig.savefig(args.plot_output, dpi=220)
    args.json_output.write_text(json.dumps(payload, indent=2))

    print(f"Saved inspection JSON to {args.json_output}")
    print(f"Saved comparison plot to {args.plot_output}")
    print("Spline derivatives at delta0:")
    print(json.dumps(payload["spline_values"], indent=2))

    if args.derivative_order > 3:
        print(
            f"Requested derivative order {args.derivative_order}, but the stored TuneMap is cubic. "
            "Use the local polynomial fits in the JSON for orders 4+."
        )
    else:
        print(
            f"Requested derivative order {args.derivative_order}: "
            f"Qx={eval_spline_derivative(tm, 'QX', args.derivative_order, args.delta0)} "
            f"Qy={eval_spline_derivative(tm, 'QY', args.derivative_order, args.delta0)}"
        )


if __name__ == "__main__":
    main()
