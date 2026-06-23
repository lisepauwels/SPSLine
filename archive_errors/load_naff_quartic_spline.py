#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from naff_quartic_spline import NAFFQuarticSpline


DEFAULT_SPLINE = Path(__file__).resolve().parent / "naff_quartic_spline" / "naff_h0_quartic_spline_fit.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basic loader / evaluator for the saved NAFF quartic spline."
    )
    parser.add_argument("--spline", type=Path, default=DEFAULT_SPLINE)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--derivative-order", type=int, default=0)
    parser.add_argument("--extrapolate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spline = NAFFQuarticSpline.load(args.spline)
    if args.derivative_order == 0:
        qx, qy = spline(args.delta, extrapolate=args.extrapolate)
        print(f"delta={args.delta:+.6e}")
        print(f"Qx={qx}")
        print(f"Qy={qy}")
    else:
        qx, qy = spline.derivative(args.derivative_order, args.delta, extrapolate=args.extrapolate)
        print(f"delta={args.delta:+.6e}")
        print(f"d^{args.derivative_order}Qx/delta^{args.derivative_order} = {qx}")
        print(f"d^{args.derivative_order}Qy/delta^{args.derivative_order} = {qy}")


if __name__ == "__main__":
    main()
