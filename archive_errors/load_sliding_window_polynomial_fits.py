#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial

DEFAULT_JSON_PATH = Path(
    "/Users/lisepauwels/phd/data/sps-measurements/results_chroma/"
    "analysis_chroma_harmonics/sliding_window_tunes/polynomial_fits/"
    "sliding_window_polynomial_fits.json"
)


class SlidingWindowPolynomialFits:
    def __init__(self, json_path: str | Path = DEFAULT_JSON_PATH):
        self.path = Path(json_path)
        self.payload = json.loads(self.path.read_text())
        self._fits = {
            (entry["plane"].upper(), int(entry["harmonic_index"])): entry
            for entry in self.payload["fits"]
        }

    def available_keys(self) -> list[tuple[str, int]]:
        return sorted(self._fits)

    def available_harmonics(self) -> list[int]:
        return sorted({harmonic for _, harmonic in self._fits})

    def entry(self, plane: str, harmonic_index: int) -> dict[str, object]:
        plane = plane.upper()
        if plane not in {"QX", "QY"}:
            raise ValueError("plane must be 'Qx' or 'Qy'.")
        key = (plane, int(harmonic_index))
        if key not in self._fits:
            raise KeyError(f"No fit stored for {plane} h{harmonic_index}.")
        return self._fits[key]

    def degree(self, plane: str, harmonic_index: int) -> int:
        return int(self.entry(plane, harmonic_index)["chosen_degree"])

    def coefficients(self, plane: str, harmonic_index: int) -> np.ndarray:
        coeff = self.entry(plane, harmonic_index)["coefficients_ascending"]
        return np.asarray(coeff, dtype=float)

    def polynomial(self, plane: str, harmonic_index: int) -> Polynomial:
        return Polynomial(self.coefficients(plane, harmonic_index))

    def evaluate(self, plane: str, harmonic_index: int, delta):
        return self.polynomial(plane, harmonic_index)(delta)

    def derivative(self, plane: str, harmonic_index: int, order: int, delta):
        if order < 0:
            raise ValueError("Derivative order must be non-negative.")
        return self.polynomial(plane, harmonic_index).deriv(order)(delta)

    def evaluate_pair(self, harmonic_index: int, delta):
        qx = self.evaluate("Qx", harmonic_index, delta)
        qy = self.evaluate("Qy", harmonic_index, delta)
        return qx, qy

    def derivative_pair(self, harmonic_index: int, order: int, delta):
        dqx = self.derivative("Qx", harmonic_index, order, delta)
        dqy = self.derivative("Qy", harmonic_index, order, delta)
        return dqx, dqy


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH, help="Path to sliding_window_polynomial_fits.json")
    parser.add_argument("--plane", choices=["Qx", "Qy"], help="Plane to evaluate")
    parser.add_argument("--harmonic", type=int, default=0, help="Harmonic index h0..h4")
    parser.add_argument("--delta", type=float, default=0.0, help="Delta value at which to evaluate")
    parser.add_argument("--derivative-order", type=int, default=0, help="Derivative order to evaluate")
    parser.add_argument("--list", action="store_true", help="List available fits and exit")
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()
    family = SlidingWindowPolynomialFits(args.json)

    if args.list:
        print(f"JSON: {family.path}")
        for plane, harmonic in family.available_keys():
            print(f"{plane} h{harmonic}: degree {family.degree(plane, harmonic)}")
        return

    if args.plane is None:
        parser.error("--plane is required unless --list is used.")

    if args.derivative_order == 0:
        value = family.evaluate(args.plane, args.harmonic, args.delta)
        print(float(value))
    else:
        value = family.derivative(args.plane, args.harmonic, args.derivative_order, args.delta)
        print(float(value))


if __name__ == "__main__":
    main()
