#!/usr/bin/env python3
from __future__ import annotations

"""
Minimal loader for the 4th-order polynomial fit to the NAFF h0 branch.

The coefficients were obtained by fitting directly to the measured h0 point cloud
from:
    /Users/lisepauwels/phd/data/sps-measurements/results_chroma/
    analysis_chroma_harmonics/tables/tune_sweep_map_by_harmonic.csv

Convention:
    Q(delta) = a0 + a1*delta + a2*delta**2 + a3*delta**3 + a4*delta**4
"""

from math import factorial
from typing import Iterable

import numpy as np


# Ascending powers of delta
QX_COEFF = np.array(
    [
        20.135535824977346,
        1.3522157005522508,
        -501.5724668554806,
        -98516.12238238202,
        40325329.202421896,
    ],
    dtype=float,
)

QY_COEFF = np.array(
    [
        20.178981746444332,
        1.4191477402189625,
        -886.8119292050295,
        38144.506688347676,
        43524952.068756945,
    ],
    dtype=float,
)


def _eval_poly(coeff: np.ndarray, delta):
    delta = np.asarray(delta, dtype=float)
    out = np.zeros_like(delta, dtype=float)
    for power, c in enumerate(coeff):
        out += c * delta**power
    return float(out) if out.ndim == 0 else out


def _eval_poly_derivative(coeff: np.ndarray, order: int, delta):
    if order < 0:
        raise ValueError("Derivative order must be non-negative.")
    delta = np.asarray(delta, dtype=float)
    out = np.zeros_like(delta, dtype=float)
    for power, c in enumerate(coeff):
        if power < order:
            continue
        prefactor = factorial(power) / factorial(power - order)
        out += c * prefactor * delta ** (power - order)
    return float(out) if out.ndim == 0 else out


def qx_poly(delta):
    return _eval_poly(QX_COEFF, delta)


def qy_poly(delta):
    return _eval_poly(QY_COEFF, delta)


def qx_poly_derivative(order: int, delta):
    return _eval_poly_derivative(QX_COEFF, order, delta)


def qy_poly_derivative(order: int, delta):
    return _eval_poly_derivative(QY_COEFF, order, delta)


def qx_qy_poly(delta):
    return qx_poly(delta), qy_poly(delta)


def qx_qy_poly_derivative(order: int, delta):
    return qx_poly_derivative(order, delta), qy_poly_derivative(order, delta)


if __name__ == "__main__":
    delta0 = 0.0
    qx0, qy0 = qx_qy_poly(delta0)
    dqx0, dqy0 = qx_qy_poly_derivative(1, delta0)
    print(f"delta={delta0:+.6e}")
    print(f"Qx={qx0}")
    print(f"Qy={qy0}")
    print(f"dQx/ddelta={dqx0}")
    print(f"dQy/ddelta={dqy0}")
