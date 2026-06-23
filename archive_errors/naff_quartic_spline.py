#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline, splrep


def _make_strictly_increasing(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    unique, counts = np.unique(x, return_counts=True)
    for value, count in zip(unique, counts):
        if count <= 1:
            continue
        idx = np.where(x == value)[0]
        offsets = np.linspace(-0.5, 0.5, count) * eps
        x[idx] = value + offsets
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Failed to construct a strictly increasing abscissa for spline fitting.")
    return x, order


@dataclass
class NAFFQuarticSpline:
    qx_spline: BSpline
    qy_spline: BSpline
    delta_min: float
    delta_max: float

    @classmethod
    def from_points(
        cls,
        delta: np.ndarray,
        qx: np.ndarray,
        qy: np.ndarray,
        smoothing_scale: float = 1.0,
        k: int = 4,
    ) -> "NAFFQuarticSpline":
        if k != 4:
            raise ValueError("This helper is intended for quartic splines, so k must be 4.")
        delta_sorted, order = _make_strictly_increasing(np.asarray(delta, dtype=float))
        qx_sorted = np.asarray(qx, dtype=float)[order]
        qy_sorted = np.asarray(qy, dtype=float)[order]

        qx_sigma = float(np.std(qx_sorted))
        qy_sigma = float(np.std(qy_sorted))
        s_qx = max(1e-16, smoothing_scale * len(delta_sorted) * qx_sigma * qx_sigma)
        s_qy = max(1e-16, smoothing_scale * len(delta_sorted) * qy_sigma * qy_sigma)

        tck_qx = splrep(delta_sorted, qx_sorted, k=k, s=s_qx)
        tck_qy = splrep(delta_sorted, qy_sorted, k=k, s=s_qy)

        qx_spline = BSpline(*tck_qx, extrapolate=False)
        qy_spline = BSpline(*tck_qy, extrapolate=False)
        return cls(
            qx_spline=qx_spline,
            qy_spline=qy_spline,
            delta_min=float(delta_sorted.min()),
            delta_max=float(delta_sorted.max()),
        )

    def _check_bounds(self, delta, extrapolate: bool) -> np.ndarray:
        scalar = np.isscalar(delta)
        d = np.atleast_1d(np.asarray(delta, dtype=float))
        if not extrapolate:
            oob = (d < self.delta_min) | (d > self.delta_max)
            if oob.any():
                raise ValueError(
                    f"delta values {d[oob]} are outside [{self.delta_min:.4g}, {self.delta_max:.4g}]"
                )
        return d if not scalar else d

    def __call__(self, delta, extrapolate: bool = False):
        scalar = np.isscalar(delta)
        d = np.atleast_1d(np.asarray(delta, dtype=float))
        if not extrapolate:
            oob = (d < self.delta_min) | (d > self.delta_max)
            if oob.any():
                raise ValueError(
                    f"delta values {d[oob]} are outside [{self.delta_min:.4g}, {self.delta_max:.4g}]"
                )
        qx_spl = self.qx_spline if extrapolate else BSpline(self.qx_spline.t, self.qx_spline.c, self.qx_spline.k, extrapolate=False)
        qy_spl = self.qy_spline if extrapolate else BSpline(self.qy_spline.t, self.qy_spline.c, self.qy_spline.k, extrapolate=False)
        qx = qx_spl(d)
        qy = qy_spl(d)
        if scalar:
            return float(qx[0]), float(qy[0])
        return qx, qy

    def derivative(self, order: int, delta, extrapolate: bool = False):
        if order < 0 or order > 4:
            raise ValueError("Quartic spline derivative order must be between 0 and 4.")
        scalar = np.isscalar(delta)
        d = np.atleast_1d(np.asarray(delta, dtype=float))
        if not extrapolate:
            oob = (d < self.delta_min) | (d > self.delta_max)
            if oob.any():
                raise ValueError(
                    f"delta values {d[oob]} are outside [{self.delta_min:.4g}, {self.delta_max:.4g}]"
                )
        qx = self.qx_spline.derivative(order)(d)
        qy = self.qy_spline.derivative(order)(d)
        if scalar:
            return float(qx[0]), float(qy[0])
        return qx, qy

    def sample(self, n: int = 400):
        delta = np.linspace(self.delta_min, self.delta_max, n)
        qx, qy = self(delta)
        return delta, qx, qy

    def derivative_sample(self, order: int, n: int = 400):
        delta = np.linspace(self.delta_min, self.delta_max, n)
        qx, qy = self.derivative(order, delta)
        return delta, qx, qy

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez(
            path,
            qx_t=self.qx_spline.t,
            qx_c=self.qx_spline.c,
            qx_k=np.asarray([self.qx_spline.k], dtype=int),
            qy_t=self.qy_spline.t,
            qy_c=self.qy_spline.c,
            qy_k=np.asarray([self.qy_spline.k], dtype=int),
            delta_min=np.asarray([self.delta_min], dtype=float),
            delta_max=np.asarray([self.delta_max], dtype=float),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NAFFQuarticSpline":
        data = np.load(path)
        qx_spline = BSpline(data["qx_t"], data["qx_c"], int(data["qx_k"][0]), extrapolate=False)
        qy_spline = BSpline(data["qy_t"], data["qy_c"], int(data["qy_k"][0]), extrapolate=False)
        return cls(
            qx_spline=qx_spline,
            qy_spline=qy_spline,
            delta_min=float(data["delta_min"][0]),
            delta_max=float(data["delta_max"][0]),
        )
