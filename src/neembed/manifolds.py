"""Manifold integration points for neembed."""

from __future__ import annotations

import math

import geoopt
import torch


def get_manifold(
    name: str,
    curvature: float = 1.0,
) -> geoopt.PoincareBall | geoopt.Lorentz:
    """Return a supported Geoopt manifold with consistent curvature semantics."""
    if curvature <= 0 or not math.isfinite(curvature):
        raise ValueError("curvature must be positive and finite")

    if name == "poincare":
        return geoopt.PoincareBall(c=curvature)
    if name == "lorentz":
        # Geoopt Lorentz ``k`` is the squared hyperboloid radius. A hyperboloid
        # of radius sqrt(k) has sectional curvature -1/k, so public curvature
        # magnitude ``c`` maps to k = 1/c. Geoopt strongly recommends double
        # precision for Lorentz geometry because Minkowski operations can be
        # numerically unstable in float32.
        k = torch.tensor(1.0 / curvature, dtype=torch.float64)
        return geoopt.Lorentz(k=k)

    raise ValueError(f"Unsupported manifold: {name}")
