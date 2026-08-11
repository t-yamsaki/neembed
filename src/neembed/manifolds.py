"""Manifold integration points for neembed."""

from __future__ import annotations

import math

import geoopt


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
        # magnitude ``c`` maps to k = 1/c.
        return geoopt.Lorentz(k=1.0 / curvature)

    raise ValueError(f"Unsupported manifold: {name}")
