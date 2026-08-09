"""Manifold integration points for neembed."""

import math

import geoopt


def get_manifold(name: str, curvature: float = 1.0) -> geoopt.PoincareBall:
    """Return the Geoopt manifold used by the minimal neembed v0.1 API."""
    if name != "poincare":
        raise ValueError(f"Unsupported manifold: {name}")
    if curvature <= 0 or not math.isfinite(curvature):
        raise ValueError("curvature must be positive and finite")

    return geoopt.PoincareBall(c=curvature)
