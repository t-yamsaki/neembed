"""Manifold integration points for neembed."""

import geoopt


def get_manifold(name: str, curvature: float = 1.0) -> geoopt.PoincareBall:
    """Return the Geoopt manifold used by the minimal neembed v0.1 API."""
    if name != "poincare":
        raise ValueError(f"Unsupported manifold: {name}")

    return geoopt.PoincareBall(c=curvature)
