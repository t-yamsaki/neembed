"""Tests for the minimal Geoopt-backed manifold integration."""

import geoopt
import pytest
import torch

from neembed.manifolds import get_manifold


def test_get_manifold_constructs_poincare_ball() -> None:
    manifold = get_manifold("poincare", curvature=2.0)

    assert isinstance(manifold, geoopt.PoincareBall)
    assert float(manifold.c) == pytest.approx(2.0)


def test_poincare_ball_exposes_expmap0_and_geodesic_distance() -> None:
    manifold = get_manifold("poincare", curvature=1.0)
    tangent_vectors = torch.tensor(
        [[0.10, 0.20], [0.20, -0.10]],
        dtype=torch.float64,
    )

    points = manifold.expmap0(tangent_vectors)
    distance = manifold.dist(points[0], points[1])

    assert torch.isfinite(points).all()
    assert torch.isfinite(distance)
    assert float(distance) >= 0.0


def test_get_manifold_rejects_unsupported_manifold() -> None:
    with pytest.raises(ValueError, match="Unsupported manifold: sphere"):
        get_manifold("sphere")
