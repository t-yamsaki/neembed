"""Cross-geometry regression tests for equivalent hyperbolic models."""

from __future__ import annotations

import itertools
import math

import pytest
import torch

from neembed.manifolds import get_manifold


CURVATURES = (0.25, 1.0, 2.0)
DTYPE_TOLERANCES = (
    (torch.float64, 1e-9, 1e-9),
    (torch.float32, 5e-5, 5e-5),
)


def _manifolds(curvature: float, dtype: torch.dtype):
    """Return matching Poincare and Lorentz manifolds at one public curvature."""
    poincare = get_manifold("poincare", curvature=curvature).to(dtype=dtype)
    lorentz = get_manifold("lorentz", curvature=curvature).to(dtype=dtype)
    return poincare, lorentz


def _poincare_points(curvature: float, dtype: torch.dtype) -> torch.Tensor:
    """Return deterministic moderate-scale points safely inside the ball."""
    radius = 1.0 / math.sqrt(curvature)
    normalized = torch.tensor(
        [
            [0.00, 0.00],
            [0.12, -0.08],
            [-0.18, 0.21],
            [0.25, 0.05],
        ],
        dtype=dtype,
    )
    return normalized * radius


def _poincare_to_lorentz(
    poincare,
    points: torch.Tensor,
) -> torch.Tensor:
    """Convert corresponding points for tests using Geoopt's stereographic map.

    Geoopt ``inv_sproj`` returns spatial coordinates followed by the extra
    hyperboloid coordinate. ``geoopt.Lorentz`` uses the time-like coordinate
    first, so the test-only conversion only reorders that final coordinate.
    """
    projected = poincare.inv_sproj(points)
    return torch.cat((projected[..., -1:], projected[..., :-1]), dim=-1)


@pytest.mark.parametrize("curvature", CURVATURES)
@pytest.mark.parametrize("dtype,atol,rtol", DTYPE_TOLERANCES)
def test_origin_and_radial_distances_match_between_models(
    curvature: float,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    poincare, lorentz = _manifolds(curvature, dtype)
    points = _poincare_points(curvature, dtype)
    lorentz_points = _poincare_to_lorentz(poincare, points)

    poincare_distances = poincare.dist(points[0], points[1:])
    lorentz_distances = lorentz.dist(lorentz_points[0], lorentz_points[1:])

    assert torch.isfinite(poincare_distances).all()
    assert torch.isfinite(lorentz_distances).all()
    assert torch.allclose(
        poincare_distances,
        lorentz_distances,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize("curvature", CURVATURES)
@pytest.mark.parametrize("dtype,atol,rtol", DTYPE_TOLERANCES)
def test_pairwise_distances_match_for_corresponding_nontrivial_points(
    curvature: float,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    poincare, lorentz = _manifolds(curvature, dtype)
    points = _poincare_points(curvature, dtype)
    lorentz_points = _poincare_to_lorentz(poincare, points)

    for left, right in itertools.combinations(range(1, len(points)), 2):
        poincare_distance = poincare.dist(points[left], points[right])
        lorentz_distance = lorentz.dist(lorentz_points[left], lorentz_points[right])

        assert torch.isfinite(poincare_distance)
        assert torch.isfinite(lorentz_distance)
        assert torch.allclose(
            poincare_distance,
            lorentz_distance,
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.parametrize("curvature", CURVATURES)
@pytest.mark.parametrize("dtype,atol,rtol", DTYPE_TOLERANCES)
def test_corresponding_points_satisfy_both_manifold_constraints(
    curvature: float,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    poincare, _ = _manifolds(curvature, dtype)
    points = _poincare_points(curvature, dtype)
    lorentz_points = _poincare_to_lorentz(poincare, points)

    radius = torch.tensor(1.0 / math.sqrt(curvature), dtype=dtype)
    ball_norms = torch.linalg.vector_norm(points, dim=-1)
    quad_form = (
        -lorentz_points[:, 0].square()
        + lorentz_points[:, 1:].square().sum(dim=-1)
    )
    expected_quad_form = torch.full_like(quad_form, -1.0 / curvature)

    assert torch.all(ball_norms < radius)
    assert torch.allclose(
        quad_form,
        expected_quad_form,
        atol=atol,
        rtol=rtol,
    )


def test_geodesic_regression_would_detect_euclidean_distance_fallback() -> None:
    curvature = 1.0
    poincare, lorentz = _manifolds(curvature, torch.float64)
    points = _poincare_points(curvature, torch.float64)
    lorentz_points = _poincare_to_lorentz(poincare, points)

    left, right = 1, 2
    poincare_geodesic = poincare.dist(points[left], points[right])
    lorentz_geodesic = lorentz.dist(lorentz_points[left], lorentz_points[right])
    poincare_euclidean = torch.linalg.vector_norm(points[left] - points[right])
    lorentz_ambient_euclidean = torch.linalg.vector_norm(
        lorentz_points[left] - lorentz_points[right]
    )

    assert torch.allclose(
        poincare_geodesic,
        lorentz_geodesic,
        atol=1e-9,
        rtol=1e-9,
    )
    assert not torch.isclose(
        poincare_geodesic,
        poincare_euclidean,
        atol=1e-3,
        rtol=1e-3,
    )
    assert not torch.isclose(
        lorentz_geodesic,
        lorentz_ambient_euclidean,
        atol=1e-3,
        rtol=1e-3,
    )
