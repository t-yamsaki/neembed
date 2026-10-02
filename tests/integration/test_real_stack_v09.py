"""Opt-in v0.9 constant-curvature acceptance against the real third-party stack."""

from __future__ import annotations

import math
import os

import pytest
import torch

from neembed import ManifoldMultipleNegativesRankingLoss, ManifoldSentenceTransformer


pytestmark = [
    pytest.mark.real_stack,
    pytest.mark.skipif(
        os.environ.get("NEEMBED_REAL_STACK") != "1",
        reason="set NEEMBED_REAL_STACK=1 to run real dependency tests",
    ),
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 8
CURVATURE = 0.5
ANCHORS = ("dog", "cat")
POSITIVES = ("animal", "feline")

GEOMETRY_CASES = (
    (
        "euclidean",
        {"manifold": "euclidean", "sectional_curvature": 0.0},
        0.0,
        torch.float32,
    ),
    (
        "poincare",
        {"manifold": "poincare", "curvature": CURVATURE},
        -CURVATURE,
        torch.float32,
    ),
    (
        "lorentz",
        {"manifold": "lorentz", "curvature": CURVATURE},
        -CURVATURE,
        torch.float64,
    ),
    (
        "sphere_projection",
        {"manifold": "sphere_projection", "sectional_curvature": CURVATURE},
        CURVATURE,
        torch.float64,
    ),
    (
        "stereographic_negative",
        {"manifold": "stereographic", "sectional_curvature": -CURVATURE},
        -CURVATURE,
        torch.float64,
    ),
    (
        "stereographic_zero",
        {"manifold": "stereographic", "sectional_curvature": 0.0},
        0.0,
        torch.float64,
    ),
    (
        "stereographic_positive",
        {"manifold": "stereographic", "sectional_curvature": CURVATURE},
        CURVATURE,
        torch.float64,
    ),
)


def _assert_finite_backward(
    model: ManifoldSentenceTransformer,
    loss: torch.Tensor,
) -> None:
    assert loss.ndim == 0
    assert math.isfinite(float(loss.detach()))
    model.zero_grad(set_to_none=True)
    loss.backward()
    projection_gradients = [
        parameter.grad
        for parameter in model.projection.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert projection_gradients, "expected gradients on the trainable projection"
    assert all(
        bool(torch.isfinite(gradient).all()) for gradient in projection_gradients
    )
    assert any(
        bool(torch.count_nonzero(gradient).item())
        for gradient in projection_gradients
    ), "expected a non-zero gradient on the trainable projection"


@pytest.mark.parametrize(
    ("case_name", "model_kwargs", "expected_sectional", "expected_dtype"),
    GEOMETRY_CASES,
    ids=[case[0] for case in GEOMETRY_CASES],
)
def test_v09_real_stack_constant_curvature_matrix(
    case_name: str,
    model_kwargs: dict[str, object],
    expected_sectional: float,
    expected_dtype: torch.dtype,
) -> None:
    torch.manual_seed(61)
    model = ManifoldSentenceTransformer(
        MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
        **model_kwargs,
    )

    embeddings = model(ANCHORS)
    assert embeddings.shape == (
        len(ANCHORS),
        EMBEDDING_DIM + (1 if case_name == "lorentz" else 0),
    )
    assert embeddings.dtype == expected_dtype
    assert bool(torch.isfinite(embeddings).all())

    distance = model.distance(embeddings[0], embeddings[1])
    assert distance.ndim == 0
    assert math.isfinite(float(distance.detach()))
    assert float(distance.detach()) >= 0.0

    if case_name in {"poincare", "lorentz"}:
        assert math.isclose(model.curvature, CURVATURE, rel_tol=1e-6, abs_tol=1e-7)
        with pytest.raises(AttributeError):
            _ = model.sectional_curvature
    else:
        assert math.isclose(
            model.sectional_curvature,
            expected_sectional,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        with pytest.raises(AttributeError):
            _ = model.curvature

    if case_name in {
        "sphere_projection",
        "stereographic_negative",
        "stereographic_zero",
        "stereographic_positive",
    }:
        assert model.manifold.k.dtype == torch.float64

    loss = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)(
        ANCHORS,
        POSITIVES,
    )
    _assert_finite_backward(model, loss)
