"""Numerical stability tests for the v0.1 Poincare path."""

import math

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed.losses import ManifoldMultipleNegativesRankingLoss
from neembed.manifolds import get_manifold
from neembed.model import ManifoldSentenceTransformer


class StabilitySentenceTransformer(nn.Module):
    """Small trainable encoder that can emit challenging tangent vectors."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [2.0, -1.0, 0.5],
                        [-1.5, 1.0, 0.5],
                        [0.75, 1.5, -1.0],
                        [1.0, 0.25, 1.5],
                    ]
                )
            )

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = [
            [float(len(sentence)), float(index + 1), -1.0]
            for index, sentence in enumerate(sentences)
        ]
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "sentence_embedding": 5.0 * self.linear(features["input_features"])
        }


@pytest.mark.parametrize("curvature", [1e-3, 1.0, 10.0])
def test_expmap0_keeps_large_tangent_vectors_finite_and_inside_ball(
    curvature: float,
) -> None:
    manifold = get_manifold("poincare", curvature=curvature)
    tangent = torch.tensor(
        [[1.0, -2.0], [25.0, 10.0], [1000.0, -500.0]],
        dtype=torch.float64,
    )

    points = manifold.expmap0(tangent)
    radius = 1.0 / math.sqrt(curvature)

    assert torch.isfinite(points).all()
    assert bool((points.norm(dim=-1) < radius).all())


@pytest.mark.parametrize("curvature", [1e-3, 1.0, 10.0])
def test_geodesic_distance_is_finite_for_near_boundary_points(
    curvature: float,
) -> None:
    manifold = get_manifold("poincare", curvature=curvature)
    tangent = torch.tensor(
        [[40.0, 0.0], [0.0, 40.0], [-40.0, 0.0]],
        dtype=torch.float64,
    )
    points = manifold.expmap0(tangent)

    distances = manifold.dist(points[:, None, :], points[None, :, :])

    assert torch.isfinite(distances).all()
    assert bool((distances >= 0).all())
    assert torch.allclose(torch.diag(distances), torch.zeros(3, dtype=distances.dtype))


@pytest.mark.parametrize(
    ("curvature", "temperature"),
    [(1e-3, 10.0), (1.0, 0.1), (10.0, 1e-3)],
)
def test_inference_distance_warmup_then_loss_backward_remain_finite(
    monkeypatch,
    curvature: float,
    temperature: float,
) -> None:
    monkeypatch.setattr(
        model_module,
        "SentenceTransformer",
        StabilitySentenceTransformer,
    )
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        "fake-model",
        embedding_dim=2,
        curvature=curvature,
    )
    loss_fn = ManifoldMultipleNegativesRankingLoss(
        model=model,
        temperature=temperature,
    )
    anchors = ["Shiba Inu", "salmon", "sparrow", "oak"]
    positives = ["dog", "fish", "bird", "plant"]

    warm_embeddings = model.encode(
        ["Shiba Inu", "dog"],
        convert_to_tensor=True,
    )
    public_distance = model.distance(warm_embeddings[0], warm_embeddings[1])

    anchor_embeddings = model(anchors)
    positive_embeddings = model(positives)
    distances = model.manifold.dist(
        anchor_embeddings[:, None, :],
        positive_embeddings[None, :, :],
    )
    loss = loss_fn(anchors, positives)
    loss.backward()
    radius = 1.0 / math.sqrt(curvature)

    assert torch.isfinite(public_distance)
    assert not public_distance.requires_grad
    assert not torch.is_inference(public_distance)
    assert torch.isfinite(anchor_embeddings).all()
    assert torch.isfinite(positive_embeddings).all()
    assert bool((anchor_embeddings.norm(dim=-1) < radius).all())
    assert bool((positive_embeddings.norm(dim=-1) < radius).all())
    assert torch.isfinite(distances).all()
    assert torch.isfinite(loss)

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "curvature",
    [0.0, -1.0, float("nan"), float("inf"), -float("inf")],
)
def test_get_manifold_rejects_invalid_curvature(curvature: float) -> None:
    with pytest.raises(ValueError, match="curvature must be positive and finite"):
        get_manifold("poincare", curvature=curvature)


@pytest.mark.parametrize(
    "temperature",
    [0.0, -0.1, float("nan"), float("inf"), -float("inf")],
)
def test_loss_rejects_invalid_temperature(monkeypatch, temperature: float) -> None:
    monkeypatch.setattr(
        model_module,
        "SentenceTransformer",
        StabilitySentenceTransformer,
    )
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)

    with pytest.raises(ValueError, match="temperature must be positive and finite"):
        ManifoldMultipleNegativesRankingLoss(
            model=model,
            temperature=temperature,
        )
