"""Tests for the manifold multiple-negatives ranking objective."""

import neembed
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import neembed.model as model_module
from neembed.losses import ManifoldMultipleNegativesRankingLoss
from neembed.model import ManifoldSentenceTransformer


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used to test the loss without model downloads."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.linear = nn.Linear(3, 4, bias=False)

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = [[float(len(sentence)), 1.0, -1.0] for sentence in sentences]
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def _make_model(
    monkeypatch,
    *,
    manifold: str = "poincare",
) -> ManifoldSentenceTransformer:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    return ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold,
        embedding_dim=2,
    )


def test_loss_is_exported_from_package() -> None:
    assert (
        neembed.ManifoldMultipleNegativesRankingLoss
        is ManifoldMultipleNegativesRankingLoss
    )


@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_loss_requires_positive_temperature(monkeypatch, temperature: float) -> None:
    model = _make_model(monkeypatch)

    with pytest.raises(ValueError, match="temperature must be positive"):
        ManifoldMultipleNegativesRankingLoss(model=model, temperature=temperature)


def test_loss_uses_diagonal_pairs_as_positive_targets(monkeypatch) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch)
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.5)
    anchors = ["Shiba Inu", "dog", "mammal"]
    positives = ["dog", "mammal", "animal"]

    actual = loss_fn(anchors, positives)

    anchor_embeddings = model(anchors)
    positive_embeddings = model(positives)
    distances = model.manifold.dist(
        anchor_embeddings[:, None, :],
        positive_embeddings[None, :, :],
    )
    expected = F.cross_entropy(
        -distances / 0.5,
        torch.arange(len(anchors), device=distances.device),
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_loss_is_finite_scalar_and_backpropagates(monkeypatch, manifold: str) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)

    loss = loss_fn(["dog", "cat"], ["mammal", "animal"])
    loss.backward()

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert model.projection.weight.grad is not None
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()
