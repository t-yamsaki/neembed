"""Tests for the opt-in symmetric manifold ranking objective."""

import neembed
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import neembed.model as model_module
from neembed.losses import ManifoldMultipleNegativesRankingLoss
from neembed.model import ManifoldSentenceTransformer
from neembed.symmetric_loss import ManifoldSymmetricMultipleNegativesRankingLoss


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used without external model downloads."""

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


def _forward_loss(
    model: ManifoldSentenceTransformer,
    anchors: list[str],
    positives: list[str],
    negatives: list[str] | None,
    *,
    temperature: float,
) -> torch.Tensor:
    anchor_embeddings = model(anchors)
    positive_embeddings = model(positives)
    candidates = positive_embeddings
    if negatives is not None:
        candidates = torch.cat((positive_embeddings, model(negatives)), dim=0)
    distances = model.manifold.dist(
        anchor_embeddings[:, None, :],
        candidates[None, :, :],
    )
    targets = torch.arange(len(anchors), device=distances.device)
    return F.cross_entropy(-distances / temperature, targets)


def _reverse_loss(
    model: ManifoldSentenceTransformer,
    anchors: list[str],
    positives: list[str],
    *,
    temperature: float,
) -> torch.Tensor:
    anchor_embeddings = model(anchors)
    positive_embeddings = model(positives)
    distances = model.manifold.dist(
        positive_embeddings[:, None, :],
        anchor_embeddings[None, :, :],
    )
    targets = torch.arange(len(anchors), device=distances.device)
    return F.cross_entropy(-distances / temperature, targets)


def test_symmetric_ranking_loss_is_exported() -> None:
    assert (
        neembed.ManifoldSymmetricMultipleNegativesRankingLoss
        is ManifoldSymmetricMultipleNegativesRankingLoss
    )
    assert "ManifoldSymmetricMultipleNegativesRankingLoss" in neembed.__all__


def test_existing_ranking_loss_remains_one_directional(monkeypatch) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch)
    anchors = ["dog", "cat"]
    positives = ["mammal", "animal"]
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.5)

    actual = loss_fn(anchors, positives)
    expected = _forward_loss(
        model,
        anchors,
        positives,
        None,
        temperature=0.5,
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_symmetric_loss_matches_mean_of_both_directions(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    anchors = ["dog", "kitten", "sparrow"]
    positives = ["mammal", "feline", "bird"]
    loss_fn = ManifoldSymmetricMultipleNegativesRankingLoss(
        model=model,
        temperature=0.5,
    )

    actual = loss_fn(anchors, positives)
    expected = 0.5 * (
        _forward_loss(model, anchors, positives, None, temperature=0.5)
        + _reverse_loss(model, anchors, positives, temperature=0.5)
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_explicit_negatives_affect_only_forward_candidate_pool(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    anchors = ["dog", "cat"]
    positives = ["mammal", "animal"]
    negatives = ["vehicle", "building"]
    loss_fn = ManifoldSymmetricMultipleNegativesRankingLoss(
        model=model,
        temperature=0.5,
    )

    actual = loss_fn(anchors, positives, negatives)
    forward = _forward_loss(
        model,
        anchors,
        positives,
        negatives,
        temperature=0.5,
    )
    reverse = _reverse_loss(model, anchors, positives, temperature=0.5)
    expected = 0.5 * (forward + reverse)

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_symmetric_loss_is_finite_and_backpropagates(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss_fn = ManifoldSymmetricMultipleNegativesRankingLoss(
        model=model,
        temperature=0.2,
    )

    loss = loss_fn(
        ["dog", "cat"],
        ["mammal", "animal"],
        ["vehicle", "building"],
    )
    loss.backward()

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert model.projection.weight.grad is not None
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()


@pytest.mark.parametrize(
    ("anchors", "positives", "negatives", "message"),
    [
        ([], [], None, "must not be empty"),
        (["a", "b"], ["p"], None, "anchors and positives"),
        (["a", "b"], ["p1", "p2"], ["n"], "anchors and negatives"),
    ],
)
def test_symmetric_loss_rejects_malformed_aligned_batches(
    monkeypatch,
    anchors: list[str],
    positives: list[str],
    negatives: list[str] | None,
    message: str,
) -> None:
    model = _make_model(monkeypatch)
    loss_fn = ManifoldSymmetricMultipleNegativesRankingLoss(model=model)

    with pytest.raises(ValueError, match=message):
        loss_fn(anchors, positives, negatives)
