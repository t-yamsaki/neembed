"""Tests for the manifold triplet objective."""

import math

import neembed
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import neembed.model as model_module
from neembed.losses import ManifoldTripletLoss
from neembed.model import ManifoldSentenceTransformer
from neembed.trainer import ManifoldTrainer


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used to test losses without model downloads."""

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


class _LineManifold:
    def dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.abs(a - b).sum(dim=-1)


class _LookupModel(nn.Module):
    """Tiny deterministic model for exact zero/positive margin cases."""

    def __init__(self) -> None:
        super().__init__()
        self.manifold = _LineManifold()
        self.values = {
            "anchor": 0.0,
            "positive-close": 1.0,
            "positive-far": 3.0,
            "negative-close": 1.0,
            "negative-far": 3.0,
        }

    def forward(self, sentences) -> torch.Tensor:
        return torch.tensor([[self.values[text]] for text in sentences])


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


def test_triplet_loss_is_exported_from_package() -> None:
    assert neembed.ManifoldTripletLoss is ManifoldTripletLoss
    assert "ManifoldTripletLoss" in neembed.__all__


@pytest.mark.parametrize("margin", [-0.1, float("inf"), float("nan")])
def test_triplet_loss_requires_non_negative_finite_margin(
    monkeypatch,
    margin: float,
) -> None:
    model = _make_model(monkeypatch)

    with pytest.raises(ValueError, match="margin must be non-negative and finite"):
        ManifoldTripletLoss(model=model, margin=margin)


def test_triplet_loss_validates_aligned_non_empty_batches(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    loss_fn = ManifoldTripletLoss(model=model)

    with pytest.raises(ValueError, match="must not be empty"):
        loss_fn([], [], [])
    with pytest.raises(ValueError, match="anchors and positives"):
        loss_fn(["a"], ["p", "p2"], ["n"])
    with pytest.raises(ValueError, match="anchors and negatives"):
        loss_fn(["a"], ["p"], ["n", "n2"])
    with pytest.raises(ValueError, match="not a string"):
        loss_fn("anchor", ["p"], ["n"])


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_triplet_loss_matches_direct_geodesic_formula(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss_fn = ManifoldTripletLoss(model=model, margin=0.25)
    anchors = ["dog", "cat"]
    positives = ["mammal", "feline"]
    negatives = ["vehicle", "airplane"]

    actual = loss_fn(anchors, positives, negatives)

    anchor_embeddings = model(anchors)
    positive_embeddings = model(positives)
    negative_embeddings = model(negatives)
    expected = F.relu(
        model.manifold.dist(anchor_embeddings, positive_embeddings)
        - model.manifold.dist(anchor_embeddings, negative_embeddings)
        + 0.25
    ).mean()

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_triplet_loss_is_finite_scalar_and_backpropagates(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss_fn = ManifoldTripletLoss(model=model, margin=1.0)

    loss = loss_fn(
        ["dog", "cat"],
        ["mammal", "feline"],
        ["vehicle", "airplane"],
    )
    loss.backward()

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert model.projection.weight.grad is not None
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()


def test_triplet_loss_has_exact_zero_and_positive_controlled_cases() -> None:
    model = _LookupModel()
    loss_fn = ManifoldTripletLoss(model=model, margin=1.0)

    zero_loss = loss_fn(
        ["anchor"],
        ["positive-close"],
        ["negative-far"],
    )
    positive_loss = loss_fn(
        ["anchor"],
        ["positive-far"],
        ["negative-close"],
    )

    assert zero_loss.item() == pytest.approx(0.0)
    assert positive_loss.item() == pytest.approx(3.0)


def test_triplet_loss_works_with_existing_three_sequence_trainer(monkeypatch) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch)
    loss_fn = ManifoldTripletLoss(model=model, margin=1.0)
    trainer = ManifoldTrainer(model=model, loss=loss_fn, verbose=False)

    history = trainer.fit(
        [
            (
                ["dog", "cat"],
                ["mammal", "feline"],
                ["vehicle", "airplane"],
            )
        ],
        epochs=1,
    )

    assert len(history) == 1
    assert math.isfinite(history[0])
