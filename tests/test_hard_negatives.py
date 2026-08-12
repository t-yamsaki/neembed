"""Regression tests for explicit hard negatives in manifold ranking training."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import neembed.model as model_module
from neembed.losses import ManifoldMultipleNegativesRankingLoss
from neembed.manifolds import get_manifold
from neembed.model import ManifoldSentenceTransformer
from neembed.trainer import ManifoldTrainer


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


class ControlledManifoldModel(nn.Module):
    """Tiny model with known tangent locations for loss-ordering checks."""

    def __init__(self, manifold: str) -> None:
        super().__init__()
        self.manifold_name = manifold
        self.manifold = get_manifold(manifold)
        self.locations = nn.Parameter(
            torch.tensor(
                [
                    [0.00, 0.00],
                    [0.10, 0.00],
                    [0.12, 0.00],
                    [0.80, 0.00],
                ],
                dtype=torch.float32,
            )
        )
        self.index = {
            "anchor": 0,
            "positive": 1,
            "hard-negative": 2,
            "easy-negative": 3,
        }

    def forward(self, sentences: list[str]) -> torch.Tensor:
        indices = torch.tensor(
            [self.index[sentence] for sentence in sentences],
            device=self.locations.device,
        )
        tangent = self.locations[indices]
        if self.manifold_name == "lorentz":
            tangent = tangent.to(dtype=torch.float64)
            tangent = torch.cat((torch.zeros_like(tangent[..., :1]), tangent), dim=-1)
        return self.manifold.expmap0(tangent)


def test_two_sequence_loss_keeps_existing_formula(monkeypatch) -> None:
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
def test_explicit_negatives_are_appended_to_geodesic_candidate_pool(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.5)
    anchors = ["dog", "cat"]
    positives = ["mammal", "animal"]
    negatives = ["vehicle", "building"]

    actual = loss_fn(anchors, positives, negatives)

    anchor_embeddings = model(anchors)
    positive_embeddings = model(positives)
    negative_embeddings = model(negatives)
    candidates = torch.cat((positive_embeddings, negative_embeddings), dim=0)
    distances = model.manifold.dist(
        anchor_embeddings[:, None, :],
        candidates[None, :, :],
    )
    expected = F.cross_entropy(
        -distances / 0.5,
        torch.arange(len(anchors), device=distances.device),
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_explicit_negative_loss_is_finite_and_backpropagates(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)

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


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_closer_explicit_negative_increases_loss(manifold: str) -> None:
    model = ControlledManifoldModel(manifold)
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.2)

    hard_loss = loss_fn(["anchor"], ["positive"], ["hard-negative"])
    easy_loss = loss_fn(["anchor"], ["positive"], ["easy-negative"])

    assert torch.isfinite(hard_loss)
    assert torch.isfinite(easy_loss)
    assert hard_loss > easy_loss


@pytest.mark.parametrize(
    ("anchors", "positives", "negatives", "message"),
    [
        ([], [], None, "must not be empty"),
        (["a", "b"], ["p"], None, "anchors and positives"),
        (["a", "b"], ["p1", "p2"], ["n"], "anchors and negatives"),
    ],
)
def test_loss_rejects_malformed_aligned_batches(
    monkeypatch,
    anchors: list[str],
    positives: list[str],
    negatives: list[str] | None,
    message: str,
) -> None:
    model = _make_model(monkeypatch)
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model)

    with pytest.raises(ValueError, match=message):
        loss_fn(anchors, positives, negatives)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_trainer_accepts_three_sequence_batches_and_updates_parameters(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.5)
    trainer = ManifoldTrainer(
        model=model,
        loss=loss,
        learning_rate=1e-2,
        verbose=False,
    )
    batches = [
        (
            ["dog", "cat"],
            ["mammal", "animal"],
            ["vehicle", "building"],
        )
    ]
    before = model.projection.weight.detach().clone()

    history = trainer.fit(batches)

    assert len(history) == 1
    assert torch.isfinite(torch.tensor(history[0]))
    assert not torch.allclose(before, model.projection.weight.detach())


def test_trainer_rejects_batch_with_unsupported_arity(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    loss = ManifoldMultipleNegativesRankingLoss(model=model)
    trainer = ManifoldTrainer(model=model, loss=loss, verbose=False)

    with pytest.raises(ValueError, match="two or three aligned sequences"):
        trainer.fit([(["dog"],)])  # type: ignore[list-item]
