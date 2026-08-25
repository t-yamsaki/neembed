"""Tests for graded manifold distance regression."""

import neembed
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import neembed.model as model_module
from neembed.losses import ManifoldDistanceMSELoss
from neembed.model import ManifoldSentenceTransformer


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
    """Tiny deterministic model for exact distance-regression cases."""

    def __init__(self) -> None:
        super().__init__()
        self.manifold = _LineManifold()
        self.values = {
            "origin": 0.0,
            "near": 1.0,
            "far": 3.0,
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


def test_distance_mse_loss_is_exported_from_package() -> None:
    assert neembed.ManifoldDistanceMSELoss is ManifoldDistanceMSELoss
    assert "ManifoldDistanceMSELoss" in neembed.__all__


def test_distance_mse_matches_exact_target_distances() -> None:
    model = _LookupModel()
    loss_fn = ManifoldDistanceMSELoss(model=model)

    loss = loss_fn(
        ["origin", "origin"],
        ["near", "far"],
        [1.0, 3.0],
    )

    assert loss.item() == pytest.approx(0.0)


def test_distance_mse_has_controlled_nonzero_error() -> None:
    model = _LookupModel()
    loss_fn = ManifoldDistanceMSELoss(model=model)

    loss = loss_fn(
        ["origin", "origin"],
        ["near", "far"],
        [0.0, 2.0],
    )

    assert loss.item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "target_distance",
    [
        1.0,
        [1.0, 1.0],
        torch.tensor([1.0, 1.0]),
    ],
)
def test_distance_mse_accepts_scalar_list_and_tensor_targets(target_distance) -> None:
    model = _LookupModel()
    loss_fn = ManifoldDistanceMSELoss(model=model)

    loss = loss_fn(
        ["origin", "origin"],
        ["near", "near"],
        target_distance,
    )

    assert loss.item() == pytest.approx(0.0)


def test_distance_mse_validates_aligned_non_empty_text_pairs(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    loss_fn = ManifoldDistanceMSELoss(model=model)

    with pytest.raises(ValueError, match="must not be empty"):
        loss_fn([], [], 0.0)
    with pytest.raises(ValueError, match="same length"):
        loss_fn(["a"], ["b", "c"], 0.0)
    with pytest.raises(ValueError, match="not a string"):
        loss_fn("anchor", ["other"], 0.0)
    with pytest.raises(ValueError, match="not a string"):
        loss_fn(["anchor"], "other", 0.0)


@pytest.mark.parametrize(
    "target_distance",
    [
        [0.1],
        torch.tensor([[0.1, 0.2]]),
    ],
)
def test_distance_mse_requires_one_target_per_pair_for_non_scalars(
    monkeypatch,
    target_distance,
) -> None:
    model = _make_model(monkeypatch)
    loss_fn = ManifoldDistanceMSELoss(model=model)

    with pytest.raises(ValueError, match="scalar or one value per aligned pair"):
        loss_fn(
            ["a1", "a2"],
            ["b1", "b2"],
            target_distance,
        )


@pytest.mark.parametrize(
    "target_distance, message",
    [
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (-0.1, "non-negative"),
        ([0.1, -0.2], "non-negative"),
        (True, "real numeric"),
        (1.0 + 2.0j, "real numeric"),
        ("not-a-number", "real numeric"),
    ],
)
def test_distance_mse_rejects_invalid_target_values(
    monkeypatch,
    target_distance,
    message: str,
) -> None:
    model = _make_model(monkeypatch)
    loss_fn = ManifoldDistanceMSELoss(model=model)

    with pytest.raises(ValueError, match=message):
        loss_fn(["anchor"], ["other"], target_distance)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_distance_mse_matches_direct_geodesic_formula(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss_fn = ManifoldDistanceMSELoss(model=model)
    texts_a = ["dog", "kitten"]
    texts_b = ["mammal", "vehicle"]
    target_distance = torch.tensor([0.25, 0.75])

    actual = loss_fn(texts_a, texts_b, target_distance)

    embeddings_a = model(texts_a)
    embeddings_b = model(texts_b)
    predicted_distance = model.manifold.dist(embeddings_a, embeddings_b)
    expected = F.mse_loss(
        predicted_distance,
        target_distance.to(
            device=predicted_distance.device,
            dtype=predicted_distance.dtype,
        ),
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_distance_mse_is_finite_scalar_and_backpropagates(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold=manifold)
    loss_fn = ManifoldDistanceMSELoss(model=model)

    loss = loss_fn(
        ["dog", "kitten"],
        ["mammal", "vehicle"],
        [0.5, 1.0],
    )
    loss.backward()

    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert model.projection.weight.grad is not None
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()
