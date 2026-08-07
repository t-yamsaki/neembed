"""Tests for the minimal manifold-valued sentence encoder."""

import neembed
import torch
from torch import nn

import neembed.model as model_module
from neembed.model import ManifoldSentenceTransformer


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used to test neembed without model downloads."""

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


def _patch_encoder(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)


def test_model_is_exported_from_package() -> None:
    assert neembed.ManifoldSentenceTransformer is ManifoldSentenceTransformer


def test_forward_with_projection_returns_finite_points_inside_ball(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2, curvature=1.0)

    embeddings = model(["Shiba Inu", "dog"])

    assert embeddings.shape == (2, 2)
    assert torch.isfinite(embeddings).all()
    assert torch.linalg.vector_norm(embeddings, dim=-1).lt(1.0).all()
    assert isinstance(model.projection, nn.Linear)


def test_forward_without_projection_preserves_encoder_dimension(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=None)

    embeddings = model(["dog", "mammal"])

    assert embeddings.shape == (2, 4)
    assert isinstance(model.projection, nn.Identity)


def test_gradients_flow_through_projection_and_manifold_map(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)

    loss = model(["dog", "mammal"]).square().sum()
    loss.backward()

    assert model.projection.weight.grad is not None
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()
