"""Tests for the minimal manifold-valued sentence encoder."""

import neembed
import numpy as np
import torch
from torch import nn

import neembed.model as model_module
from neembed.evaluator import ManifoldEmbeddingEvaluator
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


def test_lorentz_forward_adds_one_ambient_coordinate_and_satisfies_constraint(
    monkeypatch,
) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold="lorentz",
        embedding_dim=2,
        curvature=2.0,
    )

    embeddings = model(["Shiba Inu", "dog"])
    quad_form = -embeddings[:, 0].square() + embeddings[:, 1:].square().sum(dim=-1)

    assert model.embedding_dim == 2
    assert embeddings.shape == (2, 3)
    assert torch.isfinite(embeddings).all()
    # This path intentionally follows the model's float32 dtype. Geoopt's
    # Lorentz operations accumulate about 1e-4 constraint error at this scale;
    # precision-specific geometry regression belongs in #47.
    assert torch.allclose(
        quad_form,
        torch.full_like(quad_form, -0.5),
        atol=2e-4,
        rtol=2e-4,
    )


def test_gradients_flow_through_projection_and_manifold_map(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)

    loss = model(["dog", "mammal"]).square().sum()
    loss.backward()

    assert model.projection.weight.grad is not None
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()


def test_encode_batch_returns_numpy_by_default(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)

    embeddings = model.encode(["Shiba Inu", "dog"])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 2)
    assert np.isfinite(embeddings).all()


def test_encode_single_text_returns_one_embedding(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)

    embedding = model.encode("dog")

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (2,)
    assert np.isfinite(embedding).all()


def test_encode_can_return_inference_tensor(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)
    model.train()

    embeddings = model.encode(["dog", "mammal"], convert_to_tensor=True)

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 2)
    assert not embeddings.requires_grad
    assert not model.training


def test_lorentz_encode_distance_and_evaluator_return_finite_values(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold="lorentz",
        embedding_dim=2,
    )
    embeddings = model.encode(["a", "long"], convert_to_tensor=True)

    distance = model.distance(embeddings[0], embeddings[1])
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["a", "long"],
        positives=["bb", "longer"],
    )
    metrics = evaluator()

    assert embeddings.shape == (2, 3)
    assert torch.isfinite(embeddings).all()
    assert torch.isfinite(distance)
    assert float(distance) >= 0.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_distance_is_finite_nonnegative_zero_on_self_and_symmetric(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)
    embeddings = model.encode(["Shiba Inu", "dog"])

    distance_ab = model.distance(embeddings[0], embeddings[1])
    distance_ba = model.distance(embeddings[1], embeddings[0])
    distance_aa = model.distance(embeddings[0], embeddings[0])

    assert torch.isfinite(distance_ab)
    assert float(distance_ab) >= 0.0
    assert torch.allclose(distance_ab, distance_ba, atol=1e-6)
    assert torch.allclose(distance_aa, torch.zeros_like(distance_aa), atol=1e-6)


def test_distance_returns_regular_tensor_without_grad_tracking(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)
    embeddings = model.encode(["Shiba Inu", "dog"], convert_to_tensor=True)

    distance = model.distance(embeddings[0], embeddings[1])

    assert torch.isfinite(distance)
    assert not distance.requires_grad
    assert not torch.is_inference(distance)
