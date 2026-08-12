"""Tests for opt-in manifold-valued prototype parameters."""

import math

import geoopt
import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import ManifoldPrototypes, ManifoldSentenceTransformer


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


def _make_model(monkeypatch, manifold: str) -> ManifoldSentenceTransformer:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    return ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold,
        embedding_dim=2,
        curvature=1.0,
    )


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_prototypes_are_valid_geoopt_manifold_parameters(monkeypatch, manifold_name) -> None:
    torch.manual_seed(0)
    model = _make_model(monkeypatch, manifold_name)
    prototypes = ManifoldPrototypes(model, num_prototypes=3, init_std=0.05)

    expected_ambient_dim = 3 if manifold_name == "lorentz" else 2
    assert prototypes.num_prototypes == 3
    assert prototypes.embedding_dim == 2
    assert prototypes.ambient_dim == expected_ambient_dim
    assert prototypes.manifold_name == manifold_name
    assert isinstance(prototypes.prototypes, geoopt.ManifoldParameter)
    assert prototypes.prototypes.manifold is model.manifold
    assert prototypes.prototypes.shape == (3, expected_ambient_dim)
    assert model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-5,
        rtol=1e-5,
    )
    assert not hasattr(model, "prototypes")


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_sentence_to_prototype_distances_are_finite_and_differentiable(
    monkeypatch,
    manifold_name,
) -> None:
    torch.manual_seed(1)
    model = _make_model(monkeypatch, manifold_name)
    prototypes = ManifoldPrototypes(model, num_prototypes=3)
    embeddings = model(["Shiba Inu", "dog"])

    distances = prototypes(embeddings)
    loss = distances.mean()
    loss.backward()

    assert distances.shape == (2, 3)
    assert torch.isfinite(distances).all()
    assert prototypes.prototypes.grad is not None
    assert torch.isfinite(prototypes.prototypes.grad).all()
    assert torch.count_nonzero(prototypes.prototypes.grad) > 0


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_riemannian_optimizer_updates_prototypes_and_preserves_manifold(
    monkeypatch,
    manifold_name,
) -> None:
    torch.manual_seed(2)
    model = _make_model(monkeypatch, manifold_name)
    prototypes = ManifoldPrototypes(model, num_prototypes=2)
    target = model(["target"]).detach()
    before = prototypes.prototypes.detach().clone()
    optimizer = geoopt.optim.RiemannianAdam(
        prototypes.parameters(),
        lr=1e-2,
        stabilize=1,
    )

    optimizer.zero_grad()
    loss = prototypes(target).sum()
    loss.backward()
    optimizer.step()

    assert not torch.equal(before, prototypes.prototypes.detach())
    assert model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-5,
        rtol=1e-5,
    )
    assert [name for name, _ in prototypes.named_parameters()] == ["prototypes"]


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_state_dict_round_trip_preserves_prototype_points(monkeypatch, manifold_name) -> None:
    torch.manual_seed(3)
    source_model = _make_model(monkeypatch, manifold_name)
    source = ManifoldPrototypes(source_model, num_prototypes=3)
    state = source.state_dict()

    torch.manual_seed(4)
    target_model = _make_model(monkeypatch, manifold_name)
    target = ManifoldPrototypes(target_model, num_prototypes=3)
    target.load_state_dict(state)

    assert list(state) == ["prototypes"]
    assert isinstance(target.prototypes, geoopt.ManifoldParameter)
    assert target.prototypes.manifold is target_model.manifold
    assert torch.allclose(source.prototypes, target.prototypes)
    assert target_model.manifold.check_point_on_manifold(
        target.prototypes,
        atol=1e-5,
        rtol=1e-5,
    )


def test_prototypes_reject_invalid_configuration(monkeypatch) -> None:
    model = _make_model(monkeypatch, "poincare")

    with pytest.raises(ValueError, match="num_prototypes must be positive"):
        ManifoldPrototypes(model, num_prototypes=0)
    for init_std in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="init_std must be positive and finite"):
            ManifoldPrototypes(model, num_prototypes=2, init_std=init_std)


def test_prototypes_reject_learnable_curvature_until_joint_optimization_is_supported(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold="poincare",
        embedding_dim=2,
        learnable_curvature=True,
    )

    with pytest.raises(ValueError, match="fixed curvature"):
        ManifoldPrototypes(model, num_prototypes=2)
