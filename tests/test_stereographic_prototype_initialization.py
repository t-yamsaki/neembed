"""Regression coverage for stereographic prototype float64 initialization."""

import geoopt
import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import ManifoldPrototypes, ManifoldSentenceTransformer


class _FakeSentenceTransformer(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
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
    manifold_name: str,
) -> ManifoldSentenceTransformer:
    monkeypatch.setattr(model_module, "SentenceTransformer", _FakeSentenceTransformer)
    sectional_curvature = 0.5 if manifold_name == "sphere_projection" else -0.5
    return ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        sectional_curvature=sectional_curvature,
    )


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_stereographic_prototypes_start_on_float64_geometry(
    monkeypatch,
    manifold_name: str,
) -> None:
    torch.manual_seed(10)
    model = _make_model(monkeypatch, manifold_name)
    prototypes = ManifoldPrototypes(model, num_prototypes=3, init_std=0.05)

    assert isinstance(prototypes.prototypes, geoopt.ManifoldParameter)
    assert prototypes.prototypes.manifold is model.manifold
    assert prototypes.prototypes.dtype == torch.float64
    assert prototypes.prototypes.device == model.manifold.k.device
    assert model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-8,
        rtol=1e-8,
    )


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_stereographic_prototype_optimizer_step_stays_float64_without_transfer(
    monkeypatch,
    manifold_name: str,
) -> None:
    torch.manual_seed(11)
    model = _make_model(monkeypatch, manifold_name)
    prototypes = ManifoldPrototypes(model, num_prototypes=2, init_std=0.05)
    before = prototypes.prototypes.detach().clone()
    optimizer = geoopt.optim.RiemannianAdam(
        prototypes.parameters(),
        lr=1e-2,
        stabilize=1,
    )

    optimizer.zero_grad()
    distances = prototypes(model(["target", "other"]))
    loss = distances.mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert prototypes.prototypes.grad is not None
    assert prototypes.prototypes.grad.dtype == torch.float64
    assert torch.isfinite(prototypes.prototypes.grad).all()
    assert torch.count_nonzero(prototypes.prototypes.grad) > 0

    optimizer.step()

    assert prototypes.prototypes.dtype == torch.float64
    assert prototypes.prototypes.device == model.manifold.k.device
    assert not torch.equal(before, prototypes.prototypes.detach())
    assert model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-8,
        rtol=1e-8,
    )


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_stereographic_prototype_state_dict_round_trip_preserves_float64(
    monkeypatch,
    manifold_name: str,
) -> None:
    torch.manual_seed(12)
    source_model = _make_model(monkeypatch, manifold_name)
    source = ManifoldPrototypes(source_model, num_prototypes=3, init_std=0.05)
    state = source.state_dict()

    assert state["prototypes"].dtype == torch.float64

    torch.manual_seed(13)
    target_model = _make_model(monkeypatch, manifold_name)
    target = ManifoldPrototypes(target_model, num_prototypes=3, init_std=0.05)
    assert target.prototypes.dtype == torch.float64

    target.load_state_dict(state)

    assert isinstance(target.prototypes, geoopt.ManifoldParameter)
    assert target.prototypes.manifold is target_model.manifold
    assert target.prototypes.dtype == torch.float64
    assert target.prototypes.device == target_model.manifold.k.device
    assert torch.allclose(source.prototypes, target.prototypes, atol=0.0, rtol=0.0)
