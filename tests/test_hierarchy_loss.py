"""Tests for the minimal hierarchy-aware manifold prototype objective."""

import math

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import (
    ManifoldPrototypeHierarchyLoss,
    ManifoldPrototypes,
    ManifoldSentenceTransformer,
)


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


def _make_model(monkeypatch, manifold_name: str) -> ManifoldSentenceTransformer:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    return ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        curvature=1.0,
    )


def _make_loss(monkeypatch, manifold_name: str):
    model = _make_model(monkeypatch, manifold_name)
    prototypes = ManifoldPrototypes(model, num_prototypes=3, init_std=0.05)
    loss = ManifoldPrototypeHierarchyLoss(
        model,
        prototypes,
        prototype_ids=("animal", "dog", "cat"),
        parent_relations=(("dog", "animal"), ("cat", "animal")),
        margin=0.1,
    )
    return model, prototypes, loss


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_loss_is_finite_and_reaches_model_and_prototype_parameters(
    monkeypatch,
    manifold_name,
) -> None:
    torch.manual_seed(11)
    model, prototypes, loss = _make_loss(monkeypatch, manifold_name)

    value = loss(
        ["Shiba Inu", "Siamese cat"],
        ["dog", "cat"],
    )
    value.backward()

    assert value.ndim == 0
    assert math.isfinite(float(value.detach()))
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.encoder.linear.weight.grad).all()
    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert prototypes.prototypes.grad is not None
    assert torch.isfinite(prototypes.prototypes.grad).all()
    assert torch.count_nonzero(prototypes.prototypes.grad) > 0


def _ordered_points(
    prototypes: ManifoldPrototypes,
    offsets: tuple[float, float, float],
) -> torch.Tensor:
    tangent = torch.zeros_like(prototypes.prototypes)
    coordinate = 1 if prototypes.manifold_name == "lorentz" else 0
    tangent[:, coordinate] = torch.tensor(
        offsets,
        dtype=tangent.dtype,
        device=tangent.device,
    )
    return prototypes.manifold.expmap0(tangent)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_term_prefers_correctly_ordered_toy_configuration(
    monkeypatch,
    manifold_name,
) -> None:
    torch.manual_seed(12)
    model = _make_model(monkeypatch, manifold_name)
    prototypes = ManifoldPrototypes(model, num_prototypes=3, init_std=0.05)
    loss = ManifoldPrototypeHierarchyLoss(
        model,
        prototypes,
        prototype_ids=("root", "leaf", "unrelated"),
        parent_relations=(("leaf", "root"),),
        margin=0.2,
    )

    with torch.no_grad():
        prototypes.prototypes.copy_(
            _ordered_points(prototypes, (0.0, 0.05, 0.8))
        )
    correct = loss(["root sentence"], ["root"]).detach()

    with torch.no_grad():
        prototypes.prototypes.copy_(
            _ordered_points(prototypes, (0.0, 0.8, 0.05))
        )
    corrupted = loss(["root sentence"], ["root"]).detach()

    assert torch.isfinite(correct)
    assert torch.isfinite(corrupted)
    assert correct < corrupted


def test_hierarchy_loss_rejects_invalid_prototype_identifiers(monkeypatch) -> None:
    model = _make_model(monkeypatch, "poincare")
    prototypes = ManifoldPrototypes(model, num_prototypes=3)

    with pytest.raises(ValueError, match="exactly one identifier"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=("root", "leaf"),
            parent_relations=(("leaf", "root"),),
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=("root", "", "other"),
            parent_relations=(("other", "root"),),
        )
    with pytest.raises(ValueError, match="unique"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=("root", "leaf", "leaf"),
            parent_relations=(("leaf", "root"),),
        )
    with pytest.raises(ValueError, match="reference identifiers"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=("root", "leaf", "other"),
            parent_relations=(("leaf", "missing"),),
        )


def test_hierarchy_loss_rejects_malformed_relations(monkeypatch) -> None:
    model = _make_model(monkeypatch, "poincare")
    prototypes = ManifoldPrototypes(model, num_prototypes=3)
    ids = ("root", "leaf", "other")

    with pytest.raises(ValueError, match="child_id, parent_id"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=ids,
            parent_relations=(("leaf", "root", "extra"),),
        )
    with pytest.raises(ValueError, match="own parent"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=ids,
            parent_relations=(("leaf", "leaf"),),
        )
    with pytest.raises(ValueError, match="at most one parent"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=ids,
            parent_relations=(("leaf", "root"), ("leaf", "other")),
        )
    with pytest.raises(ValueError, match="acyclic"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=ids,
            parent_relations=(("leaf", "root"), ("root", "leaf")),
        )
    with pytest.raises(ValueError, match="at least one relation"):
        ManifoldPrototypeHierarchyLoss(
            model,
            prototypes,
            prototype_ids=ids,
            parent_relations=(),
        )


def test_hierarchy_loss_rejects_invalid_batch_assignments(monkeypatch) -> None:
    _, _, loss = _make_loss(monkeypatch, "poincare")

    with pytest.raises(ValueError, match="must not be empty"):
        loss([], [])
    with pytest.raises(ValueError, match="same length"):
        loss(["dog", "cat"], ["dog"])
    with pytest.raises(ValueError, match="unknown prototype identifier"):
        loss(["dog"], ["missing"])


def test_hierarchy_loss_rejects_invalid_weights(monkeypatch) -> None:
    model = _make_model(monkeypatch, "poincare")
    prototypes = ManifoldPrototypes(model, num_prototypes=3)
    kwargs = {
        "model": model,
        "prototypes": prototypes,
        "prototype_ids": ("root", "leaf", "other"),
        "parent_relations": (("leaf", "root"),),
    }

    for margin in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="margin must be non-negative and finite"):
            ManifoldPrototypeHierarchyLoss(**kwargs, margin=margin)
    for hierarchy_weight in (-0.1, math.inf, math.nan):
        with pytest.raises(
            ValueError,
            match="hierarchy_weight must be non-negative and finite",
        ):
            ManifoldPrototypeHierarchyLoss(
                **kwargs,
                hierarchy_weight=hierarchy_weight,
            )
