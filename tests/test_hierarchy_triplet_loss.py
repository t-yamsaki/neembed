"""Tests for directed parent-child-unrelated hierarchy supervision."""

import math

import pytest
import torch
from torch import nn

import neembed
from neembed import ManifoldHierarchyTripletLoss
from neembed.manifolds import get_manifold


class _ToyHierarchyTripletModel(nn.Module):
    """Tiny trainable manifold model with controllable tangent vectors."""

    def __init__(
        self,
        manifold_name: str,
        labels: list[str],
        tangent_vectors: list[list[float]],
    ) -> None:
        super().__init__()
        self.manifold_name = manifold_name
        self.manifold = get_manifold(manifold_name, curvature=1.0)
        self.index = {label: position for position, label in enumerate(labels)}
        dtype = torch.float64 if manifold_name == "lorentz" else torch.float32
        self.tangents = nn.Parameter(torch.tensor(tangent_vectors, dtype=dtype))

    def forward(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        spatial = torch.stack([self.tangents[self.index[text]] for text in texts])
        if self.manifold_name == "lorentz":
            spatial = torch.cat((spatial.new_zeros((len(texts), 1)), spatial), dim=-1)
        return self.manifold.expmap0(spatial)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_triplet_prefers_child_over_unrelated(manifold_name: str) -> None:
    labels = ["parent", "child", "unrelated"]
    correct_model = _ToyHierarchyTripletModel(
        manifold_name,
        labels,
        [[0.10, 0.0], [0.25, 0.0], [-0.35, 0.0]],
    )
    incorrect_model = _ToyHierarchyTripletModel(
        manifold_name,
        labels,
        [[0.10, 0.0], [0.25, 0.0], [0.12, 0.0]],
    )

    correct = ManifoldHierarchyTripletLoss(
        correct_model,
        margin=0.05,
        radial_margin=0.05,
    )(["parent"], ["child"], ["unrelated"])
    incorrect = ManifoldHierarchyTripletLoss(
        incorrect_model,
        margin=0.05,
        radial_margin=0.05,
    )(["parent"], ["child"], ["unrelated"])

    assert torch.isfinite(correct)
    assert torch.isfinite(incorrect)
    assert float(correct.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(incorrect.detach()) > float(correct.detach())


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_triplet_penalizes_direction_reversal(manifold_name: str) -> None:
    labels = ["parent", "child", "unrelated"]
    outward_model = _ToyHierarchyTripletModel(
        manifold_name,
        labels,
        [[0.10, 0.0], [0.25, 0.0], [-0.35, 0.0]],
    )
    reversed_model = _ToyHierarchyTripletModel(
        manifold_name,
        labels,
        [[0.25, 0.0], [0.10, 0.0], [-0.35, 0.0]],
    )

    outward = ManifoldHierarchyTripletLoss(
        outward_model,
        margin=0.0,
        radial_margin=0.05,
    )(["parent"], ["child"], ["unrelated"])
    reversed_value = ManifoldHierarchyTripletLoss(
        reversed_model,
        margin=0.0,
        radial_margin=0.05,
    )(["parent"], ["child"], ["unrelated"])

    assert float(outward.detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(reversed_value.detach()) > 0.0


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_triplet_has_finite_forward_and_backward(manifold_name: str) -> None:
    model = _ToyHierarchyTripletModel(
        manifold_name,
        ["parent", "child", "unrelated"],
        [[0.20, 0.05], [0.10, 0.02], [0.16, -0.02]],
    )
    loss = ManifoldHierarchyTripletLoss(
        model,
        margin=0.1,
        radial_margin=0.1,
    )

    value = loss(["parent"], ["child"], ["unrelated"])
    value.backward()

    assert value.ndim == 0
    assert math.isfinite(float(value.detach()))
    assert model.tangents.grad is not None
    assert torch.isfinite(model.tangents.grad).all()
    assert torch.count_nonzero(model.tangents.grad) > 0


def test_hierarchy_triplet_validates_hyperparameters_and_alignment() -> None:
    model = _ToyHierarchyTripletModel(
        "poincare",
        ["parent", "child", "unrelated"],
        [[0.10, 0.0], [0.20, 0.0], [-0.30, 0.0]],
    )

    for margin in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="margin must be non-negative and finite"):
            ManifoldHierarchyTripletLoss(model, margin=margin)
    for radial_margin in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="radial_margin"):
            ManifoldHierarchyTripletLoss(model, radial_margin=radial_margin)
    for radial_weight in (0.0, -0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="radial_weight must be positive and finite"):
            ManifoldHierarchyTripletLoss(model, radial_weight=radial_weight)

    loss = ManifoldHierarchyTripletLoss(model)
    with pytest.raises(ValueError, match="parents must be a sequence"):
        loss("parent", ["child"], ["unrelated"])
    with pytest.raises(ValueError, match="must not be empty"):
        loss([], [], [])
    with pytest.raises(ValueError, match="parents and children"):
        loss(["parent"], [], ["unrelated"])
    with pytest.raises(ValueError, match="parents and unrelated"):
        loss(["parent"], ["child"], [])


def test_hierarchy_triplet_loss_is_public() -> None:
    assert neembed.ManifoldHierarchyTripletLoss is ManifoldHierarchyTripletLoss
    assert "ManifoldHierarchyTripletLoss" in neembed.__all__
