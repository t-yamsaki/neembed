"""Tests for the small retrieval-plus-hierarchy composite objective."""

import math

import pytest
import torch
from torch import nn

import neembed
from neembed import (
    ManifoldRadialOrderLoss,
    ManifoldRetrievalHierarchyLoss,
    ManifoldTrainer,
    ManifoldTripletLoss,
)
from neembed.manifolds import get_manifold


class _FixedLoss(nn.Module):
    def __init__(self, value: float, arity: int) -> None:
        super().__init__()
        self.value = float(value)
        self.arity = arity
        self.calls = 0

    def forward(self, *args: object) -> torch.Tensor:
        assert len(args) == self.arity
        self.calls += 1
        return torch.tensor(self.value)


class _FailingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, *args: object) -> torch.Tensor:
        self.calls += 1
        raise AssertionError("hierarchy loss should not be evaluated")


class _ParameterizedLoss(nn.Module):
    def __init__(self, initial: float, arity: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(initial))
        self.arity = arity

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        assert len(values) == self.arity
        total = torch.stack([value.reshape(()) for value in values]).sum()
        return self.scale * total


class _ToyManifoldModel(nn.Module):
    """Tiny trainable model with controllable tangent vectors."""

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


def test_composite_loss_matches_exact_weighted_formula_and_components() -> None:
    retrieval = _FixedLoss(2.0, arity=2)
    hierarchy = _FixedLoss(3.0, arity=3)
    loss = ManifoldRetrievalHierarchyLoss(
        retrieval,
        hierarchy,
        hierarchy_weight=0.25,
    )

    retrieval_inputs = (["anchor"], ["positive"])
    hierarchy_inputs = (["parent"], ["child"], ["unrelated"])
    total = loss(retrieval_inputs, hierarchy_inputs)
    retrieval_value, hierarchy_value = loss.component_losses(
        retrieval_inputs,
        hierarchy_inputs,
    )

    assert float(total) == pytest.approx(2.0 + 0.25 * 3.0)
    assert float(retrieval_value) == pytest.approx(2.0)
    assert float(hierarchy_value) == pytest.approx(3.0)


def test_zero_hierarchy_weight_is_true_retrieval_only_path() -> None:
    retrieval = _FixedLoss(1.75, arity=2)
    hierarchy = _FailingLoss()
    loss = ManifoldRetrievalHierarchyLoss(
        retrieval,
        hierarchy,
        hierarchy_weight=0.0,
    )

    direct = retrieval(["anchor"], ["positive"])
    total = loss((["anchor"], ["positive"]), None)

    assert torch.equal(total, direct)
    assert hierarchy.calls == 0


def test_active_components_receive_finite_gradients_and_optimizer_updates() -> None:
    retrieval = _ParameterizedLoss(initial=1.0, arity=2)
    hierarchy = _ParameterizedLoss(initial=2.0, arity=3)
    loss = ManifoldRetrievalHierarchyLoss(
        retrieval,
        hierarchy,
        hierarchy_weight=0.5,
    )
    optimizer = torch.optim.SGD(loss.parameters(), lr=0.1)

    before_retrieval = retrieval.scale.detach().clone()
    before_hierarchy = hierarchy.scale.detach().clone()
    optimizer.zero_grad()
    total = loss(
        (torch.tensor(1.0), torch.tensor(2.0)),
        (torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)),
    )
    total.backward()

    assert retrieval.scale.grad is not None
    assert hierarchy.scale.grad is not None
    assert torch.isfinite(retrieval.scale.grad)
    assert torch.isfinite(hierarchy.scale.grad)
    assert torch.count_nonzero(retrieval.scale.grad) > 0
    assert torch.count_nonzero(hierarchy.scale.grad) > 0

    optimizer.step()
    assert not torch.equal(retrieval.scale.detach(), before_retrieval)
    assert not torch.equal(hierarchy.scale.detach(), before_hierarchy)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_composite_supports_existing_manifold_losses_and_trainer(
    manifold_name: str,
) -> None:
    labels = ["parent", "child", "unrelated"]
    model = _ToyManifoldModel(
        manifold_name,
        labels,
        [[0.40, 0.0], [0.20, 0.0], [0.35, 0.0]],
    )
    retrieval = ManifoldTripletLoss(model, margin=0.1)
    hierarchy = ManifoldRadialOrderLoss(model, margin=0.1)
    loss = ManifoldRetrievalHierarchyLoss(
        retrieval,
        hierarchy,
        hierarchy_weight=0.5,
    )

    retrieval_inputs = (["parent"], ["child"], ["unrelated"])
    hierarchy_inputs = (["parent"], ["child"])
    retrieval_value, hierarchy_value = loss.component_losses(
        retrieval_inputs,
        hierarchy_inputs,
    )
    total = loss(retrieval_inputs, hierarchy_inputs)

    assert torch.isfinite(retrieval_value)
    assert torch.isfinite(hierarchy_value)
    assert torch.isfinite(total)
    assert float(total.detach()) == pytest.approx(
        float((retrieval_value + 0.5 * hierarchy_value).detach()),
        rel=1e-6,
        abs=1e-7,
    )

    total.backward()
    assert model.tangents.grad is not None
    assert torch.isfinite(model.tangents.grad).all()
    assert torch.count_nonzero(model.tangents.grad) > 0

    model.tangents.grad = None
    trainer = ManifoldTrainer(model, loss, learning_rate=1e-3, verbose=False)
    history = trainer.fit([(retrieval_inputs, hierarchy_inputs)], epochs=1)
    assert len(history) == 1
    assert math.isfinite(history[0])


def test_composite_validates_weight_and_argument_bundles() -> None:
    retrieval = _FixedLoss(1.0, arity=1)
    hierarchy = _FixedLoss(1.0, arity=1)

    for weight in (-0.1, math.inf, math.nan, True):
        with pytest.raises(ValueError, match="non-negative and finite"):
            ManifoldRetrievalHierarchyLoss(
                retrieval,
                hierarchy,
                hierarchy_weight=weight,
            )

    loss = ManifoldRetrievalHierarchyLoss(retrieval, hierarchy)
    with pytest.raises(ValueError, match="retrieval_inputs"):
        loss("not-a-bundle", (["hierarchy"],))
    with pytest.raises(ValueError, match="retrieval_inputs"):
        loss((), (["hierarchy"],))
    with pytest.raises(ValueError, match="hierarchy_inputs"):
        loss((["retrieval"],), None)
    with pytest.raises(ValueError, match="hierarchy_inputs"):
        loss((["retrieval"],), "not-a-bundle")


def test_invalid_hierarchy_bundle_is_rejected_before_retrieval_runs() -> None:
    retrieval = _FixedLoss(1.0, arity=1)
    hierarchy = _FixedLoss(1.0, arity=1)
    loss = ManifoldRetrievalHierarchyLoss(retrieval, hierarchy)

    for hierarchy_inputs in (None, (), "not-a-bundle"):
        with pytest.raises(ValueError, match="hierarchy_inputs"):
            loss((["retrieval"],), hierarchy_inputs)
        assert retrieval.calls == 0
        assert hierarchy.calls == 0


def test_composite_loss_is_public() -> None:
    assert neembed.ManifoldRetrievalHierarchyLoss is ManifoldRetrievalHierarchyLoss
    assert "ManifoldRetrievalHierarchyLoss" in neembed.__all__
