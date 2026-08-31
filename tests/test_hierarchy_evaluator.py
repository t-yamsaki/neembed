"""Tests for hierarchy structure evaluation metrics."""

import math

import pytest
import torch
from torch import nn

import neembed
from neembed import ManifoldHierarchyEvaluator
from neembed.manifolds import get_manifold


class _ToyHierarchyModel(nn.Module):
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

    def encode(
        self,
        texts: tuple[str, ...],
        *,
        convert_to_tensor: bool,
    ) -> torch.Tensor:
        assert convert_to_tensor
        self.eval()
        spatial = torch.stack([self.tangents[self.index[text]] for text in texts])
        if self.manifold_name == "lorentz":
            spatial = torch.cat(
                (spatial.new_zeros((len(texts), 1)), spatial),
                dim=-1,
            )
        return self.manifold.expmap0(spatial)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_evaluator_perfect_structure(manifold_name: str) -> None:
    labels = ["root", "child", "grandchild"]
    model = _ToyHierarchyModel(
        manifold_name,
        labels,
        [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
    )
    evaluator = ManifoldHierarchyEvaluator(
        model=model,
        node_ids=labels,
        texts=labels,
        parent_child_edges=[("root", "child"), ("child", "grandchild")],
        depths={"root": 0, "child": 1, "grandchild": 2},
        contract="tree",
    )

    metrics = evaluator()

    assert metrics["parent_child_radial_order_accuracy"] == pytest.approx(1.0)
    assert metrics["mean_radial_order_violation"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["depth_radius_spearman"] == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_evaluator_detects_reversed_structure(manifold_name: str) -> None:
    labels = ["root", "child", "grandchild"]
    model = _ToyHierarchyModel(
        manifold_name,
        labels,
        [[0.4, 0.0], [0.2, 0.0], [0.0, 0.0]],
    )
    evaluator = ManifoldHierarchyEvaluator(
        model=model,
        node_ids=labels,
        texts=labels,
        parent_child_edges=[("root", "child"), ("child", "grandchild")],
        depths={"root": 0, "child": 1, "grandchild": 2},
    )

    metrics = evaluator()

    assert metrics["parent_child_radial_order_accuracy"] == pytest.approx(0.0)
    assert metrics["mean_radial_order_violation"] > 0.0
    assert metrics["depth_radius_spearman"] == pytest.approx(-1.0)
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_evaluator_has_stable_tie_and_root_behavior(
    manifold_name: str,
) -> None:
    labels = ["root", "child"]
    model = _ToyHierarchyModel(
        manifold_name,
        labels,
        [[0.0, 0.0], [0.0, 0.0]],
    )
    evaluator = ManifoldHierarchyEvaluator(
        model=model,
        node_ids=labels,
        texts=labels,
        parent_child_edges=[("root", "child")],
        depths={"root": 0, "child": 1},
    )

    metrics = evaluator()

    assert metrics["parent_child_radial_order_accuracy"] == pytest.approx(0.0)
    assert metrics["mean_radial_order_violation"] == pytest.approx(0.0)
    assert metrics["depth_radius_spearman"] == pytest.approx(0.0)
    assert all(math.isfinite(value) for value in metrics.values())


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_hierarchy_evaluator_handles_repeated_depth_ties(manifold_name: str) -> None:
    labels = ["root", "left", "right"]
    model = _ToyHierarchyModel(
        manifold_name,
        labels,
        [[0.0, 0.0], [0.25, 0.0], [0.0, 0.25]],
    )
    evaluator = ManifoldHierarchyEvaluator(
        model=model,
        node_ids=labels,
        texts=labels,
        parent_child_edges=[("root", "left"), ("root", "right")],
        depths={"root": 0, "left": 1, "right": 1},
    )

    metrics = evaluator()

    assert metrics["parent_child_radial_order_accuracy"] == pytest.approx(1.0)
    assert metrics["mean_radial_order_violation"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["depth_radius_spearman"] == pytest.approx(1.0)


def test_hierarchy_evaluator_is_no_grad_and_restores_mode() -> None:
    labels = ["root", "child"]
    model = _ToyHierarchyModel("poincare", labels, [[0.0, 0.0], [0.2, 0.0]])
    model.train()
    evaluator = ManifoldHierarchyEvaluator(
        model=model,
        node_ids=labels,
        texts=labels,
        parent_child_edges=[("root", "child")],
    )

    metrics = evaluator()

    assert model.training
    assert model.tangents.grad is None
    assert all(math.isfinite(value) for value in metrics.values())

    model.eval()
    evaluator()
    assert not model.training


def test_hierarchy_evaluator_reuses_hierarchy_validation() -> None:
    model = _ToyHierarchyModel(
        "poincare",
        ["a", "b"],
        [[0.0, 0.0], [0.2, 0.0]],
    )

    with pytest.raises(ValueError, match="same number"):
        ManifoldHierarchyEvaluator(
            model=model,
            node_ids=["a", "b"],
            texts=["a"],
            parent_child_edges=[("a", "b")],
        )
    with pytest.raises(ValueError, match="at least one parent-child edge"):
        ManifoldHierarchyEvaluator(
            model=model,
            node_ids=["a", "b"],
            texts=["a", "b"],
            parent_child_edges=[],
        )
    with pytest.raises(ValueError, match="acyclic"):
        ManifoldHierarchyEvaluator(
            model=model,
            node_ids=["a", "b"],
            texts=["a", "b"],
            parent_child_edges=[("a", "b"), ("b", "a")],
        )
    with pytest.raises(ValueError, match="increase along hierarchy"):
        ManifoldHierarchyEvaluator(
            model=model,
            node_ids=["a", "b"],
            texts=["a", "b"],
            parent_child_edges=[("a", "b")],
            depths={"a": 1, "b": 1},
        )


def test_hierarchy_evaluator_omits_depth_metric_without_depths() -> None:
    labels = ["root", "child"]
    model = _ToyHierarchyModel("poincare", labels, [[0.0, 0.0], [0.2, 0.0]])
    evaluator = ManifoldHierarchyEvaluator(
        model=model,
        node_ids=labels,
        texts=labels,
        parent_child_edges=[("root", "child")],
    )

    metrics = evaluator()

    assert "depth_radius_spearman" not in metrics


def test_hierarchy_evaluator_is_public() -> None:
    assert neembed.ManifoldHierarchyEvaluator is ManifoldHierarchyEvaluator
    assert "ManifoldHierarchyEvaluator" in neembed.__all__
