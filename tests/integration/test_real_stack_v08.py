"""Opt-in v0.8 hierarchy acceptance against the real third-party stack."""

from __future__ import annotations

import math
import os

import pytest
import torch

from neembed import (
    ManifoldDepthLoss,
    ManifoldHierarchyEvaluator,
    ManifoldHierarchyTripletLoss,
    ManifoldRadialOrderLoss,
    ManifoldRetrievalHierarchyLoss,
    ManifoldSentenceTransformer,
    ManifoldTripletLoss,
)


pytestmark = [
    pytest.mark.real_stack,
    pytest.mark.skipif(
        os.environ.get("NEEMBED_REAL_STACK") != "1",
        reason="set NEEMBED_REAL_STACK=1 to run real dependency tests",
    ),
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NODE_IDS = ("root", "animal", "dog", "cat")
TEXTS = ("root concept", "animal", "dog", "cat")
EDGES = (("root", "animal"), ("animal", "dog"), ("animal", "cat"))
DEPTHS = {"root": 0, "animal": 1, "dog": 2, "cat": 2}
PARENTS = ("root concept", "animal", "animal")
CHILDREN = ("animal", "dog", "cat")
UNRELATED = ("vehicle", "volcano", "airplane")
RETRIEVAL = (("dog", "cat"), ("animal", "animal"), ("vehicle", "volcano"))


def _make_model(manifold_name: str) -> ManifoldSentenceTransformer:
    torch.manual_seed(47)
    return ManifoldSentenceTransformer(
        MODEL_NAME,
        manifold=manifold_name,
        embedding_dim=16,
        curvature=1.0,
    )


def _assert_finite_backward(model: ManifoldSentenceTransformer, loss: torch.Tensor) -> None:
    assert loss.ndim == 0
    assert math.isfinite(float(loss.detach()))
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_v08_real_stack_hierarchy_losses_and_evaluator(manifold_name: str) -> None:
    model = _make_model(manifold_name)

    _assert_finite_backward(
        model,
        ManifoldRadialOrderLoss(model=model, margin=0.05)(PARENTS, CHILDREN),
    )
    _assert_finite_backward(
        model,
        ManifoldDepthLoss(model=model, radial_scale=0.5)(
            TEXTS,
            tuple(DEPTHS[node_id] for node_id in NODE_IDS),
        ),
    )
    _assert_finite_backward(
        model,
        ManifoldHierarchyTripletLoss(
            model=model,
            margin=0.05,
            radial_margin=0.05,
            radial_weight=0.5,
        )(PARENTS, CHILDREN, UNRELATED),
    )

    retrieval_loss = ManifoldTripletLoss(model=model, margin=0.05)
    hierarchy_loss = ManifoldRadialOrderLoss(model=model, margin=0.05)
    composite = ManifoldRetrievalHierarchyLoss(
        retrieval_loss,
        hierarchy_loss,
        hierarchy_weight=0.25,
    )
    _assert_finite_backward(
        model,
        composite(RETRIEVAL, (PARENTS, CHILDREN)),
    )

    evaluator = ManifoldHierarchyEvaluator(
        model=model,
        node_ids=NODE_IDS,
        texts=TEXTS,
        parent_child_edges=EDGES,
        depths=DEPTHS,
        contract="tree",
    )
    metrics = evaluator()
    assert set(metrics) == {
        "parent_child_radial_order_accuracy",
        "mean_radial_order_violation",
        "depth_radius_spearman",
    }
    assert all(math.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["parent_child_radial_order_accuracy"] <= 1.0
    assert metrics["mean_radial_order_violation"] >= 0.0
    assert -1.0 <= metrics["depth_radius_spearman"] <= 1.0
