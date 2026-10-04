"""Opt-in v0.10 product acceptance against actual third-party dependencies."""

from __future__ import annotations

import math
import os
from pathlib import Path
import runpy

import geoopt
import pytest
import torch

from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldHierarchyEvaluator,
    ManifoldHierarchyTripletLoss,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldRetrievalHierarchyLoss,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
    exact_corpus_search,
)
from neembed.manifolds import get_manifold


pytestmark = [
    pytest.mark.real_stack,
    pytest.mark.skipif(
        os.environ.get("NEEMBED_REAL_STACK") != "1",
        reason="set NEEMBED_REAL_STACK=1 to run real dependency tests",
    ),
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERIES = ("dog", "cat")
POSITIVES = ("animal", "feline")
CORPUS = ("animal", "feline", "vehicle")
PRODUCT_CASES = [
    pytest.param([
        {"name": "hierarchy", "manifold": "poincare", "intrinsic_dim": 2,
         "curvature": 0.5, "scale": 1.5},
        {"name": "residual", "manifold": "euclidean", "intrinsic_dim": 2,
         "scale": 0.75},
    ], torch.float32, id="poincare_euclidean"),
    pytest.param([
        {"name": "hierarchy", "manifold": "lorentz", "intrinsic_dim": 2,
         "curvature": 0.5, "scale": 1.5},
        {"name": "semantic", "manifold": "sphere_projection", "intrinsic_dim": 2,
         "sectional_curvature": 0.25, "scale": 0.75},
        {"name": "residual", "manifold": "euclidean", "intrinsic_dim": 2},
    ], torch.float64, id="lorentz_sphere_euclidean"),
    pytest.param([
        {"name": "negative", "manifold": "stereographic", "intrinsic_dim": 2,
         "sectional_curvature": -0.5, "scale": 0.75},
        {"name": "flat", "manifold": "stereographic", "intrinsic_dim": 2,
         "sectional_curvature": 0.0, "scale": 1.25},
        {"name": "positive", "manifold": "stereographic", "intrinsic_dim": 2,
         "sectional_curvature": 0.25, "scale": 1.5},
    ], torch.float64, id="signed_stereographic"),
]


def _assert_active_backward(model, loss):
    assert loss.ndim == 0 and torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    loss.backward()
    for parameter in (model.projection.weight, *model.encoder.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.projection.weight.grad is not None
    assert torch.count_nonzero(model.projection.weight.grad) > 0
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in model.encoder.parameters()
    )


def _independent_product(model, texts):
    """Map the same encoder features with unscaled Geoopt origin maps."""
    with torch.no_grad():
        features = model.encoder.preprocess(list(texts))
        tangent = model.projection(model.encoder(features)["sentence_embedding"])
        chunks = tangent.split(
            [c.intrinsic_dim for c in model.product_config.components], dim=-1,
        )
        components, points = [], []
        for config, chunk in zip(model.product_config.components, chunks):
            base = get_manifold(
                config.manifold, config.curvature or 1.0,
                sectional_curvature=config.sectional_curvature,
            ).to(dtype=model.manifold.dtype)
            chunk = chunk.to(dtype=model.manifold.dtype)
            if config.manifold == "lorentz":
                chunk = torch.cat((torch.zeros_like(chunk[:, :1]), chunk), dim=-1)
            points.append(chunk if config.manifold == "euclidean" else base.expmap0(chunk))
            scaled = geoopt.Scaled(base, config.scale).to(dtype=model.manifold.dtype)
            components.append((scaled, config.ambient_dim))
        independent = geoopt.ProductManifold(*components)
        return independent, independent.pack_point(*points)


@pytest.mark.parametrize(("config", "expected_dtype"), PRODUCT_CASES)
def test_v10_real_stack_product_scaling_training_and_checkpoint(
    config, expected_dtype, tmp_path,
):
    torch.manual_seed(79)
    model = ManifoldSentenceTransformer(MODEL_NAME, product_config=config).cpu()
    model.eval()
    points = model.encode(CORPUS, convert_to_tensor=True)
    assert points.shape == (len(CORPUS), model.product_config.ambient_dim)
    assert points.dtype == expected_dtype
    assert torch.isfinite(points).all()
    assert model.manifold.check_point_on_manifold(points)

    independent, expected_points = _independent_product(model, CORPUS)
    torch.testing.assert_close(points, expected_points)
    distances = model.distance(points[:, None], points[None])
    torch.testing.assert_close(distances, independent.dist(points[:, None], points[None]))
    diagnostics = model.product_distance_diagnostics(points[0], points[1])
    torch.testing.assert_close(diagnostics["total_distance"], distances[0, 1])
    for i, component in enumerate(model.product_config.components):
        torch.testing.assert_close(
            diagnostics["component_distances"][component.name],
            independent.manifolds[i].dist(
                independent.take_submanifold_value(points[0], i),
                independent.take_submanifold_value(points[1], i),
            ),
        )
    assert not any(p.requires_grad for p in model.manifold.parameters())

    retrieval = ManifoldMultipleNegativesRankingLoss(model, temperature=0.1)
    _assert_active_backward(model, retrieval(QUERIES, POSITIVES))
    hierarchy_inputs = (("animal", "animal"), QUERIES, ("vehicle", "volcano"))
    has_hierarchy = config[0]["name"] == "hierarchy"
    if has_hierarchy:
        hierarchy = ManifoldHierarchyTripletLoss(
            model, component="hierarchy", margin=10.0, radial_margin=10.0,
        )
        _assert_active_backward(model, hierarchy(*hierarchy_inputs))
        # A pure component objective has no direct gradient on other projection rows.
        assert torch.count_nonzero(model.projection.weight.grad[2:]) == 0
        loss = ManifoldRetrievalHierarchyLoss(retrieval, hierarchy, hierarchy_weight=0.3)
        batches = [((QUERIES, POSITIVES), hierarchy_inputs)]
    else:
        loss, batches = retrieval, [(QUERIES, POSITIVES)]
    trainer = ManifoldTrainer(model, loss, learning_rate=1e-4, verbose=False)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    before = model.projection.weight.detach().clone()
    assert all(math.isfinite(value) for value in trainer.fit(batches, epochs=1))
    assert not torch.equal(before, model.projection.weight)

    evaluator_kwargs = dict(
        query_ids=("dog", "cat"), queries=QUERIES,
        corpus_ids=("animal", "feline", "vehicle"), corpus=CORPUS,
        relevance={"dog": ("animal",), "cat": ("feline",)},
        recall_at_k=(1, 3), query_chunk_size=1, corpus_chunk_size=2,
    )
    metrics = ManifoldCorpusRetrievalEvaluator(model=model, **evaluator_kwargs)()
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in metrics.values())
    if has_hierarchy:
        hierarchy_metrics = ManifoldHierarchyEvaluator(
            model=model, node_ids=("animal", "dog", "cat"), texts=("animal", *QUERIES),
            parent_child_edges=(("animal", "dog"), ("animal", "cat")),
            depths={"animal": 0, "dog": 1, "cat": 1}, component="hierarchy",
        )()
        assert all(math.isfinite(value) for value in hierarchy_metrics.values())

    before = model.encode(CORPUS, convert_to_tensor=True)
    rankings = exact_corpus_search(model, QUERIES, CORPUS, top_k=3)
    report = model.product_distance_diagnostics(before[0], before[1:])
    model.save_pretrained(tmp_path / "product")
    loaded = ManifoldSentenceTransformer.from_pretrained(tmp_path / "product").cpu()
    assert loaded.product_config == model.product_config
    after = loaded.encode(CORPUS, convert_to_tensor=True)
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(
        loaded.distance(after[:, None], after[None]),
        model.distance(before[:, None], before[None]),
    )
    assert exact_corpus_search(loaded, QUERIES, CORPUS, top_k=3) == rankings
    assert ManifoldCorpusRetrievalEvaluator(model=loaded, **evaluator_kwargs)() == metrics
    restored = loaded.product_distance_diagnostics(after[0], after[1:])
    torch.testing.assert_close(restored["total_distance"], report["total_distance"])
    for name in report["component_distances"]:
        torch.testing.assert_close(
            restored["component_distances"][name], report["component_distances"][name],
        )
    if has_hierarchy:
        ManifoldHierarchyTripletLoss(loaded, component="hierarchy")


def test_v10_real_stack_regression_example(tmp_path):
    namespace = runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / "v10_mixed_curvature_workflow.py"),
    )
    result = namespace["run_example"](
        MODEL_NAME, epochs=1, output_path=tmp_path / "example",
    )
    assert result["projection_updated"]
    assert result["after"] == result["save_load"]["loaded_evaluation"]
    assert all(result["save_load"][key] for key in (
        "embeddings_match", "distances_match", "component_diagnostics_match",
    ))
