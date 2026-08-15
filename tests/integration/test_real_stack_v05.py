"""Opt-in v0.5 retrieval acceptance against the real third-party stack."""

from __future__ import annotations

import math
import os

import geoopt
import pytest
import torch

from neembed import (
    ManifoldEmbeddingEvaluator,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeAssignmentEvaluator,
    ManifoldPrototypes,
    ManifoldSentenceTransformer,
)


pytestmark = [
    pytest.mark.real_stack,
    pytest.mark.skipif(
        os.environ.get("NEEMBED_REAL_STACK") != "1",
        reason="set NEEMBED_REAL_STACK=1 to run real dependency tests",
    ),
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ANCHORS = ("Shiba Inu", "Siamese cat", "sparrow")
POSITIVES = ("dog", "cat", "bird")
EXPLICIT_NEGATIVES = ("cat", "dog", "dog")


def _make_model(manifold_name: str) -> ManifoldSentenceTransformer:
    torch.manual_seed(29)
    return ManifoldSentenceTransformer(
        MODEL_NAME,
        manifold=manifold_name,
        embedding_dim=16,
        curvature=1.0,
    )


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_v05_real_stack_retrieval_and_prototype_contracts(manifold_name: str) -> None:
    model = _make_model(manifold_name)
    ranking_loss = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)

    # The original positive-pair contract and the v0.5 explicit-negative contract
    # must both run against the real Sentence Transformer / Geoopt stack. In eval
    # mode, adding finite explicit candidates strictly increases the same
    # cross-entropy objective because the aligned target numerator is unchanged.
    model.eval()
    with torch.no_grad():
        pair_loss = ranking_loss(ANCHORS, POSITIVES)
        hard_negative_loss = ranking_loss(
            ANCHORS,
            POSITIVES,
            EXPLICIT_NEGATIVES,
        )

    assert torch.isfinite(pair_loss)
    assert torch.isfinite(hard_negative_loss)
    assert float(hard_negative_loss) > float(pair_loss)

    # The three-sequence path must remain differentiable through the real model.
    model.train()
    model.zero_grad(set_to_none=True)
    training_loss = ranking_loss(ANCHORS, POSITIVES, EXPLICIT_NEGATIVES)
    training_loss.backward()
    projection_grad = model.projection.weight.grad

    assert torch.isfinite(training_loss)
    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert torch.count_nonzero(projection_grad) > 0

    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=ANCHORS,
        positives=POSITIVES,
        recall_at_k=(1, 2, 3),
    )
    retrieval = evaluator()

    assert retrieval["retrieval_accuracy"] == pytest.approx(retrieval["recall_at_1"])
    for key in ("recall_at_1", "recall_at_2", "recall_at_3", "mrr"):
        assert 0.0 <= retrieval[key] <= 1.0
    assert retrieval["recall_at_1"] <= retrieval["recall_at_2"]
    assert retrieval["recall_at_2"] <= retrieval["recall_at_3"]
    assert math.isfinite(retrieval["mean_positive_distance"])
    assert math.isfinite(retrieval["mean_negative_distance"])

    ranked = model.rank(
        "Shiba Inu",
        ("dog", "cat", "bird", "vehicle"),
        top_k=3,
    )
    ranked_distances = [item["distance"] for item in ranked]

    assert len(ranked) == 3
    assert ranked_distances == sorted(ranked_distances)
    assert all(math.isfinite(distance) and distance >= 0.0 for distance in ranked_distances)

    # Optimize true manifold-valued prototypes with the supported Geoopt path,
    # then evaluate nearest-prototype assignments through the public v0.5 API.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    target_embeddings = model.encode(("dog", "cat"), convert_to_tensor=True)
    prototypes = ManifoldPrototypes(model, num_prototypes=2, init_std=0.05)
    prototypes_before = prototypes.prototypes.detach().clone()
    optimizer = geoopt.optim.RiemannianAdam(
        prototypes.parameters(),
        lr=1e-2,
        stabilize=1,
    )

    optimizer.zero_grad()
    prototype_loss = prototypes(target_embeddings).diagonal().mean()
    prototype_loss.backward()
    optimizer.step()

    assert torch.isfinite(prototype_loss)
    assert not torch.equal(prototypes_before, prototypes.prototypes.detach())
    assert model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-5,
        rtol=1e-5,
    )

    assignment_evaluator = ManifoldPrototypeAssignmentEvaluator(
        model=model,
        prototypes=prototypes,
        prototype_ids=("dog", "cat"),
        sentences=("dog", "cat"),
        expected_prototype_ids=("dog", "cat"),
    )
    assignment = assignment_evaluator()

    assert 0.0 <= assignment["assignment_accuracy"] <= 1.0
    assert math.isfinite(assignment["mean_assigned_prototype_distance"])
    assert assignment["mean_assigned_prototype_distance"] >= 0.0
