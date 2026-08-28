"""Opt-in v0.7 retrieval-objective acceptance against the real third-party stack."""

from __future__ import annotations

import math
import os

import pytest
import torch

from neembed import (
    ManifoldDistanceMSELoss,
    ManifoldGradedCorpusRetrievalEvaluator,
    ManifoldMarginMSELoss,
    ManifoldSentenceTransformer,
    ManifoldSymmetricMultipleNegativesRankingLoss,
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
ANCHORS = ("Shiba Inu", "Siamese cat")
POSITIVES = ("dog", "cat")
NEGATIVES = ("airplane", "volcano")
QUERY_IDS = ("q-dog", "q-cat")
CORPUS_IDS = ("dog", "cat", "wolf", "tiger", "bird")
CORPUS = CORPUS_IDS
GRADED_RELEVANCE = {
    "q-dog": {"dog": 3.0, "wolf": 1.0},
    "q-cat": {"cat": 3.0, "tiger": 2.0},
}


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
def test_v07_real_stack_objectives_and_graded_evaluation(manifold_name: str) -> None:
    model = _make_model(manifold_name)

    _assert_finite_backward(
        model,
        ManifoldTripletLoss(model=model, margin=0.1)(ANCHORS, POSITIVES, NEGATIVES),
    )
    _assert_finite_backward(
        model,
        ManifoldMarginMSELoss(model=model)(
            ANCHORS,
            POSITIVES,
            NEGATIVES,
            (0.2, 0.1),
        ),
    )
    _assert_finite_backward(
        model,
        ManifoldDistanceMSELoss(model=model)(ANCHORS, POSITIVES, (0.5, 0.5)),
    )
    _assert_finite_backward(
        model,
        ManifoldSymmetricMultipleNegativesRankingLoss(
            model=model,
            temperature=0.1,
        )(ANCHORS, POSITIVES, NEGATIVES),
    )

    evaluator = ManifoldGradedCorpusRetrievalEvaluator(
        model=model,
        query_ids=QUERY_IDS,
        queries=ANCHORS,
        corpus_ids=CORPUS_IDS,
        corpus=CORPUS,
        graded_relevance=GRADED_RELEVANCE,
        recall_at_k=(1, 3),
        ndcg_at_k=(1, 3),
        query_chunk_size=1,
        corpus_chunk_size=2,
    )
    metrics = evaluator()
    assert set(metrics) == {
        "mrr",
        "recall_at_1",
        "recall_at_3",
        "ndcg_at_1",
        "ndcg_at_3",
    }
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in metrics.values())
    assert metrics["recall_at_1"] <= metrics["recall_at_3"]
