"""Opt-in v0.6 exact retrieval acceptance against the real third-party stack."""

from __future__ import annotations

import math
import os

import pytest
import torch

from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldSentenceTransformer,
    exact_corpus_search,
    mine_hard_negatives,
)


pytestmark = [
    pytest.mark.real_stack,
    pytest.mark.skipif(
        os.environ.get("NEEMBED_REAL_STACK") != "1",
        reason="set NEEMBED_REAL_STACK=1 to run real dependency tests",
    ),
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERY_IDS = ("q-dog", "q-cat")
QUERIES = ("Shiba Inu", "Siamese cat")
CORPUS_IDS = ("dog", "cat", "wolf", "tiger", "bird")
CORPUS = CORPUS_IDS
RELEVANCE = {
    "q-dog": ("dog",),
    "q-cat": ("cat", "tiger"),
}


def _make_model(manifold_name: str) -> ManifoldSentenceTransformer:
    torch.manual_seed(41)
    return ManifoldSentenceTransformer(
        MODEL_NAME,
        manifold=manifold_name,
        embedding_dim=16,
        curvature=1.0,
    )


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_v06_real_stack_exact_corpus_evaluation_and_mining(manifold_name: str) -> None:
    model = _make_model(manifold_name)

    ranked = exact_corpus_search(
        model,
        QUERIES,
        CORPUS,
        top_k=3,
        query_chunk_size=1,
        corpus_chunk_size=2,
    )
    assert len(ranked) == len(QUERIES)
    assert all(len(results) == 3 for results in ranked)
    for results in ranked:
        distances = [float(item["distance"]) for item in results]
        assert distances == sorted(distances)
        assert all(math.isfinite(distance) and distance >= 0.0 for distance in distances)

    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=QUERY_IDS,
        queries=QUERIES,
        corpus_ids=CORPUS_IDS,
        corpus=CORPUS,
        relevance=RELEVANCE,
        recall_at_k=(1, 3),
        query_chunk_size=1,
        corpus_chunk_size=2,
    )
    metrics = evaluator()
    assert set(metrics) == {"mrr", "recall_at_1", "recall_at_3"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert metrics["recall_at_1"] <= metrics["recall_at_3"]

    batch_positive_ids = set().union(*RELEVANCE.values())
    extra_exclusions = {
        query_id: tuple(sorted(batch_positive_ids - set(RELEVANCE[query_id])))
        for query_id in QUERY_IDS
    }
    mined = mine_hard_negatives(
        model,
        QUERIES,
        CORPUS,
        query_ids=QUERY_IDS,
        corpus_ids=CORPUS_IDS,
        positive_corpus_ids=RELEVANCE,
        excluded_corpus_ids=extra_exclusions,
        num_negatives=1,
        query_chunk_size=1,
        corpus_chunk_size=2,
    )

    assert len(mined) == len(QUERIES)
    for items in mined:
        assert len(items) == 1
        item = items[0]
        assert item["corpus_id"] not in batch_positive_ids
        assert item["candidate"] == CORPUS[int(item["index"])]
        distance = float(item["distance"])
        assert math.isfinite(distance)
        assert distance >= 0.0
