"""Compact v0.6 exact retrieval and offline hard-negative workflow.

Run from the repository root::

    python examples/v06_exact_retrieval_workflow.py

The example uses only the tiny text data defined below. It is an engineering
reference that reports inspectable diagnostics; it is not a benchmark or a
claim that training improves retrieval quality on this tiny corpus.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
    exact_corpus_search,
    mine_hard_negatives,
)


QUERY_IDS = ("query-dog", "query-cat", "query-bird")
QUERIES = ("Shiba Inu", "Siamese cat", "sparrow")

CORPUS_IDS = ("dog", "cat", "bird", "vehicle", "aircraft")
CORPUS = ("dog", "cat", "bird", "vehicle", "airplane")

RELEVANCE = {
    "query-dog": ("dog",),
    "query-cat": ("cat",),
    "query-bird": ("bird",),
}
TRAIN_POSITIVES = ("dog", "cat", "bird")
TRAIN_POSITIVE_IDS = tuple(
    dict.fromkeys(corpus_id for relevant_ids in RELEVANCE.values() for corpus_id in relevant_ids)
)


def _retrieval_diagnostics(
    model: ManifoldSentenceTransformer,
    *,
    top_k: int,
    query_chunk_size: int,
    corpus_chunk_size: int,
) -> dict[str, Any]:
    search_results = exact_corpus_search(
        model,
        QUERIES,
        CORPUS,
        top_k=top_k,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )
    recall_at_k = (1,) if top_k == 1 else (1, top_k)
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=QUERY_IDS,
        queries=QUERIES,
        corpus_ids=CORPUS_IDS,
        corpus=CORPUS,
        relevance=RELEVANCE,
        recall_at_k=recall_at_k,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )
    return {
        "search_results": search_results,
        "retrieval": evaluator(),
    }


def run_example(
    model_name_or_path: str,
    *,
    epochs: int = 1,
    seed: int = 29,
    embedding_dim: int = 8,
    learning_rate: float = 1e-4,
    top_k: int = 3,
    query_chunk_size: int = 2,
    corpus_chunk_size: int = 3,
) -> dict[str, Any]:
    """Run the v0.6 reference workflow and return diagnostics."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    if top_k <= 0 or top_k > len(CORPUS):
        raise ValueError("top_k must be between 1 and the corpus size")

    torch.manual_seed(seed)
    model = ManifoldSentenceTransformer(
        model_name_or_path,
        manifold="poincare",
        embedding_dim=embedding_dim,
        curvature=1.0,
    )

    before = _retrieval_diagnostics(
        model,
        top_k=top_k,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )

    batch_positive_exclusions = {
        query_id: tuple(
            corpus_id
            for corpus_id in TRAIN_POSITIVE_IDS
            if corpus_id not in RELEVANCE[query_id]
        )
        for query_id in QUERY_IDS
    }
    mined = mine_hard_negatives(
        model,
        QUERIES,
        CORPUS,
        query_ids=QUERY_IDS,
        corpus_ids=CORPUS_IDS,
        positive_corpus_ids=RELEVANCE,
        excluded_corpus_ids=batch_positive_exclusions,
        num_negatives=1,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )
    mined_negative_texts = tuple(items[0]["candidate"] for items in mined)

    loss = ManifoldMultipleNegativesRankingLoss(model, temperature=0.1)
    trainer = ManifoldTrainer(
        model,
        loss,
        learning_rate=learning_rate,
        verbose=False,
    )
    training_history = trainer.fit(
        [(QUERIES, TRAIN_POSITIVES, mined_negative_texts)],
        epochs=epochs,
    )

    after = _retrieval_diagnostics(
        model,
        top_k=top_k,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )

    return {
        "manifold": model.manifold_name,
        "seed": seed,
        "before": before,
        "mined_negatives": [items[0] for items in mined],
        "final_training_loss": float(training_history[-1]),
        "after": after,
    }


def _validate_regression(results: dict[str, Any]) -> None:
    if results["manifold"] != "poincare":
        raise RuntimeError("v0.6 reference example must use Poincare geometry")
    if not math.isfinite(results["final_training_loss"]):
        raise RuntimeError("training loss became non-finite")

    for stage_name in ("before", "after"):
        stage = results[stage_name]
        if any(
            not math.isfinite(result["distance"])
            for ranking in stage["search_results"]
            for result in ranking
        ):
            raise RuntimeError(f"{stage_name} search produced a non-finite distance")
        if any(not math.isfinite(value) for value in stage["retrieval"].values()):
            raise RuntimeError(f"{stage_name} retrieval produced a non-finite metric")

    for query_id, mined in zip(QUERY_IDS, results["mined_negatives"], strict=True):
        if not math.isfinite(mined["distance"]):
            raise RuntimeError("hard-negative mining produced a non-finite distance")
        if mined["corpus_id"] in RELEVANCE[query_id]:
            raise RuntimeError("hard-negative mining returned a known positive")
        if mined["corpus_id"] in TRAIN_POSITIVE_IDS:
            raise RuntimeError("hard-negative mining returned another batch positive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model name or local path.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    results = run_example(args.model, epochs=args.epochs)
    _validate_regression(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    print("Diagnostics only: this tiny run is not a retrieval-quality claim.")


if __name__ == "__main__":
    main()
