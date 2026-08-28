"""Deterministic v0.7 retrieval-objective comparison workflow.

Run from the repository root::

    python examples/v07_objective_comparison.py

The example keeps the encoder architecture, Poincare manifold, local data, seed,
and exact retrieval evaluation fixed while swapping four objectives: MNRL,
Triplet, MarginMSE, and DistanceMSE. It reports engineering regression
diagnostics only; the tiny data are not a benchmark and the output must not be
interpreted as evidence that one objective is better than another.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from neembed import (
    ManifoldDistanceMSELoss,
    ManifoldGradedCorpusRetrievalEvaluator,
    ManifoldMarginMSELoss,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
    ManifoldTripletLoss,
)


QUERY_IDS = ("query-dog", "query-cat", "query-bird")
QUERIES = ("Shiba Inu", "Siamese cat", "sparrow")

TRAIN_POSITIVES = ("dog", "cat", "bird")
TRAIN_NEGATIVES = ("vehicle", "building", "tool")
TEACHER_MARGINS = (1.0, 1.0, 1.0)
TARGET_DISTANCES = (0.5, 0.5, 0.5)

CORPUS_IDS = (
    "dog",
    "canine",
    "cat",
    "feline",
    "bird",
    "avian",
    "vehicle",
    "building",
    "tool",
)
CORPUS = CORPUS_IDS
GRADED_RELEVANCE = {
    "query-dog": {"dog": 3.0, "canine": 1.0},
    "query-cat": {"cat": 3.0, "feline": 1.0},
    "query-bird": {"bird": 3.0, "avian": 1.0},
}

OBJECTIVE_NAMES = ("mnrl", "triplet", "margin_mse", "distance_mse")


def _build_model(
    model_name_or_path: str,
    *,
    seed: int,
    embedding_dim: int,
) -> ManifoldSentenceTransformer:
    # Reset before every objective so all variants start from identical model
    # initialization when the encoder implementation is deterministic under
    # torch.manual_seed().
    torch.manual_seed(seed)
    return ManifoldSentenceTransformer(
        model_name_or_path,
        manifold="poincare",
        embedding_dim=embedding_dim,
        curvature=1.0,
    )


def _evaluate(
    model: ManifoldSentenceTransformer,
    *,
    query_chunk_size: int,
    corpus_chunk_size: int,
) -> dict[str, float]:
    evaluator = ManifoldGradedCorpusRetrievalEvaluator(
        model=model,
        query_ids=QUERY_IDS,
        queries=QUERIES,
        corpus_ids=CORPUS_IDS,
        corpus=CORPUS,
        graded_relevance=GRADED_RELEVANCE,
        recall_at_k=(1, 3),
        ndcg_at_k=(1, 3),
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )
    return evaluator()


def _objective_and_batch(
    name: str,
    model: ManifoldSentenceTransformer,
):
    if name == "mnrl":
        return (
            ManifoldMultipleNegativesRankingLoss(model, temperature=0.1),
            (QUERIES, TRAIN_POSITIVES, TRAIN_NEGATIVES),
        )
    if name == "triplet":
        return (
            ManifoldTripletLoss(model, margin=0.1),
            (QUERIES, TRAIN_POSITIVES, TRAIN_NEGATIVES),
        )
    if name == "margin_mse":
        return (
            ManifoldMarginMSELoss(model),
            (QUERIES, TRAIN_POSITIVES, TRAIN_NEGATIVES, TEACHER_MARGINS),
        )
    if name == "distance_mse":
        return (
            ManifoldDistanceMSELoss(model),
            (QUERIES, TRAIN_POSITIVES, TARGET_DISTANCES),
        )
    raise ValueError(f"unknown objective: {name}")


def _run_objective(
    name: str,
    model_name_or_path: str,
    *,
    epochs: int,
    seed: int,
    embedding_dim: int,
    learning_rate: float,
    query_chunk_size: int,
    corpus_chunk_size: int,
) -> dict[str, Any]:
    model = _build_model(
        model_name_or_path,
        seed=seed,
        embedding_dim=embedding_dim,
    )
    before = _evaluate(
        model,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )

    loss, batch = _objective_and_batch(name, model)
    trainer = ManifoldTrainer(
        model,
        loss,
        learning_rate=learning_rate,
        verbose=False,
    )
    history = trainer.fit([batch], epochs=epochs)

    after = _evaluate(
        model,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )
    return {
        "before": before,
        "final_training_loss": float(history[-1]),
        "after": after,
    }


def run_comparison(
    model_name_or_path: str,
    *,
    epochs: int = 1,
    seed: int = 37,
    embedding_dim: int = 8,
    learning_rate: float = 1e-4,
    query_chunk_size: int = 2,
    corpus_chunk_size: int = 3,
) -> dict[str, Any]:
    """Run all objective variants under one fixed Poincare setup."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    for value, name in (
        (query_chunk_size, "query_chunk_size"),
        (corpus_chunk_size, "corpus_chunk_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    objectives = {
        name: _run_objective(
            name,
            model_name_or_path,
            epochs=epochs,
            seed=seed,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            query_chunk_size=query_chunk_size,
            corpus_chunk_size=corpus_chunk_size,
        )
        for name in OBJECTIVE_NAMES
    }
    return {
        "manifold": "poincare",
        "seed": seed,
        "objectives": objectives,
    }


def _validate_regression(results: dict[str, Any]) -> None:
    if results["manifold"] != "poincare":
        raise RuntimeError("v0.7 objective comparison must use Poincare geometry")

    objective_results = results["objectives"]
    if tuple(objective_results) != OBJECTIVE_NAMES:
        raise RuntimeError("objective comparison output is incomplete")

    reference_before = objective_results[OBJECTIVE_NAMES[0]]["before"]
    for name in OBJECTIVE_NAMES:
        diagnostics = objective_results[name]
        if diagnostics["before"] != reference_before:
            raise RuntimeError("objective variants did not start from identical diagnostics")
        if not math.isfinite(diagnostics["final_training_loss"]):
            raise RuntimeError(f"{name} training loss became non-finite")
        for stage_name in ("before", "after"):
            metrics = diagnostics[stage_name]
            if any(not math.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"{name} {stage_name} metrics became non-finite")
            if any(not 0.0 <= value <= 1.0 for value in metrics.values()):
                raise RuntimeError(f"{name} {stage_name} metric left [0, 1]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model name or local path.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    results = run_comparison(args.model, epochs=args.epochs)
    _validate_regression(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(
        "Diagnostics only: this tiny run is an engineering regression reference, "
        "not a benchmark or objective-superiority claim."
    )


if __name__ == "__main__":
    main()
