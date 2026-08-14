"""Tiny v0.5 retrieval and hard-negative regression workflow.

This is an engineering/regression example, not a benchmark or a claim that one
geometry or training recipe is generally superior. The user-facing reference
run uses Poincare geometry and a tiny public text corpus defined in this file.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import geoopt
import torch

from neembed import (
    ManifoldEmbeddingEvaluator,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeAssignmentEvaluator,
    ManifoldPrototypeHierarchyLoss,
    ManifoldPrototypes,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)


TRAIN_ANCHORS = ("Shiba Inu", "Siamese cat", "sparrow")
TRAIN_POSITIVES = ("dog", "cat", "bird")
HARD_NEGATIVES = ("cat", "dog", "dog")

EVAL_ANCHORS = ("Akita", "Persian cat", "robin")
EVAL_POSITIVES = ("dog", "cat", "bird")

RERANK_QUERY = "Shiba Inu"
RERANK_CANDIDATES = ("dog", "cat", "bird", "vehicle")

PROTOTYPE_IDS = ("animal", "dog", "cat", "bird")
PROTOTYPE_RELATIONS = (
    ("dog", "animal"),
    ("cat", "animal"),
    ("bird", "animal"),
)
PROTOTYPE_SENTENCES = (
    "living animal",
    "Shiba Inu",
    "Siamese cat",
    "sparrow",
)
PROTOTYPE_ASSIGNMENTS = ("animal", "dog", "cat", "bird")


def _initial_ranking_losses(
    model: ManifoldSentenceTransformer,
    loss: ManifoldMultipleNegativesRankingLoss,
) -> tuple[float, float]:
    """Measure paired-only and explicit-negative objectives on identical weights."""
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            paired_only = float(loss(TRAIN_ANCHORS, TRAIN_POSITIVES).detach())
            with_hard_negatives = float(
                loss(TRAIN_ANCHORS, TRAIN_POSITIVES, HARD_NEGATIVES).detach()
            )
    finally:
        model.train(was_training)
    return paired_only, with_hard_negatives


def _train_prototypes(
    model: ManifoldSentenceTransformer,
    prototypes: ManifoldPrototypes,
    *,
    epochs: int,
    learning_rate: float,
) -> float:
    """Fit only prototype points while leaving the retrieval model fixed."""
    hierarchy_loss = ManifoldPrototypeHierarchyLoss(
        model,
        prototypes,
        prototype_ids=PROTOTYPE_IDS,
        parent_relations=PROTOTYPE_RELATIONS,
        margin=0.1,
    )
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    try:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        optimizer = geoopt.optim.RiemannianAdam(
            prototypes.parameters(),
            lr=learning_rate,
            stabilize=1,
        )
        trainer = ManifoldTrainer(
            model,
            hierarchy_loss,
            optimizer=optimizer,
            verbose=False,
        )
        history = trainer.fit(
            [(PROTOTYPE_SENTENCES, PROTOTYPE_ASSIGNMENTS)],
            epochs=epochs,
        )
    finally:
        for parameter, original_requires_grad in zip(
            model.parameters(),
            requires_grad,
            strict=True,
        ):
            parameter.requires_grad_(original_requires_grad)
    return float(history[-1])


def run_example(
    model_name_or_path: str,
    *,
    epochs: int = 2,
    prototype_epochs: int = 5,
    seed: int = 23,
    embedding_dim: int = 8,
    learning_rate: float = 1e-4,
    prototype_learning_rate: float = 1e-2,
) -> dict[str, Any]:
    """Run the compact v0.5 reference workflow and return diagnostics."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if prototype_epochs <= 0:
        raise ValueError("prototype_epochs must be positive")
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    if prototype_learning_rate <= 0 or not math.isfinite(prototype_learning_rate):
        raise ValueError("prototype_learning_rate must be positive and finite")

    torch.manual_seed(seed)
    model = ManifoldSentenceTransformer(
        model_name_or_path,
        manifold="poincare",
        embedding_dim=embedding_dim,
        curvature=1.0,
    )
    ranking_loss = ManifoldMultipleNegativesRankingLoss(model, temperature=0.1)
    paired_only_loss, hard_negative_loss = _initial_ranking_losses(model, ranking_loss)

    trainer = ManifoldTrainer(
        model,
        ranking_loss,
        learning_rate=learning_rate,
        weight_decay=0.0,
        verbose=False,
    )
    training_history = trainer.fit(
        [(TRAIN_ANCHORS, TRAIN_POSITIVES, HARD_NEGATIVES)],
        epochs=epochs,
    )

    retrieval_metrics = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=EVAL_ANCHORS,
        positives=EVAL_POSITIVES,
        recall_at_k=(1, 2, 3),
    )()
    ranked_results = model.rank(
        RERANK_QUERY,
        RERANK_CANDIDATES,
        top_k=3,
    )

    embedding_pair = model.encode(
        EVAL_ANCHORS[:2],
        convert_to_tensor=True,
    )
    embedding_distance = float(model.distance(embedding_pair[0], embedding_pair[1]))

    prototypes = ManifoldPrototypes(
        model,
        num_prototypes=len(PROTOTYPE_IDS),
        init_std=0.05,
    )
    prototype_training_loss = _train_prototypes(
        model,
        prototypes,
        epochs=prototype_epochs,
        learning_rate=prototype_learning_rate,
    )
    prototype_metrics = ManifoldPrototypeAssignmentEvaluator(
        model=model,
        prototypes=prototypes,
        prototype_ids=PROTOTYPE_IDS,
        sentences=PROTOTYPE_SENTENCES,
        expected_prototype_ids=PROTOTYPE_ASSIGNMENTS,
    )()

    return {
        "seed": seed,
        "manifold": "poincare",
        "final_training_loss": float(training_history[-1]),
        "initial_pair_loss": paired_only_loss,
        "initial_hard_negative_loss": hard_negative_loss,
        "hard_negative_loss_delta": hard_negative_loss - paired_only_loss,
        "retrieval": retrieval_metrics,
        "ranked_results": ranked_results,
        "embedding_distance": embedding_distance,
        "prototype_training_loss": prototype_training_loss,
        "prototype": prototype_metrics,
    }


def _validate_results(results: dict[str, Any]) -> None:
    scalar_keys = (
        "final_training_loss",
        "initial_pair_loss",
        "initial_hard_negative_loss",
        "hard_negative_loss_delta",
        "embedding_distance",
        "prototype_training_loss",
    )
    for key in scalar_keys:
        if not math.isfinite(results[key]):
            raise RuntimeError(f"{key} became non-finite")
    if results["hard_negative_loss_delta"] <= 0.0:
        raise RuntimeError("explicit hard negatives did not increase the initial objective")
    if not math.isfinite(results["prototype"]["mean_assigned_prototype_distance"]):
        raise RuntimeError("prototype assignment distance became non-finite")
    if any(not math.isfinite(item["distance"]) for item in results["ranked_results"]):
        raise RuntimeError("reranking produced a non-finite distance")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model name or local path.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--prototype-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    results = run_example(
        args.model,
        epochs=args.epochs,
        prototype_epochs=args.prototype_epochs,
        seed=args.seed,
    )
    _validate_results(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    print("Diagnostics only: this tiny example is not a performance-superiority claim.")


if __name__ == "__main__":
    main()
