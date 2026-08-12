"""Tiny v0.4 learnable-structure regression example.

This script is an engineering regression check, not a performance benchmark or
claim that learnable curvature or trainable prototypes are generally superior.
It compares the same tiny hierarchy with fixed structure and with the opt-in
v0.4 learnable-structure path.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import geoopt
import torch

from neembed import (
    ManifoldPrototypeHierarchyLoss,
    ManifoldPrototypes,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)


PROTOTYPE_IDS = ("animal", "dog", "cat")
PARENT_RELATIONS = (("dog", "animal"), ("cat", "animal"))
TRAIN_SENTENCES = (
    "Shiba Inu",
    "Golden retriever",
    "Siamese cat",
    "Persian cat",
    "living animal",
    "creature",
)
TRAIN_ASSIGNMENTS = ("dog", "dog", "cat", "cat", "animal", "animal")


def _assignment_diagnostics(
    model: ManifoldSentenceTransformer,
    prototypes: ManifoldPrototypes,
) -> tuple[float, bool, float]:
    with torch.no_grad():
        embeddings = model(TRAIN_SENTENCES)
        prototype_distances = prototypes(embeddings)
        predicted = prototype_distances.argmin(dim=1).tolist()
        expected = [PROTOTYPE_IDS.index(label) for label in TRAIN_ASSIGNMENTS]
        accuracy = sum(
            predicted_index == expected_index
            for predicted_index, expected_index in zip(predicted, expected, strict=True)
        ) / len(expected)
        embedding_distance = model.manifold.dist(embeddings[0], embeddings[1])
        finite_distances = bool(
            torch.isfinite(prototype_distances).all().item()
            and torch.isfinite(embedding_distance).item()
        )
    return float(accuracy), finite_distances, float(embedding_distance)


def run_configuration(
    model_name_or_path: str,
    *,
    learnable_structure: bool,
    epochs: int,
    seed: int,
    embedding_dim: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Run one matched tiny-hierarchy configuration and return diagnostics."""
    torch.manual_seed(seed)
    model = ManifoldSentenceTransformer(
        model_name_or_path,
        manifold="poincare",
        embedding_dim=embedding_dim,
        curvature=1.0,
        learnable_curvature=learnable_structure,
    )
    prototypes = ManifoldPrototypes(
        model,
        num_prototypes=len(PROTOTYPE_IDS),
        init_std=0.05,
    )
    if not learnable_structure:
        prototypes.prototypes.requires_grad_(False)

    loss = ManifoldPrototypeHierarchyLoss(
        model,
        prototypes,
        prototype_ids=PROTOTYPE_IDS,
        parent_relations=PARENT_RELATIONS,
        margin=0.1,
    )
    initial_curvature = model.curvature
    initial_prototypes = prototypes.prototypes.detach().clone()

    if learnable_structure:
        optimizer = geoopt.optim.RiemannianAdam(
            [parameter for parameter in loss.parameters() if parameter.requires_grad],
            lr=learning_rate,
            stabilize=1,
        )
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rate,
        )

    trainer = ManifoldTrainer(
        model,
        loss,
        optimizer=optimizer,
        verbose=False,
    )
    history = trainer.fit(
        [(TRAIN_SENTENCES, TRAIN_ASSIGNMENTS)],
        epochs=epochs,
    )

    assignment_accuracy, finite_distances, embedding_distance = _assignment_diagnostics(
        model,
        prototypes,
    )
    final_curvature = model.curvature
    prototype_shift = float(
        torch.linalg.vector_norm(
            prototypes.prototypes.detach() - initial_prototypes
        ).cpu()
    )
    prototypes_valid = model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-5,
        rtol=1e-5,
    )

    return {
        "final_training_loss": float(history[-1]),
        "assignment_accuracy": assignment_accuracy,
        "initial_curvature": initial_curvature,
        "final_curvature": final_curvature,
        "curvature_delta": final_curvature - initial_curvature,
        "prototype_shift": prototype_shift,
        "prototypes_valid": bool(prototypes_valid),
        "finite_distances": finite_distances,
        "embedding_distance": embedding_distance,
        "intrinsic_dim": model.embedding_dim,
        "ambient_dim": prototypes.ambient_dim,
    }


def run_benchmark(
    model_name_or_path: str,
    *,
    epochs: int = 2,
    seed: int = 17,
    embedding_dim: int = 8,
    learning_rate: float = 1e-4,
) -> dict[str, dict[str, Any]]:
    """Compare matched fixed and learnable structure on the tiny hierarchy."""
    fixed = run_configuration(
        model_name_or_path,
        learnable_structure=False,
        epochs=epochs,
        seed=seed,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
    )
    learnable = run_configuration(
        model_name_or_path,
        learnable_structure=True,
        epochs=epochs,
        seed=seed,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
    )
    return {
        "fixed_structure": fixed,
        "learnable_structure": learnable,
    }


def _print_results(results: dict[str, dict[str, Any]]) -> None:
    for name, result in results.items():
        print(name)
        for key in (
            "final_training_loss",
            "assignment_accuracy",
            "initial_curvature",
            "final_curvature",
            "curvature_delta",
            "prototype_shift",
            "prototypes_valid",
            "finite_distances",
            "embedding_distance",
            "intrinsic_dim",
            "ambient_dim",
        ):
            print(f"  {key}: {result[key]}")
    print("Diagnostics only: these toy results are not a superiority claim.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model name or local path.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    results = run_benchmark(args.model, epochs=args.epochs)
    learnable = results["learnable_structure"]
    if not math.isfinite(learnable["curvature_delta"]):
        raise RuntimeError("learnable curvature became non-finite")
    _print_results(results)


if __name__ == "__main__":
    main()
