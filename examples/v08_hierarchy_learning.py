"""Deterministic v0.8 retrieval-only vs hierarchy-aware learning workflow.

Run from the repository root::

    python examples/v08_hierarchy_learning.py

The example uses one tiny caller-owned taxonomy and identical Poincare model
initialization for two engineering-regression paths. ``retrieval_only`` repeats
the same retrieval triplet objective, while ``hierarchy_aware`` combines that
retrieval objective with radial-order, depth, and directed hierarchy supervision
through :class:`neembed.ManifoldRetrievalHierarchyLoss`.

The taxonomy, depths, edges, and directed negatives below are all explicit
caller-owned data. Nothing is inferred from text labels. The output is a
regression diagnostic only and must not be interpreted as evidence that one
training path is better than the other.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from neembed import (
    ManifoldDepthLoss,
    ManifoldEmbeddingEvaluator,
    ManifoldHierarchyEvaluator,
    ManifoldHierarchyTripletLoss,
    ManifoldRadialOrderLoss,
    ManifoldRetrievalHierarchyLoss,
    ManifoldSentenceTransformer,
    ManifoldTripletLoss,
)


NODE_IDS = (
    "root",
    "animal",
    "dog",
    "cat",
    "shiba-inu",
    "siamese-cat",
    "artifact",
    "vehicle",
    "car",
)
NODE_TEXTS = (
    "root concept",
    "animal",
    "dog",
    "cat",
    "Shiba Inu",
    "Siamese cat",
    "artifact",
    "vehicle",
    "car",
)
TEXT_BY_ID = dict(zip(NODE_IDS, NODE_TEXTS))

PARENT_CHILD_EDGES = (
    ("root", "animal"),
    ("root", "artifact"),
    ("animal", "dog"),
    ("animal", "cat"),
    ("dog", "shiba-inu"),
    ("cat", "siamese-cat"),
    ("artifact", "vehicle"),
    ("vehicle", "car"),
)
DEPTHS = {
    "root": 0,
    "animal": 1,
    "artifact": 1,
    "dog": 2,
    "cat": 2,
    "vehicle": 2,
    "shiba-inu": 3,
    "siamese-cat": 3,
    "car": 3,
}
DIRECTED_TRIPLETS = (
    ("animal", "dog", "vehicle"),
    ("animal", "cat", "vehicle"),
    ("dog", "shiba-inu", "car"),
    ("cat", "siamese-cat", "car"),
    ("artifact", "vehicle", "dog"),
    ("vehicle", "car", "cat"),
)

RETRIEVAL_ANCHORS = ("Shiba Inu", "Siamese cat")
RETRIEVAL_POSITIVES = ("dog", "cat")
RETRIEVAL_NEGATIVES = ("car", "vehicle")
RETRIEVAL_INPUTS = (
    RETRIEVAL_ANCHORS,
    RETRIEVAL_POSITIVES,
    RETRIEVAL_NEGATIVES,
)

RADIAL_INPUTS = (
    tuple(TEXT_BY_ID[parent_id] for parent_id, _ in PARENT_CHILD_EDGES),
    tuple(TEXT_BY_ID[child_id] for _, child_id in PARENT_CHILD_EDGES),
)
DEPTH_INPUTS = (
    NODE_TEXTS,
    tuple(DEPTHS[node_id] for node_id in NODE_IDS),
)
DIRECTED_INPUTS = (
    tuple(TEXT_BY_ID[parent_id] for parent_id, _, _ in DIRECTED_TRIPLETS),
    tuple(TEXT_BY_ID[child_id] for _, child_id, _ in DIRECTED_TRIPLETS),
    tuple(TEXT_BY_ID[negative_id] for _, _, negative_id in DIRECTED_TRIPLETS),
)

HIERARCHY_PHASES = ("radial", "depth", "directed")


def _build_model(
    model_name_or_path: str,
    *,
    seed: int,
    embedding_dim: int,
) -> ManifoldSentenceTransformer:
    torch.manual_seed(seed)
    return ManifoldSentenceTransformer(
        model_name_or_path,
        manifold="poincare",
        embedding_dim=embedding_dim,
        curvature=1.0,
    )


def _evaluate(model: ManifoldSentenceTransformer) -> dict[str, dict[str, float]]:
    retrieval = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=RETRIEVAL_ANCHORS,
        positives=RETRIEVAL_POSITIVES,
        recall_at_k=(1, 2),
    )()
    hierarchy = ManifoldHierarchyEvaluator(
        model=model,
        node_ids=NODE_IDS,
        texts=NODE_TEXTS,
        parent_child_edges=PARENT_CHILD_EDGES,
        depths=DEPTHS,
        contract="tree",
    )()
    return {"retrieval": retrieval, "hierarchy": hierarchy}


def _optimizer_step(
    optimizer: torch.optim.Optimizer,
    loss,
    *inputs,
) -> float:
    optimizer.zero_grad()
    value = loss(*inputs)
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise RuntimeError("training loss must be a finite scalar")
    value.backward()
    optimizer.step()
    return float(value.detach())


def _train_retrieval_only(
    model: ManifoldSentenceTransformer,
    *,
    epochs: int,
    learning_rate: float,
) -> dict[str, Any]:
    retrieval_loss = ManifoldTripletLoss(model, margin=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    training_losses: list[float] = []
    for _ in range(epochs):
        # Match the hierarchy-aware path's three optimizer updates per epoch
        # without adding hierarchy supervision to this comparison arm.
        for _phase in HIERARCHY_PHASES:
            training_losses.append(
                _optimizer_step(optimizer, retrieval_loss, *RETRIEVAL_INPUTS)
            )

    with torch.no_grad():
        final_retrieval_loss = float(retrieval_loss(*RETRIEVAL_INPUTS))
    return {
        "training_losses": training_losses,
        "final_retrieval_loss": final_retrieval_loss,
    }


def _hierarchy_composites(
    model: ManifoldSentenceTransformer,
    *,
    hierarchy_weight: float,
):
    retrieval_loss = ManifoldTripletLoss(model, margin=0.1)
    return (
        (
            "radial",
            ManifoldRetrievalHierarchyLoss(
                retrieval_loss,
                ManifoldRadialOrderLoss(model, margin=0.05),
                hierarchy_weight=hierarchy_weight,
            ),
            RADIAL_INPUTS,
        ),
        (
            "depth",
            ManifoldRetrievalHierarchyLoss(
                retrieval_loss,
                ManifoldDepthLoss(model, radial_scale=0.2),
                hierarchy_weight=hierarchy_weight,
            ),
            DEPTH_INPUTS,
        ),
        (
            "directed",
            ManifoldRetrievalHierarchyLoss(
                retrieval_loss,
                ManifoldHierarchyTripletLoss(
                    model,
                    margin=0.1,
                    radial_margin=0.05,
                    radial_weight=1.0,
                ),
                hierarchy_weight=hierarchy_weight,
            ),
            DIRECTED_INPUTS,
        ),
    )


def _train_hierarchy_aware(
    model: ManifoldSentenceTransformer,
    *,
    epochs: int,
    learning_rate: float,
    hierarchy_weight: float,
) -> dict[str, Any]:
    composites = _hierarchy_composites(
        model,
        hierarchy_weight=hierarchy_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    training_losses: list[float] = []
    for _ in range(epochs):
        for _name, composite, hierarchy_inputs in composites:
            training_losses.append(
                _optimizer_step(
                    optimizer,
                    composite,
                    RETRIEVAL_INPUTS,
                    hierarchy_inputs,
                )
            )

    objective_diagnostics: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for name, composite, hierarchy_inputs in composites:
            retrieval_value, hierarchy_value = composite.component_losses(
                RETRIEVAL_INPUTS,
                hierarchy_inputs,
            )
            total_value = composite(RETRIEVAL_INPUTS, hierarchy_inputs)
            objective_diagnostics[name] = {
                "hierarchy_weight": composite.hierarchy_weight,
                "retrieval_loss": float(retrieval_value),
                "hierarchy_loss": float(hierarchy_value),
                "total_loss": float(total_value),
            }

    return {
        "training_losses": training_losses,
        "objective_diagnostics": objective_diagnostics,
    }


def run_hierarchy_learning(
    model_name_or_path: str,
    *,
    epochs: int = 2,
    seed: int = 83,
    embedding_dim: int = 8,
    learning_rate: float = 1e-4,
    hierarchy_weight: float = 0.25,
) -> dict[str, Any]:
    """Run matched retrieval-only and hierarchy-aware Poincare regressions."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if (
        isinstance(embedding_dim, bool)
        or not isinstance(embedding_dim, int)
        or embedding_dim <= 0
    ):
        raise ValueError("embedding_dim must be a positive integer")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    if hierarchy_weight < 0 or not math.isfinite(hierarchy_weight):
        raise ValueError("hierarchy_weight must be non-negative and finite")

    retrieval_only_model = _build_model(
        model_name_or_path,
        seed=seed,
        embedding_dim=embedding_dim,
    )
    hierarchy_aware_model = _build_model(
        model_name_or_path,
        seed=seed,
        embedding_dim=embedding_dim,
    )

    retrieval_only_before = _evaluate(retrieval_only_model)
    hierarchy_aware_before = _evaluate(hierarchy_aware_model)

    retrieval_only_training = _train_retrieval_only(
        retrieval_only_model,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    hierarchy_aware_training = _train_hierarchy_aware(
        hierarchy_aware_model,
        epochs=epochs,
        learning_rate=learning_rate,
        hierarchy_weight=hierarchy_weight,
    )

    return {
        "manifold": "poincare",
        "seed": seed,
        "supervision": {
            "node_ids": list(NODE_IDS),
            "parent_child_edges": [list(edge) for edge in PARENT_CHILD_EDGES],
            "depths": dict(DEPTHS),
            "directed_triplets": [list(triplet) for triplet in DIRECTED_TRIPLETS],
        },
        "retrieval_only": {
            "before": retrieval_only_before,
            **retrieval_only_training,
            "after": _evaluate(retrieval_only_model),
        },
        "hierarchy_aware": {
            "before": hierarchy_aware_before,
            **hierarchy_aware_training,
            "after": _evaluate(hierarchy_aware_model),
        },
    }


def _validate_regression(results: dict[str, Any]) -> None:
    if results["manifold"] != "poincare":
        raise RuntimeError("v0.8 hierarchy learning example must use Poincare geometry")
    if results["retrieval_only"]["before"] != results["hierarchy_aware"]["before"]:
        raise RuntimeError("comparison paths did not start from identical diagnostics")

    for variant_name in ("retrieval_only", "hierarchy_aware"):
        variant = results[variant_name]
        if not variant["training_losses"]:
            raise RuntimeError(f"{variant_name} did not execute training updates")
        if any(not math.isfinite(value) for value in variant["training_losses"]):
            raise RuntimeError(f"{variant_name} training loss became non-finite")
        for stage_name in ("before", "after"):
            for metric_group in variant[stage_name].values():
                if any(not math.isfinite(value) for value in metric_group.values()):
                    raise RuntimeError(
                        f"{variant_name} {stage_name} metrics became non-finite"
                    )

    if not math.isfinite(results["retrieval_only"]["final_retrieval_loss"]):
        raise RuntimeError("retrieval-only diagnostic became non-finite")

    diagnostics = results["hierarchy_aware"]["objective_diagnostics"]
    if tuple(diagnostics) != HIERARCHY_PHASES:
        raise RuntimeError("hierarchy objective diagnostics are incomplete")
    for name, values in diagnostics.items():
        if any(not math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"{name} hierarchy diagnostic became non-finite")
        expected_total = (
            values["retrieval_loss"]
            + values["hierarchy_weight"] * values["hierarchy_loss"]
        )
        if not math.isclose(values["total_loss"], expected_total, rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError(f"{name} composite diagnostic violated its formula")

    for variant_name in ("retrieval_only", "hierarchy_aware"):
        hierarchy_metrics = results[variant_name]["after"]["hierarchy"]
        accuracy = hierarchy_metrics["parent_child_radial_order_accuracy"]
        violation = hierarchy_metrics["mean_radial_order_violation"]
        association = hierarchy_metrics["depth_radius_spearman"]
        if not 0.0 <= accuracy <= 1.0:
            raise RuntimeError("hierarchy radial-order accuracy left [0, 1]")
        if violation < 0.0:
            raise RuntimeError("hierarchy violation must remain non-negative")
        if not -1.0 <= association <= 1.0:
            raise RuntimeError("depth-radius association left [-1, 1]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model name or local path.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    results = run_hierarchy_learning(args.model, epochs=args.epochs)
    _validate_regression(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(
        "Diagnostics only: this tiny taxonomy is an engineering regression "
        "reference, not a benchmark or hierarchy-aware superiority claim."
    )


if __name__ == "__main__":
    main()
