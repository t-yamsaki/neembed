"""Deterministic v0.10 mixed-curvature engineering regression.

Run from the repository root::

    python examples/v10_mixed_curvature_workflow.py

Roles, hierarchy labels and fixed scales are explicit caller choices. Retrieval
uses the full Poincare x SphereProjection x Euclidean product; hierarchy uses
only the named Poincare component. This tiny run is not a benchmark or a claim
that mixed curvature is superior. CI substitutes a deterministic tiny encoder
with local save/load support, without downloading a pretrained model.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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


PRODUCT_CONFIG = [
    {"name": "hierarchy", "manifold": "poincare", "intrinsic_dim": 2,
     "curvature": 0.5, "scale": 1.5},
    {"name": "semantic", "manifold": "sphere_projection", "intrinsic_dim": 2,
     "sectional_curvature": 0.25, "scale": 0.75},
    {"name": "residual", "manifold": "euclidean", "intrinsic_dim": 2, "scale": 1.0},
]
HIERARCHY_COMPONENT = "hierarchy"

QUERIES = ("Shiba Inu", "Siamese cat")
QUERY_IDS = ("q-dog", "q-cat")
POSITIVES = ("dog", "cat")
NEGATIVES = ("car", "vehicle")
CORPUS = ("dog", "canine", "cat", "feline", "car", "vehicle")
CORPUS_IDS = ("dog", "canine", "cat", "feline", "car", "vehicle")
RELEVANCE = {"q-dog": ("dog", "canine"), "q-cat": ("cat", "feline")}

NODE_IDS = ("root", "animal", "dog", "cat", "vehicle", "car")
NODE_TEXTS = ("root concept", "animal", "dog", "cat", "vehicle", "car")
EDGES = (("root", "animal"), ("animal", "dog"), ("animal", "cat"),
         ("root", "vehicle"), ("vehicle", "car"))
DEPTHS = {"root": 0, "animal": 1, "dog": 2, "cat": 2, "vehicle": 1, "car": 2}
RETRIEVAL_INPUTS = (QUERIES, POSITIVES, NEGATIVES)
HIERARCHY_INPUTS = (("animal", "animal", "vehicle"), ("dog", "cat", "car"),
                    ("car", "vehicle", "dog"))


def _component_diagnostics(model: ManifoldSentenceTransformer) -> dict[str, Any]:
    queries = model.encode(QUERIES, convert_to_tensor=True)
    positives = model.encode(POSITIVES, convert_to_tensor=True)
    report = model.product_distance_diagnostics(queries, positives)
    return {
        "total_distance": report["total_distance"].cpu().tolist(),
        "component_distances": {
            name: values.cpu().tolist()
            for name, values in report["component_distances"].items()
        },
    }


def _evaluate(model: ManifoldSentenceTransformer) -> dict[str, Any]:
    retrieval = ManifoldCorpusRetrievalEvaluator(
        model=model, query_ids=QUERY_IDS, queries=QUERIES,
        corpus_ids=CORPUS_IDS, corpus=CORPUS, relevance=RELEVANCE,
        recall_at_k=(1, 3), query_chunk_size=1, corpus_chunk_size=2,
    )()
    hierarchy = ManifoldHierarchyEvaluator(
        model=model, node_ids=NODE_IDS, texts=NODE_TEXTS,
        parent_child_edges=EDGES, depths=DEPTHS, contract="tree",
        component=HIERARCHY_COMPONENT,
    )()
    return {
        "retrieval": retrieval,
        "hierarchy": hierarchy,
        "component_diagnostics": _component_diagnostics(model),
        "search_results": exact_corpus_search(
            model, QUERIES, CORPUS, top_k=3,
            query_chunk_size=1, corpus_chunk_size=2,
        ),
    }


def _save_load(model: ManifoldSentenceTransformer, output_path: Path) -> dict[str, Any]:
    before = model.encode(NODE_TEXTS, convert_to_tensor=True)
    distances = model.distance(before[:, None], before[None])
    diagnostics = _component_diagnostics(model)
    model.save_pretrained(output_path)
    loaded = ManifoldSentenceTransformer.from_pretrained(output_path).cpu()
    if loaded.product_config != model.product_config:
        raise RuntimeError("product component identity changed after loading")
    after = loaded.encode(NODE_TEXTS, convert_to_tensor=True)
    torch.testing.assert_close(after, before)
    torch.testing.assert_close(loaded.distance(after[:, None], after[None]), distances)
    loaded_diagnostics = _component_diagnostics(loaded)
    torch.testing.assert_close(
        torch.tensor(loaded_diagnostics["total_distance"], dtype=torch.float64),
        torch.tensor(diagnostics["total_distance"], dtype=torch.float64),
    )
    for name, values in diagnostics["component_distances"].items():
        torch.testing.assert_close(torch.tensor(loaded_diagnostics["component_distances"][name], dtype=torch.float64),
                                   torch.tensor(values, dtype=torch.float64))
    # Reconstruct the evaluator with the persisted stable component name.
    loaded_metrics = _evaluate(loaded)
    return {
        "product_config": loaded.product_config.to_dict(),
        "embeddings_match": True,
        "distances_match": True,
        "component_diagnostics_match": True,
        "loaded_evaluation": loaded_metrics,
    }


def run_example(
    model_name_or_path: str,
    *,
    epochs: int = 2,
    seed: int = 71,
    learning_rate: float = 1e-3,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train one explicitly assigned product and verify its local checkpoint."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(learning_rate, bool) or learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be positive and finite")
    torch.manual_seed(seed)
    model = ManifoldSentenceTransformer(model_name_or_path, product_config=PRODUCT_CONFIG).cpu()
    before = _evaluate(model)
    initial_projection = model.projection.weight.detach().clone()
    retrieval = ManifoldMultipleNegativesRankingLoss(model, temperature=0.5)
    hierarchy = ManifoldHierarchyTripletLoss(
        model, component=HIERARCHY_COMPONENT, margin=0.2,
        radial_margin=0.1, radial_weight=0.5,
    )
    joint = ManifoldRetrievalHierarchyLoss(retrieval, hierarchy, hierarchy_weight=0.3)
    trainer = ManifoldTrainer(model, joint, learning_rate=learning_rate,
                              weight_decay=0.0, verbose=False)
    training_losses = trainer.fit([(RETRIEVAL_INPUTS, HIERARCHY_INPUTS)], epochs=epochs)
    after = _evaluate(model)
    with torch.no_grad():
        retrieval_value, hierarchy_value = joint.component_losses(RETRIEVAL_INPUTS, HIERARCHY_INPUTS)
    if output_path is None:
        with TemporaryDirectory(prefix="neembed-v10-") as directory:
            save_load = _save_load(model, Path(directory))
    else:
        save_load = _save_load(model, Path(output_path))
    result = {
        "seed": seed,
        "product_config": model.product_config.to_dict(),
        "hierarchy_component": HIERARCHY_COMPONENT,
        "scale_policy": "fixed",
        "before": before,
        "training_losses": training_losses,
        "projection_updated": not torch.equal(initial_projection, model.projection.weight),
        "objective_diagnostics": {
            "retrieval_loss": float(retrieval_value),
            "hierarchy_loss": float(hierarchy_value),
            "hierarchy_weight": joint.hierarchy_weight,
            "total_loss": float(retrieval_value + joint.hierarchy_weight * hierarchy_value),
        },
        "after": after,
        "save_load": save_load,
    }
    _validate_regression(result)
    return result


def _validate_regression(results: dict[str, Any]) -> None:
    """Check finite diagnostics and explicit fixed component identity."""
    def finite(value):
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                finite(item)
        elif isinstance(value, (float, int)) and not math.isfinite(value):
            raise RuntimeError("mixed-curvature regression produced a non-finite value")

    finite(results)
    if results["product_config"] != results["save_load"]["product_config"]:
        raise RuntimeError("saved product configuration differs")
    expected_names = [component["name"] for component in PRODUCT_CONFIG]
    for stage in ("before", "after"):
        if list(results[stage]["component_diagnostics"]["component_distances"]) != expected_names:
            raise RuntimeError("component diagnostic order changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence Transformer model name or local path.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--output", type=Path, help="Keep the checkpoint in this directory.")
    args = parser.parse_args()
    result = run_example(args.model, epochs=args.epochs, seed=args.seed, output_path=args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    print("Diagnostics only: not a benchmark or mixed-curvature superiority claim. "
          "Roles and fixed scales are caller-assigned; learnable scales are not supported.")


if __name__ == "__main__":
    main()
