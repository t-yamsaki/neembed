"""Deterministic v0.9 constant-curvature engineering comparison.

Run from the repository root::

    python examples/v09_constant_curvature_comparison.py

The example holds the encoder choice, tiny retrieval data, projection dimension,
seed, MNRL objective, temperature, and retrieval evaluator fixed while changing
only the configured constant-curvature geometry. It reports engineering
regression diagnostics; the tiny data are not a benchmark and the output must
not be interpreted as evidence that one geometry is superior to another.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from neembed import (
    ManifoldGradedCorpusRetrievalEvaluator,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
)


QUERY_IDS = ("query-dog", "query-cat", "query-bird")
QUERIES = ("Shiba Inu", "Siamese cat", "sparrow")
TRAIN_POSITIVES = ("dog", "cat", "bird")
TRAIN_NEGATIVES = ("vehicle", "building", "tool")

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

LEGACY_CURVATURE = 0.5
SIGNED_CURVATURE = 0.5
TEMPERATURE = 0.1

GEOMETRY_CONFIGS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "euclidean",
        {"manifold": "euclidean", "sectional_curvature": 0.0},
    ),
    (
        "poincare",
        {"manifold": "poincare", "curvature": LEGACY_CURVATURE},
    ),
    (
        "lorentz",
        {"manifold": "lorentz", "curvature": LEGACY_CURVATURE},
    ),
    (
        "sphere_projection",
        {
            "manifold": "sphere_projection",
            "sectional_curvature": SIGNED_CURVATURE,
        },
    ),
    (
        "stereographic_negative",
        {
            "manifold": "stereographic",
            "sectional_curvature": -SIGNED_CURVATURE,
        },
    ),
    (
        "stereographic_zero",
        {"manifold": "stereographic", "sectional_curvature": 0.0},
    ),
    (
        "stereographic_positive",
        {
            "manifold": "stereographic",
            "sectional_curvature": SIGNED_CURVATURE,
        },
    ),
)
VARIANT_NAMES = tuple(name for name, _ in GEOMETRY_CONFIGS)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _first_manifold_state(model: ManifoldSentenceTransformer) -> torch.Tensor | None:
    for parameter in model.manifold.parameters():
        return parameter
    for buffer in model.manifold.buffers():
        return buffer
    return None


def _build_model(
    model_name_or_path: str,
    *,
    seed: int,
    embedding_dim: int,
    config: dict[str, Any],
) -> ManifoldSentenceTransformer:
    # Reset before every geometry so deterministic encoders and projection layers
    # start from the same ordinary Euclidean parameters in the CI regression.
    torch.manual_seed(seed)
    return ManifoldSentenceTransformer(
        model_name_or_path,
        embedding_dim=embedding_dim,
        **config,
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


def _metadata(
    model: ManifoldSentenceTransformer,
    sample_embedding: torch.Tensor,
) -> dict[str, Any]:
    if model.manifold_name in {"poincare", "lorentz"}:
        curvature_api = "curvature"
        curvature_value = model.curvature
        sectional_curvature = -curvature_value
    else:
        curvature_api = "sectional_curvature"
        curvature_value = model.sectional_curvature
        sectional_curvature = curvature_value

    manifold_state = _first_manifold_state(model)
    return {
        "manifold": model.manifold_name,
        "curvature_api": curvature_api,
        "curvature_value": float(curvature_value),
        "sectional_curvature": float(sectional_curvature),
        "embedding_dtype": _dtype_name(sample_embedding.dtype),
        "embedding_device": sample_embedding.device.type,
        "manifold_state_dtype": (
            None if manifold_state is None else _dtype_name(manifold_state.dtype)
        ),
        "manifold_state_device": (
            None if manifold_state is None else manifold_state.device.type
        ),
    }


def _run_geometry(
    name: str,
    config: dict[str, Any],
    model_name_or_path: str,
    *,
    seed: int,
    embedding_dim: int,
    temperature: float,
    query_chunk_size: int,
    corpus_chunk_size: int,
) -> dict[str, Any]:
    model = _build_model(
        model_name_or_path,
        seed=seed,
        embedding_dim=embedding_dim,
        config=config,
    )
    objective = ManifoldMultipleNegativesRankingLoss(
        model,
        temperature=temperature,
    )
    loss = objective(QUERIES, TRAIN_POSITIVES, TRAIN_NEGATIVES)
    metrics = _evaluate(
        model,
        query_chunk_size=query_chunk_size,
        corpus_chunk_size=corpus_chunk_size,
    )
    sample_embedding = model.encode(QUERIES[0], convert_to_tensor=True)
    return {
        "name": name,
        "loss": float(loss.detach().cpu()),
        "retrieval": metrics,
        "metadata": _metadata(model, sample_embedding),
    }


def run_comparison(
    model_name_or_path: str,
    *,
    seed: int = 59,
    embedding_dim: int = 4,
    temperature: float = TEMPERATURE,
    query_chunk_size: int = 2,
    corpus_chunk_size: int = 3,
) -> dict[str, Any]:
    """Run one fixed retrieval regression across supported constant curvatures."""
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    for value, label in (
        (query_chunk_size, "query_chunk_size"),
        (corpus_chunk_size, "corpus_chunk_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")

    variants = {
        name: _run_geometry(
            name,
            config,
            model_name_or_path,
            seed=seed,
            embedding_dim=embedding_dim,
            temperature=temperature,
            query_chunk_size=query_chunk_size,
            corpus_chunk_size=corpus_chunk_size,
        )
        for name, config in GEOMETRY_CONFIGS
    }
    return {
        "seed": seed,
        "objective": "mnrl",
        "temperature": float(temperature),
        "embedding_dim": embedding_dim,
        "variants": variants,
    }


def _validate_regression(results: dict[str, Any]) -> None:
    if results["objective"] != "mnrl":
        raise RuntimeError("v0.9 curvature comparison must keep MNRL fixed")
    variants = results["variants"]
    if tuple(variants) != VARIANT_NAMES:
        raise RuntimeError("constant-curvature comparison output is incomplete")

    expected_sectional = {
        "euclidean": 0.0,
        "poincare": -LEGACY_CURVATURE,
        "lorentz": -LEGACY_CURVATURE,
        "sphere_projection": SIGNED_CURVATURE,
        "stereographic_negative": -SIGNED_CURVATURE,
        "stereographic_zero": 0.0,
        "stereographic_positive": SIGNED_CURVATURE,
    }
    expected_manifold = {
        "euclidean": "euclidean",
        "poincare": "poincare",
        "lorentz": "lorentz",
        "sphere_projection": "sphere_projection",
        "stereographic_negative": "stereographic",
        "stereographic_zero": "stereographic",
        "stereographic_positive": "stereographic",
    }

    for name in VARIANT_NAMES:
        diagnostics = variants[name]
        if diagnostics["name"] != name:
            raise RuntimeError(f"{name} variant label changed")
        if not math.isfinite(diagnostics["loss"]):
            raise RuntimeError(f"{name} loss became non-finite")

        metrics = diagnostics["retrieval"]
        if any(not math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"{name} retrieval metrics became non-finite")
        if any(not 0.0 <= value <= 1.0 for value in metrics.values()):
            raise RuntimeError(f"{name} retrieval metric left [0, 1]")

        metadata = diagnostics["metadata"]
        if metadata["manifold"] != expected_manifold[name]:
            raise RuntimeError(f"{name} manifold metadata changed")
        if metadata["sectional_curvature"] != expected_sectional[name]:
            raise RuntimeError(f"{name} sectional-curvature metadata changed")
        if name in {"poincare", "lorentz"}:
            if metadata["curvature_api"] != "curvature":
                raise RuntimeError(f"{name} must retain the legacy curvature API")
        elif metadata["curvature_api"] != "sectional_curvature":
            raise RuntimeError(f"{name} must use signed sectional_curvature metadata")

        if name in {
            "lorentz",
            "sphere_projection",
            "stereographic_negative",
            "stereographic_zero",
            "stereographic_positive",
        } and metadata["embedding_dtype"] != "float64":
            raise RuntimeError(f"{name} geometry output must use float64")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model name or local path.",
    )
    parser.add_argument("--seed", type=int, default=59)
    args = parser.parse_args()

    results = run_comparison(args.model, seed=args.seed)
    _validate_regression(results)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(
        "Diagnostics only: this tiny run is an engineering regression reference, "
        "not a benchmark or geometry-superiority claim."
    )


if __name__ == "__main__":
    main()
