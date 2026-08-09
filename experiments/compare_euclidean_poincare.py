"""Compare the original Euclidean encoder with Poincare fine-tuning."""

import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from neembed import (
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 0
EMBEDDING_DIM = 32
CURVATURE = 1.0
TEMPERATURE = 0.1
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
EPOCHS = 10
BATCH_SIZE = 5

TRAIN_PAIRS = [
    ("Beagle", "dog"),
    ("Maine Coon", "cat"),
    ("eagle", "bird"),
    ("tuna", "fish"),
    ("maple", "plant"),
    ("Poodle", "dog"),
    ("Persian cat", "cat"),
    ("robin", "bird"),
    ("trout", "fish"),
    ("rose", "plant"),
]

EVALUATION_PAIRS = [
    ("Shiba Inu", "dog"),
    ("Siamese cat", "cat"),
    ("sparrow", "bird"),
    ("salmon", "fish"),
    ("oak", "plant"),
]


def set_seed(seed: int) -> None:
    """Seed random sources used by the comparison."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_batches() -> list[tuple[list[str], list[str]]]:
    batches: list[tuple[list[str], list[str]]] = []
    for start in range(0, len(TRAIN_PAIRS), BATCH_SIZE):
        batch = TRAIN_PAIRS[start : start + BATCH_SIZE]
        batches.append(
            (
                [anchor for anchor, _ in batch],
                [positive for _, positive in batch],
            )
        )
    return batches


def _parent_retrieval_accuracy(distances: torch.Tensor) -> float:
    targets = torch.arange(distances.shape[0], device=distances.device)
    predictions = distances.argmin(dim=1)
    return float((predictions == targets).float().mean())


def _evaluate_euclidean(model: SentenceTransformer) -> tuple[float, float]:
    anchors = [anchor for anchor, _ in EVALUATION_PAIRS]
    candidates = [positive for _, positive in EVALUATION_PAIRS]

    started = time.perf_counter()
    anchor_embeddings = model.encode(anchors, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    anchor_embeddings = F.normalize(anchor_embeddings, dim=-1)
    candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
    distances = 1.0 - anchor_embeddings @ candidate_embeddings.T
    elapsed = time.perf_counter() - started

    return _parent_retrieval_accuracy(distances), elapsed


def _evaluate_poincare(
    model: ManifoldSentenceTransformer,
) -> tuple[float, float]:
    anchors = [anchor for anchor, _ in EVALUATION_PAIRS]
    candidates = [positive for _, positive in EVALUATION_PAIRS]

    started = time.perf_counter()
    anchor_embeddings = model.encode(anchors, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    distances = torch.stack(
        [
            torch.stack(
                [model.distance(anchor, candidate) for candidate in candidate_embeddings]
            )
            for anchor in anchor_embeddings
        ]
    )
    elapsed = time.perf_counter() - started

    return _parent_retrieval_accuracy(distances), elapsed


def run_experiment() -> dict[str, object]:
    """Run the minimal Euclidean-vs-Poincare hierarchy comparison."""
    set_seed(SEED)

    euclidean_model = SentenceTransformer(MODEL_NAME)
    euclidean_accuracy, euclidean_eval_seconds = _evaluate_euclidean(euclidean_model)

    set_seed(SEED)
    poincare_model = ManifoldSentenceTransformer(
        MODEL_NAME,
        manifold="poincare",
        embedding_dim=EMBEDDING_DIM,
        curvature=CURVATURE,
    )
    loss = ManifoldMultipleNegativesRankingLoss(
        model=poincare_model,
        temperature=TEMPERATURE,
    )
    trainer = ManifoldTrainer(
        model=poincare_model,
        loss=loss,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        verbose=False,
    )

    training_started = time.perf_counter()
    history = trainer.fit(_train_batches(), epochs=EPOCHS)
    training_seconds = time.perf_counter() - training_started
    poincare_accuracy, poincare_eval_seconds = _evaluate_poincare(poincare_model)

    return {
        "metadata": {
            "model": MODEL_NAME,
            "seed": SEED,
            "metric": "parent_retrieval_accuracy",
            "train_pairs": [
                {"anchor": anchor, "positive": positive}
                for anchor, positive in TRAIN_PAIRS
            ],
            "evaluation_pairs": [
                {"anchor": anchor, "positive": positive}
                for anchor, positive in EVALUATION_PAIRS
            ],
            "poincare_config": {
                "embedding_dim": EMBEDDING_DIM,
                "curvature": CURVATURE,
                "temperature": TEMPERATURE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
            },
        },
        "results": {
            "euclidean_pretrained": {
                "distance": "cosine",
                "parent_retrieval_accuracy": euclidean_accuracy,
                "training_seconds": 0.0,
                "evaluation_seconds": euclidean_eval_seconds,
            },
            "poincare_finetuned": {
                "distance": "poincare_geodesic",
                "parent_retrieval_accuracy": poincare_accuracy,
                "final_training_loss": float(history[-1]),
                "training_seconds": training_seconds,
                "evaluation_seconds": poincare_eval_seconds,
            },
        },
    }


def main() -> None:
    """Run the comparison and emit one machine-readable JSON result."""
    print(json.dumps(run_experiment(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
