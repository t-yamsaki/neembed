"""Reproducible Euclidean-vs-Poincare benchmark for neembed v0.2."""

from collections.abc import Sequence
import json
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn

from neembed import (
    ManifoldEmbeddingEvaluator,
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
    """Seed random sources used by the benchmark."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_batches() -> list[tuple[list[str], list[str]]]:
    """Return deterministic batches with unique positive candidates per batch."""
    batches: list[tuple[list[str], list[str]]] = []
    for start in range(0, len(TRAIN_PAIRS), BATCH_SIZE):
        batch = TRAIN_PAIRS[start : start + BATCH_SIZE]
        positives = [positive for _, positive in batch]
        if len(set(positives)) != len(positives):
            raise ValueError("benchmark batches must not contain duplicate positives")
        batches.append(
            (
                [anchor for anchor, _ in batch],
                positives,
            )
        )
    return batches


class EuclideanSentenceTransformer(nn.Module):
    """Sentence Transformer baseline with a learned Euclidean projection.

    This benchmark-local wrapper mirrors neembed's encoder plus projection path,
    but keeps the output in normalized Euclidean space and uses cosine distance.
    It intentionally is not part of neembed's public API.
    """

    def __init__(self, model_name_or_path: str, *, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = SentenceTransformer(model_name_or_path)
        encoder_dim = self.encoder.get_embedding_dimension()
        if encoder_dim is None:
            raise ValueError("Sentence Transformer embedding dimension is unknown")
        self.projection = nn.Linear(encoder_dim, embedding_dim)
        self.projection.to(self.encoder.device)

    def forward(self, sentences: Sequence[str]) -> torch.Tensor:
        """Encode texts as unit-normalized Euclidean embeddings."""
        features = self.encoder.preprocess(list(sentences))
        features = {
            key: value.to(self.encoder.device) if torch.is_tensor(value) else value
            for key, value in features.items()
        }
        encoder_output: dict[str, Any] = self.encoder(features)
        projected = self.projection(encoder_output["sentence_embedding"])
        return F.normalize(projected, dim=-1)

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        convert_to_tensor: bool = False,
    ) -> Any:
        """Encode texts for evaluator compatibility."""
        single_input = isinstance(sentences, str)
        batch = [sentences] if single_input else list(sentences)
        self.eval()
        with torch.inference_mode():
            embeddings = self(batch)
        if single_input:
            embeddings = embeddings[0]
        if convert_to_tensor:
            return embeddings
        return embeddings.cpu().numpy()

    def distance(self, a: Any, b: Any) -> torch.Tensor:
        """Return cosine distance for broadcast-compatible embedding tensors."""
        reference = next(self.parameters())
        a_tensor = torch.as_tensor(a, device=reference.device, dtype=reference.dtype)
        b_tensor = torch.as_tensor(b, device=reference.device, dtype=reference.dtype)
        return 1.0 - F.cosine_similarity(a_tensor, b_tensor, dim=-1)


class EuclideanMultipleNegativesRankingLoss(nn.Module):
    """Benchmark-local cosine counterpart to neembed's geodesic ranking loss."""

    def __init__(
        self,
        *,
        model: EuclideanSentenceTransformer,
        temperature: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.temperature = temperature

    def forward(
        self,
        anchors: Sequence[str],
        positives: Sequence[str],
    ) -> torch.Tensor:
        """Return in-batch multiple-negatives loss using cosine similarity."""
        anchor_embeddings = self.model(anchors)
        positive_embeddings = self.model(positives)
        logits = anchor_embeddings @ positive_embeddings.T / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets)


def _train_euclidean(
    model: EuclideanSentenceTransformer,
) -> list[float]:
    """Train the Euclidean baseline with the same schedule as neembed."""
    loss = EuclideanMultipleNegativesRankingLoss(
        model=model,
        temperature=TEMPERATURE,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[float] = []

    for _ in range(EPOCHS):
        model.train()
        total_loss = 0.0
        steps = 0
        for anchors, positives in _train_batches():
            optimizer.zero_grad()
            batch_loss = loss(anchors, positives)
            batch_loss.backward()
            optimizer.step()
            total_loss += float(batch_loss.detach())
            steps += 1
        history.append(total_loss / steps)

    return history


def _evaluator(model: nn.Module) -> ManifoldEmbeddingEvaluator:
    """Build the shared v0.2 evaluator for the fixed held-out task."""
    return ManifoldEmbeddingEvaluator(
        model=model,  # type: ignore[arg-type]
        anchors=[anchor for anchor, _ in EVALUATION_PAIRS],
        positives=[positive for _, positive in EVALUATION_PAIRS],
    )


def _result(
    *,
    distance: str,
    metrics: dict[str, float],
    final_training_loss: float,
) -> dict[str, float | str]:
    """Combine shared evaluator metrics with the final training loss."""
    return {
        "distance": distance,
        **metrics,
        "final_training_loss": final_training_loss,
    }


def run_benchmark() -> dict[str, object]:
    """Run the deterministic tiny hierarchy retrieval benchmark."""
    set_seed(SEED)
    euclidean_model = EuclideanSentenceTransformer(
        MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
    )
    euclidean_history = _train_euclidean(euclidean_model)
    euclidean_metrics = _evaluator(euclidean_model)()

    set_seed(SEED)
    poincare_model = ManifoldSentenceTransformer(
        MODEL_NAME,
        manifold="poincare",
        embedding_dim=EMBEDDING_DIM,
        curvature=CURVATURE,
    )
    poincare_loss = ManifoldMultipleNegativesRankingLoss(
        model=poincare_model,
        temperature=TEMPERATURE,
    )
    poincare_trainer = ManifoldTrainer(
        model=poincare_model,
        loss=poincare_loss,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        verbose=False,
    )
    poincare_history = poincare_trainer.fit(_train_batches(), epochs=EPOCHS)
    poincare_metrics = _evaluator(poincare_model)()

    return {
        "metadata": {
            "benchmark": "tiny_hierarchy_retrieval",
            "model": MODEL_NAME,
            "seed": SEED,
            "embedding_dim": EMBEDDING_DIM,
            "temperature": TEMPERATURE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "curvature": CURVATURE,
            "train_pairs": [
                {"anchor": anchor, "positive": positive}
                for anchor, positive in TRAIN_PAIRS
            ],
            "evaluation_pairs": [
                {"anchor": anchor, "positive": positive}
                for anchor, positive in EVALUATION_PAIRS
            ],
        },
        "results": {
            "euclidean_finetuned": _result(
                distance="cosine",
                metrics=euclidean_metrics,
                final_training_loss=float(euclidean_history[-1]),
            ),
            "poincare_finetuned": _result(
                distance="poincare_geodesic",
                metrics=poincare_metrics,
                final_training_loss=float(poincare_history[-1]),
            ),
        },
    }


def run_experiment() -> dict[str, object]:
    """Backward-compatible alias for the original experiment entry point."""
    return run_benchmark()


def main() -> None:
    """Run the benchmark and emit one stable machine-readable JSON result."""
    print(json.dumps(run_benchmark(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
