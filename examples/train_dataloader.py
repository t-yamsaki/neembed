"""Train neembed with an ordinary PyTorch DataLoader."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from neembed import (
    ManifoldEmbeddingEvaluator,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)


def set_seed(seed: int) -> None:
    """Seed the random sources used by this CPU example."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    """Train for multiple epochs from a re-iterable DataLoader."""
    set_seed(0)

    train_pairs = [
        ("Shiba Inu", "dog"),
        ("Siamese cat", "cat"),
        ("sparrow", "bird"),
        ("salmon", "fish"),
        ("oak", "plant"),
        ("rose", "flower"),
        ("Python", "programming language"),
        ("violin", "instrument"),
    ]
    train_dataloader = DataLoader(
        train_pairs,
        batch_size=4,
        shuffle=False,
    )

    model = ManifoldSentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        manifold="poincare",
        embedding_dim=32,
        curvature=1.0,
    )
    loss = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)
    trainer = ManifoldTrainer(model=model, loss=loss)
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["Shiba Inu", "Siamese cat", "sparrow", "salmon"],
        positives=["dog", "cat", "bird", "fish"],
    )

    history = trainer.fit(
        train_dataloader,
        epochs=2,
        evaluator=evaluator,
    )

    print("\nValidation history:")
    for epoch, record in enumerate(history, start=1):
        metrics = record["validation"]
        print(
            f"Epoch {epoch}: "
            f"retrieval_accuracy={metrics['retrieval_accuracy']:.3f}, "
            f"mean_positive_distance={metrics['mean_positive_distance']:.3f}, "
            f"mean_negative_distance={metrics['mean_negative_distance']:.3f}"
        )


if __name__ == "__main__":
    main()
