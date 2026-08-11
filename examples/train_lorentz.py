"""Minimal Lorentz training and evaluation example for neembed."""

import math
import random

import numpy as np
import torch

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
    """Train and evaluate the standard neembed workflow on Lorentz geometry."""
    set_seed(0)

    train_batches = [
        (
            ["Shiba Inu", "Siamese cat", "sparrow", "salmon"],
            ["dog", "cat", "bird", "fish"],
        ),
        (
            ["dog", "cat", "bird", "fish"],
            ["mammal", "feline", "vertebrate", "aquatic animal"],
        ),
    ]

    model = ManifoldSentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        manifold="lorentz",
        embedding_dim=32,
        curvature=1.0,
    )
    loss = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)
    trainer = ManifoldTrainer(model=model, loss=loss)
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["Beagle", "Maine Coon", "eagle", "tuna"],
        positives=["dog", "cat", "bird", "fish"],
    )

    history = trainer.fit(
        train_batches,
        epochs=2,
        evaluator=evaluator,
    )
    final_record = history[-1]
    final_loss = float(final_record["train_loss"])
    metrics = final_record["validation"]

    if not math.isfinite(final_loss) or not all(
        math.isfinite(float(value)) for value in metrics.values()
    ):
        raise RuntimeError("Lorentz example produced a non-finite training result")

    print("\nFinal validation:")
    print(f"train_loss={final_loss:.6f}")
    print(f"retrieval_accuracy={metrics['retrieval_accuracy']:.3f}")
    print(f"mean_positive_distance={metrics['mean_positive_distance']:.3f}")
    print(f"mean_negative_distance={metrics['mean_negative_distance']:.3f}")

    terms = ["Shiba Inu", "dog", "mammal"]
    embeddings = model.encode(terms, convert_to_tensor=True)
    ambient_dim = embeddings.shape[-1]
    if ambient_dim != model.embedding_dim + 1:
        raise RuntimeError("Lorentz output must add one ambient time-like coordinate")

    print("\nLorentz representation:")
    print(
        f"intrinsic_dimension={model.embedding_dim} "
        f"ambient_dimension={ambient_dim} dtype={embeddings.dtype}"
    )
    distance = model.distance(embeddings[0], embeddings[1])
    print(f"Shiba Inu -> dog geodesic_distance={float(distance):.6f}")


if __name__ == "__main__":
    main()
