"""Minimal end-to-end Poincaré fine-tuning example for neembed."""

from neembed import (
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)


def main() -> None:
    """Train on a tiny hierarchy-style dataset and print geodesic distances."""
    train_batches = [
        (
            ["Shiba Inu", "Siamese cat", "sparrow", "salmon"],
            ["dog", "cat", "bird", "fish"],
        ),
        (
            ["dog", "sparrow", "salmon", "oak"],
            ["mammal", "vertebrate", "aquatic animal", "plant"],
        ),
    ]

    model = ManifoldSentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        manifold="poincare",
        embedding_dim=32,
        curvature=1.0,
    )
    loss = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)
    trainer = ManifoldTrainer(model=model, loss=loss)

    trainer.fit(train_batches, epochs=2)

    terms = ["Shiba Inu", "dog", "mammal", "animal", "car"]
    embeddings = model.encode(terms)
    by_term = dict(zip(terms, embeddings))

    print("\nRepresentative geodesic distances:")
    for other in ("dog", "animal", "car"):
        distance = model.distance(by_term["Shiba Inu"], by_term[other])
        print(f"Shiba Inu -> {other}: {float(distance):.6f}")


if __name__ == "__main__":
    main()
