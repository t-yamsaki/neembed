"""Minimal hierarchy-aware prototype training example."""

import geoopt

from neembed import (
    ManifoldPrototypeHierarchyLoss,
    ManifoldPrototypes,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)


model = ManifoldSentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    manifold="poincare",
    embedding_dim=16,
)
prototype_ids = ["animal", "dog", "cat"]
parent_relations = [("dog", "animal"), ("cat", "animal")]
prototypes = ManifoldPrototypes(model, num_prototypes=len(prototype_ids))
loss = ManifoldPrototypeHierarchyLoss(
    model,
    prototypes,
    prototype_ids=prototype_ids,
    parent_relations=parent_relations,
    margin=0.1,
)
optimizer = geoopt.optim.RiemannianAdam(
    [parameter for parameter in loss.parameters() if parameter.requires_grad],
    lr=1e-3,
)
trainer = ManifoldTrainer(
    model,
    loss,
    optimizer=optimizer,
)

train_batches = [
    (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
    (["Golden retriever", "Persian cat"], ["dog", "cat"]),
]

if __name__ == "__main__":
    print(trainer.fit(train_batches, epochs=1))
