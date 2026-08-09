# neembed

**Fine-tune pretrained sentence embedding models in non-Euclidean spaces with Geoopt.**

[日本語](docs/README_ja.md)

> **Status:** v0.1 is implemented and being prepared for its first public PyPI release. The API is intentionally small and may still evolve before a stable 1.0 release.

`neembed` is a lightweight integration layer between pretrained text embedding models and manifold-valued representations.

```text
Pretrained Sentence Encoder
        │
        ▼
Euclidean sentence embedding
        │
        ▼
Projection head (optional)
        │
        ▼
Tangent-space representation
        │
        ▼
Geoopt expmap₀
        │
        ▼
Non-Euclidean embedding
        │
        ▼
Geodesic-distance loss
```

Rather than reimplementing Riemannian geometry, `neembed` delegates manifold operations to [Geoopt](https://geoopt.readthedocs.io/) and focuses on making existing Sentence Transformer models easy to fine-tune in non-Euclidean spaces.

## Why neembed?

Most pretrained sentence embedding models represent text in Euclidean vector spaces. This works well for many semantic similarity and retrieval tasks, but hierarchical or tree-like data can be awkward to represent in a flat space.

Typical examples include:

- taxonomies and concept hierarchies
- knowledge graphs
- hierarchical labels
- tree-like semantic relations

Hyperbolic spaces are especially attractive for hierarchical data because they can represent rapidly expanding tree-like structures compactly.

## Design goals

1. **Reuse pretrained embedding models.** Do not rebuild encoders already provided by Sentence Transformers or Hugging Face.
2. **Reuse Geoopt for geometry.** Do not maintain custom exponential maps, logarithmic maps, or geodesic-distance implementations without a compelling reason.
3. **Keep the public API small.** The core concepts are `Model`, `Loss`, and `Trainer`.
4. **Start with one geometry.** v0.1 supports the Poincaré ball only.
5. **Separate manifold-valued outputs from manifold-valued parameters.** Manifold-valued outputs alone do not require a Riemannian optimizer.

## v0.1 scope

The first release includes:

- Sentence Transformers as the pretrained encoder backend
- Poincaré ball embeddings through Geoopt
- optional lower-dimensional tangent-space projection
- geodesic distance
- in-batch multiple-negatives ranking / InfoNCE-style loss
- ordinary `AdamW` fine-tuning
- `encode()` and `distance()` inference helpers
- local `save_pretrained()` / `from_pretrained()` round trips
- numerical stability coverage for the Poincaré path
- a runnable end-to-end training example
- a reproducible Euclidean-vs-Poincaré baseline experiment

Not included in v0.1:

- custom manifold mathematics
- generic manifold registries
- Lorentz, spherical, SPD, or product manifolds
- learnable curvature
- manifold-valued trainable prototypes or classifiers
- distributed contrastive training
- vector database / ANN functionality
- a replacement for Sentence Transformers

## Installation

After `v0.1.0` is published to PyPI:

```bash
pip install neembed
```

For development:

```bash
git clone https://github.com/t-yamsaki/neembed.git
cd neembed
pip install -e ".[dev]"
```

Runtime dependencies are intentionally limited to:

```text
torch
sentence-transformers
geoopt
```

## Quick start

```python
from neembed import (
    ManifoldSentenceTransformer,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldTrainer,
)

model = ManifoldSentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    manifold="poincare",
    embedding_dim=64,
    curvature=1.0,
)

loss = ManifoldMultipleNegativesRankingLoss(
    model=model,
    temperature=0.1,
)

trainer = ManifoldTrainer(
    model=model,
    loss=loss,
)

train_batches = [
    (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
    (["dog", "cat"], ["mammal", "feline"]),
]

trainer.fit(train_batches, epochs=1)

embeddings = model.encode([
    "Shiba Inu",
    "dog",
    "mammal",
])

distance = model.distance(embeddings[0], embeddings[1])
print(float(distance))
```

Each anchor is paired with the positive at the same batch index. Because off-diagonal positive candidates are used as in-batch negatives, avoid duplicate positives within the same batch.

For a complete runnable workflow, see [examples/train_poincare.py](examples/train_poincare.py):

```bash
python examples/train_poincare.py
```

## Model architecture

For the v0.1 Poincaré implementation:

```text
SentenceTransformer
        │
        │ pretrained sentence embedding h ∈ R^d
        ▼
Linear projection (optional)
        │
        │ tangent vector v ∈ T₀M
        ▼
Geoopt expmap₀
        │
        │ z ∈ M
        ▼
Poincaré embedding
```

Conceptually,

$$
h = f_\theta(x),
$$

$$
v = Wh,
$$

$$
z = \mathrm{Exp}_0^c(v),
$$

where $f_\theta$ is the pretrained sentence encoder, $W$ is an optional projection layer, and $z$ is the final manifold-valued embedding.

## Training objective

For a batch of anchor-positive pairs, `neembed` replaces Euclidean/cosine similarity with negative geodesic distance:

$$
s_{ij} = -\frac{d_{\mathcal M}(z_i, z_j^+)}{\tau}.
$$

The in-batch objective is then

$$
\mathcal L_i = -\log\frac{\exp(s_{ii})}{\sum_j \exp(s_{ij})}.
$$

This is the manifold-aware analogue of a multiple-negatives ranking / InfoNCE objective.

## Why AdamW is enough for v0.1

The encoder and optional projection are ordinary Euclidean PyTorch parameters. Only the output representation is mapped onto the manifold, so gradients can flow through Geoopt's map and geodesic distance while the trainable parameters are optimized with ordinary `AdamW`.

A Riemannian optimizer becomes relevant when trainable parameters themselves live on a manifold, such as trainable hierarchy nodes or manifold prototypes. That is outside the v0.1 scope.

## Saving and loading

```python
model.save_pretrained("./saved_model")

loaded = ManifoldSentenceTransformer.from_pretrained("./saved_model")
```

A saved model contains the underlying Sentence Transformer state plus the neembed projection and manifold configuration.

## Validation experiment

The first comparison experiment evaluates the original pretrained Euclidean encoder and the same encoder after Poincaré fine-tuning on the same held-out hierarchy retrieval pairs:

```bash
python experiments/compare_euclidean_poincare.py
```

See [experiments/README.md](experiments/README.md) for the fixed configuration and interpretation notes. The experiment is intentionally small and is not a claim that hyperbolic embeddings are universally superior.

## Numerical considerations

Poincaré embeddings can become numerically sensitive near the ball boundary. v0.1 delegates projection and geometry to Geoopt and includes tests for:

- finite embeddings for representative and large tangent vectors
- valid Poincaré ball domain
- finite near-boundary geodesic distances
- finite gradients through the forward/loss path
- curvature and temperature edge cases

## Roadmap

### v0.1 — Minimal hyperbolic fine-tuning

- [x] Sentence Transformer backbone
- [x] Poincaré ball through Geoopt
- [x] optional projection head
- [x] geodesic distance
- [x] manifold multiple-negatives ranking loss
- [x] minimal trainer
- [x] `encode()`
- [x] save / load
- [x] numerical stability tests
- [x] minimal end-to-end example
- [x] Euclidean baseline experiment

### v0.2 — More objectives and geometries

- [ ] triplet loss
- [ ] explicit hard negatives
- [ ] Lorentz model
- [ ] evaluation utilities
- [ ] Hugging Face Dataset examples

### v0.3 — Learnable manifold structure

- [ ] learnable curvature
- [ ] `geoopt.ManifoldParameter`
- [ ] RiemannianAdam
- [ ] manifold prototypes
- [ ] hierarchical objectives

### Later

- [ ] spherical embeddings
- [ ] product manifolds
- [ ] mixed-curvature representations
- [ ] distributed contrastive learning
- [ ] retrieval / reranking integrations

## What neembed is not

`neembed` is not intended to be:

- a new Riemannian geometry framework
- a replacement for Geoopt
- a replacement for Sentence Transformers
- a full hyperbolic neural-network library
- a vector database

Its role is deliberately smaller:

> **Turn pretrained sentence embedding models into manifold-valued embedding models with Geoopt.**

## Contributing

The project prioritizes a small, well-tested surface area. Before proposing a large abstraction or subsystem, consider:

1. Is it required for the target manifold fine-tuning workflow?
2. Can Geoopt or Sentence Transformers already provide it?
3. Can it be added later without breaking the core API?

Small, well-tested additions are preferred over broad framework-level abstractions.

## References

- [Geoopt documentation](https://geoopt.readthedocs.io/)
- [Sentence Transformers documentation](https://www.sbert.net/)
- [Creating Custom Sentence Transformer Models](https://www.sbert.net/docs/sentence_transformer/usage/custom_models.html)

## License

`neembed` is released under the [MIT License](LICENSE).

Third-party dependencies retain their own licenses. In particular, neembed depends on Geoopt rather than vendoring or relicensing Geoopt source code.
