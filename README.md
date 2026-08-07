# neembed

**Fine-tune pretrained sentence embedding models in non-Euclidean spaces with Geoopt.**

[日本語](docs/README_ja.md)

> **Status:** Early development. The API shown below describes the intended minimal v0.1 interface and may change before the first stable release.

`neembed` is a lightweight integration layer between pretrained text embedding models and manifold-aware optimization.

The core idea is deliberately small:

```text
Pretrained Sentence Encoder
        │
        ▼
Euclidean sentence embedding
        │
        ▼
Projection head
        │
        ▼
Tangent-space representation
        │
        ▼
Geoopt manifold map
        │
        ▼
Non-Euclidean embedding
        │
        ▼
Geodesic-distance loss
```

Rather than reimplementing Riemannian geometry, `neembed` delegates manifold operations to [Geoopt](https://geoopt.readthedocs.io/) and focuses on making existing embedding models easy to fine-tune in non-Euclidean spaces.

## Why neembed?

Most pretrained sentence embedding models represent text in Euclidean vector spaces. This works well for many semantic similarity and retrieval tasks, but some data have structures that are naturally difficult to represent in a flat space.

Examples include:

- taxonomies and concept hierarchies
- knowledge graphs
- hierarchical labels
- tree-like semantic relations
- data with latent spherical or mixed-curvature structure

Hyperbolic spaces are particularly attractive for hierarchical data because they can represent rapidly expanding tree-like structures compactly.

`neembed` aims to make experiments with these geometries feel close to ordinary Sentence Transformers fine-tuning.

## Design goals

`neembed` follows a few intentionally strict design principles:

1. **Reuse pretrained embedding models.**  
   Do not rebuild encoders that Hugging Face or Sentence Transformers already provide.

2. **Reuse Geoopt for geometry.**  
   Do not maintain custom implementations of exponential maps, logarithmic maps, geodesic distances, or Riemannian optimizers unless there is a compelling reason.

3. **Keep the public API small.**  
   The core concepts should remain `Model`, `Loss`, and `Trainer`.

4. **Start with one geometry.**  
   v0.1 focuses on the Poincaré ball before adding Lorentz, spherical, product, or mixed-curvature spaces.

5. **Separate manifold-valued outputs from manifold-valued parameters.**  
   Producing embeddings on a manifold does not automatically require a Riemannian optimizer.

## Scope

### v0.1

The initial release is intentionally narrow:

- Sentence Transformers as the primary pretrained encoder backend
- Poincaré ball embeddings via Geoopt
- optional projection to a lower-dimensional tangent space
- geodesic distance
- in-batch contrastive / multiple-negatives ranking objective
- Euclidean fine-tuning with `AdamW`
- encoding and distance computation

### Not in v0.1

The following are deliberately postponed:

- custom implementations of manifold mathematics
- arbitrary manifold registries
- Lorentz, sphere, SPD, or product manifolds
- learnable curvature
- manifold prototypes
- Riemannian classifiers
- distributed contrastive training
- custom vector databases or ANN indexes
- large configuration frameworks
- a general-purpose replacement for Sentence Transformers

## Installation

Once released:

```bash
pip install neembed
```

For development:

```bash
git clone https://github.com/<YOUR_USERNAME>/neembed.git
cd neembed
pip install -e ".[dev]"
```

Core dependencies are expected to include:

```text
torch
sentence-transformers
geoopt
```

## Quick start

The intended v0.1 API is:

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

trainer.fit(train_dataset)

embeddings = model.encode([
    "Shiba Inu",
    "dog",
    "mammal",
    "animal",
])

distance = model.distance(
    embeddings[0],
    embeddings[1],
)
```

The goal is for non-Euclidean fine-tuning to require only a small conceptual change from ordinary sentence-embedding fine-tuning.

## Model architecture

For the initial Poincaré implementation:

```text
SentenceTransformer
        │
        │ pretrained sentence embedding h ∈ R^d
        ▼
Linear projection (optional)
        │
        │ tangent vector v ∈ T₀M
        ▼
Scale / stabilization
        │
        ▼
expmap₀
        │
        │ z ∈ M
        ▼
Poincaré embedding
```

Conceptually,

\[
h = f_\theta(x),
\]

\[
v = Wh,
\]

\[
z = \operatorname{Exp}_0^c(v),
\]

where:

- \(f_\theta\) is the pretrained sentence encoder,
- \(W\) is an optional projection layer,
- \(v\) is a tangent-space representation,
- \(z\) is the final manifold-valued embedding.

The Transformer itself remains an ordinary PyTorch model. Only the final representation is mapped onto the manifold.

## Training objective

The initial contrastive objective uses geodesic distance instead of cosine similarity or Euclidean distance.

For a batch of anchor-positive pairs, define:

\[
s_{ij}
=
-\frac{d_{\mathcal M}(z_i, z_j^+)}{\tau},
\]

where \(d_{\mathcal M}\) is the geodesic distance on the manifold and \(\tau\) is the temperature.

The loss is then:

\[
\mathcal L_i
=
-\log
\frac{\exp(s_{ii})}
{\sum_j \exp(s_{ij})}.
\]

This provides a manifold-aware analogue of in-batch multiple-negatives ranking / InfoNCE training.

## Optimizers

A non-Euclidean output does **not** by itself require a Riemannian optimizer.

In the minimal model:

```text
Transformer parameters   ┐
Projection parameters    ├─ Euclidean parameters → AdamW
                         │
Output embeddings        └─ mapped onto a manifold
```

Gradients flow through the manifold map and geodesic loss back into the ordinary model parameters.

A Riemannian optimizer such as Geoopt's `RiemannianAdam` becomes useful when the model contains trainable parameters that themselves live on a manifold, for example:

- manifold prototypes
- trainable hierarchy nodes
- class centroids
- manifold-valued entity embeddings

That functionality is outside the initial v0.1 scope.

## Proposed package structure

```text
neembed/
├── pyproject.toml
├── README.md
├── docs/
│   └── README_ja.md
├── src/
│   └── neembed/
│       ├── __init__.py
│       ├── model.py
│       ├── manifolds.py
│       ├── losses.py
│       └── trainer.py
├── examples/
│   └── train_poincare.py
└── tests/
    ├── test_model.py
    ├── test_losses.py
    └── test_training.py
```

### `model.py`

Owns the integration between a pretrained sentence encoder and a manifold-valued output.

```python
model = ManifoldSentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    manifold="poincare",
    embedding_dim=64,
    curvature=1.0,
)
```

### `manifolds.py`

Provides a deliberately thin interface over Geoopt.

For v0.1, only the Poincaré ball is required.

```python
import geoopt

def get_manifold(name: str, curvature: float = 1.0):
    if name == "poincare":
        return geoopt.PoincareBall(c=curvature)

    raise ValueError(f"Unsupported manifold: {name}")
```

The package should avoid creating its own general manifold abstraction unless real use cases require one.

### `losses.py`

Contains losses defined in terms of manifold geometry.

Initial target:

```text
ManifoldMultipleNegativesRankingLoss
```

Possible later additions:

```text
ManifoldTripletLoss
ManifoldContrastiveLoss
HierarchyLoss
```

### `trainer.py`

Contains a small PyTorch training loop rather than trying to replace the full Hugging Face or Sentence Transformers training stack.

Its responsibilities should stay limited to:

- forward passes
- loss computation
- backward propagation
- optimizer steps
- validation hooks
- checkpointing

## Data

The library itself should not require a custom dataset format.

A pair-based dataset may conceptually contain:

```json
{"anchor": "Shiba Inu", "positive": "dog"}
{"anchor": "dog", "positive": "mammal"}
{"anchor": "mammal", "positive": "vertebrate"}
```

but users should be free to supply ordinary PyTorch or Hugging Face datasets.

Typical positive relations include:

- child → parent
- document → category
- query → relevant document
- paraphrase pairs
- synonym pairs
- clicked query-document pairs

## Evaluation

Useful evaluation metrics depend on the task.

For retrieval:

- Recall@K
- MRR
- nDCG

For hierarchical embeddings:

- parent retrieval accuracy
- ancestor retrieval accuracy
- hierarchy reconstruction
- Spearman correlation between manifold radius and known hierarchy depth

For generic representation learning:

- downstream classification
- clustering quality
- comparison against the original Euclidean encoder

A strong experiment should always include the original pretrained Euclidean model as a baseline.

## Numerical considerations

Hyperbolic models can become unstable when embeddings approach the boundary of the Poincaré ball.

Practical controls may include:

- tangent-vector scaling
- projection / stabilization through Geoopt
- conservative learning rates
- curvature sweeps
- temperature sweeps
- gradient clipping

For deep hierarchies or difficult optimization regimes, a Lorentz-model backend may become a useful future extension.

## Roadmap

### v0.1 — Minimal hyperbolic fine-tuning

- [ ] Sentence Transformer backbone
- [ ] Poincaré ball through Geoopt
- [ ] projection head
- [ ] geodesic distance
- [ ] manifold multiple-negatives ranking loss
- [ ] minimal trainer
- [ ] `encode()`
- [ ] save / load
- [ ] basic tests
- [ ] minimal example

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

Its role is much smaller:

> **Turn pretrained sentence embedding models into manifold-valued embedding models with Geoopt.**

## Contributing

The project is currently focused on keeping the v0.1 surface area small.

Before proposing a large abstraction or new subsystem, prefer an implementation that answers:

1. Is this required for Poincaré fine-tuning?
2. Can Geoopt or Sentence Transformers already do it?
3. Can this be added later without breaking the core API?

Small, well-tested additions are preferred over broad framework-level abstractions.

## References

- [Geoopt documentation](https://geoopt.readthedocs.io/)
- [Sentence Transformers documentation](https://www.sbert.net/)
- [Creating Custom Sentence Transformer Models](https://www.sbert.net/docs/sentence_transformer/usage/custom_models.html)

## License

TBD.
