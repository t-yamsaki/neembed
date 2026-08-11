# neembed

**Fine-tune pretrained sentence embedding models in non-Euclidean spaces with Geoopt.**

[Documentation](https://neembed.readthedocs.io/en/latest/) · [日本語](docs/README_ja.md)

> **Status:** v0.2.0 adds evaluation, epoch validation, DataLoader interoperability examples, and a reproducible Euclidean-vs-Poincaré benchmark. Development toward v0.3.0 adds Lorentz / Hyperboloid embeddings as the second supported hyperbolic geometry. The API remains intentionally small and may still evolve before a stable 1.0 release.

`neembed` is a lightweight integration layer between pretrained Sentence Transformer models and manifold-valued representations. It keeps the pretrained encoder intact, optionally projects its Euclidean output, and delegates hyperbolic geometry to Geoopt.

```text
Pretrained Sentence Encoder
        ↓
Euclidean sentence embedding
        ↓
Projection head (optional)
        ↓
Tangent-space representation
        ↓
Geoopt manifold map
        ↓
Non-Euclidean embedding
```

## Why non-Euclidean embeddings?

Hierarchical and tree-like relations can be awkward to represent in a flat Euclidean space. Hyperbolic spaces are a natural fit for rapidly expanding structures such as:

- taxonomies and concept hierarchies
- knowledge graphs
- hierarchical labels
- tree-like semantic relations

The current development API supports the Poincaré ball and Lorentz / Hyperboloid models. The published v0.2.0 release remains Poincaré-only; Lorentz support is targeted for v0.3.0.

## v0.2

The v0.2 release includes:

- Sentence Transformers as the pretrained encoder backend
- Poincaré-ball embeddings through Geoopt
- optional lower-dimensional tangent-space projection
- geodesic distance
- manifold-aware multiple-negatives ranking / InfoNCE-style loss
- ordinary `AdamW` fine-tuning
- `encode()` and `distance()` inference helpers
- `save_pretrained()` / `from_pretrained()` local persistence
- `ManifoldEmbeddingEvaluator` with retrieval and distance metrics
- optional epoch-end validation in `ManifoldTrainer`
- ordinary PyTorch `DataLoader` interoperability example
- a reproducible Euclidean-vs-Poincaré engineering benchmark

Detailed behavior, assumptions, and API signatures live in the [documentation](https://neembed.readthedocs.io/en/latest/).

## Installation

```bash
pip install neembed-geoopt
```

The PyPI distribution is named `neembed-geoopt`; the Python import package remains `neembed`.

For development:

```bash
git clone https://github.com/t-yamsaki/neembed.git
cd neembed
pip install -e ".[dev]"
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

trainer = ManifoldTrainer(model=model, loss=loss)

train_batches = [
    (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
    (["dog", "cat"], ["mammal", "feline"]),
]

trainer.fit(train_batches, epochs=1)

embeddings = model.encode(["Shiba Inu", "dog", "mammal"])
distance = model.distance(embeddings[0], embeddings[1])
print(float(distance))
```

Each anchor is paired with the positive at the same batch index. Because off-diagonal candidates become in-batch negatives, avoid duplicate positives within one batch. See the [Training guide](https://neembed.readthedocs.io/en/latest/user_guide/training.html) for the objective and batching details.

## Documentation

The full guide is hosted on Read the Docs:

- [Installation](https://neembed.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start](https://neembed.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Architecture](https://neembed.readthedocs.io/en/latest/user_guide/architecture.html)
- [Training](https://neembed.readthedocs.io/en/latest/user_guide/training.html)
- [Evaluation](https://neembed.readthedocs.io/en/latest/user_guide/evaluation.html)
- [Inference](https://neembed.readthedocs.io/en/latest/user_guide/inference.html)
- [Saving and Loading](https://neembed.readthedocs.io/en/latest/user_guide/saving_loading.html)
- [API Reference](https://neembed.readthedocs.io/en/latest/#api-reference)

## Examples and validation

Run the end-to-end example:

```bash
python examples/train_poincare.py
```

- [examples/train_poincare.py](examples/train_poincare.py) contains the minimal workflow.
- [examples/train_dataloader.py](examples/train_dataloader.py) shows ordinary PyTorch `DataLoader` training and epoch validation.
- [experiments/README.md](experiments/README.md) documents the reproducible Euclidean-vs-Poincaré benchmark and its interpretation limits.

## License

`neembed` is released under the [MIT License](LICENSE). Third-party dependencies retain their own licenses.
