# neembed

**Fine-tune pretrained sentence embedding models in non-Euclidean spaces with Geoopt.**

[Documentation](https://neembed.readthedocs.io/en/latest/) · [日本語](docs/README_ja.md)

> **Status:** Package version v0.5.0 adds caller-supplied explicit hard-negative training, Recall@K / MRR evaluation, small in-memory geodesic reranking, and prototype-assignment evaluation while preserving the v0.4 learnable-structure path and the original fixed-curvature model-only workflow. The API remains intentionally small and may still evolve before a stable 1.0 release.

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

The current API supports the Poincaré ball and Lorentz / Hyperboloid models through the same model, loss, trainer, evaluator, and sentence-model save/load workflow.

## Current scope

v0.4 keeps the fixed-curvature v0.3 path backward-compatible:

- Sentence Transformers as the pretrained encoder backend
- Poincaré-ball and Lorentz / Hyperboloid embeddings through Geoopt
- optional lower-dimensional tangent-space projection
- shared public curvature semantics across the two hyperbolic models
- geodesic distance and manifold-aware multiple-negatives ranking loss
- ordinary `AdamW` fine-tuning for the model-only path
- `encode()` / `distance()`, `ManifoldEmbeddingEvaluator`, DataLoader interoperability, and local save/load
- geometry-consistency regressions and a matched Euclidean-vs-Poincaré-vs-Lorentz engineering benchmark

v0.4 also adds:

- opt-in fixed vs learnable curvature for Poincaré and Lorentz
- true trainable manifold-valued `ManifoldPrototypes`
- `ManifoldPrototypeHierarchyLoss` for sentence assignments plus parent-child structure
- an explicit caller-supplied Geoopt Riemannian optimizer path for manifold parameters
- joint learnable-curvature / prototype training through Geoopt stabilization
- a compact fixed-vs-learnable structure regression example

v0.5 adds a focused retrieval workflow without introducing a retrieval framework:

- optional caller-supplied explicit hard negatives while preserving the original `(anchors, positives)` training contract
- Recall@K and MRR on the aligned retrieval evaluator
- `model.rank()` for small in-memory geodesic reranking
- nearest-prototype assignment evaluation for learned manifold structure
- one reproducible v0.5 retrieval regression example

A manifold-valued **output** does not by itself require Riemannian optimization: encoder/projection parameters and learnable curvature are not manifold-valued points. Detailed parameter, optimizer, persistence, and numerical behavior lives in the [Learnable structure guide](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html). The end-to-end retrieval composition and its in-memory boundary are documented in the [Retrieval workflow guide](https://neembed.readthedocs.io/en/latest/user_guide/retrieval.html).

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

Each anchor is paired with the positive at the same batch index. Because off-diagonal candidates become in-batch negatives, avoid duplicate positives within one batch. This model-only path keeps the ordinary AdamW behavior even though its outputs lie on a manifold. See the [Training guide](https://neembed.readthedocs.io/en/latest/user_guide/training.html) for the objective and batching details, the [Retrieval workflow guide](https://neembed.readthedocs.io/en/latest/user_guide/retrieval.html) for optional explicit negatives and retrieval evaluation, and the [Learnable structure guide](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html) before adding trainable manifold prototypes.

## Documentation

The full guide is hosted on Read the Docs:

- [Installation](https://neembed.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start](https://neembed.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Architecture](https://neembed.readthedocs.io/en/latest/user_guide/architecture.html)
- [Learnable structure](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html)
- [Training](https://neembed.readthedocs.io/en/latest/user_guide/training.html)
- [Retrieval workflow](https://neembed.readthedocs.io/en/latest/user_guide/retrieval.html)
- [Evaluation](https://neembed.readthedocs.io/en/latest/user_guide/evaluation.html)
- [Inference](https://neembed.readthedocs.io/en/latest/user_guide/inference.html)
- [Saving and Loading](https://neembed.readthedocs.io/en/latest/user_guide/saving_loading.html)
- [API Reference](https://neembed.readthedocs.io/en/latest/#api-reference)

## Examples and validation

Run the main references from the repository root:

```bash
python examples/train_poincare.py
python examples/train_lorentz.py
python examples/v04_learnable_structure.py
python examples/v05_retrieval_workflow.py
```

- [examples/train_poincare.py](examples/train_poincare.py) — minimal Poincaré workflow
- [examples/train_lorentz.py](examples/train_lorentz.py) — Lorentz train/evaluate/inference with intrinsic-vs-ambient dimensions
- [examples/train_dataloader.py](examples/train_dataloader.py) — ordinary PyTorch `DataLoader` training and epoch validation
- [examples/train_hierarchy.py](examples/train_hierarchy.py) — minimal hierarchy-aware prototype objective
- [examples/v04_learnable_structure.py](examples/v04_learnable_structure.py) — fixed-vs-learnable structure regression diagnostics; not a superiority benchmark
- [examples/v05_retrieval_workflow.py](examples/v05_retrieval_workflow.py) — explicit hard negatives, Recall@K / MRR, in-memory geodesic reranking, and prototype assignment in one Poincaré regression workflow; not a research benchmark
- [experiments/README.md](experiments/README.md) — reproducible Euclidean-vs-Poincaré-vs-Lorentz engineering benchmark and interpretation limits

## License

`neembed` is released under the [MIT License](LICENSE). Third-party dependencies retain their own licenses.
