# Experiments

## Euclidean baseline

The [Euclidean baseline experiment](compare_euclidean_poincare.py) is the first minimal validation experiment for neembed. It compares:

- the original pretrained Sentence Transformer using cosine distance, and
- the same pretrained encoder after neembed Poincaré fine-tuning using geodesic distance.

Both variants are evaluated on the same held-out child-to-parent retrieval pairs. The experiment reports parent retrieval accuracy in one JSON object together with the training and evaluation runtime notes needed to interpret the comparison. On CUDA, evaluation timing synchronizes the active device immediately before and after each timed section so asynchronous kernels are included in the reported elapsed time.

Run it from the repository root:

```bash
python experiments/compare_euclidean_poincare.py
```

The comparison is intentionally small and is not a benchmark or a claim that hyperbolic embeddings are universally better.

### Fixed configuration

- encoder: `sentence-transformers/all-MiniLM-L6-v2`
- seed: `0`
- Poincaré embedding dimension: `32`
- curvature: `1.0`
- temperature: `0.1`
- optimizer: `AdamW`
- learning rate: `2e-5`
- weight decay: `0.01`
- epochs: `10`
- batch size: `5`
- metric: parent retrieval accuracy

The emitted JSON also contains the exact train and evaluation pairs so the comparison can be reproduced without a separate dataset file.
