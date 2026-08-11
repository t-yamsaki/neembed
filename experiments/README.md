# Experiments

## Euclidean vs Poincaré benchmark

The [v0.2 comparison benchmark](compare_euclidean_poincare.py) is a small engineering and regression reference for neembed. It compares two models built from the same pretrained Sentence Transformer:

- a learned 32-dimensional Euclidean projection trained with cosine multiple-negatives ranking, and
- a learned 32-dimensional neembed Poincaré projection trained with geodesic multiple-negatives ranking.

Both variants use the same train pairs, held-out evaluation pairs, batch size, number of epochs, temperature, optimizer family, learning rate, and weight decay. The geometry and corresponding distance are the intended differences.

Run it from the repository root:

```bash
python experiments/compare_euclidean_poincare.py
```

The command emits one JSON object containing the fixed configuration, exact train/evaluation pairs, and these metrics for each variant:

- `retrieval_accuracy`
- `mean_positive_distance`
- `mean_negative_distance`
- `final_training_loss`

The three evaluation metrics are computed through the shared v0.2 `ManifoldEmbeddingEvaluator`; the Euclidean benchmark wrapper exposes the same minimal `encode()` / `distance()` contract only inside this experiment.

### Fixed configuration

- encoder: `sentence-transformers/all-MiniLM-L6-v2`
- seed: `0`
- embedding dimension: `32` for both variants
- Poincaré curvature: `1.0`
- temperature: `0.1`
- optimizer: `AdamW`
- learning rate: `2e-5`
- weight decay: `0.01`
- epochs: `10`
- batch size: `5`

The benchmark fixes Python, NumPy, and PyTorch random seeds and keeps the task definition in the repository. Timing is intentionally omitted because runtime measurements are hardware-dependent and make the output less useful as a deterministic regression reference.

### Interpretation limits

This is a tiny controlled hierarchy-retrieval task, not a leaderboard or a research result. A higher score for either geometry in this benchmark does **not** establish general superiority. The benchmark exists to make the Euclidean/Poincaré comparison reproducible and to catch behavioral regressions as neembed adds future geometry support.
