# Experiments

## Euclidean vs Poincaré vs Lorentz benchmark

The [v0.3 comparison benchmark](compare_euclidean_poincare.py) is a small engineering and regression reference for neembed. It compares three models built from the same pretrained Sentence Transformer:

- a learned 32-dimensional Euclidean projection trained with cosine multiple-negatives ranking,
- a learned 32-dimensional neembed Poincaré projection trained with geodesic multiple-negatives ranking, and
- a learned 32-dimensional neembed Lorentz projection trained with geodesic multiple-negatives ranking.

All three variants use the same train pairs, held-out evaluation pairs, intrinsic projection dimension, batch size, number of epochs, temperature, optimizer family, learning rate, weight decay, and fixed seed. Poincaré and Lorentz also use the same public curvature magnitude. The geometry and corresponding distance are the intended differences.

For Lorentz, `embedding_dim=32` still means a 32-dimensional intrinsic projection. The hyperboloid representation adds one time-like coordinate, so the Lorentz output is 33-dimensional. The benchmark records this difference in `metadata.ambient_dimensions`; it is not a larger intrinsic model capacity.

Run it from the repository root:

```bash
python experiments/compare_euclidean_poincare.py
```

The command emits one JSON object containing the fixed configuration, exact train/evaluation pairs, ambient dimensions, and these metrics for each variant:

- `retrieval_accuracy`
- `mrr`
- `recall_at_1`
- `mean_positive_distance`
- `mean_negative_distance`
- `final_training_loss`

The five evaluation metrics are computed through the shared `ManifoldEmbeddingEvaluator`; the Euclidean benchmark wrapper exposes the same minimal `encode()` / `distance()` contract only inside this experiment. In the current one-positive-per-anchor evaluation contract, `recall_at_1` is equivalent to `retrieval_accuracy`.

### Fixed configuration

- encoder: `sentence-transformers/all-MiniLM-L6-v2`
- seed: `0`
- intrinsic embedding dimension: `32` for all variants
- ambient dimensions: Euclidean `32`, Poincaré `32`, Lorentz `33`
- hyperbolic curvature magnitude: `1.0` for both Poincaré and Lorentz
- temperature: `0.1`
- optimizer: `AdamW`
- learning rate: `2e-5`
- weight decay: `0.01`
- epochs: `10`
- batch size: `5`

The benchmark fixes Python, NumPy, and PyTorch random seeds and resets the same seed before constructing each variant. It also keeps the task definition in the repository. Timing is intentionally omitted because runtime measurements are hardware-dependent and make the output less useful as a deterministic regression reference.

### Interpretation limits

This is a tiny controlled hierarchy-retrieval task, not a leaderboard or a research result. A higher score for any geometry in this benchmark does **not** establish general superiority. Raw mean distances also use geometry-specific distance scales, so they should not be treated as directly comparable universal quality scores across cosine, Poincaré, and Lorentz spaces. The benchmark exists to make the three paths reproducible under controlled conditions and to catch behavioral regressions as neembed evolves.
