"""Evaluation helpers for manifold-valued sentence embeddings."""

from collections.abc import Sequence

import torch

from neembed.model import ManifoldSentenceTransformer


class ManifoldEmbeddingEvaluator:
    """Evaluate aligned anchor-positive pairs using manifold geodesic distance.

    Each anchor is matched against every positive candidate. The positive at the
    same index is treated as the correct retrieval target, while off-diagonal
    candidates are treated as negatives.

    Args:
        model: Manifold sentence embedding model to evaluate.
        anchors: Anchor texts. At least two aligned pairs are required.
        positives: Positive texts aligned by index with ``anchors``.
        recall_at_k: Positive integer cutoffs for Recall@K metrics. The default
            reports Recall@1. A cutoff greater than or equal to the candidate
            count has Recall@K equal to 1 because the aligned target is always
            present in the candidate pool.

    Raises:
        ValueError: If the anchor and positive counts differ, fewer than two
            aligned pairs are provided, or ``recall_at_k`` is empty, duplicated,
            or contains a non-positive integer.
    """

    def __init__(
        self,
        *,
        model: ManifoldSentenceTransformer,
        anchors: Sequence[str],
        positives: Sequence[str],
        recall_at_k: Sequence[int] = (1,),
    ) -> None:
        self.model = model
        self.anchors = list(anchors)
        self.positives = list(positives)
        self.recall_at_k = tuple(recall_at_k)

        if len(self.anchors) != len(self.positives):
            raise ValueError("anchors and positives must contain the same number of items")
        if len(self.anchors) < 2:
            raise ValueError("evaluation requires at least two aligned pairs")
        if not self.recall_at_k:
            raise ValueError("recall_at_k must contain at least one cutoff")
        if any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0
            for k in self.recall_at_k
        ):
            raise ValueError("recall_at_k values must be positive integers")
        if len(set(self.recall_at_k)) != len(self.recall_at_k):
            raise ValueError("recall_at_k values must be unique")

    def __call__(self) -> dict[str, float]:
        """Return retrieval, ranking, and aligned/off-diagonal distance metrics.

        Returns:
            A dictionary containing the existing ``retrieval_accuracy``,
            ``mean_positive_distance``, and ``mean_negative_distance`` metrics,
            plus ``mrr`` and one ``recall_at_<k>`` key per configured cutoff.

        Notes:
            Evaluation does not track gradients. Because ``model.encode()`` enters
            evaluation mode, the model's original train/eval mode is restored
            before this method returns. Candidate ranks are ordered by ascending
            manifold geodesic distance with stable index-order tie handling.
        """
        was_training = self.model.training
        try:
            with torch.no_grad():
                anchor_embeddings = self.model.encode(
                    self.anchors,
                    convert_to_tensor=True,
                )
                positive_embeddings = self.model.encode(
                    self.positives,
                    convert_to_tensor=True,
                )

                distances = self.model.distance(
                    anchor_embeddings.unsqueeze(1),
                    positive_embeddings.unsqueeze(0),
                )

                pair_count = len(self.anchors)
                targets = torch.arange(pair_count, device=distances.device)
                positive_distances = distances.diagonal()

                # Compute the exact stable rank of each aligned target without
                # materializing an N x N int64 argsort result. Candidates with a
                # smaller distance rank first; equal-distance candidates preserve
                # their original index order.
                target_distances = positive_distances.unsqueeze(1)
                closer_counts = (distances < target_distances).sum(dim=1)
                candidate_indices = torch.arange(
                    pair_count,
                    device=distances.device,
                ).unsqueeze(0)
                equal_before_counts = (
                    (distances == target_distances)
                    & (candidate_indices < targets.unsqueeze(1))
                ).sum(dim=1)
                target_ranks = closer_counts + equal_before_counts + 1

                negative_mask = ~torch.eye(
                    pair_count,
                    dtype=torch.bool,
                    device=distances.device,
                )
                negative_distances = distances[negative_mask]

                metrics = {
                    "retrieval_accuracy": float(
                        (target_ranks == 1).float().mean().item()
                    ),
                    "mean_positive_distance": float(
                        positive_distances.mean().item()
                    ),
                    "mean_negative_distance": float(
                        negative_distances.mean().item()
                    ),
                    "mrr": float(
                        target_ranks.float().reciprocal().mean().item()
                    ),
                }
                for k in self.recall_at_k:
                    metrics[f"recall_at_{k}"] = float(
                        (target_ranks <= k).float().mean().item()
                    )
                return metrics
        finally:
            self.model.train(was_training)
