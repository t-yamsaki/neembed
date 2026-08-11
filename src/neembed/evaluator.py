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

    Raises:
        ValueError: If the anchor and positive counts differ or fewer than two
            aligned pairs are provided.
    """

    def __init__(
        self,
        *,
        model: ManifoldSentenceTransformer,
        anchors: Sequence[str],
        positives: Sequence[str],
    ) -> None:
        self.model = model
        self.anchors = list(anchors)
        self.positives = list(positives)

        if len(self.anchors) != len(self.positives):
            raise ValueError("anchors and positives must contain the same number of items")
        if len(self.anchors) < 2:
            raise ValueError("evaluation requires at least two aligned pairs")

    def __call__(self) -> dict[str, float]:
        """Return retrieval accuracy and aligned/off-diagonal distance means.

        Returns:
            A dictionary containing ``retrieval_accuracy``,
            ``mean_positive_distance``, and ``mean_negative_distance``.

        Notes:
            Evaluation does not track gradients. Because ``model.encode()`` enters
            evaluation mode, the model's original train/eval mode is restored
            before this method returns.
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
                retrieved = distances.argmin(dim=1)
                positive_distances = distances.diagonal()
                negative_mask = ~torch.eye(
                    pair_count,
                    dtype=torch.bool,
                    device=distances.device,
                )
                negative_distances = distances[negative_mask]

                return {
                    "retrieval_accuracy": float(
                        (retrieved == targets).float().mean().item()
                    ),
                    "mean_positive_distance": float(
                        positive_distances.mean().item()
                    ),
                    "mean_negative_distance": float(
                        negative_distances.mean().item()
                    ),
                }
        finally:
            self.model.train(was_training)
