"""Losses for manifold-valued sentence embeddings."""

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn

from neembed.model import ManifoldSentenceTransformer


class ManifoldMultipleNegativesRankingLoss(nn.Module):
    """In-batch ranking loss based on manifold geodesic distance.

    Each anchor is paired with the positive at the same batch index. All
    off-diagonal positive candidates are treated as negatives, so duplicate
    positives within one batch should be avoided.

    Args:
        model: Manifold sentence model used to encode anchors and positives.
        temperature: Positive, finite temperature used to scale distance logits.
    """

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be positive and finite")

        self.model = model
        self.temperature = float(temperature)

    def forward(
        self,
        anchors: Sequence[str],
        positives: Sequence[str],
    ) -> torch.Tensor:
        """Return the contrastive loss for aligned anchor-positive pairs.

        Args:
            anchors: Batch of anchor texts.
            positives: Batch of positive texts aligned by index with ``anchors``.

        Returns:
            A scalar cross-entropy loss built from pairwise geodesic distances.
        """
        anchor_embeddings = self.model(anchors)
        positive_embeddings = self.model(positives)

        distances = self.model.manifold.dist(
            anchor_embeddings[:, None, :],
            positive_embeddings[None, :, :],
        )
        logits = -distances / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets)
