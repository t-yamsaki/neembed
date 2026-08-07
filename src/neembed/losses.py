"""Losses for manifold-valued sentence embeddings."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from neembed.model import ManifoldSentenceTransformer


class ManifoldMultipleNegativesRankingLoss(nn.Module):
    """InfoNCE-style in-batch ranking loss using manifold geodesic distance."""

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.model = model
        self.temperature = float(temperature)

    def forward(
        self,
        anchors: Sequence[str],
        positives: Sequence[str],
    ) -> torch.Tensor:
        """Return the in-batch contrastive loss for aligned anchor-positive pairs."""
        anchor_embeddings = self.model(anchors)
        positive_embeddings = self.model(positives)

        distances = self.model.manifold.dist(
            anchor_embeddings[:, None, :],
            positive_embeddings[None, :, :],
        )
        logits = -distances / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets)
