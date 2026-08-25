"""Symmetric retrieval objective for manifold sentence embeddings."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from neembed.losses import ManifoldMultipleNegativesRankingLoss


class ManifoldSymmetricMultipleNegativesRankingLoss(
    ManifoldMultipleNegativesRankingLoss
):
    """Bidirectional multiple-negatives ranking over manifold geodesic distance.

    The loss averages two cross-entropy terms with the same aligned diagonal
    targets: ``anchor -> positive`` and ``positive -> anchor``. Both directions
    use the configured Geoopt manifold distance and the inherited temperature.

    Explicit negatives remain caller-supplied and are appended only to the
    forward ``anchor -> candidate`` pool. They are not added to the reverse
    direction because they have no aligned reverse target; the reverse term uses
    only the positive texts as queries and the anchors as candidates. This keeps
    target assignments unambiguous while preserving the existing hard-negative
    semantics in the forward direction.

    Use :class:`neembed.ManifoldMultipleNegativesRankingLoss` when only the
    original one-directional objective is desired.
    """

    def forward(
        self,
        anchors: Sequence[str],
        positives: Sequence[str],
        negatives: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Return the mean of forward and reverse retrieval losses."""
        if len(anchors) == 0:
            raise ValueError("anchors and positives must not be empty")
        if len(anchors) != len(positives):
            raise ValueError("anchors and positives must have the same length")
        if negatives is not None and len(anchors) != len(negatives):
            raise ValueError("anchors and negatives must have the same length")

        anchor_embeddings = self.model(anchors)
        positive_embeddings = self.model(positives)
        candidate_embeddings = positive_embeddings

        if negatives is not None:
            negative_embeddings = self.model(negatives)
            candidate_embeddings = torch.cat(
                (positive_embeddings, negative_embeddings),
                dim=0,
            )

        targets = torch.arange(len(anchors), device=anchor_embeddings.device)

        forward_distances = self.model.manifold.dist(
            anchor_embeddings[:, None, :],
            candidate_embeddings[None, :, :],
        )
        forward_loss = F.cross_entropy(
            -forward_distances / self.temperature,
            targets,
        )

        reverse_distances = self.model.manifold.dist(
            positive_embeddings[:, None, :],
            anchor_embeddings[None, :, :],
        )
        reverse_loss = F.cross_entropy(
            -reverse_distances / self.temperature,
            targets,
        )

        return 0.5 * (forward_loss + reverse_loss)
