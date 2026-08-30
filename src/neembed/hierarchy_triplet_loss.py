"""Directed hierarchy triplet loss for manifold sentence embeddings."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn

from neembed.model import ManifoldSentenceTransformer


class ManifoldHierarchyTripletLoss(nn.Module):
    """Prefer parent-child proximity while preserving hierarchy direction.

    For each aligned ``(parent, child, unrelated)`` triplet, the objective sums
    two hinge penalties:

    ``relu(d(parent, child) - d(parent, unrelated) + margin)``

    and

    ``radial_weight * relu(radius(parent) + radial_margin - radius(child))``.

    The first term makes the declared child closer to its parent than the
    unrelated node using the configured manifold geodesic distance ``d``. The
    second term makes the objective directional by requiring the parent to lie
    radially inside the child. ``radius(x)`` is Geoopt's geodesic distance from
    the configured manifold origin via ``manifold.dist0(x)``.

    The loss consumes caller-owned aligned text triplets directly, so it does
    not parse or persist hierarchy IDs or edges. Callers that construct triplets
    from explicit tree/DAG metadata can validate that metadata separately before
    forming batches.

    Args:
        model: Manifold sentence model used to encode aligned hierarchy triplets.
        margin: Non-negative, finite geodesic ranking margin separating the
            child from the unrelated node.
        radial_margin: Non-negative, finite geodesic radial margin requiring the
            child to lie farther from the origin than its parent.
        radial_weight: Positive, finite multiplier for the directional radial
            penalty.
    """

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        margin: float = 0.1,
        *,
        radial_margin: float = 0.1,
        radial_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if margin < 0 or not math.isfinite(margin):
            raise ValueError("margin must be non-negative and finite")
        if radial_margin < 0 or not math.isfinite(radial_margin):
            raise ValueError("radial_margin must be non-negative and finite")
        if radial_weight <= 0 or not math.isfinite(radial_weight):
            raise ValueError("radial_weight must be positive and finite")

        self.model = model
        self.margin = float(margin)
        self.radial_margin = float(radial_margin)
        self.radial_weight = float(radial_weight)

    def forward(
        self,
        parents: Sequence[str],
        children: Sequence[str],
        unrelated: Sequence[str],
    ) -> torch.Tensor:
        """Return the mean directed hierarchy triplet loss.

        Args:
            parents: Batch of parent texts.
            children: Batch of child texts aligned by index with ``parents``.
            unrelated: Batch of unrelated texts aligned by index with ``parents``.

        Returns:
            Scalar mean of geodesic ranking and radial-order penalties.
        """
        for name, values in (
            ("parents", parents),
            ("children", children),
            ("unrelated", unrelated),
        ):
            if isinstance(values, (str, bytes)):
                raise ValueError(f"{name} must be a sequence of texts, not a string")

        if len(parents) == 0:
            raise ValueError("parents, children, and unrelated must not be empty")
        if len(parents) != len(children):
            raise ValueError("parents and children must have the same length")
        if len(parents) != len(unrelated):
            raise ValueError("parents and unrelated must have the same length")

        parent_embeddings = self.model(parents)
        child_embeddings = self.model(children)
        unrelated_embeddings = self.model(unrelated)

        child_distances = self.model.manifold.dist(
            parent_embeddings,
            child_embeddings,
        )
        unrelated_distances = self.model.manifold.dist(
            parent_embeddings,
            unrelated_embeddings,
        )
        ranking_penalty = F.relu(
            child_distances - unrelated_distances + self.margin
        )

        parent_radii = self.model.manifold.dist0(parent_embeddings)
        child_radii = self.model.manifold.dist0(child_embeddings)
        radial_penalty = F.relu(
            parent_radii + self.radial_margin - child_radii
        )

        return (ranking_penalty + self.radial_weight * radial_penalty).mean()
