"""Hierarchy direction loss based on radial distance from the manifold origin."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn

from neembed.model import ManifoldSentenceTransformer


class ManifoldRadialOrderLoss(nn.Module):
    """Require parent texts to remain radially inside their child texts.

    For each aligned ``(parent, child)`` pair, the loss penalizes violations of

    ``radius(parent) + margin <= radius(child)``

    with ``relu(radius(parent) + margin - radius(child))`` and averages the
    penalties over the batch. ``radius(x)`` is Geoopt's geodesic distance from
    the configured manifold origin via ``manifold.dist0(x)``; it is not an
    ambient Euclidean norm.

    The Poincare origin is the zero vector in ball coordinates. The Lorentz
    origin is ``(sqrt(k), 0, ..., 0)`` on the hyperboloid, where Geoopt's
    squared-radius parameter ``k`` equals ``1 / curvature`` under neembed's
    public curvature convention.

    Args:
        model: Manifold sentence model used to encode aligned parent-child pairs.
        margin: Non-negative, finite geodesic radial margin. ``0`` permits equal
            parent and child radii; positive values require the child to be at
            least ``margin`` farther from the origin.
    """

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        margin: float = 0.1,
    ) -> None:
        super().__init__()
        if margin < 0 or not math.isfinite(margin):
            raise ValueError("margin must be non-negative and finite")

        self.model = model
        self.margin = float(margin)

    def forward(
        self,
        parents: Sequence[str],
        children: Sequence[str],
    ) -> torch.Tensor:
        """Return the mean radial-order violation for aligned hierarchy pairs.

        Args:
            parents: Batch of parent texts.
            children: Batch of child texts aligned by index with ``parents``.

        Returns:
            Scalar mean hinge loss in geodesic distance units.
        """
        for name, values in (("parents", parents), ("children", children)):
            if isinstance(values, (str, bytes)):
                raise ValueError(f"{name} must be a sequence of texts, not a string")

        if len(parents) == 0:
            raise ValueError("parents and children must not be empty")
        if len(parents) != len(children):
            raise ValueError("parents and children must have the same length")

        parent_embeddings = self.model(parents)
        child_embeddings = self.model(children)
        parent_radii = self.model.manifold.dist0(parent_embeddings)
        child_radii = self.model.manifold.dist0(child_embeddings)
        return F.relu(parent_radii + self.margin - child_radii).mean()
