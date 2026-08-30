"""Hierarchy depth supervision based on radial distance from the manifold origin."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn

from neembed.model import ManifoldSentenceTransformer


class ManifoldDepthLoss(nn.Module):
    """Regress caller-owned hierarchy depths to geodesic radial targets.

    Each non-negative integer depth ``d`` is mapped to the target radius
    ``d * radial_scale``. The observed radius is Geoopt's geodesic distance from
    the configured manifold origin via ``model.manifold.dist0``. The loss is the
    mean squared error between observed and target radii.

    Depth ``0`` therefore pulls roots to the manifold origin. Nodes with the same
    depth share the same target radius but are otherwise unconstrained: this loss
    does not impose angular similarity or parent-child adjacency. Depth metadata
    is supplied on every forward call and is not stored in model state.

    For Poincare geometry the origin is the zero vector in ball coordinates. For
    Lorentz geometry the origin is ``(sqrt(k), 0, ..., 0)`` on the hyperboloid,
    where Geoopt's ``k`` equals ``1 / curvature`` under neembed's public
    curvature convention.

    If callers maintain separate node IDs, they should align the ``texts`` and
    ``depths`` sequences externally; IDs are not interpreted or persisted by the
    loss.

    Args:
        model: Manifold sentence model used to encode the supervised texts.
        radial_scale: Positive, finite geodesic radius assigned to one hierarchy
            depth step. A depth ``d`` has target radius ``d * radial_scale``.
    """

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        radial_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(radial_scale, bool)
            or radial_scale <= 0
            or not math.isfinite(radial_scale)
        ):
            raise ValueError("radial_scale must be positive and finite")

        self.model = model
        self.radial_scale = float(radial_scale)

    @staticmethod
    def _normalize_depths(
        depths: Sequence[int | float] | torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        if isinstance(depths, (str, bytes)):
            raise ValueError("depths must contain one non-negative integer per text")
        if not torch.is_tensor(depths) and isinstance(depths, Sequence):
            if any(isinstance(depth, bool) for depth in depths):
                raise ValueError("depth values must be non-negative integers")

        try:
            target = torch.as_tensor(depths)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("depths must contain real numeric values") from exc

        if target.dtype == torch.bool or torch.is_complex(target):
            raise ValueError("depth values must be non-negative integers")
        if target.ndim != 1 or target.shape[0] != batch_size:
            raise ValueError("depths must contain one value per text")

        if torch.is_floating_point(target):
            if not bool(torch.isfinite(target).all()):
                raise ValueError("depth values must be finite")
            if not bool(torch.equal(target, torch.floor(target))):
                raise ValueError("depth values must be non-negative integers")
        if bool((target < 0).any()):
            raise ValueError("depth values must be non-negative integers")
        return target

    def forward(
        self,
        texts: Sequence[str],
        depths: Sequence[int | float] | torch.Tensor,
    ) -> torch.Tensor:
        """Return radial target MSE for aligned texts and hierarchy depths.

        Args:
            texts: Non-empty batch of texts ordered to match ``depths``.
            depths: One caller-owned non-negative integer depth per text. Integer-
                valued floating tensors such as ``[0.0, 1.0]`` are accepted.

        Returns:
            Scalar mean squared error in squared geodesic-distance units.
        """
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be a sequence of strings, not a string")
        if len(texts) == 0:
            raise ValueError("texts and depths must not be empty")

        target_depths = self._normalize_depths(depths, batch_size=len(texts))
        embeddings = self.model(texts)
        radii = self.model.manifold.dist0(embeddings)
        target_radii = target_depths.to(
            device=radii.device,
            dtype=radii.dtype,
        ) * self.radial_scale
        return F.mse_loss(radii, target_radii)
