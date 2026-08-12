"""Trainable manifold-valued prototypes backed by Geoopt."""

from __future__ import annotations

import math

import geoopt
import torch
from torch import nn

from neembed.model import ManifoldSentenceTransformer


class ManifoldPrototypes(nn.Module):
    """Represent trainable prototype points on a model's configured manifold.

    Args:
        model: Sentence model whose manifold, curvature, device, and intrinsic
            dimension define the prototype geometry.
        num_prototypes: Number of trainable prototype points.
        init_std: Standard deviation used by Geoopt's manifold-native random
            initializer.

    Notes:
        Prototype points are true :class:`geoopt.ManifoldParameter` values. They
        therefore require a Geoopt Riemannian optimizer for safe updates. For
        mixed training, build one Geoopt optimizer over the ordinary model
        parameters and these prototype parameters, then pass it explicitly to
        :class:`neembed.ManifoldTrainer`.

        ``embedding_dim`` is intrinsic. Poincare prototypes have that many
        coordinates; Lorentz prototypes have one additional ambient time-like
        coordinate, matching the sentence-model output contract.

        Prototypes may share a manifold whose curvature is learnable. In that
        joint path, use Geoopt optimizer stabilization after every step (for
        example ``geoopt.optim.RiemannianAdam(..., stabilize=1)``) so prototype
        coordinates are projected back onto the manifold after the curvature
        parameter changes. neembed delegates both the Riemannian update and the
        stabilization projection to Geoopt.

        Prototype coordinates are external to
        :meth:`neembed.ManifoldSentenceTransformer.save_pretrained`; persist this
        module with its own ``state_dict()`` and reconstruct it on a compatible
        reloaded model before loading that state.
    """

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        num_prototypes: int,
        *,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if num_prototypes <= 0:
            raise ValueError("num_prototypes must be positive")
        if init_std <= 0 or not math.isfinite(init_std):
            raise ValueError("init_std must be positive and finite")

        self.num_prototypes = int(num_prototypes)
        self.embedding_dim = model.embedding_dim
        self.manifold_name = model.manifold_name
        self.ambient_dim = self.embedding_dim + int(self.manifold_name == "lorentz")

        manifold = model.manifold
        initial = manifold.random_normal(
            self.num_prototypes,
            self.ambient_dim,
            std=init_std,
        )
        self.prototypes = geoopt.ManifoldParameter(initial, manifold=manifold)

    @property
    def manifold(self):
        """Return the Geoopt manifold associated with the prototype parameter."""
        return self.prototypes.manifold

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return geodesic distances from embeddings to every prototype.

        Args:
            embeddings: One manifold embedding with shape ``(ambient_dim,)`` or
                a batch whose final dimension is ``ambient_dim``.

        Returns:
            Distances with one prototype dimension appended. A batch of shape
            ``(batch_size, ambient_dim)`` produces ``(batch_size, num_prototypes)``.
        """
        if embeddings.ndim == 0 or embeddings.shape[-1] != self.ambient_dim:
            raise ValueError(
                f"embeddings must have final dimension {self.ambient_dim}, "
                f"got shape {tuple(embeddings.shape)}"
            )
        return self.manifold.dist(embeddings.unsqueeze(-2), self.prototypes)
