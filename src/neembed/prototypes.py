"""Trainable manifold-valued prototypes backed by Geoopt."""

from __future__ import annotations

import math

import geoopt
import torch
from torch import nn

from neembed.model import ManifoldSentenceTransformer, _select_geometry_device


_STEREOGRAPHIC_DOUBLE_MANIFOLDS = {"sphere_projection", "stereographic"}


def _infer_apply_device(fn, source_device: torch.device | str) -> torch.device:
    """Infer the device requested by a PyTorch ``_apply`` transform."""
    probe = torch.empty(0, dtype=torch.uint8, device=torch.device(source_device))
    return fn(probe).device


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

        SphereProjection and Stereographic prototypes follow their shared
        manifold's float64 geometry device from initialization onward. In
        particular, they remain on CPU when the associated sentence model uses
        the Apple MPS CPU fallback.

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
        random_kwargs = {"std": init_std}
        if self.manifold_name in _STEREOGRAPHIC_DOUBLE_MANIFOLDS:
            random_kwargs.update(
                dtype=torch.float64,
                device=manifold.k.device,
            )
        initial = manifold.random_normal(
            self.num_prototypes,
            self.ambient_dim,
            **random_kwargs,
        )
        self.prototypes = geoopt.ManifoldParameter(initial, manifold=manifold)

    def _ensure_stereographic_parameter(self, manifold) -> None:
        """Restore Geoopt metadata and the fixed float64 prototype dtype."""
        parameter = self._parameters["prototypes"]
        if not isinstance(parameter, geoopt.ManifoldParameter):
            grad = parameter.grad
            parameter = geoopt.ManifoldParameter(
                parameter.detach(),
                manifold=manifold,
                requires_grad=parameter.requires_grad,
            )
            if grad is not None:
                parameter.grad = grad
            self._parameters["prototypes"] = parameter

        with torch.no_grad():
            if parameter.dtype != torch.float64:
                parameter.data = parameter.data.to(dtype=torch.float64)
            if parameter.grad is not None and parameter.grad.dtype != torch.float64:
                parameter.grad.data = parameter.grad.data.to(dtype=torch.float64)

    def _apply(self, fn, recurse: bool = True):
        """Apply transforms without sending fixed-double prototypes to MPS."""
        if self.manifold_name not in _STEREOGRAPHIC_DOUBLE_MANIFOLDS:
            return super()._apply(fn, recurse=recurse)

        prototypes = self.prototypes
        manifold = prototypes.manifold
        target_device = _infer_apply_device(fn, prototypes.device)
        geometry_device = _select_geometry_device(self.manifold_name, target_device)

        if target_device.type != "mps":
            # Let PyTorch preserve the exact transform semantics for CPU, CUDA,
            # meta, and ``to_empty``. This makes the requested prototype device
            # independent of whether a containing module visits the prototypes or
            # the sentence model first.
            result = super()._apply(fn, recurse=recurse)
            self._ensure_stereographic_parameter(manifold)
            return result

        # MPS cannot represent the float64 prototype. Exclude only this parameter
        # from the supplied transform while allowing any other module state to
        # follow normal ``_apply`` semantics.
        self._parameters.pop("prototypes")
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._parameters["prototypes"] = prototypes

        if prototypes.device.type == "meta":
            # A transform that successfully maps a meta probe to MPS is a
            # materializing transform such as ``to_empty``. Materialize the
            # protected prototype directly on the CPU geometry fallback.
            materialized = torch.empty(
                prototypes.shape,
                device=geometry_device,
                dtype=torch.float64,
            )
            self._parameters["prototypes"] = geoopt.ManifoldParameter(
                materialized,
                manifold=manifold,
                requires_grad=prototypes.requires_grad,
            )
        else:
            with torch.no_grad():
                prototypes.data = prototypes.data.to(
                    device=geometry_device,
                    dtype=torch.float64,
                )
                if prototypes.grad is not None:
                    prototypes.grad.data = prototypes.grad.data.to(
                        device=geometry_device,
                        dtype=torch.float64,
                    )

        self._ensure_stereographic_parameter(manifold)
        return result

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