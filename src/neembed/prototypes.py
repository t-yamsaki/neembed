"""Trainable manifold-valued prototypes backed by Geoopt."""

from __future__ import annotations

import math

import geoopt
import torch
from torch import nn

from neembed.model import (
    ManifoldSentenceTransformer,
    _apply_closure_values,
    _fixed_double_apply_fn,
)


_STEREOGRAPHIC_DOUBLE_MANIFOLDS = {"sphere_projection", "stereographic"}


def _infer_apply_device(fn, source_device: torch.device | str) -> torch.device:
    """Infer the target device of a standard PyTorch ``_apply`` transform."""
    source_device = torch.device(source_device)
    values = _apply_closure_values(fn)
    qualname = getattr(fn, "__qualname__", "")

    if "device" in values:
        device = values["device"]
        if device is None:
            if "Module.cuda" in qualname:
                return torch.device("cuda")
            if "Module.xpu" in qualname:
                return torch.device("xpu")
            if "Module.mtia" in qualname:
                return torch.device("mtia")
            return source_device
        if isinstance(device, int):
            if "Module.cuda" in qualname:
                return torch.device("cuda", device)
            if "Module.xpu" in qualname:
                return torch.device("xpu", device)
            if "Module.mtia" in qualname:
                return torch.device("mtia", device)
        return torch.device(device)

    if "Module.cpu" in qualname:
        return torch.device("cpu")
    if "Module.cuda" in qualname:
        return torch.device("cuda")
    if "Module.xpu" in qualname:
        return torch.device("xpu")
    if "Module.mtia" in qualname:
        return torch.device("mtia")

    probe = torch.empty(0, dtype=torch.uint8, device=source_device)
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

    def _apply_prototype_parameter(self, fn, manifold) -> None:
        """Apply a protected transform while preserving parameter identity."""
        parameter = self._parameters["prototypes"]
        gradient = parameter.grad
        with torch.no_grad():
            applied = fn(parameter)
            gradient_applied = None if gradient is None else fn(gradient)

        parameter.grad = None
        if applied is not parameter:
            try:
                parameter.data = applied
            except RuntimeError:
                replacement = geoopt.ManifoldParameter(
                    applied,
                    manifold=manifold,
                    requires_grad=parameter.requires_grad,
                )
                try:
                    torch.utils.swap_tensors(parameter, replacement)
                except Exception:
                    parameter.grad = gradient
                    raise

        if gradient_applied is not None:
            parameter.grad = gradient_applied.requires_grad_(gradient.requires_grad)

        if not isinstance(parameter, geoopt.ManifoldParameter):
            raise RuntimeError("prototype transform lost ManifoldParameter metadata")
        if parameter.manifold is not manifold:
            raise RuntimeError("prototype transform changed the shared manifold")
        if parameter.dtype != torch.float64:
            raise RuntimeError("fixed stereographic prototypes must remain float64")
        if parameter.grad is not None and parameter.grad.dtype != torch.float64:
            raise RuntimeError("stereographic prototype gradients must remain float64")

    def _apply(self, fn, recurse: bool = True):
        """Apply transforms without narrowing or sending prototypes to MPS."""
        if self.manifold_name not in _STEREOGRAPHIC_DOUBLE_MANIFOLDS:
            return super()._apply(fn, recurse=recurse)

        prototypes = self.prototypes
        manifold = prototypes.manifold
        target_device = _infer_apply_device(fn, prototypes.device)
        geometry_fn = _fixed_double_apply_fn(
            self.manifold_name,
            fn,
            target_device,
        )

        # Preserve normal PyTorch semantics for any non-prototype module state.
        # The manifold-valued parameter is handled separately so its identity,
        # Geoopt metadata, gradient, float64 dtype, and MPS CPU fallback remain
        # intact across dtype casts, device moves, meta transitions, and to_empty.
        self._parameters.pop("prototypes")
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._parameters["prototypes"] = prototypes

        self._apply_prototype_parameter(geometry_fn, manifold)
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