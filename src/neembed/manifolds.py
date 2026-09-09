"""Manifold integration points for neembed."""

from __future__ import annotations

import math

import geoopt
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


class _PositiveScalar(nn.Module):
    """Keep a scalar parameter positive without changing Geoopt manifold math."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.softplus(value)

    def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
        # Stable inverse-softplus for positive values.
        return value + torch.log(-torch.expm1(-value))


def get_manifold(
    name: str,
    curvature: float = 1.0,
    learnable: bool = False,
    *,
    sectional_curvature: float | None = None,
) -> geoopt.PoincareBall | geoopt.Lorentz | geoopt.Euclidean | geoopt.SphereProjection:
    """Return a supported Geoopt manifold with backward-compatible curvature semantics."""
    if name in {"poincare", "lorentz"}:
        if sectional_curvature is not None:
            raise ValueError(
                "sectional_curvature must be None for poincare and lorentz"
            )
        if curvature <= 0 or not math.isfinite(curvature):
            raise ValueError("curvature must be positive and finite")

        if name == "poincare":
            return geoopt.PoincareBall(c=curvature, learnable=learnable)

        # Geoopt Lorentz ``k`` is the squared hyperboloid radius. A hyperboloid
        # of radius sqrt(k) has sectional curvature -1/k, so public curvature
        # magnitude ``c`` maps to k = 1/c. Geoopt strongly recommends double
        # precision for Lorentz geometry because Minkowski operations can be
        # numerically unstable in float32.
        k = torch.tensor(1.0 / curvature, dtype=torch.float64)
        manifold = geoopt.Lorentz(k=k, learnable=learnable)
        if learnable:
            # Unlike PoincareBall, Geoopt's Lorentz ``k`` parameter is
            # unconstrained. A standard PyTorch parametrization keeps the
            # squared radius positive while all manifold operations remain
            # Geoopt's implementation.
            parametrize.register_parametrization(manifold, "k", _PositiveScalar())
        return manifold

    if name == "euclidean":
        if learnable:
            raise ValueError("learnable_curvature is not supported for euclidean")
        if curvature != 1.0:
            raise ValueError(
                "curvature is a legacy poincare/lorentz argument and must remain "
                "at its default value for euclidean"
            )
        if sectional_curvature is None:
            sectional_curvature = 0.0
        if not math.isfinite(sectional_curvature) or sectional_curvature != 0.0:
            raise ValueError("euclidean sectional_curvature must be exactly 0.0")
        # ndim=1 makes the final embedding dimension one Euclidean manifold
        # point, so dist() returns the vector L2 norm rather than per-coordinate
        # absolute differences.
        return geoopt.Euclidean(ndim=1)

    if name == "sphere_projection":
        if learnable:
            raise ValueError(
                "learnable_curvature is not supported for sphere_projection"
            )
        if curvature != 1.0:
            raise ValueError(
                "curvature is a legacy poincare/lorentz argument and must remain "
                "at its default value for sphere_projection"
            )
        if sectional_curvature is None:
            raise ValueError("sphere_projection requires sectional_curvature")
        if (
            not math.isfinite(sectional_curvature)
            or sectional_curvature <= 0.0
        ):
            raise ValueError(
                "sphere_projection sectional_curvature must be positive and finite"
            )
        return geoopt.SphereProjection(k=sectional_curvature, learnable=False)

    raise ValueError(f"Unsupported manifold: {name}")
