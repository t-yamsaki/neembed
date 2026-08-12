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
) -> geoopt.PoincareBall | geoopt.Lorentz:
    """Return a supported Geoopt manifold with consistent curvature semantics."""
    if curvature <= 0 or not math.isfinite(curvature):
        raise ValueError("curvature must be positive and finite")

    if name == "poincare":
        return geoopt.PoincareBall(c=curvature, learnable=learnable)
    if name == "lorentz":
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

    raise ValueError(f"Unsupported manifold: {name}")
