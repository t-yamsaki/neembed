"""Runtime helpers for flat mixed-curvature product embeddings."""

from __future__ import annotations

import geoopt
import torch

from neembed.manifolds import get_manifold
from neembed.product_config import ProductConfig


_DOUBLE_COMPONENT_MANIFOLDS = {
    "lorentz",
    "sphere_projection",
    "stereographic",
}


def product_requires_double(config: ProductConfig) -> bool:
    """Return whether any component requires the product geometry to use float64."""
    return any(
        component.manifold in _DOUBLE_COMPONENT_MANIFOLDS
        for component in config.components
    )


def product_geometry_device(
    config: ProductConfig,
    encoder_device: torch.device | str,
) -> torch.device:
    """Return the common device used by every component in the packed product."""
    device = torch.device(encoder_device)
    if device.type == "mps" and product_requires_double(config):
        return torch.device("cpu")
    return device


def product_geometry_dtype(
    config: ProductConfig,
    projection_dtype: torch.dtype,
) -> torch.dtype:
    """Return the common dtype required by Geoopt ``ProductManifold``."""
    if product_requires_double(config):
        return torch.float64
    return projection_dtype



def validate_product_scale(scale: torch.Tensor) -> None:
    """Require both the scale and Geoopt's squared metric factor to be usable."""
    squared = scale.square()
    if not bool(torch.isfinite(scale) & (scale > 0)
                & torch.isfinite(squared) & (squared > 0)):
        raise ValueError(
            "component scale and its square must be positive and finite in geometry dtype"
        )


def build_product_manifold(
    config: ProductConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> geoopt.ProductManifold:
    """Construct a Geoopt product using the normalized component contract."""
    device = torch.device(device)
    manifolds_with_shape = []
    for component in config.components:
        if component.manifold in {"poincare", "lorentz"}:
            manifold = get_manifold(
                component.manifold,
                component.curvature,
                False,
            )
        else:
            manifold = get_manifold(
                component.manifold,
                1.0,
                False,
                sectional_curvature=component.sectional_curvature,
            )
        manifold.to(device=device, dtype=dtype)
        if component.scale != 1.0:
            scale = torch.tensor(component.scale, device="cpu", dtype=dtype)
            validate_product_scale(scale)
            manifold = geoopt.Scaled(manifold, learnable=False).to(
                device=device, dtype=dtype,
            )
            # Scaled creates its buffer in the default dtype. Copy the configured
            # value after conversion to avoid quantizing float64 scales.
            manifold.scale.copy_(scale)
        manifolds_with_shape.append((manifold, component.ambient_dim))

    return geoopt.ProductManifold(*manifolds_with_shape)


def map_product_tangent(
    manifold: geoopt.ProductManifold,
    config: ProductConfig,
    tangent: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Split projected features, map each component, and pack one product point."""
    tangent = tangent.to(device=torch.device(device), dtype=dtype)
    chunks = torch.split(
        tangent,
        [component.intrinsic_dim for component in config.components],
        dim=-1,
    )

    points = []
    for component, component_manifold, chunk in zip(
        config.components,
        manifold.manifolds,
        chunks,
    ):
        if component.manifold == "euclidean":
            point = chunk
        else:
            if component.manifold == "lorentz":
                chunk = torch.cat(
                    (torch.zeros_like(chunk[..., :1]), chunk),
                    dim=-1,
                )
            # Scaling weights distances between the same encoded points. Using
            # Scaled.expmap0 would also divide the tangent by the scale.
            base_manifold = (
                component_manifold.base
                if isinstance(component_manifold, geoopt.Scaled)
                else component_manifold
            )
            point = base_manifold.expmap0(chunk)
        points.append(point)

    return manifold.pack_point(*points)
