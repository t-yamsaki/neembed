"""Shared component selection for radial product hierarchy supervision."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from neembed.model import ManifoldSentenceTransformer


def _resolve_hierarchy_component(
    model: ManifoldSentenceTransformer,
    component: str | int | None,
) -> int | None:
    """Resolve an explicit Poincare/Lorentz component; preserve single paths."""
    if getattr(model, "manifold_name", None) != "product":
        if component is not None:
            raise ValueError("component selection requires a product model")
        return None
    if component is None:
        raise ValueError(
            "product hierarchy supervision requires an explicit component name or index"
        )
    components = model.product_config.components
    if isinstance(component, str):
        names = [item.name for item in components]
        if component not in names:
            raise ValueError(f"unknown hierarchy component name: {component!r}")
        index = names.index(component)
    elif isinstance(component, int) and not isinstance(component, bool):
        if not 0 <= component < len(components):
            raise ValueError("hierarchy component index is out of range")
        index = component
    else:
        raise ValueError("component must be a stable name or a non-negative integer index")
    if components[index].manifold not in {"poincare", "lorentz"}:
        raise ValueError("hierarchy component must use poincare or lorentz geometry")
    return index


def _hierarchy_geometry(
    model: ManifoldSentenceTransformer,
    embeddings: torch.Tensor,
    component_index: int | None,
):
    """Return native/scaled geometry and its points without detaching gradients."""
    if component_index is None:
        return model.manifold, embeddings
    return (
        model.manifold.manifolds[component_index],
        model.manifold.take_submanifold_value(embeddings, component_index),
    )
