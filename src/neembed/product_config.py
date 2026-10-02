"""Normalized public configuration contract for v0.10 product geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any


_PRODUCT_SCHEMA_TYPE = "product"
_PRODUCT_SCHEMA_VERSION = 1
_SUPPORTED_COMPONENT_MANIFOLDS = {
    "euclidean",
    "poincare",
    "lorentz",
    "sphere_projection",
    "stereographic",
}
_COMPONENT_KEYS = {
    "name",
    "manifold",
    "intrinsic_dim",
    "curvature",
    "sectional_curvature",
    "scale",
}


def _positive_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return normalized


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


@dataclass(frozen=True)
class ProductComponentConfig:
    """One normalized component in an ordered mixed-curvature product.

    ``intrinsic_dim`` is the width consumed from the encoder projection before
    geometry mapping. ``ambient_dim`` is the packed point width after mapping;
    Lorentz contributes one additional time-like coordinate.

    ``scale`` is persisted as part of the v0.10 configuration contract but is
    not applied to distances until scaled product geometry support is added.
    """

    name: str
    manifold: str
    intrinsic_dim: int
    curvature: float | None = None
    sectional_curvature: float | None = None
    scale: float = 1.0

    @property
    def projection_dim(self) -> int:
        """Width consumed from the ordinary encoder projection."""
        return self.intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        """Width occupied by this component in the packed manifold point."""
        if self.manifold == "lorentz":
            return self.intrinsic_dim + 1
        return self.intrinsic_dim

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible persistence metadata."""
        result: dict[str, Any] = {
            "name": self.name,
            "manifold": self.manifold,
            "intrinsic_dim": self.intrinsic_dim,
        }
        if self.curvature is not None:
            result["curvature"] = self.curvature
        if self.sectional_curvature is not None:
            result["sectional_curvature"] = self.sectional_curvature
        result["scale"] = self.scale
        return result


@dataclass(frozen=True)
class ProductConfig:
    """Normalized ordered configuration for a flat product of vector geometries."""

    components: tuple[ProductComponentConfig, ...]

    @property
    def projection_dim(self) -> int:
        """Total encoder projection width required before component splitting."""
        return sum(component.projection_dim for component in self.components)

    @property
    def ambient_dim(self) -> int:
        """Total width of the future packed ProductManifold point."""
        return sum(component.ambient_dim for component in self.components)

    @property
    def component_names(self) -> tuple[str, ...]:
        """Stable component identities in product order."""
        return tuple(component.name for component in self.components)

    def to_dict(self) -> dict[str, Any]:
        """Return versioned deterministic JSON-compatible persistence metadata."""
        return {
            "type": _PRODUCT_SCHEMA_TYPE,
            "version": _PRODUCT_SCHEMA_VERSION,
            "components": [component.to_dict() for component in self.components],
        }

    @classmethod
    def from_dict(cls, metadata: Mapping[str, Any]) -> ProductConfig:
        """Validate persisted product metadata and return its normalized form."""
        if not isinstance(metadata, Mapping):
            raise ValueError("product metadata must be a mapping")

        allowed_keys = {"type", "version", "components"}
        unknown = set(metadata) - allowed_keys
        if unknown:
            joined = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(f"unsupported product metadata field(s): {joined}")
        if metadata.get("type") != _PRODUCT_SCHEMA_TYPE:
            raise ValueError("product metadata type must be 'product'")
        version = metadata.get("version")
        if isinstance(version, bool) or version != _PRODUCT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported product metadata version: {version!r}; "
                f"expected {_PRODUCT_SCHEMA_VERSION}"
            )
        if "components" not in metadata:
            raise ValueError("product metadata requires components")
        return _normalize_components(metadata["components"])


def _normalize_component(
    component: ProductComponentConfig | Mapping[str, Any],
    *,
    index: int,
) -> ProductComponentConfig:
    if isinstance(component, ProductComponentConfig):
        raw: Mapping[str, Any] = component.to_dict()
    elif isinstance(component, Mapping):
        raw = component
    else:
        raise ValueError(f"product component {index} must be a mapping")

    unknown = set(raw) - _COMPONENT_KEYS
    if unknown:
        joined = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(
            f"product component {index} has unsupported field(s): {joined}"
        )

    manifold = raw.get("manifold")
    if manifold not in _SUPPORTED_COMPONENT_MANIFOLDS:
        raise ValueError(
            f"product component {index} has unsupported manifold: {manifold!r}"
        )

    intrinsic_dim = raw.get("intrinsic_dim")
    if (
        isinstance(intrinsic_dim, bool)
        or not isinstance(intrinsic_dim, int)
        or intrinsic_dim <= 0
    ):
        raise ValueError(
            f"product component {index} intrinsic_dim must be a positive integer"
        )

    name = raw.get("name", f"component_{index}")
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(
            f"product component {index} name must be a non-empty trimmed string"
        )

    scale = _positive_finite(
        raw.get("scale", 1.0),
        field=f"product component {index} scale",
    )

    curvature: float | None = None
    sectional_curvature: float | None = None

    if manifold in {"poincare", "lorentz"}:
        if raw.get("sectional_curvature") is not None:
            raise ValueError(
                f"product component {index} {manifold} uses curvature, not "
                "sectional_curvature"
            )
        curvature = _positive_finite(
            raw.get("curvature", 1.0),
            field=f"product component {index} curvature",
        )
    elif manifold == "euclidean":
        if raw.get("curvature") is not None:
            raise ValueError(
                f"product component {index} euclidean does not accept curvature"
            )
        raw_sectional = raw.get("sectional_curvature", 0.0)
        sectional_curvature = _finite(
            raw_sectional,
            field=f"product component {index} sectional_curvature",
        )
        if sectional_curvature != 0.0:
            raise ValueError(
                f"product component {index} euclidean sectional_curvature "
                "must be exactly 0.0"
            )
    elif manifold == "sphere_projection":
        if raw.get("curvature") is not None:
            raise ValueError(
                f"product component {index} sphere_projection does not accept curvature"
            )
        if raw.get("sectional_curvature") is None:
            raise ValueError(
                f"product component {index} sphere_projection requires "
                "sectional_curvature"
            )
        sectional_curvature = _positive_finite(
            raw["sectional_curvature"],
            field=f"product component {index} sectional_curvature",
        )
    else:
        if raw.get("curvature") is not None:
            raise ValueError(
                f"product component {index} stereographic does not accept curvature"
            )
        if raw.get("sectional_curvature") is None:
            raise ValueError(
                f"product component {index} stereographic requires "
                "sectional_curvature"
            )
        sectional_curvature = _finite(
            raw["sectional_curvature"],
            field=f"product component {index} sectional_curvature",
        )

    return ProductComponentConfig(
        name=name,
        manifold=manifold,
        intrinsic_dim=intrinsic_dim,
        curvature=curvature,
        sectional_curvature=sectional_curvature,
        scale=scale,
    )


def _normalize_components(components: Any) -> ProductConfig:
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise ValueError("product components must be a non-empty ordered sequence")
    if not components:
        raise ValueError("product components must contain at least one component")

    normalized = tuple(
        _normalize_component(component, index=index)
        for index, component in enumerate(components)
    )
    names = [component.name for component in normalized]
    if len(names) != len(set(names)):
        raise ValueError("product component names must be unique")
    return ProductConfig(normalized)


def normalize_product_config(
    config: ProductConfig
    | Sequence[ProductComponentConfig | Mapping[str, Any]]
    | Mapping[str, Any],
) -> ProductConfig:
    """Normalize public product input or versioned persistence metadata.

    A plain ordered component sequence is intended for constructor-facing use.
    A mapping is interpreted as the exact versioned form returned by
    :meth:`ProductConfig.to_dict`, so load-time validation cannot silently accept
    unknown schema versions or fields.
    """
    if isinstance(config, ProductConfig):
        return config
    if isinstance(config, Mapping):
        return ProductConfig.from_dict(config)
    return _normalize_components(config)
