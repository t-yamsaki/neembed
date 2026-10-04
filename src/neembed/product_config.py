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
_PERSISTED_COMPONENT_BASE_KEYS = {
    "name",
    "manifold",
    "intrinsic_dim",
    "scale",
}


def _positive_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive finite number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be a positive finite number") from exc
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return normalized


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be finite")
    return normalized


def _validate_manifold_name(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a supported manifold name string")
    if value not in _SUPPORTED_COMPONENT_MANIFOLDS:
        raise ValueError(f"{field} has unsupported manifold: {value!r}")
    return value


@dataclass(frozen=True)
class ProductComponentConfig:
    """One normalized component in an ordered mixed-curvature product.

    ``intrinsic_dim`` is the width consumed from the encoder projection before
    geometry mapping. ``ambient_dim`` is the packed point width after mapping;
    Lorentz contributes one additional time-like coordinate.

    ``scale`` is a fixed positive distance multiplier, applied through Geoopt
    ``Scaled`` without changing component dimensions or encoded coordinates.
    """

    name: str
    manifold: str
    intrinsic_dim: int
    curvature: float | None = None
    sectional_curvature: float | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("name must be a non-empty trimmed string")
        manifold = _validate_manifold_name(self.manifold, field="manifold")
        object.__setattr__(self, "manifold", manifold)
        if (
            isinstance(self.intrinsic_dim, bool)
            or not isinstance(self.intrinsic_dim, int)
            or self.intrinsic_dim <= 0
        ):
            raise ValueError("intrinsic_dim must be a positive integer")

        object.__setattr__(self, "scale", _positive_finite(self.scale, field="scale"))

        if manifold in {"poincare", "lorentz"}:
            if self.sectional_curvature is not None:
                raise ValueError(
                    f"{manifold} uses curvature, not sectional_curvature"
                )
            curvature = 1.0 if self.curvature is None else self.curvature
            object.__setattr__(
                self,
                "curvature",
                _positive_finite(curvature, field="curvature"),
            )
            return

        if self.curvature is not None:
            raise ValueError(f"{manifold} does not accept curvature")

        if manifold == "euclidean":
            sectional = 0.0 if self.sectional_curvature is None else self.sectional_curvature
            sectional = _finite(sectional, field="sectional_curvature")
            if sectional != 0.0:
                raise ValueError("euclidean sectional_curvature must be exactly 0.0")
            object.__setattr__(self, "sectional_curvature", sectional)
            return

        if self.sectional_curvature is None:
            raise ValueError(f"{manifold} requires sectional_curvature")
        sectional = _finite(self.sectional_curvature, field="sectional_curvature")
        if manifold == "sphere_projection" and sectional <= 0.0:
            raise ValueError(
                "sphere_projection sectional_curvature must be positive and finite"
            )
        object.__setattr__(self, "sectional_curvature", sectional)

    @property
    def projection_dim(self) -> int:
        """Width consumed from the ordinary encoder projection."""
        return self.intrinsic_dim

    @property
    def ambient_dim(self) -> int:
        """Width occupied by this component in the packed manifold point."""
        return self.intrinsic_dim + 1 if self.manifold == "lorentz" else self.intrinsic_dim

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

    def __post_init__(self) -> None:
        if isinstance(self.components, (str, bytes)) or not isinstance(
            self.components, Sequence
        ):
            raise ValueError("components must be a non-empty ordered sequence")
        components = tuple(self.components)
        if not components:
            raise ValueError("components must contain at least one component")
        if not all(isinstance(component, ProductComponentConfig) for component in components):
            raise ValueError("components must contain ProductComponentConfig values")
        names = [component.name for component in components]
        if len(names) != len(set(names)):
            raise ValueError("component names must be unique")
        object.__setattr__(self, "components", components)

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
        unknown = set(metadata) - {"type", "version", "components"}
        if unknown:
            joined = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(f"unsupported product metadata field(s): {joined}")
        if metadata.get("type") != _PRODUCT_SCHEMA_TYPE:
            raise ValueError("product metadata type must be 'product'")
        version = metadata.get("version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != _PRODUCT_SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported product metadata version: {version!r}; "
                f"expected integer {_PRODUCT_SCHEMA_VERSION}"
            )
        if "components" not in metadata:
            raise ValueError("product metadata requires components")
        return _load_persisted_components(metadata["components"])


def _normalize_component(
    component: ProductComponentConfig | Mapping[str, Any],
    *,
    index: int,
) -> ProductComponentConfig:
    if isinstance(component, ProductComponentConfig):
        return component
    if not isinstance(component, Mapping):
        raise ValueError(f"product component {index} must be a mapping")

    unknown = set(component) - _COMPONENT_KEYS
    if unknown:
        joined = ", ".join(sorted(str(key) for key in unknown))
        raise ValueError(
            f"product component {index} has unsupported field(s): {joined}"
        )

    try:
        return ProductComponentConfig(
            name=component.get("name", f"component_{index}"),
            manifold=component.get("manifold"),
            intrinsic_dim=component.get("intrinsic_dim"),
            curvature=component.get("curvature"),
            sectional_curvature=component.get("sectional_curvature"),
            scale=component.get("scale", 1.0),
        )
    except ValueError as exc:
        raise ValueError(f"product component {index} {exc}") from exc


def _normalize_components(components: Any) -> ProductConfig:
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise ValueError("product components must be a non-empty ordered sequence")
    if not components:
        raise ValueError("product components must contain at least one component")
    return ProductConfig(
        tuple(
            _normalize_component(component, index=index)
            for index, component in enumerate(components)
        )
    )


def _load_persisted_components(components: Any) -> ProductConfig:
    """Load the exact normalized component schema without constructor defaults."""
    if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
        raise ValueError("product metadata components must be a non-empty ordered sequence")
    if not components:
        raise ValueError("product metadata components must contain at least one component")

    normalized: list[ProductComponentConfig] = []
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise ValueError(f"persisted product component {index} must be a mapping")

        manifold = _validate_manifold_name(
            component.get("manifold"),
            field=f"persisted product component {index} manifold",
        )

        expected = set(_PERSISTED_COMPONENT_BASE_KEYS)
        if manifold in {"poincare", "lorentz"}:
            expected.add("curvature")
        else:
            expected.add("sectional_curvature")

        missing = expected - set(component)
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(
                f"persisted product component {index} missing required field(s): {joined}"
            )
        unknown = set(component) - expected
        if unknown:
            joined = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(
                f"persisted product component {index} has unsupported field(s): {joined}"
            )

        normalized.append(_normalize_component(component, index=index))

    return ProductConfig(tuple(normalized))


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
