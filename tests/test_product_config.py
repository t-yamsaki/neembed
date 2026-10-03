"""Regression tests for the v0.10 mixed-curvature product configuration contract."""

from inspect import signature
import math

import pytest

from neembed import (
    ManifoldSentenceTransformer,
    ProductComponentConfig,
    ProductConfig,
    normalize_product_config,
)


def test_normalize_product_config_represents_h8_s8_r16_unambiguously() -> None:
    config = normalize_product_config(
        [
            {
                "name": "hierarchy",
                "manifold": "poincare",
                "intrinsic_dim": 8,
                "curvature": 0.5,
            },
            {
                "name": "spherical",
                "manifold": "sphere_projection",
                "intrinsic_dim": 8,
                "sectional_curvature": 0.25,
                "scale": 2.0,
            },
            {
                "name": "residual",
                "manifold": "euclidean",
                "intrinsic_dim": 16,
            },
        ]
    )

    assert isinstance(config, ProductConfig)
    assert config.component_names == ("hierarchy", "spherical", "residual")
    assert config.projection_dim == 32
    assert config.ambient_dim == 32
    assert config.components[0] == ProductComponentConfig(
        name="hierarchy",
        manifold="poincare",
        intrinsic_dim=8,
        curvature=0.5,
        scale=1.0,
    )
    assert config.components[1].sectional_curvature == 0.25
    assert config.components[1].scale == 2.0
    assert config.components[2].sectional_curvature == 0.0
    assert config.components[2].scale == 1.0


def test_product_config_tracks_projection_vs_lorentz_ambient_width() -> None:
    config = normalize_product_config(
        [
            {"name": "hyperboloid", "manifold": "lorentz", "intrinsic_dim": 8},
            {"name": "flat", "manifold": "euclidean", "intrinsic_dim": 16},
        ]
    )

    assert config.components[0].projection_dim == 8
    assert config.components[0].ambient_dim == 9
    assert config.components[1].projection_dim == 16
    assert config.components[1].ambient_dim == 16
    assert config.projection_dim == 24
    assert config.ambient_dim == 25


def test_product_config_assigns_deterministic_default_names() -> None:
    config = normalize_product_config(
        [
            {"manifold": "poincare", "intrinsic_dim": 4},
            {"manifold": "stereographic", "intrinsic_dim": 6, "sectional_curvature": 0.0},
        ]
    )

    assert config.component_names == ("component_0", "component_1")


def test_constructor_normalization_keeps_convenient_component_defaults() -> None:
    config = normalize_product_config(
        [{"manifold": "poincare", "intrinsic_dim": 2}]
    )

    assert config.components[0] == ProductComponentConfig(
        name="component_0",
        manifold="poincare",
        intrinsic_dim=2,
        curvature=1.0,
        scale=1.0,
    )


def test_product_config_persistence_round_trip_is_deterministic() -> None:
    config = normalize_product_config(
        [
            {"name": "negative", "manifold": "poincare", "intrinsic_dim": 8, "curvature": 0.5},
            {
                "name": "signed",
                "manifold": "stereographic",
                "intrinsic_dim": 8,
                "sectional_curvature": 0.125,
                "scale": 0.75,
            },
        ]
    )

    metadata = config.to_dict()
    assert metadata == {
        "type": "product",
        "version": 1,
        "components": [
            {
                "name": "negative",
                "manifold": "poincare",
                "intrinsic_dim": 8,
                "curvature": 0.5,
                "scale": 1.0,
            },
            {
                "name": "signed",
                "manifold": "stereographic",
                "intrinsic_dim": 8,
                "sectional_curvature": 0.125,
                "scale": 0.75,
            },
        ],
    }

    loaded = ProductConfig.from_dict(metadata)
    assert loaded == config
    assert loaded.to_dict() == metadata
    assert normalize_product_config(metadata) == config


@pytest.mark.parametrize(
    ("component", "missing_field"),
    [
        (
            {
                "name": "negative",
                "manifold": "poincare",
                "intrinsic_dim": 2,
                "curvature": 1.0,
                "scale": 1.0,
            },
            "name",
        ),
        (
            {
                "name": "negative",
                "manifold": "poincare",
                "intrinsic_dim": 2,
                "curvature": 1.0,
                "scale": 1.0,
            },
            "curvature",
        ),
        (
            {
                "name": "negative",
                "manifold": "poincare",
                "intrinsic_dim": 2,
                "curvature": 1.0,
                "scale": 1.0,
            },
            "scale",
        ),
        (
            {
                "name": "flat",
                "manifold": "euclidean",
                "intrinsic_dim": 2,
                "sectional_curvature": 0.0,
                "scale": 1.0,
            },
            "sectional_curvature",
        ),
    ],
)
def test_product_config_load_rejects_missing_persisted_fields(
    component, missing_field
) -> None:
    truncated = dict(component)
    truncated.pop(missing_field)

    with pytest.raises(ValueError, match=rf"missing required field.*{missing_field}"):
        ProductConfig.from_dict(
            {"type": "product", "version": 1, "components": [truncated]}
        )


@pytest.mark.parametrize("version", [1.0, True, "1", None])
def test_product_config_load_requires_integer_schema_version(version) -> None:
    component = {
        "name": "flat",
        "manifold": "euclidean",
        "intrinsic_dim": 2,
        "sectional_curvature": 0.0,
        "scale": 1.0,
    }

    with pytest.raises(ValueError, match="unsupported product metadata version"):
        ProductConfig.from_dict(
            {"type": "product", "version": version, "components": [component]}
        )


@pytest.mark.parametrize("manifold", [[], {}, ["poincare"]])
def test_constructor_rejects_non_string_manifold_values(manifold) -> None:
    with pytest.raises(ValueError, match="manifold.*string"):
        normalize_product_config([{"manifold": manifold, "intrinsic_dim": 2}])


@pytest.mark.parametrize("manifold", [[], {}, ["poincare"]])
def test_persistence_rejects_non_string_manifold_values(manifold) -> None:
    component = {
        "name": "broken",
        "manifold": manifold,
        "intrinsic_dim": 2,
        "curvature": 1.0,
        "scale": 1.0,
    }

    with pytest.raises(ValueError, match="persisted product component 0 manifold.*string"):
        ProductConfig.from_dict(
            {"type": "product", "version": 1, "components": [component]}
        )


def test_direct_component_construction_converts_numeric_overflow_to_value_error() -> None:
    huge = 10**10000

    with pytest.raises(ValueError, match="curvature must be a positive finite number"):
        ProductComponentConfig(
            name="negative",
            manifold="poincare",
            intrinsic_dim=2,
            curvature=huge,
        )

    with pytest.raises(ValueError, match="sectional_curvature must be finite"):
        ProductComponentConfig(
            name="signed",
            manifold="stereographic",
            intrinsic_dim=2,
            sectional_curvature=huge,
        )

    with pytest.raises(ValueError, match="scale must be a positive finite number"):
        ProductComponentConfig(
            name="flat",
            manifold="euclidean",
            intrinsic_dim=2,
            scale=huge,
        )


def test_constructor_and_persistence_convert_numeric_overflow_to_value_error() -> None:
    huge = 10**10000

    with pytest.raises(ValueError, match="curvature must be a positive finite number"):
        normalize_product_config(
            [{"manifold": "poincare", "intrinsic_dim": 2, "curvature": huge}]
        )

    persisted = {
        "type": "product",
        "version": 1,
        "components": [
            {
                "name": "signed",
                "manifold": "stereographic",
                "intrinsic_dim": 2,
                "sectional_curvature": huge,
                "scale": 1.0,
            }
        ],
    }
    with pytest.raises(ValueError, match="sectional_curvature must be finite"):
        ProductConfig.from_dict(persisted)


@pytest.mark.parametrize(
    ("component", "message"),
    [
        ({"manifold": "product", "intrinsic_dim": 8}, "unsupported manifold"),
        ({"manifold": "spd", "intrinsic_dim": 8}, "unsupported manifold"),
        ({"manifold": "euclidean", "intrinsic_dim": 0}, "positive integer"),
        ({"manifold": "euclidean", "intrinsic_dim": -1}, "positive integer"),
        ({"manifold": "euclidean", "intrinsic_dim": 1.5}, "positive integer"),
        ({"manifold": "euclidean", "intrinsic_dim": True}, "positive integer"),
        ({"manifold": "poincare", "intrinsic_dim": 8, "curvature": 0.0}, "positive finite"),
        ({"manifold": "poincare", "intrinsic_dim": 8, "curvature": math.inf}, "positive finite"),
        (
            {"manifold": "poincare", "intrinsic_dim": 8, "sectional_curvature": -1.0},
            "uses curvature",
        ),
        (
            {"manifold": "lorentz", "intrinsic_dim": 8, "sectional_curvature": -1.0},
            "uses curvature",
        ),
        ({"manifold": "euclidean", "intrinsic_dim": 8, "curvature": 1.0}, "does not accept curvature"),
        (
            {"manifold": "euclidean", "intrinsic_dim": 8, "sectional_curvature": 0.5},
            "exactly 0.0",
        ),
        ({"manifold": "sphere_projection", "intrinsic_dim": 8}, "requires sectional_curvature"),
        (
            {"manifold": "sphere_projection", "intrinsic_dim": 8, "sectional_curvature": 0.0},
            "positive and finite",
        ),
        ({"manifold": "stereographic", "intrinsic_dim": 8}, "requires sectional_curvature"),
        (
            {"manifold": "stereographic", "intrinsic_dim": 8, "sectional_curvature": math.nan},
            "must be finite",
        ),
        ({"manifold": "euclidean", "intrinsic_dim": 8, "scale": 0.0}, "positive finite"),
        ({"manifold": "euclidean", "intrinsic_dim": 8, "scale": math.inf}, "positive finite"),
        ({"manifold": "euclidean", "intrinsic_dim": 8, "components": []}, "unsupported field"),
    ],
)
def test_product_config_rejects_invalid_components(component, message) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_product_config([component])


def test_product_config_rejects_empty_and_duplicate_component_identity() -> None:
    with pytest.raises(ValueError, match="at least one"):
        normalize_product_config([])

    with pytest.raises(ValueError, match="names must be unique"):
        normalize_product_config(
            [
                {"name": "shared", "manifold": "euclidean", "intrinsic_dim": 2},
                {"name": "shared", "manifold": "euclidean", "intrinsic_dim": 3},
            ]
        )

    with pytest.raises(ValueError, match="trimmed string"):
        normalize_product_config(
            [{"name": " hierarchy ", "manifold": "euclidean", "intrinsic_dim": 2}]
        )


def test_product_config_load_rejects_unknown_schema_and_fields() -> None:
    valid_component = {
        "name": "flat",
        "manifold": "euclidean",
        "intrinsic_dim": 4,
        "sectional_curvature": 0.0,
        "scale": 1.0,
    }

    with pytest.raises(ValueError, match="type must be 'product'"):
        ProductConfig.from_dict(
            {"type": "single", "version": 1, "components": [valid_component]}
        )
    with pytest.raises(ValueError, match="unsupported product metadata version"):
        ProductConfig.from_dict(
            {"type": "product", "version": 2, "components": [valid_component]}
        )
    with pytest.raises(ValueError, match="unsupported product metadata field"):
        ProductConfig.from_dict(
            {
                "type": "product",
                "version": 1,
                "components": [valid_component],
                "nested": {},
            }
        )


def test_single_manifold_constructor_contract_is_unchanged() -> None:
    parameters = signature(ManifoldSentenceTransformer).parameters

    assert parameters["manifold"].default == "poincare"
    assert parameters["embedding_dim"].default is None
    assert parameters["curvature"].default == 1.0
    assert parameters["learnable_curvature"].default is False
    assert parameters["sectional_curvature"].default is None
    assert "product_config" not in parameters
