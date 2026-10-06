"""Versioned local sentence-model metadata, separate from ProductConfig."""

from collections.abc import Mapping
from importlib.metadata import version
import json
from pathlib import Path
import re

import torch

from neembed.product_config import ProductComponentConfig, ProductConfig


FORMAT_VERSION = 1
DTYPES = {name: getattr(torch, name) for name in ("float16", "bfloat16", "float32", "float64")}
_GEOMETRY_KEYS = {"manifold", "embedding_dim", "curvature", "learnable_curvature",
                  "sectional_curvature", "product_config"}
_VERSIONED_KEYS = {"format_version", "neembed_version", "input_config", "dtypes", "base_model"}


def _module_dtype(module):
    dtypes = {value.dtype for value in (*module.parameters(), *module.buffers()) if value.is_floating_point()}
    if len(dtypes) > 1:
        raise ValueError("saved modules must use a uniform floating dtype")
    if not dtypes:
        return None
    name = str(dtypes.pop()).removeprefix("torch.")
    if name not in DTYPES:
        raise ValueError(f"unsupported saved dtype: {name!r}")
    return name


def validate_state_dtype(state, expected_dtype, *, filename):
    if not isinstance(state, Mapping) or any(
        not torch.is_tensor(value) or (value.is_floating_point() and
            str(value.dtype).removeprefix("torch.") != expected_dtype)
        for value in state.values()
    ):
        raise ValueError(f"{filename} does not match the saved dtype")


def _model_id(value):
    if _hub_id(value) is None or Path(value).exists():
        return None
    return _hub_id(value)


def _hub_id(value):
    # Persist only a Hub-style ID, never URLs, absolute/cache paths or credentials.
    if isinstance(value, str) and len(value) <= 96 and re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*(/[A-Za-z0-9_-][A-Za-z0-9_.-]*)?", value):
        return value
    return None


def _revision(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_./-]+", value) else None


def source_metadata(encoder, model_name_or_path, revision):
    card = getattr(encoder, "model_card_data", None)
    model_id = _model_id(model_name_or_path) or _model_id(getattr(card, "base_model", None))
    known_revision = None
    if model_id is not None and getattr(card, "base_model", None) == model_id:
        known_revision = _revision(getattr(card, "base_model_revision", None))
    return {"model_id": model_id, "revision": (known_revision or _revision(revision)) if model_id else None}


def new_metadata(model):
    prompts = getattr(model.encoder, "prompts", {})
    if not isinstance(prompts, Mapping):
        raise ValueError("encoder.prompts must be a mapping")
    return {
        "format_version": FORMAT_VERSION,
        "neembed_version": version("neembed-geoopt"),
        "input_config": {
            "prompts": dict(prompts),
            "default_prompt_name": getattr(model.encoder, "default_prompt_name", None),
        },
        "dtypes": {name: _module_dtype(module) for name, module in (
            ("encoder", model.encoder), ("projection", model.projection), ("geometry", model.manifold),
        )},
        "base_model": dict(model._base_model),
    }


def _keys(value, required, optional=(), *, field):
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f"invalid {field} fields; missing: {sorted(missing)}, unknown: {sorted(unknown)}")


def validate_config(config):
    """Reject malformed metadata before loading any encoder; return geometry kwargs."""
    if not isinstance(config, Mapping):
        raise ValueError("neembed_config.json must contain a JSON object")
    versioned = "format_version" in config
    if versioned:
        saved_version = config["format_version"]
        if type(saved_version) is not int or saved_version != FORMAT_VERSION:
            raise ValueError(f"unsupported neembed format_version: {saved_version!r}; expected {FORMAT_VERSION}")
    _keys(config, {"manifold", "embedding_dim"} | (_VERSIONED_KEYS if versioned else set()),
          _GEOMETRY_KEYS, field="neembed metadata")
    dim = config["embedding_dim"]
    if dim is not None and (type(dim) is not int or dim <= 0):
        raise ValueError("embedding_dim must be a positive integer or null")
    name = config["manifold"]
    if not isinstance(name, str):
        raise ValueError("manifold must be a supported name string")
    kwargs = {"manifold": name, "embedding_dim": dim}
    allowed = {"manifold", "embedding_dim"}
    if name == "product":
        if "product_config" not in config:
            raise ValueError("product metadata requires product_config")
        product = ProductConfig.from_dict(config["product_config"])
        if dim != product.projection_dim:
            raise ValueError("saved embedding_dim does not match product_config.projection_dim")
        kwargs["product_config"] = product
        allowed.add("product_config")
    else:
        field = "curvature" if name in ("poincare", "lorentz") else "sectional_curvature"
        if field not in config:
            raise ValueError(f"{name} metadata requires {field}")
        component = ProductComponentConfig(name="geometry", manifold=name, intrinsic_dim=dim or 1,
                                           **{field: config[field]})
        # Persisted curvature is required; null must not trigger constructor defaults.
        if config[field] is None:
            raise ValueError(f"{field} must be a finite number")
        kwargs[field] = getattr(component, field)
        allowed.add(field)
        if name in ("poincare", "lorentz"):
            learnable = config.get("learnable_curvature", False)
            if type(learnable) is not bool:
                raise ValueError("learnable_curvature must be a boolean")
            kwargs["learnable_curvature"] = learnable
            allowed.add("learnable_curvature")
    if config.keys() & (_GEOMETRY_KEYS - allowed):
        raise ValueError("saved geometry contains conflicting fields")
    if not versioned:
        return kwargs

    if not isinstance(config["neembed_version"], str) or not config["neembed_version"].strip():
        raise ValueError("neembed_version must be a non-empty string")
    inputs = config["input_config"]
    _keys(inputs, {"prompts", "default_prompt_name"}, field="input_config")
    prompts = inputs["prompts"]
    if not isinstance(prompts, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in prompts.items()):
        raise ValueError("input_config.prompts must map names to strings")
    default = inputs["default_prompt_name"]
    if default is not None and (not isinstance(default, str) or default not in prompts):
        raise ValueError("default_prompt_name must be null or a saved prompt name")
    dtypes = config["dtypes"]
    _keys(dtypes, {"encoder", "projection", "geometry"}, field="dtypes")
    for field, dtype in dtypes.items():
        if dtype is not None and (not isinstance(dtype, str) or dtype not in DTYPES):
            raise ValueError(f"unsupported dtypes.{field}: {dtype!r}")
    if (dim is None) != (dtypes["projection"] is None):
        raise ValueError("projection dtype does not match embedding_dim")
    if name in ("sphere_projection", "stereographic") or (
        name == "product" and any(c.manifold in ("lorentz", "sphere_projection", "stereographic") for c in kwargs["product_config"].components)
    ):
        if dtypes["geometry"] != "float64":
            raise ValueError("fixed-double geometry dtype must be float64")
    elif name in ("poincare", "lorentz") and dtypes["geometry"] is None:
        raise ValueError("curved geometry requires a floating dtype")
    source = config["base_model"]
    _keys(source, {"model_id", "revision"}, field="base_model")
    if source["model_id"] is not None and _hub_id(source["model_id"]) != source["model_id"]:
        raise ValueError("base_model.model_id must be a Hub-style ID or null")
    if source["revision"] is not None and (
        source["model_id"] is None or _revision(source["revision"]) != source["revision"]
    ):
        raise ValueError("base_model.revision requires a model ID and a valid revision string")
    return kwargs


def read_config(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate metadata field: {key!r}")
            result[key] = value
        return result

    try:
        config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid neembed_config.json: {exc.msg}") from exc
    return config, validate_config(config)
