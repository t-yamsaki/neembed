"""Shared opt-in input preprocessing for training and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from neembed.model import ManifoldSentenceTransformer


InputOptions = Mapping[str, Mapping[str, str | None]]


def _normalize_input_options(
    input_options: InputOptions | None,
    input_names: Sequence[str],
) -> dict[str, dict[str, str | None]]:
    """Copy input-specific forward options and reject misspelled input/keys.

    Prompt selection and value validation remain in the model's common forward
    path. Unspecified options (including None values) keep legacy model calls.
    """
    if input_options is None:
        return {}
    if not isinstance(input_options, Mapping):
        raise TypeError("input_options must be a mapping of text input names to options")
    normalized = {}
    for name, options in input_options.items():
        if name not in input_names:
            raise ValueError(f"unknown text input in input_options: {name!r}")
        if not isinstance(options, Mapping):
            raise TypeError(f"input_options[{name!r}] must be a mapping")
        for key in options:
            if key not in ("task", "prompt_name", "prompt"):
                raise ValueError(f"unknown forward option for {name!r}: {key!r}")
        normalized[name] = {
            key: value for key, value in options.items() if value is not None
        }
    return normalized


def _encode_input(
    model: ManifoldSentenceTransformer,
    texts: Sequence[str],
    input_options: InputOptions,
    name: str,
) -> torch.Tensor:
    """Apply one input's options through differentiable model forward."""
    return model(texts, **input_options.get(name, {}))


def _encode_inference_input(
    model: ManifoldSentenceTransformer,
    texts: str | Sequence[str],
    options: Mapping[str, str | None] | None = None,
    *,
    convert_to_tensor: bool = False,
) -> Any:
    """Apply the same input options using inference encoding.

    Leave conversion unspecified on the CPU staging path so legacy encoders
    with encode(texts) signatures continue to work when no options are given.
    """
    kwargs: dict[str, Any] = dict(options or {})
    if convert_to_tensor:
        kwargs["convert_to_tensor"] = True
    return model.encode(texts, **kwargs)
