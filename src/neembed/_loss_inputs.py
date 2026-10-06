"""Shared opt-in input preprocessing for text objectives."""

from collections.abc import Mapping, Sequence

import torch

from neembed.model import ManifoldSentenceTransformer


LossInputOptions = Mapping[str, Mapping[str, str | None]]


def _normalize_input_options(
    input_options: LossInputOptions | None,
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
    input_options: LossInputOptions,
    name: str,
) -> torch.Tensor:
    """Apply one input's options through differentiable model forward."""
    return model(texts, **input_options.get(name, {}))
