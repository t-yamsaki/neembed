"""Smoke test for the hierarchy training example without model downloads."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

import neembed.model as model_module


class FakeSentenceTransformer(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = [[float(len(sentence)), 1.0, -1.0] for sentence in sentences]
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def test_hierarchy_example_constructs_without_download(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    example_path = Path(__file__).parents[1] / "examples" / "train_hierarchy.py"
    spec = importlib.util.spec_from_file_location("train_hierarchy", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.prototype_ids == ["animal", "dog", "cat"]
    assert module.parent_relations == [("dog", "animal"), ("cat", "animal")]
    assert module.prototypes.num_prototypes == 3
