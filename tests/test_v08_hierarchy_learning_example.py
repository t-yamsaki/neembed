"""Regression tests for the v0.8 hierarchy-native learning example."""

from pathlib import Path
import math
import runpy
import sys

import pytest
import torch
from torch import nn

import neembed.model as model_module


class FakeSentenceTransformer(nn.Module):
    """Small deterministic trainable encoder used without network downloads."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.linear = nn.Linear(5, 7, bias=False)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 7

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = []
        for sentence in sentences:
            code_sum = sum(ord(char) for char in sentence)
            vowel_count = sum(char.lower() in "aeiou" for char in sentence)
            word_count = len(sentence.split())
            rows.append(
                [
                    float(len(sentence)),
                    float(word_count),
                    float(vowel_count),
                    float(code_sum % 17),
                    float(code_sum % 31),
                ]
            )
        return {"input_features": torch.tensor(rows, dtype=torch.float32)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        embedding = self.linear(features["input_features"])
        return {"sentence_embedding": self.dropout(embedding)}


def _load_example() -> dict:
    example_path = Path(__file__).parents[1] / "examples" / "v08_hierarchy_learning.py"
    return runpy.run_path(str(example_path))


def test_v08_hierarchy_learning_is_deterministic_finite_and_explicit(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    namespace = _load_example()
    run_hierarchy_learning = namespace["run_hierarchy_learning"]
    validate = namespace["_validate_regression"]

    torch.manual_seed(999)
    first = run_hierarchy_learning(
        "fake-model",
        epochs=1,
        seed=47,
        embedding_dim=4,
        learning_rate=1e-4,
        hierarchy_weight=0.25,
    )
    torch.manual_seed(123)
    second = run_hierarchy_learning(
        "fake-model",
        epochs=1,
        seed=47,
        embedding_dim=4,
        learning_rate=1e-4,
        hierarchy_weight=0.25,
    )

    validate(first)
    validate(second)
    assert first == second
    assert first["manifold"] == "poincare"

    supervision = first["supervision"]
    assert supervision["node_ids"] == list(namespace["NODE_IDS"])
    assert supervision["parent_child_edges"] == [
        list(edge) for edge in namespace["PARENT_CHILD_EDGES"]
    ]
    assert supervision["depths"] == namespace["DEPTHS"]
    assert supervision["directed_triplets"] == [
        list(triplet) for triplet in namespace["DIRECTED_TRIPLETS"]
    ]

    assert first["retrieval_only"]["before"] == first["hierarchy_aware"]["before"]
    assert len(first["retrieval_only"]["training_losses"]) == 3
    assert len(first["hierarchy_aware"]["training_losses"]) == 3
    assert all(
        math.isfinite(value)
        for variant in ("retrieval_only", "hierarchy_aware")
        for value in first[variant]["training_losses"]
    )

    diagnostics = first["hierarchy_aware"]["objective_diagnostics"]
    assert tuple(diagnostics) == ("radial", "depth", "directed")
    for values in diagnostics.values():
        assert all(math.isfinite(value) for value in values.values())
        assert values["total_loss"] == pytest.approx(
            values["retrieval_loss"]
            + values["hierarchy_weight"] * values["hierarchy_loss"]
        )

    for variant in ("retrieval_only", "hierarchy_aware"):
        assert set(first[variant]["before"]) == {"retrieval", "hierarchy"}
        assert set(first[variant]["after"]) == {"retrieval", "hierarchy"}
        hierarchy_metrics = first[variant]["after"]["hierarchy"]
        assert set(hierarchy_metrics) == {
            "parent_child_radial_order_accuracy",
            "mean_radial_order_violation",
            "depth_radius_spearman",
        }
        assert all(math.isfinite(value) for value in hierarchy_metrics.values())


def test_v08_hierarchy_learning_root_command_runs_without_download(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    example_path = Path(__file__).parents[1] / "examples" / "v08_hierarchy_learning.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(example_path), "--model", "fake-model", "--epochs", "1"],
    )

    runpy.run_path(str(example_path), run_name="__main__")

    output = capsys.readouterr().out.lower()
    assert '"retrieval_only"' in output
    assert '"hierarchy_aware"' in output
    assert '"radial"' in output
    assert '"depth"' in output
    assert '"directed"' in output
    assert '"parent_child_radial_order_accuracy"' in output
    assert '"depth_radius_spearman"' in output
    assert "not a benchmark" in output
    assert "not" in output and "superiority" in output
    assert "nan" not in output
    assert "inf" not in output
