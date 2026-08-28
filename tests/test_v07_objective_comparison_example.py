"""Regression tests for the v0.7 retrieval-objective comparison example."""

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
        self.linear = nn.Linear(4, 6, bias=False)

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 6

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = []
        for sentence in sentences:
            code_sum = sum(ord(char) for char in sentence)
            vowel_count = sum(char.lower() in "aeiou" for char in sentence)
            rows.append(
                [
                    float(len(sentence)),
                    float(vowel_count),
                    float(code_sum % 17),
                    float(code_sum % 29),
                ]
            )
        return {"input_features": torch.tensor(rows, dtype=torch.float32)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def _load_example() -> dict:
    example_path = (
        Path(__file__).parents[1] / "examples" / "v07_objective_comparison.py"
    )
    return runpy.run_path(str(example_path))


def test_v07_objective_comparison_is_deterministic_and_finite(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    namespace = _load_example()
    run_comparison = namespace["run_comparison"]
    validate = namespace["_validate_regression"]

    torch.manual_seed(999)
    first = run_comparison(
        "fake-model",
        epochs=1,
        seed=41,
        embedding_dim=4,
        learning_rate=1e-4,
    )
    torch.manual_seed(123)
    second = run_comparison(
        "fake-model",
        epochs=1,
        seed=41,
        embedding_dim=4,
        learning_rate=1e-4,
    )

    validate(first)
    validate(second)
    assert first == pytest.approx(second)

    assert first["manifold"] == "poincare"
    assert tuple(first["objectives"]) == (
        "mnrl",
        "triplet",
        "margin_mse",
        "distance_mse",
    )

    before_metrics = [
        diagnostics["before"] for diagnostics in first["objectives"].values()
    ]
    assert all(metrics == before_metrics[0] for metrics in before_metrics[1:])

    expected_metric_keys = {
        "mrr",
        "recall_at_1",
        "recall_at_3",
        "ndcg_at_1",
        "ndcg_at_3",
    }
    for diagnostics in first["objectives"].values():
        assert math.isfinite(diagnostics["final_training_loss"])
        assert set(diagnostics["before"]) == expected_metric_keys
        assert set(diagnostics["after"]) == expected_metric_keys
        assert all(
            math.isfinite(value)
            for stage in (diagnostics["before"], diagnostics["after"])
            for value in stage.values()
        )


def test_v07_objective_comparison_root_command_runs_with_fake_encoder(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    example_path = (
        Path(__file__).parents[1] / "examples" / "v07_objective_comparison.py"
    )
    monkeypatch.setattr(sys, "argv", [str(example_path), "--model", "fake-model"])

    runpy.run_path(str(example_path), run_name="__main__")

    output = capsys.readouterr().out.lower()
    assert '"mnrl"' in output
    assert '"triplet"' in output
    assert '"margin_mse"' in output
    assert '"distance_mse"' in output
    assert '"ndcg_at_3"' in output
    assert "not a benchmark" in output
    assert "not" in output and "superiority" in output
    assert "nan" not in output
    assert "inf" not in output
