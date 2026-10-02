"""Regression tests for the v0.9 constant-curvature comparison example."""

from pathlib import Path
import math
import runpy
import sys

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
                    float(len(sentence)) / 20.0,
                    float(vowel_count) / 10.0,
                    float(code_sum % 17) / 20.0,
                    float(code_sum % 29) / 30.0,
                ]
            )
        return {"input_features": torch.tensor(rows, dtype=torch.float32)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def _load_example() -> dict:
    example_path = (
        Path(__file__).parents[1] / "examples" / "v09_constant_curvature_comparison.py"
    )
    return runpy.run_path(str(example_path))


def test_v09_constant_curvature_comparison_is_deterministic_and_finite(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    namespace = _load_example()
    run_comparison = namespace["run_comparison"]
    validate = namespace["_validate_regression"]

    torch.manual_seed(999)
    first = run_comparison("fake-model", seed=61, embedding_dim=4)
    torch.manual_seed(123)
    second = run_comparison("fake-model", seed=61, embedding_dim=4)

    validate(first)
    validate(second)
    assert first == second
    assert first["objective"] == "mnrl"
    assert first["temperature"] == 0.1
    assert tuple(first["variants"]) == (
        "euclidean",
        "poincare",
        "lorentz",
        "sphere_projection",
        "stereographic_negative",
        "stereographic_zero",
        "stereographic_positive",
    )

    expected_manifold = {
        "euclidean": "euclidean",
        "poincare": "poincare",
        "lorentz": "lorentz",
        "sphere_projection": "sphere_projection",
        "stereographic_negative": "stereographic",
        "stereographic_zero": "stereographic",
        "stereographic_positive": "stereographic",
    }
    expected_sectional = {
        "euclidean": 0.0,
        "poincare": -0.5,
        "lorentz": -0.5,
        "sphere_projection": 0.5,
        "stereographic_negative": -0.5,
        "stereographic_zero": 0.0,
        "stereographic_positive": 0.5,
    }
    expected_metric_keys = {
        "mrr",
        "recall_at_1",
        "recall_at_3",
        "ndcg_at_1",
        "ndcg_at_3",
    }

    for name, diagnostics in first["variants"].items():
        assert math.isfinite(diagnostics["loss"])
        assert set(diagnostics["retrieval"]) == expected_metric_keys
        assert all(math.isfinite(value) for value in diagnostics["retrieval"].values())
        assert all(0.0 <= value <= 1.0 for value in diagnostics["retrieval"].values())

        metadata = diagnostics["metadata"]
        assert metadata["manifold"] == expected_manifold[name]
        assert math.isclose(
            metadata["sectional_curvature"],
            expected_sectional[name],
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        assert metadata["embedding_device"] == "cpu"

    assert first["variants"]["poincare"]["metadata"]["curvature_api"] == "curvature"
    assert first["variants"]["lorentz"]["metadata"]["curvature_api"] == "curvature"
    for name in (
        "euclidean",
        "sphere_projection",
        "stereographic_negative",
        "stereographic_zero",
        "stereographic_positive",
    ):
        assert (
            first["variants"][name]["metadata"]["curvature_api"]
            == "sectional_curvature"
        )

    assert first["variants"]["euclidean"]["metadata"]["embedding_dtype"] == "float32"
    assert first["variants"]["poincare"]["metadata"]["embedding_dtype"] == "float32"
    for name in (
        "lorentz",
        "sphere_projection",
        "stereographic_negative",
        "stereographic_zero",
        "stereographic_positive",
    ):
        assert first["variants"][name]["metadata"]["embedding_dtype"] == "float64"


def test_v09_constant_curvature_comparison_root_command_runs_with_fake_encoder(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    example_path = (
        Path(__file__).parents[1] / "examples" / "v09_constant_curvature_comparison.py"
    )
    monkeypatch.setattr(sys, "argv", [str(example_path), "--model", "fake-model"])

    runpy.run_path(str(example_path), run_name="__main__")

    output = capsys.readouterr().out.lower()
    for name in (
        "euclidean",
        "poincare",
        "lorentz",
        "sphere_projection",
        "stereographic_negative",
        "stereographic_zero",
        "stereographic_positive",
    ):
        assert f'"{name}"' in output
    assert '"sectional_curvature"' in output
    assert '"embedding_dtype"' in output
    assert '"ndcg_at_3"' in output
    assert "not a benchmark" in output
    assert "not" in output and "superiority" in output
    assert "nan" not in output
    assert "inf" not in output
