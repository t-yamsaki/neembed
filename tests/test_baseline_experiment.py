"""Tests for the reproducible Euclidean-vs-Poincare benchmark."""

import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

import neembed.model as model_module


class FakeSentenceTransformer(nn.Module):
    """Deterministic trainable encoder used without model downloads."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.linear = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.5, -0.25, 0.10],
                        [0.20, 0.40, -0.30],
                        [-0.10, 0.30, 0.50],
                        [0.25, -0.15, 0.35],
                    ]
                )
            )

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = [
            [float(len(sentence)), float(index + 1), -1.0]
            for index, sentence in enumerate(sentences)
        ]
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def _load_benchmark_module():
    benchmark_path = (
        Path(__file__).parents[1] / "experiments" / "compare_euclidean_poincare.py"
    )
    spec = importlib.util.spec_from_file_location("neembed_comparison_benchmark", benchmark_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_benchmark(monkeypatch, benchmark) -> None:
    monkeypatch.setattr(benchmark, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(benchmark, "EPOCHS", 2)
    monkeypatch.setattr(benchmark, "EMBEDDING_DIM", 2)


def test_benchmark_returns_directly_comparable_results(monkeypatch) -> None:
    benchmark = _load_benchmark_module()
    _patch_benchmark(monkeypatch, benchmark)

    result = benchmark.run_benchmark()
    train_pairs = result["metadata"]["train_pairs"]
    evaluation_pairs = result["metadata"]["evaluation_pairs"]

    assert result["metadata"]["benchmark"] == "tiny_hierarchy_retrieval"
    assert result["metadata"]["seed"] == 0
    assert result["metadata"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert result["metadata"]["epochs"] == 2
    assert result["metadata"]["embedding_dim"] == 2
    assert len(train_pairs) == 10
    assert len(evaluation_pairs) == 5
    assert {pair["anchor"] for pair in train_pairs}.isdisjoint(
        {pair["anchor"] for pair in evaluation_pairs}
    )

    results = result["results"]
    assert set(results) == {"euclidean_finetuned", "poincare_finetuned"}
    assert results["euclidean_finetuned"]["distance"] == "cosine"
    assert results["poincare_finetuned"]["distance"] == "poincare_geodesic"

    for variant in results.values():
        assert set(variant) == {
            "distance",
            "retrieval_accuracy",
            "mean_positive_distance",
            "mean_negative_distance",
            "final_training_loss",
        }
        assert 0.0 <= variant["retrieval_accuracy"] <= 1.0
        assert math.isfinite(variant["mean_positive_distance"])
        assert math.isfinite(variant["mean_negative_distance"])
        assert math.isfinite(variant["final_training_loss"])

    for _, positives in benchmark._train_batches():
        assert len(positives) == len(set(positives))
    json.dumps(result, allow_nan=False)


def test_benchmark_is_deterministic_with_fixed_seed(monkeypatch) -> None:
    benchmark = _load_benchmark_module()
    _patch_benchmark(monkeypatch, benchmark)

    first = benchmark.run_benchmark()
    second = benchmark.run_benchmark()

    assert first["metadata"] == second["metadata"]
    for variant in ("euclidean_finetuned", "poincare_finetuned"):
        first_result = first["results"][variant]
        second_result = second["results"][variant]
        assert first_result["distance"] == second_result["distance"]
        for metric in (
            "retrieval_accuracy",
            "mean_positive_distance",
            "mean_negative_distance",
            "final_training_loss",
        ):
            assert first_result[metric] == pytest.approx(second_result[metric], abs=1e-8)


def test_original_run_experiment_alias_and_cli_emit_benchmark_json(
    monkeypatch,
    capsys,
) -> None:
    benchmark = _load_benchmark_module()
    _patch_benchmark(monkeypatch, benchmark)

    assert benchmark.run_experiment() == benchmark.run_benchmark()

    benchmark.main()
    output = json.loads(capsys.readouterr().out)
    assert set(output["results"]) == {"euclidean_finetuned", "poincare_finetuned"}


def test_experiment_readme_documents_benchmark_command() -> None:
    readme = (
        Path(__file__).parents[1] / "experiments" / "README.md"
    ).read_text(encoding="utf-8")

    assert "[v0.2 comparison benchmark](compare_euclidean_poincare.py)" in readme
    assert "python experiments/compare_euclidean_poincare.py" in readme
    assert "not a leaderboard or a research result" in readme
