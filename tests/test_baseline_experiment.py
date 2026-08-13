"""Tests for the Euclidean/Poincare/Lorentz comparison benchmark."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "experiments" / "compare_euclidean_poincare.py"


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used to run the benchmark without downloads."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
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


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location("neembed_baseline_benchmark", BENCHMARK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load baseline benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _patch_benchmark(monkeypatch, benchmark) -> None:
    monkeypatch.setattr(benchmark, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(benchmark.model_module, "SentenceTransformer", FakeSentenceTransformer)


def test_benchmark_script_exists() -> None:
    assert BENCHMARK_PATH.is_file()


def test_benchmark_returns_directly_comparable_results(monkeypatch) -> None:
    benchmark = _load_benchmark_module()
    _patch_benchmark(monkeypatch, benchmark)

    result = benchmark.run_benchmark()
    metadata = result["metadata"]
    train_pairs = metadata["train_pairs"]
    evaluation_pairs = metadata["evaluation_pairs"]

    assert metadata["benchmark"] == "tiny_hierarchy_retrieval"
    assert metadata["seed"] == 0
    assert metadata["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert metadata["epochs"] == 2
    assert metadata["embedding_dim"] == 2
    assert metadata["curvature"] == 1.0
    assert metadata["ambient_dimensions"] == {
        "euclidean_finetuned": 2,
        "poincare_finetuned": 2,
        "lorentz_finetuned": 3,
    }
    assert len(train_pairs) == 10
    assert len(evaluation_pairs) == 5
    assert {pair["anchor"] for pair in train_pairs}.isdisjoint(
        {pair["anchor"] for pair in evaluation_pairs}
    )

    results = result["results"]
    assert set(results) == {
        "euclidean_finetuned",
        "poincare_finetuned",
        "lorentz_finetuned",
    }
    assert results["euclidean_finetuned"]["distance"] == "cosine"
    assert results["poincare_finetuned"]["distance"] == "poincare_geodesic"
    assert results["lorentz_finetuned"]["distance"] == "lorentz_geodesic"

    for variant in results.values():
        assert set(variant) == {
            "distance",
            "retrieval_accuracy",
            "mean_positive_distance",
            "mean_negative_distance",
            "mrr",
            "recall_at_1",
            "final_training_loss",
        }
        assert 0.0 <= variant["retrieval_accuracy"] <= 1.0
        assert variant["recall_at_1"] == variant["retrieval_accuracy"]
        assert 0.0 <= variant["mrr"] <= 1.0
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
    for variant in (
        "euclidean_finetuned",
        "poincare_finetuned",
        "lorentz_finetuned",
    ):
        first_result = first["results"][variant]
        second_result = second["results"][variant]
        assert first_result["distance"] == second_result["distance"]
        for metric in first_result:
            if metric == "distance":
                continue
            assert math.isclose(
                first_result[metric],
                second_result[metric],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
