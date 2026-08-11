"""Regression tests for the v0.2 Euclidean-vs-Poincare benchmark."""

import math

import pytest
import torch
from torch import nn

import neembed.model as model_module
from experiments import compare_euclidean_poincare as benchmark


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


def _patch_benchmark(monkeypatch) -> None:
    monkeypatch.setattr(benchmark, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(benchmark, "EPOCHS", 2)
    monkeypatch.setattr(benchmark, "EMBEDDING_DIM", 2)


def test_benchmark_batches_keep_in_batch_positives_unique() -> None:
    for _, positives in benchmark._train_batches():
        assert len(positives) == len(set(positives))


def test_benchmark_reports_shared_v02_metrics_for_both_geometries(monkeypatch) -> None:
    _patch_benchmark(monkeypatch)

    output = benchmark.run_benchmark()

    assert output["metadata"]["benchmark"] == "tiny_hierarchy_retrieval"
    assert output["metadata"]["seed"] == benchmark.SEED
    assert output["metadata"]["epochs"] == 2

    results = output["results"]
    assert set(results) == {"euclidean_finetuned", "poincare_finetuned"}
    assert results["euclidean_finetuned"]["distance"] == "cosine"
    assert results["poincare_finetuned"]["distance"] == "poincare_geodesic"

    for result in results.values():
        assert set(result) == {
            "distance",
            "retrieval_accuracy",
            "mean_positive_distance",
            "mean_negative_distance",
            "final_training_loss",
        }
        assert 0.0 <= result["retrieval_accuracy"] <= 1.0
        assert math.isfinite(result["mean_positive_distance"])
        assert math.isfinite(result["mean_negative_distance"])
        assert math.isfinite(result["final_training_loss"])


def test_benchmark_is_deterministic_with_fixed_seed(monkeypatch) -> None:
    _patch_benchmark(monkeypatch)

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


def test_original_run_experiment_entry_point_remains_available(monkeypatch) -> None:
    _patch_benchmark(monkeypatch)

    assert benchmark.run_experiment() == benchmark.run_benchmark()
