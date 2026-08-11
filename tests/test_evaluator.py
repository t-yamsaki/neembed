"""Tests for manifold embedding evaluation."""

import pytest
import torch
from torch import nn

import neembed
from neembed.evaluator import ManifoldEmbeddingEvaluator


class ControlledDistanceModel(nn.Module):
    """Small model stub with deterministic encoded points and distance calls."""

    def __init__(self, points: dict[str, float]) -> None:
        super().__init__()
        self.points = points
        self.distance_calls = 0
        self.grad_enabled_during_distance: bool | None = None

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_tensor: bool = False,
    ):
        self.eval()
        with torch.inference_mode():
            embeddings = torch.tensor(
                [[self.points[sentence]] for sentence in sentences],
                dtype=torch.float32,
            )
        if convert_to_tensor:
            return embeddings
        return embeddings.numpy()

    def distance(self, a, b) -> torch.Tensor:
        self.distance_calls += 1
        self.grad_enabled_during_distance = torch.is_grad_enabled()
        return torch.abs(a - b).sum(dim=-1)


def test_evaluator_is_exported_from_package() -> None:
    assert neembed.ManifoldEmbeddingEvaluator is ManifoldEmbeddingEvaluator


def test_evaluator_returns_expected_metrics_for_perfect_retrieval() -> None:
    model = ControlledDistanceModel(
        {"a0": 0.0, "a1": 10.0, "p0": 1.0, "p1": 9.0}
    )
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["a0", "a1"],
        positives=["p0", "p1"],
    )

    metrics = evaluator()

    assert metrics == {
        "retrieval_accuracy": pytest.approx(1.0),
        "mean_positive_distance": pytest.approx(1.0),
        "mean_negative_distance": pytest.approx(9.0),
    }


def test_evaluator_reports_incorrect_aligned_retrieval() -> None:
    model = ControlledDistanceModel(
        {"a0": 0.0, "a1": 10.0, "p0": 9.0, "p1": 1.0}
    )
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["a0", "a1"],
        positives=["p0", "p1"],
    )

    first = evaluator()
    second = evaluator()

    assert first == second
    assert first["retrieval_accuracy"] == pytest.approx(0.0)
    assert first["mean_positive_distance"] == pytest.approx(9.0)
    assert first["mean_negative_distance"] == pytest.approx(1.0)


def test_evaluator_uses_model_distance_without_grad_and_restores_training_mode() -> None:
    model = ControlledDistanceModel(
        {"a0": 0.0, "a1": 10.0, "p0": 1.0, "p1": 9.0}
    )
    model.train()
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["a0", "a1"],
        positives=["p0", "p1"],
    )

    evaluator()

    assert model.distance_calls == 1
    assert model.grad_enabled_during_distance is False
    assert model.training


def test_evaluator_preserves_eval_mode() -> None:
    model = ControlledDistanceModel(
        {"a0": 0.0, "a1": 10.0, "p0": 1.0, "p1": 9.0}
    )
    model.eval()
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["a0", "a1"],
        positives=["p0", "p1"],
    )

    evaluator()

    assert not model.training


def test_evaluator_rejects_mismatched_pair_counts() -> None:
    model = ControlledDistanceModel({"a0": 0.0, "a1": 1.0, "p0": 0.5})

    with pytest.raises(ValueError, match="same number"):
        ManifoldEmbeddingEvaluator(
            model=model,
            anchors=["a0", "a1"],
            positives=["p0"],
        )


def test_evaluator_rejects_fewer_than_two_pairs() -> None:
    model = ControlledDistanceModel({"a0": 0.0, "p0": 1.0})

    with pytest.raises(ValueError, match="at least two aligned pairs"):
        ManifoldEmbeddingEvaluator(
            model=model,
            anchors=["a0"],
            positives=["p0"],
        )
