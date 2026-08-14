"""Tests for manifold embedding evaluation."""

import math

import pytest
import torch
from torch import nn

import neembed
import neembed.model as model_module
from neembed.evaluator import ManifoldEmbeddingEvaluator
from neembed.model import ManifoldSentenceTransformer


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


class FakeSentenceTransformer(nn.Module):
    """Small encoder used to exercise both real manifold backends."""

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
        "mrr": pytest.approx(1.0),
        "recall_at_1": pytest.approx(1.0),
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
    assert first["mrr"] == pytest.approx(0.5)
    assert first["recall_at_1"] == pytest.approx(0.0)


def test_evaluator_reports_known_ranks_and_configurable_recall_cutoffs() -> None:
    model = ControlledDistanceModel(
        {
            "a0": 0.0,
            "a1": 17.0,
            "a2": 0.0,
            "p0": 0.0,
            "p1": 10.0,
            "p2": 20.0,
        }
    )
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["a0", "a1", "a2"],
        positives=["p0", "p1", "p2"],
        recall_at_k=(1, 2, 3, 5),
    )

    metrics = evaluator()

    assert metrics["retrieval_accuracy"] == pytest.approx(1.0 / 3.0)
    assert metrics["recall_at_1"] == pytest.approx(metrics["retrieval_accuracy"])
    assert metrics["recall_at_2"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall_at_3"] == pytest.approx(1.0)
    assert metrics["recall_at_5"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 1.0 / 3.0) / 3.0)


def test_evaluator_preserves_index_order_for_equal_distance_ties() -> None:
    model = ControlledDistanceModel(
        {
            "a0": 0.0,
            "a1": 0.0,
            "p0": -1.0,
            "p1": 1.0,
        }
    )
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["a0", "a1"],
        positives=["p0", "p1"],
        recall_at_k=(1, 2),
    )

    metrics = evaluator()

    # Both candidates tie for both anchors. Stable index ordering gives the
    # aligned targets ranks 1 and 2 respectively.
    assert metrics["retrieval_accuracy"] == pytest.approx(0.5)
    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["recall_at_2"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "recall_at_k,match",
    [
        ((), "at least one cutoff"),
        ((0,), "positive integers"),
        ((-1,), "positive integers"),
        ((1.5,), "positive integers"),
        ((True,), "positive integers"),
        ((1, 1), "unique"),
    ],
)
def test_evaluator_rejects_invalid_recall_cutoffs(recall_at_k, match: str) -> None:
    model = ControlledDistanceModel(
        {"a0": 0.0, "a1": 10.0, "p0": 1.0, "p1": 9.0}
    )

    with pytest.raises(ValueError, match=match):
        ManifoldEmbeddingEvaluator(
            model=model,
            anchors=["a0", "a1"],
            positives=["p0", "p1"],
            recall_at_k=recall_at_k,
        )


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


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_evaluator_ranking_metrics_are_finite_on_supported_manifolds(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold,
        embedding_dim=2,
    )
    evaluator = ManifoldEmbeddingEvaluator(
        model=model,
        anchors=["a", "bbbb", "cccccccc"],
        positives=["aa", "bbbbb", "ccccccccc"],
        recall_at_k=(1, 2, 5),
    )

    metrics = evaluator()

    assert set(metrics) == {
        "retrieval_accuracy",
        "mean_positive_distance",
        "mean_negative_distance",
        "mrr",
        "recall_at_1",
        "recall_at_2",
        "recall_at_5",
    }
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["recall_at_1"] == pytest.approx(metrics["retrieval_accuracy"])
    assert metrics["recall_at_5"] == pytest.approx(1.0)


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
