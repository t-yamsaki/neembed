"""Regression coverage for numerically large graded relevance values."""

import math

import pytest
import torch
from torch import nn

from neembed.graded_evaluator import ManifoldGradedCorpusRetrievalEvaluator


class ControlledCorpusModel(nn.Module):
    """Minimal model with deterministic one-dimensional encoded points."""

    def __init__(self, points: dict[str, float]) -> None:
        super().__init__()
        self.points = points

    def encode(self, sentences: list[str]):
        self.eval()
        with torch.inference_mode():
            return torch.tensor(
                [[self.points[sentence]] for sentence in sentences],
                dtype=torch.float32,
            )

    def distance(self, a, b) -> torch.Tensor:
        return torch.abs(a - b).sum(dim=-1)


def test_large_finite_grades_do_not_overflow_ndcg() -> None:
    model = ControlledCorpusModel({"q": 0.0, "lower": 0.1, "higher": 0.2})
    evaluator = ManifoldGradedCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["lower-id", "higher-id"],
        corpus=["lower", "higher"],
        graded_relevance={
            "q-id": {"lower-id": 2047.0, "higher-id": 2048.0}
        },
        recall_at_k=(1, 2),
        ndcg_at_k=(2,),
    )

    metrics = evaluator()

    lower_gain_ratio = 0.5
    expected = (
        lower_gain_ratio + 1.0 / math.log2(3.0)
    ) / (
        1.0 + lower_gain_ratio / math.log2(3.0)
    )
    assert math.isfinite(metrics["ndcg_at_2"])
    assert metrics["ndcg_at_2"] == pytest.approx(expected)
