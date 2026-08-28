"""Tests for exact graded corpus retrieval evaluation."""

import math

import pytest
import torch
from torch import nn

import neembed
from neembed.evaluator import ManifoldCorpusRetrievalEvaluator
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


def _base_kwargs(model) -> dict:
    return {
        "model": model,
        "query_ids": ["q-id"],
        "queries": ["q"],
        "corpus_ids": ["c0-id", "c1-id", "c2-id"],
        "corpus": ["c0", "c1", "c2"],
        "graded_relevance": {"q-id": {"c0-id": 3.0, "c1-id": 1.0}},
        "recall_at_k": (1, 2, 3),
        "ndcg_at_k": (1, 2, 3),
    }


def test_graded_corpus_evaluator_is_exported() -> None:
    assert (
        neembed.ManifoldGradedCorpusRetrievalEvaluator
        is ManifoldGradedCorpusRetrievalEvaluator
    )
    assert "ManifoldGradedCorpusRetrievalEvaluator" in neembed.__all__


def test_existing_binary_evaluator_output_is_unchanged() -> None:
    model = ControlledCorpusModel({"q": 0.0, "c0": 0.1, "c1": 0.2})
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["c0-id", "c1-id"],
        corpus=["c0", "c1"],
        relevance={"q-id": ["c0-id"]},
        recall_at_k=(1, 2),
    )

    assert set(evaluator()) == {"mrr", "recall_at_1", "recall_at_2"}


def test_perfect_graded_ranking_has_unit_ndcg() -> None:
    model = ControlledCorpusModel(
        {"q": 0.0, "high": 0.1, "medium": 0.2, "low": 0.3}
    )
    evaluator = ManifoldGradedCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["high-id", "medium-id", "low-id"],
        corpus=["high", "medium", "low"],
        graded_relevance={
            "q-id": {"high-id": 3.0, "medium-id": 2.0, "low-id": 1.0}
        },
        recall_at_k=(1, 3),
        ndcg_at_k=(1, 3),
    )

    metrics = evaluator()

    assert metrics["ndcg_at_1"] == pytest.approx(1.0)
    assert metrics["ndcg_at_3"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall_at_3"] == pytest.approx(1.0)


def test_partial_graded_ranking_matches_hand_computed_ndcg() -> None:
    model = ControlledCorpusModel(
        {"q": 0.0, "low": 0.1, "high": 0.2, "zero": 0.3}
    )
    evaluator = ManifoldGradedCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["low-id", "high-id", "zero-id"],
        corpus=["low", "high", "zero"],
        graded_relevance={
            "q-id": {"low-id": 1.0, "high-id": 3.0, "zero-id": 0.0}
        },
        recall_at_k=(1, 2),
        ndcg_at_k=(2,),
    )

    metrics = evaluator()

    actual_dcg = 1.0 + 7.0 / math.log2(3.0)
    ideal_dcg = 7.0 + 1.0 / math.log2(3.0)
    assert metrics["ndcg_at_2"] == pytest.approx(actual_dcg / ideal_dcg)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["recall_at_2"] == pytest.approx(1.0)


def test_zero_gain_top_k_has_zero_ndcg() -> None:
    model = ControlledCorpusModel(
        {"q": 0.0, "zero": 0.1, "positive": 0.2, "other": 0.3}
    )
    evaluator = ManifoldGradedCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["zero-id", "positive-id", "other-id"],
        corpus=["zero", "positive", "other"],
        graded_relevance={"q-id": {"zero-id": 0.0, "positive-id": 2.0}},
        recall_at_k=(1, 2),
        ndcg_at_k=(1, 2),
    )

    metrics = evaluator()

    assert metrics["ndcg_at_1"] == pytest.approx(0.0)
    assert metrics["ndcg_at_2"] > 0.0
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["recall_at_1"] == pytest.approx(0.0)
    assert metrics["recall_at_2"] == pytest.approx(1.0)


def test_ndcg_cutoff_larger_than_corpus_uses_full_corpus() -> None:
    model = ControlledCorpusModel(
        {"q": 0.0, "high": 0.1, "medium": 0.2, "low": 0.3}
    )
    kwargs = {
        "model": model,
        "query_ids": ["q-id"],
        "queries": ["q"],
        "corpus_ids": ["high-id", "medium-id", "low-id"],
        "corpus": ["high", "medium", "low"],
        "graded_relevance": {
            "q-id": {"high-id": 3.0, "medium-id": 2.0, "low-id": 1.0}
        },
        "recall_at_k": (1,),
        "ndcg_at_k": (3, 10),
    }

    metrics = ManifoldGradedCorpusRetrievalEvaluator(**kwargs)()

    assert metrics["ndcg_at_10"] == pytest.approx(metrics["ndcg_at_3"])
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)


def test_graded_evaluator_preserves_exact_corpus_index_tie_order() -> None:
    model = ControlledCorpusModel({"q": 0.0, "first": -1.0, "second": 1.0})
    evaluator = ManifoldGradedCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["first-id", "second-id"],
        corpus=["first", "second"],
        graded_relevance={"q-id": {"first-id": 1.0, "second-id": 3.0}},
        recall_at_k=(1, 2),
        ndcg_at_k=(1, 2),
        corpus_chunk_size=1,
    )

    metrics = evaluator()

    assert metrics["ndcg_at_1"] == pytest.approx(1.0 / 7.0)
    expected_at_2 = (1.0 + 7.0 / math.log2(3.0)) / (
        7.0 + 1.0 / math.log2(3.0)
    )
    assert metrics["ndcg_at_2"] == pytest.approx(expected_at_2)


def test_graded_evaluator_restores_model_training_mode() -> None:
    model = ControlledCorpusModel({"q": 0.0, "c0": 0.1, "c1": 0.2, "c2": 0.3})
    model.train()
    evaluator = ManifoldGradedCorpusRetrievalEvaluator(**_base_kwargs(model))

    evaluator()

    assert model.training


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("ndcg_at_k", (), "at least one cutoff"),
        ("ndcg_at_k", (0,), "positive integers"),
        ("ndcg_at_k", (True,), "positive integers"),
        ("ndcg_at_k", (1, 1), "unique"),
        ("graded_relevance", [], "must map query IDs"),
        (
            "graded_relevance",
            {"q-id": [("c0-id", 1.0)]},
            "must be a mapping",
        ),
        (
            "graded_relevance",
            {"q-id": {"c0-id": -1.0}},
            "finite and non-negative",
        ),
        (
            "graded_relevance",
            {"q-id": {"c0-id": float("nan")}},
            "finite and non-negative",
        ),
        (
            "graded_relevance",
            {"q-id": {"c0-id": True}},
            "real numbers",
        ),
        (
            "graded_relevance",
            {"q-id": {"c0-id": 0.0}},
            "at least one positive grade",
        ),
        (
            "graded_relevance",
            {"q-id": {"c0-id": 1.0, "missing": 0.0}},
            "unknown graded corpus IDs",
        ),
    ],
)
def test_graded_evaluator_rejects_invalid_inputs(
    field: str,
    value,
    match: str,
) -> None:
    model = ControlledCorpusModel({"q": 0.0, "c0": 0.1, "c1": 0.2, "c2": 0.3})
    kwargs = _base_kwargs(model)
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        ManifoldGradedCorpusRetrievalEvaluator(**kwargs)
