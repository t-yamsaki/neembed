"""Tests for exact corpus retrieval evaluation with explicit relevance."""

import math

import pytest
import torch
from torch import nn

import neembed
import neembed.model as model_module
from neembed.evaluator import ManifoldCorpusRetrievalEvaluator
from neembed.model import ManifoldSentenceTransformer


class ControlledCorpusModel(nn.Module):
    """Minimal model with deterministic one-dimensional encoded points."""

    def __init__(self, points: dict[str, float]) -> None:
        super().__init__()
        self.points = points
        self.distance_calls = 0

    def encode(self, sentences: list[str]):
        self.eval()
        with torch.inference_mode():
            return torch.tensor(
                [[self.points[sentence]] for sentence in sentences],
                dtype=torch.float32,
            )

    def distance(self, a, b) -> torch.Tensor:
        self.distance_calls += 1
        return torch.abs(a - b).sum(dim=-1)


class FakeSentenceTransformer(nn.Module):
    """Small encoder used to exercise both supported manifold backends."""

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


def _base_kwargs(model) -> dict:
    return {
        "model": model,
        "query_ids": ["q0"],
        "queries": ["query"],
        "corpus_ids": ["c0", "c1"],
        "corpus": ["candidate", "other"],
        "relevance": {"q0": ["c0"]},
        "recall_at_k": (1, 2),
    }


def test_corpus_evaluator_is_exported_from_package() -> None:
    assert neembed.ManifoldCorpusRetrievalEvaluator is ManifoldCorpusRetrievalEvaluator


def test_single_positive_metrics_match_hand_computed_ranks() -> None:
    model = ControlledCorpusModel(
        {
            "q0": 0.0,
            "q1": 10.0,
            "c0": 1.0,
            "c1": 9.0,
            "c2": 5.0,
        }
    )
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=["query-0", "query-1"],
        queries=["q0", "q1"],
        corpus_ids=["corpus-0", "corpus-1", "corpus-2"],
        corpus=["c0", "c1", "c2"],
        relevance={
            "query-0": ["corpus-2"],  # rank 2
            "query-1": ["corpus-0"],  # rank 3
        },
        recall_at_k=(1, 2, 3),
        query_chunk_size=1,
        corpus_chunk_size=2,
    )

    metrics = evaluator()

    assert metrics == {
        "mrr": pytest.approx((1.0 / 2.0 + 1.0 / 3.0) / 2.0),
        "recall_at_1": pytest.approx(0.0),
        "recall_at_2": pytest.approx(0.5),
        "recall_at_3": pytest.approx(1.0),
    }


def test_multi_positive_recall_uses_fraction_of_relevant_items() -> None:
    model = ControlledCorpusModel(
        {"q": 0.0, "c0": 0.0, "c1": 1.0, "c2": 2.0, "c3": 3.0}
    )
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["c0-id", "c1-id", "c2-id", "c3-id"],
        corpus=["c0", "c1", "c2", "c3"],
        relevance={"q-id": ["c1-id", "c3-id"]},
        recall_at_k=(1, 2, 3, 4, 10),
    )

    metrics = evaluator()

    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["recall_at_1"] == pytest.approx(0.0)
    assert metrics["recall_at_2"] == pytest.approx(0.5)
    assert metrics["recall_at_3"] == pytest.approx(0.5)
    assert metrics["recall_at_4"] == pytest.approx(1.0)
    assert metrics["recall_at_10"] == pytest.approx(1.0)


def test_corpus_evaluator_preserves_exact_search_tie_order() -> None:
    model = ControlledCorpusModel({"q": 0.0, "left": -1.0, "right": 1.0, "far": 3.0})
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["left-id", "right-id", "far-id"],
        corpus=["left", "right", "far"],
        relevance={"q-id": ["right-id"]},
        recall_at_k=(1, 2),
        corpus_chunk_size=1,
    )

    metrics = evaluator()

    # left and right tie. exact_corpus_search keeps corpus index order, so the
    # relevant right item is second.
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["recall_at_1"] == pytest.approx(0.0)
    assert metrics["recall_at_2"] == pytest.approx(1.0)


def test_corpus_evaluator_restores_model_training_mode() -> None:
    model = ControlledCorpusModel({"q": 0.0, "c": 1.0})
    model.train()
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q-id"],
        queries=["q"],
        corpus_ids=["c-id"],
        corpus=["c"],
        relevance={"q-id": ["c-id"]},
    )

    evaluator()

    assert model.training
    assert model.distance_calls > 0


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_corpus_evaluator_is_finite_on_supported_manifolds(
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
    model.train()
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q0", "q1"],
        queries=["a", "long query"],
        corpus_ids=["c0", "c1", "c2", "c3"],
        corpus=["aa", "bbbb", "cccccc", "dddddddd"],
        relevance={"q0": ["c0", "c1"], "q1": ["c2"]},
        recall_at_k=(1, 2, 10),
        query_chunk_size=1,
        corpus_chunk_size=2,
    )

    metrics = evaluator()

    assert set(metrics) == {"mrr", "recall_at_1", "recall_at_2", "recall_at_10"}
    assert all(math.isfinite(value) for value in metrics.values())
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    assert metrics["recall_at_10"] == pytest.approx(1.0)
    assert model.training


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("query_ids", "q0", "query_ids must be a sequence"),
        ("queries", "query", "queries must be a sequence"),
        ("corpus_ids", "c0", "corpus_ids must be a sequence"),
        ("corpus", "candidate", "corpus must be a sequence"),
        ("query_ids", [], "at least one query"),
        ("corpus_ids", [], "at least one corpus item"),
        ("query_ids", ["q0", "q0"], "query_ids must be unique"),
        ("corpus_ids", ["c0", "c0"], "corpus_ids must be unique"),
        ("query_ids", [""], "query_ids must be non-empty strings"),
        ("corpus_ids", [""], "corpus_ids must be non-empty strings"),
        ("queries", [1], "queries must contain only strings"),
        ("corpus", [1, "other"], "corpus must contain only strings"),
        ("recall_at_k", (), "at least one cutoff"),
        ("recall_at_k", (0,), "positive integers"),
        ("recall_at_k", (True,), "positive integers"),
        ("recall_at_k", (1, 1), "unique"),
        ("query_chunk_size", 0, "positive integer"),
        ("corpus_chunk_size", False, "positive integer"),
    ],
)
def test_corpus_evaluator_rejects_malformed_core_inputs(
    field: str,
    value,
    match: str,
) -> None:
    model = ControlledCorpusModel({"query": 0.0, "candidate": 1.0, "other": 2.0})
    kwargs = _base_kwargs(model)
    kwargs[field] = value

    if field == "query_ids" and value == []:
        kwargs["queries"] = []
        kwargs["relevance"] = {}
    elif field == "corpus_ids" and value == []:
        kwargs["corpus"] = []
    elif field == "query_ids" and value == ["q0", "q0"]:
        kwargs["queries"] = ["query", "query"]
    elif field == "corpus_ids" and value == ["c0", "c0"]:
        kwargs["corpus"] = ["candidate", "other"]

    with pytest.raises(ValueError, match=match):
        ManifoldCorpusRetrievalEvaluator(**kwargs)


def test_corpus_evaluator_rejects_length_mismatches() -> None:
    model = ControlledCorpusModel({"query": 0.0, "candidate": 1.0, "other": 2.0})

    kwargs = _base_kwargs(model)
    kwargs["query_ids"] = ["q0", "q1"]
    with pytest.raises(ValueError, match="query_ids and queries"):
        ManifoldCorpusRetrievalEvaluator(**kwargs)

    kwargs = _base_kwargs(model)
    kwargs["corpus_ids"] = ["c0"]
    with pytest.raises(ValueError, match="corpus_ids and corpus"):
        ManifoldCorpusRetrievalEvaluator(**kwargs)


def test_corpus_evaluator_rejects_invalid_relevance_mapping() -> None:
    model = ControlledCorpusModel({"query": 0.0, "candidate": 1.0, "other": 2.0})

    cases = [
        ({}, "missing relevance"),
        ({"q0": ["c0"], "unknown": ["c1"]}, "unknown relevance query IDs"),
        ({"q0": []}, "must not be empty"),
        ({"q0": "c0"}, "must be a sequence"),
        ({"q0": ["c0", "c0"]}, "unique corpus IDs"),
        ({"q0": ["missing"]}, "unknown relevant corpus IDs"),
        ({"q0": [""]}, "non-empty strings"),
    ]
    for relevance, match in cases:
        kwargs = _base_kwargs(model)
        kwargs["relevance"] = relevance
        with pytest.raises(ValueError, match=match):
            ManifoldCorpusRetrievalEvaluator(**kwargs)

    kwargs = _base_kwargs(model)
    kwargs["relevance"] = [("q0", ["c0"])]
    with pytest.raises(ValueError, match="relevance must be a mapping"):
        ManifoldCorpusRetrievalEvaluator(**kwargs)
