"""Tests for deterministic offline hard-negative mining."""

import math

import pytest
import torch
from torch import nn

import neembed
import neembed.model as model_module
from neembed.mining import mine_hard_negatives
from neembed.model import ManifoldSentenceTransformer


class ControlledMiningModel(nn.Module):
    """Minimal deterministic encoder/distance model for mining tests."""

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


def test_miner_is_exported_from_package() -> None:
    assert neembed.mine_hard_negatives is mine_hard_negatives


def test_miner_selects_nearest_valid_negative_and_excludes_positive_and_self() -> None:
    model = ControlledMiningModel(
        {
            "query": 0.0,
            "self": 0.0,
            "positive": 0.1,
            "hard": 0.2,
            "far": 3.0,
        }
    )

    mined = mine_hard_negatives(
        model,
        ["query"],
        ["self", "positive", "hard", "far"],
        query_ids=["q-id"],
        corpus_ids=["q-id", "p-id", "hard-id", "far-id"],
        positive_corpus_ids={"q-id": ["p-id"]},
        query_chunk_size=1,
        corpus_chunk_size=2,
    )

    assert mined == [
        [
            {
                "corpus_id": "hard-id",
                "candidate": "hard",
                "index": 2,
                "distance": pytest.approx(0.2),
            }
        ]
    ]


def test_miner_supports_multiple_negatives_and_additional_exclusions() -> None:
    model = ControlledMiningModel(
        {
            "query": 0.0,
            "positive": 0.1,
            "excluded": 0.2,
            "n2": 0.3,
            "n3": 0.4,
            "far": 2.0,
        }
    )

    mined = mine_hard_negatives(
        model,
        ["query"],
        ["positive", "excluded", "n2", "n3", "far"],
        query_ids=["q"],
        corpus_ids=["p", "excluded", "n2", "n3", "far"],
        positive_corpus_ids={"q": ["p"]},
        excluded_corpus_ids={"q": ["excluded"]},
        num_negatives=2,
        corpus_chunk_size=2,
    )

    assert [result["corpus_id"] for result in mined[0]] == ["n2", "n3"]
    assert [result["index"] for result in mined[0]] == [2, 3]
    assert [result["distance"] for result in mined[0]] == pytest.approx([0.3, 0.4])


def test_miner_preserves_corpus_order_for_equal_distance_ties() -> None:
    model = ControlledMiningModel(
        {"query": 0.0, "positive": 0.0, "left": -1.0, "right": 1.0}
    )

    mined = mine_hard_negatives(
        model,
        ["query"],
        ["positive", "left", "right"],
        query_ids=["q"],
        corpus_ids=["p", "left", "right"],
        positive_corpus_ids={"q": ["p"]},
        num_negatives=2,
        corpus_chunk_size=1,
    )

    assert [result["corpus_id"] for result in mined[0]] == ["left", "right"]


def test_miner_is_deterministic_and_restores_training_mode() -> None:
    model = ControlledMiningModel(
        {"query": 0.0, "positive": 0.5, "hard": 1.0, "far": 2.0}
    )
    model.train()
    kwargs = {
        "query_ids": ["q"],
        "corpus_ids": ["p", "hard", "far"],
        "positive_corpus_ids": {"q": ["p"]},
        "query_chunk_size": 1,
        "corpus_chunk_size": 1,
    }

    first = mine_hard_negatives(
        model,
        ["query"],
        ["positive", "hard", "far"],
        **kwargs,
    )
    second = mine_hard_negatives(
        model,
        ["query"],
        ["positive", "hard", "far"],
        **kwargs,
    )

    assert first == second
    assert model.training
    assert model.distance_calls > 0


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_miner_is_finite_on_supported_manifolds(monkeypatch, manifold: str) -> None:
    torch.manual_seed(0)
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold,
        embedding_dim=2,
    )
    model.train()

    mined = mine_hard_negatives(
        model,
        ["query"],
        ["positive", "candidate one", "candidate two"],
        query_ids=["q"],
        corpus_ids=["p", "c1", "c2"],
        positive_corpus_ids={"q": ["p"]},
        num_negatives=2,
        query_chunk_size=1,
        corpus_chunk_size=2,
    )

    assert len(mined) == 1
    assert len(mined[0]) == 2
    assert {result["corpus_id"] for result in mined[0]} == {"c1", "c2"}
    assert all(math.isfinite(float(result["distance"])) for result in mined[0])
    assert model.training


def test_miner_rejects_impossible_request_after_exclusions() -> None:
    model = ControlledMiningModel({"query": 0.0, "self": 0.0, "positive": 1.0})

    with pytest.raises(ValueError, match="only 0 valid negatives"):
        mine_hard_negatives(
            model,
            ["query"],
            ["self", "positive"],
            query_ids=["q"],
            corpus_ids=["q", "p"],
            positive_corpus_ids={"q": ["p"]},
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("num_negatives", 0, "positive integer"),
        ("num_negatives", True, "positive integer"),
        ("query_ids", ["q", "q"], "unique IDs"),
        ("corpus_ids", ["p", "p"], "unique IDs"),
        ("positive_corpus_ids", {}, "missing positive_corpus_ids"),
        ("positive_corpus_ids", {"q": ["missing"]}, "unknown positive_corpus_ids corpus IDs"),
    ],
)
def test_miner_rejects_invalid_configuration(field: str, value, match: str) -> None:
    model = ControlledMiningModel({"query": 0.0, "positive": 1.0, "other": 2.0})
    kwargs = {
        "query_ids": ["q"],
        "corpus_ids": ["p", "other"],
        "positive_corpus_ids": {"q": ["p"]},
    }
    kwargs[field] = value
    if field == "query_ids" and value == ["q", "q"]:
        queries = ["query", "query"]
    else:
        queries = ["query"]
    if field == "corpus_ids" and value == ["p", "p"]:
        corpus = ["positive", "other"]
    else:
        corpus = ["positive", "other"]

    with pytest.raises(ValueError, match=match):
        mine_hard_negatives(model, queries, corpus, **kwargs)
