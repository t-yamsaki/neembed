"""Tests for the minimal in-memory geodesic reranking helper."""

import math

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed.model import ManifoldSentenceTransformer


class FakeSentenceTransformer(nn.Module):
    """Small deterministic encoder used without model downloads."""

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


def _model(monkeypatch, *, manifold: str = "poincare") -> ManifoldSentenceTransformer:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    return ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold,
        embedding_dim=2,
    )


def test_rank_orders_known_distances_and_preserves_ties(monkeypatch) -> None:
    model = _model(monkeypatch)
    candidate_distances = {"far": 3.0, "tie-a": 1.0, "tie-b": 1.0, "near": 0.5}
    grad_enabled_during_distance: list[bool] = []

    def controlled_encode(sentences, *, convert_to_tensor=False):
        if isinstance(sentences, str):
            embeddings = torch.tensor([0.0])
        else:
            embeddings = torch.tensor(
                [[candidate_distances[sentence]] for sentence in sentences]
            )
        return embeddings if convert_to_tensor else embeddings.numpy()

    def controlled_distance(a, b):
        del a
        grad_enabled_during_distance.append(torch.is_grad_enabled())
        return b.squeeze(-1)

    monkeypatch.setattr(model, "encode", controlled_encode)
    monkeypatch.setattr(model, "distance", controlled_distance)

    results = model.rank("query", ["far", "tie-a", "tie-b", "near"])

    assert [result["candidate"] for result in results] == [
        "near",
        "tie-a",
        "tie-b",
        "far",
    ]
    assert [result["index"] for result in results] == [3, 1, 2, 0]
    assert [result["distance"] for result in results] == pytest.approx(
        [0.5, 1.0, 1.0, 3.0]
    )
    assert grad_enabled_during_distance == [False]


def test_rank_supports_top_k_and_full_candidate_list(monkeypatch) -> None:
    model = _model(monkeypatch)

    def controlled_encode(sentences, *, convert_to_tensor=False):
        if isinstance(sentences, str):
            embeddings = torch.tensor([0.0])
        else:
            values = {"a": 2.0, "b": 0.5, "c": 1.0}
            embeddings = torch.tensor([[values[sentence]] for sentence in sentences])
        return embeddings if convert_to_tensor else embeddings.numpy()

    monkeypatch.setattr(model, "encode", controlled_encode)
    monkeypatch.setattr(model, "distance", lambda a, b: b.squeeze(-1))

    assert [item["candidate"] for item in model.rank("q", ["a", "b", "c"], top_k=1)] == [
        "b"
    ]
    assert len(model.rank("q", ["a", "b", "c"], top_k=3)) == 3
    assert len(model.rank("q", ["a", "b", "c"])) == 3


def test_rank_supports_one_candidate(monkeypatch) -> None:
    model = _model(monkeypatch)
    result = model.rank("dog", ["mammal"])

    assert len(result) == 1
    assert result[0]["candidate"] == "mammal"
    assert result[0]["index"] == 0
    assert math.isfinite(result[0]["distance"])
    assert result[0]["distance"] >= 0.0


@pytest.mark.parametrize("top_k", [0, -1, 3, 1.5, True])
def test_rank_rejects_invalid_top_k(monkeypatch, top_k) -> None:
    model = _model(monkeypatch)

    with pytest.raises(ValueError, match="between 1 and the candidate count"):
        model.rank("query", ["a", "b"], top_k=top_k)


def test_rank_rejects_empty_candidate_list(monkeypatch) -> None:
    model = _model(monkeypatch)

    with pytest.raises(ValueError, match="at least one item"):
        model.rank("query", [])


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
def test_rank_uses_real_geodesic_distance_for_supported_manifolds(
    monkeypatch,
    manifold: str,
) -> None:
    torch.manual_seed(0)
    model = _model(monkeypatch, manifold=manifold)
    query = "dog"
    candidates = ["mammal", "cat", "very long animal description"]

    query_embedding = model.encode(query, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
    expected = model.distance(query_embedding.unsqueeze(0), candidate_embeddings)

    results = model.rank(query, candidates)

    assert len(results) == len(candidates)
    assert [item["distance"] for item in results] == sorted(
        item["distance"] for item in results
    )
    assert all(math.isfinite(item["distance"]) for item in results)
    for item in results:
        assert item["distance"] == pytest.approx(float(expected[item["index"]]))


def test_rank_does_not_change_encode_or_distance_contract(monkeypatch) -> None:
    model = _model(monkeypatch)
    before = model.encode(["dog", "cat"], convert_to_tensor=True)
    before_distance = model.distance(before[0], before[1])

    model.rank("dog", ["cat", "mammal"], top_k=1)

    after = model.encode(["dog", "cat"], convert_to_tensor=True)
    after_distance = model.distance(after[0], after[1])
    assert before.shape == after.shape == (2, 2)
    assert not before.requires_grad
    assert not after.requires_grad
    assert torch.isfinite(before_distance)
    assert torch.isfinite(after_distance)
