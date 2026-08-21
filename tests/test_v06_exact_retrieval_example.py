"""Regression tests for the v0.6 exact retrieval and mining example."""

import math

import pytest
import torch
from torch import nn

import neembed.model as model_module
import examples.v06_exact_retrieval_workflow as v06_example
from examples.v06_exact_retrieval_workflow import RELEVANCE, run_example


class FakeSentenceTransformer(nn.Module):
    """Small deterministic encoder used without model or network downloads."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.linear = nn.Linear(4, 6, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.20, -0.10, 0.05, 0.15],
                        [0.00, 0.25, -0.15, 0.10],
                        [-0.20, 0.05, 0.30, -0.10],
                        [0.15, 0.10, 0.00, -0.25],
                        [0.05, -0.20, 0.10, 0.30],
                        [-0.10, 0.15, 0.20, 0.05],
                    ],
                    dtype=torch.float32,
                )
            )

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 6

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        vowels = set("aeiouAEIOU")
        rows = []
        for sentence in sentences:
            code_sum = sum(ord(character) for character in sentence)
            vowel_count = sum(character in vowels for character in sentence)
            first_code = ord(sentence[0]) if sentence else 0
            rows.append(
                [
                    len(sentence) / 20.0,
                    (code_sum % 31) / 31.0,
                    vowel_count / 10.0,
                    (first_code % 17) / 17.0,
                ]
            )
        return {"input_features": torch.tensor(rows, dtype=torch.float32)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def _run(monkeypatch):
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    return run_example(
        "fake-model",
        epochs=1,
        seed=13,
        embedding_dim=3,
        learning_rate=1e-2,
        top_k=3,
        query_chunk_size=2,
        corpus_chunk_size=2,
    )


def test_v06_example_composes_public_retrieval_mining_and_training(monkeypatch) -> None:
    results = _run(monkeypatch)

    assert results["manifold"] == "poincare"
    assert math.isfinite(results["final_training_loss"])

    for stage_name in ("before", "after"):
        stage = results[stage_name]
        search_results = stage["search_results"]
        assert len(search_results) == 3
        assert all(len(ranking) == 3 for ranking in search_results)
        for ranking in search_results:
            assert [item["distance"] for item in ranking] == sorted(
                item["distance"] for item in ranking
            )
            assert all(math.isfinite(item["distance"]) for item in ranking)
            assert all({"candidate", "index", "distance"} == set(item) for item in ranking)

        retrieval = stage["retrieval"]
        assert set(retrieval) == {"mrr", "recall_at_1", "recall_at_3"}
        assert all(0.0 <= value <= 1.0 for value in retrieval.values())

    mined = results["mined_negatives"]
    assert len(mined) == 3
    for query_id, item in zip(v06_example.QUERY_IDS, mined, strict=True):
        assert {"corpus_id", "candidate", "index", "distance"} == set(item)
        assert item["corpus_id"] not in RELEVANCE[query_id]
        assert math.isfinite(item["distance"])
        assert item["distance"] >= 0.0


def test_v06_example_is_deterministic_with_fixed_seed(monkeypatch) -> None:
    first = _run(monkeypatch)
    second = _run(monkeypatch)

    assert first["final_training_loss"] == pytest.approx(second["final_training_loss"])
    assert first["before"]["retrieval"] == pytest.approx(second["before"]["retrieval"])
    assert first["after"]["retrieval"] == pytest.approx(second["after"]["retrieval"])

    for first_stage, second_stage in (
        (first["before"], second["before"]),
        (first["after"], second["after"]),
    ):
        for first_ranking, second_ranking in zip(
            first_stage["search_results"],
            second_stage["search_results"],
            strict=True,
        ):
            assert [item["index"] for item in first_ranking] == [
                item["index"] for item in second_ranking
            ]
            assert [item["distance"] for item in first_ranking] == pytest.approx(
                [item["distance"] for item in second_ranking]
            )

    assert [item["corpus_id"] for item in first["mined_negatives"]] == [
        item["corpus_id"] for item in second["mined_negatives"]
    ]
    assert [item["distance"] for item in first["mined_negatives"]] == pytest.approx(
        [item["distance"] for item in second["mined_negatives"]]
    )


def test_v06_example_documents_repository_root_command() -> None:
    assert v06_example.__doc__ is not None
    assert "python examples/v06_exact_retrieval_workflow.py" in v06_example.__doc__
