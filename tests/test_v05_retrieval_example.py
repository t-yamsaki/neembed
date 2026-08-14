"""Regression tests for the v0.5 retrieval workflow example."""

import math

import pytest
import torch
from torch import nn

import neembed.model as model_module
from examples.v05_retrieval_workflow import _train_prototypes, run_example
from neembed import ManifoldPrototypes, ManifoldSentenceTransformer


class FakeSentenceTransformer(nn.Module):
    """Small deterministic encoder used without external model downloads."""

    modes_seen: list[bool] = []

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
                    ]
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
        type(self).modes_seen.append(self.training)
        return {"sentence_embedding": self.linear(features["input_features"])}


def _run(monkeypatch):
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    return run_example(
        "fake-model",
        epochs=1,
        prototype_epochs=2,
        seed=11,
        embedding_dim=3,
        learning_rate=1e-2,
        prototype_learning_rate=2e-2,
    )


def test_v05_example_exercises_public_retrieval_workflow(monkeypatch) -> None:
    results = _run(monkeypatch)

    assert results["manifold"] == "poincare"
    assert math.isfinite(results["final_training_loss"])
    assert math.isfinite(results["embedding_distance"])
    assert results["embedding_distance"] >= 0.0

    assert results["initial_hard_negative_loss"] > results["initial_pair_loss"]
    assert results["hard_negative_loss_delta"] > 0.0

    retrieval = results["retrieval"]
    assert retrieval["retrieval_accuracy"] == pytest.approx(retrieval["recall_at_1"])
    for key in ("recall_at_1", "recall_at_2", "recall_at_3", "mrr"):
        assert 0.0 <= retrieval[key] <= 1.0

    ranked_results = results["ranked_results"]
    assert len(ranked_results) == 3
    assert [item["distance"] for item in ranked_results] == sorted(
        item["distance"] for item in ranked_results
    )
    assert all(math.isfinite(item["distance"]) for item in ranked_results)
    assert all({"candidate", "index", "distance"} == set(item) for item in ranked_results)

    prototype = results["prototype"]
    assert 0.0 <= prototype["assignment_accuracy"] <= 1.0
    assert math.isfinite(prototype["mean_assigned_prototype_distance"])
    assert prototype["mean_assigned_prototype_distance"] >= 0.0
    assert math.isfinite(results["prototype_training_loss"])


def test_v05_example_is_deterministic_with_fixed_seed(monkeypatch) -> None:
    first = _run(monkeypatch)
    second = _run(monkeypatch)

    assert [item["candidate"] for item in first["ranked_results"]] == [
        item["candidate"] for item in second["ranked_results"]
    ]
    assert [item["distance"] for item in first["ranked_results"]] == pytest.approx(
        [item["distance"] for item in second["ranked_results"]]
    )
    assert first["retrieval"] == pytest.approx(second["retrieval"])
    assert first["prototype"] == pytest.approx(second["prototype"])
    assert first["final_training_loss"] == pytest.approx(second["final_training_loss"])
    assert first["prototype_training_loss"] == pytest.approx(
        second["prototype_training_loss"]
    )


def test_prototype_stage_keeps_frozen_encoder_in_eval_mode(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    torch.manual_seed(7)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold="poincare",
        embedding_dim=3,
    )
    prototypes = ManifoldPrototypes(model, num_prototypes=4, init_std=0.05)
    model.train()
    FakeSentenceTransformer.modes_seen.clear()

    loss = _train_prototypes(
        model,
        prototypes,
        epochs=2,
        learning_rate=2e-2,
    )

    assert math.isfinite(loss)
    assert FakeSentenceTransformer.modes_seen
    assert not any(FakeSentenceTransformer.modes_seen)
    assert model.training
