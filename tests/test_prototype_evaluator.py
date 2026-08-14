"""Tests for nearest-prototype assignment evaluation."""

import math

import pytest
import torch
from torch import nn

import neembed
import neembed.model as model_module
from neembed import (
    ManifoldPrototypeAssignmentEvaluator,
    ManifoldPrototypes,
    ManifoldSentenceTransformer,
)


class FakeSentenceTransformer(nn.Module):
    """Small deterministic encoder used without external model downloads."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.linear = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.10, 0.00, 0.00],
                        [0.00, 0.10, 0.00],
                        [0.00, 0.00, 0.10],
                        [0.05, 0.02, -0.03],
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
            [float(len(sentence)), 1.0, -1.0]
            for sentence in sentences
        ]
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def _configured_model_and_prototypes(
    monkeypatch,
    *,
    manifold: str = "poincare",
    learnable_curvature: bool = False,
):
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold,
        embedding_dim=None,
        curvature=1.0,
        learnable_curvature=learnable_curvature,
    )
    prototypes = ManifoldPrototypes(model, num_prototypes=2, init_std=0.01)
    sentences = ("a", "bbbb")
    embeddings = model.encode(sentences, convert_to_tensor=True)
    with torch.no_grad():
        prototypes.prototypes.copy_(embeddings)
    return model, prototypes, sentences


def test_prototype_assignment_evaluator_is_exported() -> None:
    assert (
        neembed.ManifoldPrototypeAssignmentEvaluator
        is ManifoldPrototypeAssignmentEvaluator
    )


def test_prototype_assignment_evaluator_reports_perfect_assignment(monkeypatch) -> None:
    model, prototypes, sentences = _configured_model_and_prototypes(monkeypatch)
    evaluator = ManifoldPrototypeAssignmentEvaluator(
        model=model,
        prototypes=prototypes,
        prototype_ids=("short", "long"),
        sentences=sentences,
        expected_prototype_ids=("short", "long"),
    )

    metrics = evaluator()

    assert metrics["assignment_accuracy"] == pytest.approx(1.0)
    assert math.isfinite(metrics["mean_assigned_prototype_distance"])
    assert metrics["mean_assigned_prototype_distance"] >= 0.0


def test_prototype_assignment_evaluator_reports_incorrect_assignment(monkeypatch) -> None:
    model, prototypes, sentences = _configured_model_and_prototypes(monkeypatch)
    evaluator = ManifoldPrototypeAssignmentEvaluator(
        model=model,
        prototypes=prototypes,
        prototype_ids=("short", "long"),
        sentences=sentences,
        expected_prototype_ids=("long", "short"),
    )

    metrics = evaluator()

    assert metrics["assignment_accuracy"] == pytest.approx(0.0)
    assert math.isfinite(metrics["mean_assigned_prototype_distance"])


def test_prototype_ids_remain_explicit_index_aligned_metadata(monkeypatch) -> None:
    model, prototypes, sentences = _configured_model_and_prototypes(monkeypatch)
    evaluator = ManifoldPrototypeAssignmentEvaluator(
        model=model,
        prototypes=prototypes,
        prototype_ids=("prototype-b", "prototype-a"),
        sentences=sentences,
        expected_prototype_ids=("prototype-b", "prototype-a"),
    )

    metrics = evaluator()

    assert metrics["assignment_accuracy"] == pytest.approx(1.0)
    assert not hasattr(prototypes, "prototype_ids")


@pytest.mark.parametrize(
    "prototype_ids,expected_message",
    [
        (("only-one",), "one ID per prototype"),
        (("same", "same"), "must be unique"),
    ],
)
def test_prototype_assignment_evaluator_validates_prototype_id_alignment(
    monkeypatch,
    prototype_ids,
    expected_message: str,
) -> None:
    model, prototypes, sentences = _configured_model_and_prototypes(monkeypatch)

    with pytest.raises(ValueError, match=expected_message):
        ManifoldPrototypeAssignmentEvaluator(
            model=model,
            prototypes=prototypes,
            prototype_ids=prototype_ids,
            sentences=sentences,
            expected_prototype_ids=("short", "long"),
        )


def test_prototype_assignment_evaluator_rejects_unknown_expected_id(monkeypatch) -> None:
    model, prototypes, sentences = _configured_model_and_prototypes(monkeypatch)

    with pytest.raises(ValueError, match="unknown expected prototype IDs"):
        ManifoldPrototypeAssignmentEvaluator(
            model=model,
            prototypes=prototypes,
            prototype_ids=("short", "long"),
            sentences=sentences,
            expected_prototype_ids=("short", "unknown"),
        )


def test_prototype_assignment_evaluator_rejects_mismatched_or_empty_data(
    monkeypatch,
) -> None:
    model, prototypes, sentences = _configured_model_and_prototypes(monkeypatch)

    with pytest.raises(ValueError, match="same number of items"):
        ManifoldPrototypeAssignmentEvaluator(
            model=model,
            prototypes=prototypes,
            prototype_ids=("short", "long"),
            sentences=sentences,
            expected_prototype_ids=("short",),
        )

    with pytest.raises(ValueError, match="at least one sentence"):
        ManifoldPrototypeAssignmentEvaluator(
            model=model,
            prototypes=prototypes,
            prototype_ids=("short", "long"),
            sentences=(),
            expected_prototype_ids=(),
        )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("prototype_ids", "short", "prototype_ids must be a sequence"),
        ("sentences", "a", "sentences must be a sequence"),
        (
            "expected_prototype_ids",
            "short",
            "expected_prototype_ids must be a sequence",
        ),
    ],
)
def test_prototype_assignment_evaluator_rejects_bare_string_sequences(
    monkeypatch,
    field: str,
    value: str,
    match: str,
) -> None:
    model, prototypes, sentences = _configured_model_and_prototypes(monkeypatch)
    arguments = {
        "model": model,
        "prototypes": prototypes,
        "prototype_ids": ("short", "long"),
        "sentences": sentences,
        "expected_prototype_ids": ("short", "long"),
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=match):
        ManifoldPrototypeAssignmentEvaluator(**arguments)


@pytest.mark.parametrize("manifold", ["poincare", "lorentz"])
@pytest.mark.parametrize("learnable_curvature", [False, True])
def test_prototype_assignment_evaluation_is_no_grad_non_mutating_and_geometry_safe(
    monkeypatch,
    manifold: str,
    learnable_curvature: bool,
) -> None:
    model, prototypes, sentences = _configured_model_and_prototypes(
        monkeypatch,
        manifold=manifold,
        learnable_curvature=learnable_curvature,
    )
    model.train()
    prototypes.eval()
    curvature_before = model.curvature
    prototype_points_before = prototypes.prototypes.detach().clone()
    evaluator = ManifoldPrototypeAssignmentEvaluator(
        model=model,
        prototypes=prototypes,
        prototype_ids=("short", "long"),
        sentences=sentences,
        expected_prototype_ids=("short", "long"),
    )

    metrics = evaluator()

    assert metrics["assignment_accuracy"] == pytest.approx(1.0)
    assert math.isfinite(metrics["mean_assigned_prototype_distance"])
    assert model.training
    assert not prototypes.training
    assert model.curvature == pytest.approx(curvature_before)
    assert torch.equal(prototypes.prototypes.detach(), prototype_points_before)
    assert prototypes.prototypes.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())
    assert model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-5,
        rtol=1e-5,
    )
