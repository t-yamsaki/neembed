"""Tests for optional validation evaluation in ManifoldTrainer."""

import pytest
import torch
from torch import nn

from neembed.trainer import ManifoldTrainer


class TinyModel(nn.Module):
    """Minimal trainable model for trainer tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class TinyLoss:
    """Simple differentiable loss tied to TinyModel."""

    def __init__(self, model: TinyModel) -> None:
        self.model = model

    def __call__(self, anchors, positives) -> torch.Tensor:
        del anchors, positives
        return self.model.weight.square()


class RecordingEvaluator:
    """Record evaluation context and deliberately switch to eval mode."""

    def __init__(self, model: TinyModel) -> None:
        self.model = model
        self.calls = 0
        self.grad_enabled: list[bool] = []
        self.training_mode: list[bool] = []
        self.parameters_unchanged: list[bool] = []

    def __call__(self) -> dict[str, float]:
        self.calls += 1
        self.grad_enabled.append(torch.is_grad_enabled())
        self.training_mode.append(self.model.training)

        before = self.model.weight.detach().clone()
        self.model.eval()
        score = float(self.model.weight.detach())
        after = self.model.weight.detach().clone()
        self.parameters_unchanged.append(torch.equal(before, after))
        return {"score": score}


class RaisingEvaluator:
    """Switch modes and fail so trainer cleanup can be tested."""

    def __init__(self, model: TinyModel) -> None:
        self.model = model

    def __call__(self) -> dict[str, float]:
        self.model.eval()
        raise RuntimeError("validation failed")


def _trainer() -> tuple[TinyModel, ManifoldTrainer]:
    model = TinyModel()
    trainer = ManifoldTrainer(
        model,
        TinyLoss(model),
        learning_rate=0.1,
        weight_decay=0.0,
        verbose=False,
    )
    return model, trainer


def test_fit_without_evaluator_preserves_float_history() -> None:
    _, trainer = _trainer()
    train_batches = [(["anchor"], ["positive"])]

    history = trainer.fit(train_batches, epochs=2)

    assert len(history) == 2
    assert all(isinstance(loss, float) for loss in history)


def test_fit_runs_evaluator_after_each_epoch_and_returns_validation_history() -> None:
    model, trainer = _trainer()
    evaluator = RecordingEvaluator(model)
    train_batches = [(["anchor"], ["positive"])]

    history = trainer.fit(train_batches, epochs=2, evaluator=evaluator)

    assert len(history) == 2
    assert all(set(epoch) == {"train_loss", "validation"} for epoch in history)
    assert all(isinstance(epoch["train_loss"], float) for epoch in history)
    assert all(set(epoch["validation"]) == {"score"} for epoch in history)
    assert evaluator.calls == 2


def test_validation_runs_without_gradients_or_parameter_updates() -> None:
    model, trainer = _trainer()
    evaluator = RecordingEvaluator(model)

    trainer.fit([(["anchor"], ["positive"])], evaluator=evaluator)

    assert evaluator.grad_enabled == [False]
    assert evaluator.parameters_unchanged == [True]


def test_trainer_restores_training_mode_after_each_validation() -> None:
    model, trainer = _trainer()
    evaluator = RecordingEvaluator(model)

    trainer.fit([(["anchor"], ["positive"])], epochs=2, evaluator=evaluator)

    assert evaluator.training_mode == [True, True]
    assert model.training


def test_trainer_restores_training_mode_when_validation_raises() -> None:
    model, trainer = _trainer()

    with pytest.raises(RuntimeError, match="validation failed"):
        trainer.fit(
            [(["anchor"], ["positive"])],
            evaluator=RaisingEvaluator(model),
        )

    assert model.training
