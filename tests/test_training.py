"""Tests for the minimal Euclidean training loop."""

import math

import neembed
import torch
from torch import nn

import neembed.model as model_module
from neembed.losses import ManifoldMultipleNegativesRankingLoss
from neembed.model import ManifoldSentenceTransformer
from neembed.trainer import ManifoldTrainer


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used to test training without model downloads."""

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


def _make_trainer(monkeypatch) -> ManifoldTrainer:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)
    loss = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.5)
    return ManifoldTrainer(model=model, loss=loss, learning_rate=1e-2)


def test_trainer_is_exported_from_package() -> None:
    assert neembed.ManifoldTrainer is ManifoldTrainer


def test_trainer_uses_adamw_by_default(monkeypatch) -> None:
    trainer = _make_trainer(monkeypatch)

    assert isinstance(trainer.optimizer, torch.optim.AdamW)


def test_fit_returns_finite_epoch_losses_and_updates_parameters(
    monkeypatch,
    capsys,
) -> None:
    torch.manual_seed(0)
    trainer = _make_trainer(monkeypatch)
    batches = [
        (["dog", "cat"], ["mammal", "animal"]),
        (["wolf", "fox"], ["canine", "mammal"]),
    ]
    before = trainer.model.projection.weight.detach().clone()

    history = trainer.fit(batches, epochs=2)

    after = trainer.model.projection.weight.detach()
    output = capsys.readouterr().out

    assert len(history) == 2
    assert all(math.isfinite(value) for value in history)
    assert not torch.allclose(before, after)
    assert "Epoch 1/2" in output
    assert "Epoch 2/2" in output
