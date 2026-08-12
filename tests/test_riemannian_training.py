"""Tests for caller-supplied Riemannian optimization in the trainer."""

import math

import geoopt
import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import ManifoldPrototypes, ManifoldSentenceTransformer
from neembed.losses import ManifoldMultipleNegativesRankingLoss
from neembed.trainer import ManifoldTrainer


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used without external model downloads."""

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


class PrototypeAugmentedLoss(ManifoldMultipleNegativesRankingLoss):
    """Exercise gradients through both ordinary and manifold parameters."""

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        prototypes: ManifoldPrototypes,
    ) -> None:
        super().__init__(model=model, temperature=0.5)
        self.prototypes = prototypes

    def forward(self, anchors, positives):
        ranking_loss = super().forward(anchors, positives)
        prototype_distances = self.prototypes(self.model(anchors))
        return ranking_loss + 0.1 * prototype_distances.square().mean()


def _make_mixed_trainer(monkeypatch, manifold_name: str):
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        curvature=1.0,
    )
    prototypes = ManifoldPrototypes(model, num_prototypes=3, init_std=0.05)
    loss = PrototypeAugmentedLoss(model, prototypes)
    trainable_parameters = [
        parameter for parameter in loss.parameters() if parameter.requires_grad
    ]
    optimizer = geoopt.optim.RiemannianAdam(
        trainable_parameters,
        lr=1e-2,
        stabilize=1,
    )
    trainer = ManifoldTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        verbose=False,
    )
    return trainer, prototypes, loss, optimizer


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_supplied_riemannian_optimizer_covers_and_updates_mixed_parameters(
    monkeypatch,
    manifold_name,
) -> None:
    torch.manual_seed(7)
    trainer, prototypes, loss, optimizer = _make_mixed_trainer(
        monkeypatch,
        manifold_name,
    )

    expected_parameters = {
        id(parameter) for parameter in loss.parameters() if parameter.requires_grad
    }
    optimized_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    encoder_before = trainer.model.encoder.linear.weight.detach().clone()
    projection_before = trainer.model.projection.weight.detach().clone()
    prototypes_before = prototypes.prototypes.detach().clone()

    history = trainer.fit(
        [(["dog", "cat"], ["mammal", "animal"])],
        epochs=1,
    )

    assert trainer.optimizer is optimizer
    assert isinstance(trainer.optimizer, geoopt.optim.RiemannianAdam)
    assert optimized_parameters == expected_parameters
    assert len(history) == 1
    assert math.isfinite(history[0])

    assert not torch.allclose(
        encoder_before,
        trainer.model.encoder.linear.weight.detach(),
    )
    assert not torch.allclose(
        projection_before,
        trainer.model.projection.weight.detach(),
    )
    assert not torch.equal(prototypes_before, prototypes.prototypes.detach())

    participating_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    assert all(parameter.grad is not None for parameter in participating_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in participating_parameters
        if parameter.grad is not None
    )
    assert trainer.model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-5,
        rtol=1e-5,
    )
