"""Integration test for hierarchy loss with caller-supplied Riemannian optimizer."""

import math

import geoopt
import torch
from torch import nn

import neembed.model as model_module
from neembed import (
    ManifoldPrototypeHierarchyLoss,
    ManifoldPrototypes,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)


class FakeSentenceTransformer(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
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


def test_hierarchy_loss_trains_through_existing_trainer(monkeypatch) -> None:
    torch.manual_seed(13)
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold="poincare",
        embedding_dim=2,
        curvature=1.0,
    )
    prototypes = ManifoldPrototypes(model, num_prototypes=3, init_std=0.05)
    loss = ManifoldPrototypeHierarchyLoss(
        model,
        prototypes,
        prototype_ids=("animal", "dog", "cat"),
        parent_relations=(("dog", "animal"), ("cat", "animal")),
    )
    optimizer = geoopt.optim.RiemannianAdam(
        [parameter for parameter in loss.parameters() if parameter.requires_grad],
        lr=1e-2,
        stabilize=1,
    )
    trainer = ManifoldTrainer(model, loss, optimizer=optimizer, verbose=False)

    history = trainer.fit(
        [(["Shiba Inu", "Siamese cat"], ["dog", "cat"])],
        epochs=1,
    )

    assert len(history) == 1
    assert math.isfinite(history[0])
    assert model.manifold.check_point_on_manifold(
        prototypes.prototypes,
        atol=1e-5,
        rtol=1e-5,
    )
