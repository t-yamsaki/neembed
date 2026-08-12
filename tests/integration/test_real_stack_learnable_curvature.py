"""Opt-in real-stack acceptance for learnable curvature."""

import math
import os

import pytest
import torch

from neembed import (
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)


pytestmark = [
    pytest.mark.real_stack,
    pytest.mark.skipif(
        os.environ.get("NEEMBED_REAL_STACK") != "1",
        reason="set NEEMBED_REAL_STACK=1 to run real dependency tests",
    ),
]

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TRAIN_BATCH = (
    ["Shiba Inu", "Siamese cat", "sparrow", "salmon"],
    ["dog", "cat", "bird", "fish"],
)
GEOMETRY_TERMS = ["Shiba Inu", "dog", "mammal", "animal"]


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_real_learnable_curvature_is_finite_trainable_and_persistent(
    manifold_name: str,
    tmp_path,
) -> None:
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        MODEL_NAME,
        manifold=manifold_name,
        embedding_dim=32,
        curvature=1.0,
        learnable_curvature=True,
    )
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)
    trainer = ManifoldTrainer(
        model=model,
        loss=loss_fn,
        learning_rate=1e-3,
        weight_decay=0.0,
        verbose=False,
    )
    curvature_parameter = next(
        parameter for parameter in model.manifold.parameters() if parameter.requires_grad
    )
    initial_curvature = model.curvature

    trainer.optimizer.zero_grad()
    embeddings = model(GEOMETRY_TERMS)
    geometry_loss = embeddings.square().mean()
    geometry_loss.backward()

    assert curvature_parameter.grad is not None
    assert torch.isfinite(curvature_parameter.grad).all()
    assert torch.count_nonzero(curvature_parameter.grad) > 0

    trainer.optimizer.step()

    assert math.isfinite(model.curvature)
    assert model.curvature > 0.0
    assert model.curvature != pytest.approx(initial_curvature, abs=1e-10)
    assert torch.isfinite(model(GEOMETRY_TERMS)).all()
    assert torch.isfinite(loss_fn(*TRAIN_BATCH))

    save_path = tmp_path / manifold_name
    expected_curvature = model.curvature
    model.save_pretrained(save_path)
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)

    assert loaded.learnable_curvature
    assert loaded.curvature == pytest.approx(expected_curvature, rel=1e-6, abs=1e-7)
    assert any(parameter.requires_grad for parameter in loaded.manifold.parameters())
    assert torch.isfinite(loaded.encode(GEOMETRY_TERMS, convert_to_tensor=True)).all()
