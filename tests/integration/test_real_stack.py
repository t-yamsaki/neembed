"""Opt-in acceptance gates against the real third-party stack."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import sys

import geoopt
import numpy as np
import pytest
import sentence_transformers
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
TRAIN_BATCHES = [
    (
        ["Shiba Inu", "Siamese cat", "sparrow", "salmon"],
        ["dog", "cat", "bird", "fish"],
    ),
    (
        ["dog", "sparrow", "salmon", "oak"],
        ["mammal", "vertebrate", "aquatic animal", "plant"],
    ),
]
EVAL_ANCHORS = ["Shiba Inu", "salmon", "sparrow", "oak"]
EVAL_POSITIVES = ["dog", "fish", "bird", "plant"]
GEOMETRY_TERMS = ["Shiba Inu", "dog", "mammal", "animal"]


def set_seed(seed: int) -> None:
    """Seed every random source used by this CPU-only acceptance test."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(seed: int) -> ManifoldSentenceTransformer:
    """Construct the real model with the fixed pre-#8 gate configuration."""
    set_seed(seed)
    return ManifoldSentenceTransformer(
        MODEL_NAME,
        manifold="poincare",
        embedding_dim=32,
        curvature=1.0,
    )


def evaluate_pairs(
    model: ManifoldSentenceTransformer,
    loss_fn: ManifoldMultipleNegativesRankingLoss,
) -> tuple[float, float]:
    """Return eval-mode loss and diagonal parent retrieval accuracy."""
    model.eval()
    with torch.inference_mode():
        loss = float(loss_fn(EVAL_ANCHORS, EVAL_POSITIVES))
        anchor_embeddings = model(EVAL_ANCHORS)
        positive_embeddings = model(EVAL_POSITIVES)
        distances = model.manifold.dist(
            anchor_embeddings[:, None, :],
            positive_embeddings[None, :, :],
        )
        targets = torch.arange(len(EVAL_ANCHORS), device=distances.device)
        accuracy = float((distances.argmin(dim=1) == targets).float().mean())
    return loss, accuracy


def test_gate_1_uses_real_dependencies_from_the_virtual_environment() -> None:
    venv_root = Path(sys.prefix).resolve()
    for module in (torch, sentence_transformers, geoopt):
        module_path = Path(module.__file__).resolve()
        assert module_path.is_relative_to(venv_root), module_path
        print(f"dependency={module.__name__} path={module_path}")


def test_gate_3_real_model_geometry() -> None:
    model = make_model(0)
    embeddings = model.encode(GEOMETRY_TERMS, convert_to_tensor=True)
    norms = embeddings.norm(dim=-1)
    distance_ab = model.distance(embeddings[0], embeddings[1])
    distance_ba = model.distance(embeddings[1], embeddings[0])
    distance_aa = model.distance(embeddings[0], embeddings[0])

    assert embeddings.shape == (4, 32)
    assert torch.isfinite(embeddings).all()
    assert float(norms.max()) < 0.99
    assert torch.isfinite(distance_ab)
    assert float(distance_ab) >= 0.0
    assert float((distance_ab - distance_ba).abs()) < 1e-5
    assert float(distance_aa) < 1e-5
    print(
        "geometry "
        f"max_norm={float(norms.max()):.9f} "
        f"distance_ab={float(distance_ab):.9f} "
        f"symmetry_error={float((distance_ab - distance_ba).abs()):.9f} "
        f"self_distance={float(distance_aa):.9f}"
    )


def test_gates_4_and_5_real_gradients_and_adamw_update() -> None:
    model = make_model(0)
    loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)
    trainer = ManifoldTrainer(
        model=model,
        loss=loss_fn,
        learning_rate=2e-5,
        verbose=False,
    )
    projection_before = model.projection.weight.detach().clone()
    encoder_parameter = next(
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    )
    encoder_before = encoder_parameter.detach().clone()

    trainer.optimizer.zero_grad()
    batch_loss = loss_fn(*TRAIN_BATCHES[0])
    batch_loss.backward()

    projection_grad = model.projection.weight.grad
    encoder_grads = [
        parameter.grad
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    ]
    observed_grads = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]

    assert projection_grad is not None
    assert torch.isfinite(projection_grad).all()
    assert torch.count_nonzero(projection_grad) > 0
    assert encoder_grads
    assert all(torch.isfinite(gradient).all() for gradient in encoder_grads)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in encoder_grads)
    assert observed_grads
    assert all(torch.isfinite(gradient).all() for gradient in observed_grads)

    trainer.optimizer.step()

    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert not torch.equal(projection_before, model.projection.weight.detach())
    assert not torch.equal(encoder_before, encoder_parameter.detach())
    print(
        "optimization "
        f"projection_grad_norm={float(projection_grad.norm()):.9f} "
        f"encoder_grad_tensors={len(encoder_grads)} "
        "projection_changed=True encoder_changed=True optimizer=AdamW"
    )


def test_gates_6_and_7_learning_is_stable_across_three_seeds() -> None:
    results: list[tuple[int, float, float, float, float, float]] = []
    improved_seeds = 0

    for seed in (0, 1, 2):
        model = make_model(seed)
        loss_fn = ManifoldMultipleNegativesRankingLoss(model=model, temperature=0.1)
        trainer = ManifoldTrainer(
            model=model,
            loss=loss_fn,
            learning_rate=2e-5,
            verbose=False,
        )
        initial_loss, initial_accuracy = evaluate_pairs(model, loss_fn)
        history: list[float] = []
        epoch_max_norms: list[float] = []
        for _ in range(20):
            history.extend(trainer.fit(TRAIN_BATCHES, epochs=1))
            epoch_embeddings = model.encode(
                GEOMETRY_TERMS,
                convert_to_tensor=True,
            )
            assert torch.isfinite(epoch_embeddings).all()
            epoch_max_norm = float(epoch_embeddings.norm(dim=-1).max())
            assert epoch_max_norm < 0.99
            epoch_max_norms.append(epoch_max_norm)

        final_loss, final_accuracy = evaluate_pairs(model, loss_fn)
        max_norm = max(epoch_max_norms)

        assert len(history) == 20
        assert all(math.isfinite(loss) for loss in history)
        assert math.isfinite(initial_loss)
        assert math.isfinite(final_loss)

        improved = final_loss < initial_loss and final_accuracy >= initial_accuracy
        improved_seeds += int(improved)
        results.append(
            (
                seed,
                initial_loss,
                final_loss,
                initial_accuracy,
                final_accuracy,
                max_norm,
            )
        )
        print(
            f"seed={seed} initial_loss={initial_loss:.9f} "
            f"final_loss={final_loss:.9f} "
            f"initial_accuracy={initial_accuracy:.3f} "
            f"final_accuracy={final_accuracy:.3f} "
            f"max_norm={max_norm:.9f} improved={improved}"
        )

    assert results[0][2] < results[0][1], results
    assert results[0][4] >= results[0][3], results
    assert improved_seeds >= 2, results
    print(f"improved_seeds={improved_seeds}/3")
