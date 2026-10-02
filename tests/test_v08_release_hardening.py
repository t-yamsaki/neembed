"""Regression checks retained from the v0.8 real-stack hardening."""

from pathlib import Path
import runpy

import pytest
import torch
from torch import nn


ROOT = Path(__file__).parents[1]


class _TinyProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(1.0)


def _load_backward_helper():
    namespace = runpy.run_path(
        str(ROOT / "tests" / "integration" / "test_real_stack_v08.py")
    )
    return namespace["_assert_finite_backward"]


def test_v08_backward_gate_rejects_connected_zero_projection_gradient() -> None:
    assert_finite_backward = _load_backward_helper()
    model = _TinyProjectionModel()
    zero_loss = (model.projection.weight * 0.0).sum()

    with pytest.raises(AssertionError, match="non-zero gradient"):
        assert_finite_backward(model, zero_loss)


def test_v08_backward_gate_accepts_active_finite_projection_gradient() -> None:
    assert_finite_backward = _load_backward_helper()
    model = _TinyProjectionModel()
    active_loss = model.projection.weight.square().sum()

    assert_finite_backward(model, active_loss)
