"""The real-stack backward gate must prove active encoder/projection gradients."""

from pathlib import Path
import runpy

import pytest
import torch
from torch import nn


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("inactive", ["encoder", "projection", None])
def test_product_backward_gate_rejects_inactive_trainable_parameters(inactive):
    namespace = runpy.run_path(
        str(ROOT / "tests/integration/test_real_stack_v10.py"),
    )
    model = nn.Module()
    model.encoder = nn.Linear(1, 1, bias=False)
    model.projection = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.encoder.weight.fill_(1.0)
        model.projection.weight.fill_(1.0)
    loss = sum(
        parameter.square().sum() * (0.0 if name == inactive else 1.0)
        for name, parameter in (
            ("encoder", model.encoder.weight), ("projection", model.projection.weight),
        )
    )
    if inactive is None:
        namespace["_assert_active_backward"](model, loss)
    else:
        with pytest.raises(AssertionError):
            namespace["_assert_active_backward"](model, loss)
