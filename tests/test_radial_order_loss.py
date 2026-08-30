"""Tests for radial parent-child hierarchy supervision."""

import math

import pytest
import torch
from torch import nn

import neembed
from neembed import ManifoldRadialOrderLoss
from neembed.manifolds import get_manifold


class _ToyRadialModel(nn.Module):
    """Tiny trainable manifold model with controllable radial positions."""

    def __init__(
        self,
        manifold_name: str,
        *,
        parent_offset: float,
        child_offset: float,
    ) -> None:
        super().__init__()
        self.manifold_name = manifold_name
        self.manifold = get_manifold(manifold_name, curvature=1.0)
        dtype = torch.float64 if manifold_name == "lorentz" else torch.float32
        self.parent_tangent = nn.Parameter(
            torch.tensor([parent_offset, 0.0], dtype=dtype)
        )
        self.child_tangent = nn.Parameter(
            torch.tensor([child_offset, 0.0], dtype=dtype)
        )

    def forward(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        tangents = []
        for text in texts:
            if text == "parent":
                tangent = self.parent_tangent
            elif text == "child":
                tangent = self.child_tangent
            else:
                raise ValueError(f"unknown toy text: {text}")
            if self.manifold_name == "lorentz":
                tangent = torch.cat((tangent.new_zeros(1), tangent), dim=0)
            tangents.append(tangent)
        return self.manifold.expmap0(torch.stack(tangents))


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_radial_order_loss_distinguishes_satisfied_and_violated_pairs(
    manifold_name: str,
) -> None:
    satisfied_model = _ToyRadialModel(
        manifold_name,
        parent_offset=0.05,
        child_offset=1.0,
    )
    violated_model = _ToyRadialModel(
        manifold_name,
        parent_offset=1.0,
        child_offset=0.05,
    )

    satisfied = ManifoldRadialOrderLoss(satisfied_model, margin=0.1)(
        ["parent"],
        ["child"],
    )
    violated = ManifoldRadialOrderLoss(violated_model, margin=0.1)(
        ["parent"],
        ["child"],
    )

    assert torch.isfinite(satisfied)
    assert torch.isfinite(violated)
    assert float(satisfied.detach()) == pytest.approx(0.0, abs=1e-7)
    assert float(violated.detach()) > 0.0


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_radial_order_loss_has_finite_gradients_on_both_manifolds(
    manifold_name: str,
) -> None:
    model = _ToyRadialModel(
        manifold_name,
        parent_offset=0.8,
        child_offset=0.1,
    )
    loss = ManifoldRadialOrderLoss(model, margin=0.2)

    value = loss(["parent"], ["child"])
    value.backward()

    assert value.ndim == 0
    assert math.isfinite(float(value.detach()))
    for parameter in (model.parent_tangent, model.child_tangent):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_radial_coordinate_matches_geoopt_distance_from_origin(
    manifold_name: str,
) -> None:
    model = _ToyRadialModel(
        manifold_name,
        parent_offset=0.2,
        child_offset=0.7,
    )
    embeddings = model(["parent", "child"])
    radii = model.manifold.dist0(embeddings)
    origin = model.manifold.origin(
        *embeddings.shape,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )

    assert torch.allclose(
        radii,
        model.manifold.dist(origin, embeddings),
        rtol=1e-6,
        atol=1e-7,
    )
    if manifold_name == "poincare":
        assert torch.count_nonzero(origin) == 0
    else:
        assert torch.allclose(origin[..., 0], torch.sqrt(model.manifold.k))
        assert torch.count_nonzero(origin[..., 1:]) == 0


def test_radial_order_loss_validates_margin_and_aligned_inputs() -> None:
    model = _ToyRadialModel(
        "poincare",
        parent_offset=0.2,
        child_offset=0.7,
    )

    ManifoldRadialOrderLoss(model, margin=0.0)
    for margin in (-0.1, math.inf, math.nan):
        with pytest.raises(ValueError, match="non-negative and finite"):
            ManifoldRadialOrderLoss(model, margin=margin)

    loss = ManifoldRadialOrderLoss(model)
    with pytest.raises(ValueError, match="sequence of texts"):
        loss("parent", ["child"])
    with pytest.raises(ValueError, match="sequence of texts"):
        loss(["parent"], "child")
    with pytest.raises(ValueError, match="must not be empty"):
        loss([], [])
    with pytest.raises(ValueError, match="same length"):
        loss(["parent"], ["child", "child"])


def test_radial_order_loss_is_public() -> None:
    assert neembed.ManifoldRadialOrderLoss is ManifoldRadialOrderLoss
    assert "ManifoldRadialOrderLoss" in neembed.__all__
