"""Tests for supervised radial hierarchy depth structure."""

import math

import pytest
import torch
from torch import nn

import neembed
from neembed import ManifoldDepthLoss
from neembed.manifolds import get_manifold


class _ToyDepthModel(nn.Module):
    """Tiny trainable manifold model with controllable tangent vectors."""

    def __init__(
        self,
        manifold_name: str,
        labels: list[str],
        tangent_vectors: list[list[float]],
    ) -> None:
        super().__init__()
        self.manifold_name = manifold_name
        self.manifold = get_manifold(manifold_name, curvature=1.0)
        self.index = {label: position for position, label in enumerate(labels)}
        dtype = torch.float64 if manifold_name == "lorentz" else torch.float32
        self.tangents = nn.Parameter(torch.tensor(tangent_vectors, dtype=dtype))

    def forward(self, texts: list[str] | tuple[str, ...]) -> torch.Tensor:
        spatial = torch.stack([self.tangents[self.index[text]] for text in texts])
        if self.manifold_name == "lorentz":
            spatial = torch.cat((spatial.new_zeros((len(texts), 1)), spatial), dim=-1)
        return self.manifold.expmap0(spatial)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_depth_loss_matches_linear_radial_targets_and_depth_order(
    manifold_name: str,
) -> None:
    labels = ["root", "depth-1", "depth-2"]
    aligned_model = _ToyDepthModel(
        manifold_name,
        labels,
        [[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]],
    )
    with torch.no_grad():
        aligned_radii = aligned_model.manifold.dist0(aligned_model(labels))
        radial_scale = float(aligned_radii[1])
        assert float(aligned_radii[0]) == pytest.approx(0.0, abs=1e-7)
        assert float(aligned_radii[2]) == pytest.approx(
            2.0 * radial_scale,
            rel=1e-5,
            abs=1e-7,
        )

    aligned_loss = ManifoldDepthLoss(
        aligned_model,
        radial_scale=radial_scale,
    )(labels, [0, 1, 2])

    reversed_model = _ToyDepthModel(
        manifold_name,
        labels,
        [[0.0, 0.0], [0.4, 0.0], [0.2, 0.0]],
    )
    reversed_loss = ManifoldDepthLoss(
        reversed_model,
        radial_scale=radial_scale,
    )(labels, [0, 1, 2])

    assert torch.isfinite(aligned_loss)
    assert torch.isfinite(reversed_loss)
    assert float(aligned_loss.detach()) == pytest.approx(0.0, abs=1e-7)
    assert float(reversed_loss.detach()) > float(aligned_loss.detach())


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_depth_loss_root_and_repeated_depth_behavior(manifold_name: str) -> None:
    labels = ["root", "peer-a", "peer-b"]
    model = _ToyDepthModel(
        manifold_name,
        labels,
        [[0.0, 0.0], [0.25, 0.0], [0.0, 0.25]],
    )
    with torch.no_grad():
        radii = model.manifold.dist0(model(labels))
        radial_scale = float(radii[1])
        assert float(radii[0]) == pytest.approx(0.0, abs=1e-7)
        assert float(radii[2]) == pytest.approx(radial_scale, rel=1e-5, abs=1e-7)

    value = ManifoldDepthLoss(model, radial_scale=radial_scale)(
        labels,
        [0, 1, 1],
    )

    assert float(value.detach()) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_depth_loss_has_finite_forward_and_backward(manifold_name: str) -> None:
    model = _ToyDepthModel(
        manifold_name,
        ["node-a", "node-b"],
        [[0.10, 0.05], [0.15, -0.05]],
    )
    loss = ManifoldDepthLoss(model, radial_scale=1.0)

    value = loss(["node-a", "node-b"], [2, 3])
    value.backward()

    assert value.ndim == 0
    assert math.isfinite(float(value.detach()))
    assert model.tangents.grad is not None
    assert torch.isfinite(model.tangents.grad).all()
    assert torch.count_nonzero(model.tangents.grad) > 0


def test_depth_loss_validates_scale_and_aligned_depth_targets() -> None:
    model = _ToyDepthModel("poincare", ["node"], [[0.2, 0.0]])

    ManifoldDepthLoss(model, radial_scale=0.5)
    for radial_scale in (0.0, -0.1, math.inf, math.nan, True):
        with pytest.raises(ValueError, match="positive and finite"):
            ManifoldDepthLoss(model, radial_scale=radial_scale)

    loss = ManifoldDepthLoss(model)
    with pytest.raises(ValueError, match="sequence of strings"):
        loss("node", [1])
    with pytest.raises(ValueError, match="must not be empty"):
        loss([], [])
    with pytest.raises(ValueError, match="one value per text"):
        loss(["node"], [0, 1])

    for depths in ([-1], [1.5], [True], torch.tensor([False])):
        with pytest.raises(ValueError, match="non-negative integers"):
            loss(["node"], depths)

    for depths in ([math.inf], [math.nan]):
        with pytest.raises(ValueError, match="finite"):
            loss(["node"], depths)

    value = loss(["node"], torch.tensor([1.0]))
    assert torch.isfinite(value)


def test_depth_normalization_preserves_tensor_dtype_and_device() -> None:
    depths = torch.tensor([0.0, 1.0], dtype=torch.float32)

    normalized = ManifoldDepthLoss._normalize_depths(depths, batch_size=2)

    assert normalized.dtype == depths.dtype
    assert normalized.device == depths.device


def test_depth_loss_is_public_and_keeps_depth_metadata_external() -> None:
    model = _ToyDepthModel("poincare", ["node"], [[0.2, 0.0]])
    loss = ManifoldDepthLoss(model)

    assert neembed.ManifoldDepthLoss is ManifoldDepthLoss
    assert "ManifoldDepthLoss" in neembed.__all__
    assert all("depth" not in name for name in loss.state_dict())
