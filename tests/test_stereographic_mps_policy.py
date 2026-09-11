"""Regression coverage for stereographic float64 device selection on Apple MPS."""

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed.model import ManifoldSentenceTransformer, _select_geometry_device


def test_stereographic_float64_geometry_falls_back_to_cpu_on_mps() -> None:
    assert _select_geometry_device("sphere_projection", "mps") == torch.device("cpu")
    assert _select_geometry_device("stereographic", "mps") == torch.device("cpu")


def test_non_stereographic_device_behavior_is_unchanged_on_mps() -> None:
    assert _select_geometry_device("poincare", "mps") == torch.device("mps")
    assert _select_geometry_device("euclidean", "mps") == torch.device("mps")
    assert _select_geometry_device("lorentz", "mps") == torch.device("mps")


def test_stereographic_geometry_keeps_supported_devices() -> None:
    assert _select_geometry_device("sphere_projection", "cpu") == torch.device("cpu")
    assert _select_geometry_device("stereographic", "cuda") == torch.device("cuda")


class _FakeMpsEncoder(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    @property
    def device(self) -> torch.device:
        return torch.device("mps")

    def get_embedding_dimension(self) -> int:
        return 4


class _RecordingManifold(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("k", torch.tensor(0.5, dtype=torch.float64))
        self.requested_device: torch.device | None = None
        self.requested_dtype: torch.dtype | None = None

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.requested_device = torch.device(device)
        self.requested_dtype = kwargs.get("dtype")
        return self


class _SimulatedTransferModule(nn.Module):
    """Pretend an ordinary child module followed ``.to('mps')`` without MPS CI."""

    def __init__(self) -> None:
        super().__init__()
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def _apply(self, fn, recurse: bool = True):
        self._device = torch.device("mps")
        return self


def _make_transfer_model(manifold_name: str) -> tuple[ManifoldSentenceTransformer, _RecordingManifold]:
    model = ManifoldSentenceTransformer.__new__(ManifoldSentenceTransformer)
    nn.Module.__init__(model)
    model.encoder = _SimulatedTransferModule()
    model.projection = _SimulatedTransferModule()
    model.manifold_name = manifold_name
    manifold = _RecordingManifold()
    model.manifold = manifold
    return model, manifold


def test_constructor_does_not_move_float64_stereographic_state_to_mps(
    monkeypatch,
) -> None:
    manifold = _RecordingManifold()
    monkeypatch.setattr(model_module, "SentenceTransformer", _FakeMpsEncoder)
    monkeypatch.setattr(model_module, "get_manifold", lambda *args, **kwargs: manifold)

    original_linear_to = nn.Linear.to

    def keep_fake_projection_on_cpu(self, device=None, *args, **kwargs):
        if device is not None and torch.device(device).type == "mps":
            return self
        return original_linear_to(self, device, *args, **kwargs)

    monkeypatch.setattr(nn.Linear, "to", keep_fake_projection_on_cpu)

    ManifoldSentenceTransformer(
        "fake-model",
        manifold="sphere_projection",
        embedding_dim=2,
        sectional_curvature=0.5,
    )

    assert manifold.requested_device == torch.device("cpu")


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_model_to_mps_preserves_cpu_float64_geometry_fallback(
    manifold_name: str,
) -> None:
    model, manifold = _make_transfer_model(manifold_name)

    model.to("mps")

    assert model.encoder.device == torch.device("mps")
    assert manifold.requested_device == torch.device("cpu")
    assert manifold.requested_dtype == torch.float64
    assert model.manifold is manifold


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_parent_module_to_mps_preserves_cpu_float64_geometry_fallback(
    manifold_name: str,
) -> None:
    model, manifold = _make_transfer_model(manifold_name)
    parent = nn.Module()
    parent.add_module("model", model)

    parent.to("mps")

    assert model.encoder.device == torch.device("mps")
    assert manifold.requested_device == torch.device("cpu")
    assert manifold.requested_dtype == torch.float64
    assert model.manifold is manifold
