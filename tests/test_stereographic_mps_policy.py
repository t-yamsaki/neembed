"""Regression coverage for stereographic float64 device selection on Apple MPS."""

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

    def to(self, device, *args, **kwargs):
        self.requested_device = torch.device(device)
        return self


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
