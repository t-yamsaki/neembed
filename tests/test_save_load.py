"""Tests for local save/load round trips."""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed.model import ManifoldSentenceTransformer


class FakeSentenceTransformer(nn.Module):
    """Tiny encoder with a Sentence Transformers-like save/load surface."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

        model_path = Path(model_name_or_path)
        state_path = model_path / "encoder.pt"
        if state_path.exists():
            self.load_state_dict(torch.load(state_path, weights_only=True))

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

    def save_pretrained(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "encoder.pt")


class DeviceReportingSentenceTransformer(FakeSentenceTransformer):
    """Fake a restored encoder that Sentence Transformers placed on CUDA."""

    @property
    def device(self) -> torch.device:
        return torch.device("cuda")


class RecordingLinear(nn.Linear):
    """Record device moves without requiring a CUDA-capable test host."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.requested_device: torch.device | None = None

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        if device is not None:
            self.requested_device = torch.device(device)
        return self


class RecordingManifold(nn.Module):
    """Record the device requested for the manifold without moving tensors."""

    def __init__(self) -> None:
        super().__init__()
        self.requested_device: torch.device | None = None

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args and isinstance(args[0], (str, torch.device)):
            device = args[0]
        if device is not None:
            self.requested_device = torch.device(device)
        return self


def test_save_pretrained_round_trip_preserves_config_and_embeddings(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold="poincare",
        embedding_dim=2,
        curvature=2.0,
    )
    before = model.encode(["Shiba Inu", "dog"], convert_to_tensor=True)
    save_path = tmp_path / "saved-model"

    model.save_pretrained(save_path)
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)
    after = loaded.encode(["Shiba Inu", "dog"], convert_to_tensor=True)

    config = json.loads((save_path / "neembed_config.json").read_text(encoding="utf-8"))
    assert config == {
        "embedding_dim": 2,
        "manifold": "poincare",
        "curvature": 2.0,
    }
    assert (save_path / "encoder" / "encoder.pt").exists()
    assert (save_path / "projection.pt").exists()
    assert loaded.embedding_dim == model.embedding_dim
    assert loaded.manifold_name == model.manifold_name
    assert loaded.curvature == model.curvature
    assert torch.allclose(before, after)


def test_lorentz_save_pretrained_round_trip_preserves_config_and_embeddings(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold="lorentz",
        embedding_dim=2,
        curvature=2.0,
    )
    before = model.encode(["Shiba Inu", "dog"], convert_to_tensor=True)
    save_path = tmp_path / "saved-lorentz-model"

    model.save_pretrained(save_path)
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)
    after = loaded.encode(["Shiba Inu", "dog"], convert_to_tensor=True)

    config = json.loads((save_path / "neembed_config.json").read_text(encoding="utf-8"))
    assert config == {
        "embedding_dim": 2,
        "manifold": "lorentz",
        "curvature": 2.0,
    }
    assert loaded.embedding_dim == 2
    assert loaded.manifold_name == "lorentz"
    assert loaded.curvature == 2.0
    assert before.shape == after.shape == (2, 3)
    assert torch.allclose(before, after)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_learned_curvature_round_trip_preserves_current_value_and_trainability(
    monkeypatch,
    tmp_path,
    manifold_name: str,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        curvature=1.0,
        learnable_curvature=True,
    )
    curvature_parameter = next(
        parameter for parameter in model.manifold.parameters() if parameter.requires_grad
    )
    with torch.no_grad():
        curvature_parameter.add_(0.1)
    expected_curvature = model.curvature
    before = model.encode(["Shiba Inu", "dog"], convert_to_tensor=True)
    save_path = tmp_path / f"saved-{manifold_name}-learnable"

    model.save_pretrained(save_path)
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)
    after = loaded.encode(["Shiba Inu", "dog"], convert_to_tensor=True)

    config = json.loads((save_path / "neembed_config.json").read_text(encoding="utf-8"))
    assert config["learnable_curvature"] is True
    assert config["curvature"] == pytest.approx(expected_curvature)
    assert loaded.learnable_curvature
    assert loaded.curvature == pytest.approx(expected_curvature, rel=1e-6, abs=1e-7)
    assert any(parameter.requires_grad for parameter in loaded.manifold.parameters())
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_save_pretrained_round_trip_preserves_disabled_projection(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=None)
    before = model.encode(["dog", "mammal"], convert_to_tensor=True)
    save_path = tmp_path / "saved-without-projection"

    model.save_pretrained(save_path)
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)
    after = loaded.encode(["dog", "mammal"], convert_to_tensor=True)

    config = json.loads((save_path / "neembed_config.json").read_text(encoding="utf-8"))
    assert config["embedding_dim"] is None
    assert isinstance(loaded.projection, nn.Identity)
    assert loaded.embedding_dim == model.embedding_dim
    assert torch.allclose(before, after)


def test_from_pretrained_aligns_projection_and_manifold_with_encoder_device(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)
    save_path = tmp_path / "saved-model"
    model.save_pretrained(save_path)

    monkeypatch.setattr(
        model_module,
        "SentenceTransformer",
        DeviceReportingSentenceTransformer,
    )
    monkeypatch.setattr(model_module.nn, "Linear", RecordingLinear)
    monkeypatch.setattr(model_module, "get_manifold", lambda *args: RecordingManifold())

    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)

    assert isinstance(loaded.projection, RecordingLinear)
    assert loaded.projection.requested_device == torch.device("cuda")
    assert isinstance(loaded.manifold, RecordingManifold)
    assert loaded.manifold.requested_device == torch.device("cuda")
