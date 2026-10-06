"""Encoder loading options and offline restoration without model downloads."""

import json
from pathlib import Path
import socket

import pytest
from sentence_transformers import SentenceTransformer, models
import torch
from torch import nn

import neembed.model as model_module
from neembed import ManifoldSentenceTransformer


class RecordingEncoder(nn.Module):
    """Record loading arguments while supporting a local save/load round trip."""

    def __init__(self, model_name_or_path, **kwargs):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.loading_kwargs = kwargs
        self.linear = nn.Linear(3, 4, bias=False)
        state_path = Path(model_name_or_path) / "encoder.pt"
        if state_path.exists():
            self.load_state_dict(torch.load(state_path, weights_only=True))
        if "device" in kwargs:
            self.to(kwargs["device"])

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, sentences):
        return {"input_features": torch.tensor([
            [len(text) / 100.0, 0.1, -0.1] for text in sentences
        ])}

    def forward(self, features):
        return {"sentence_embedding": self.linear(features["input_features"])}

    def save_pretrained(self, output_path):
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "encoder.pt")


def test_default_loading_preserves_original_encoder_call(monkeypatch, tmp_path):
    monkeypatch.setattr(model_module, "SentenceTransformer", RecordingEncoder)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)
    assert model.encoder.loading_kwargs == {}
    assert model.encoder.model_name_or_path == "fake-model"

    model.save_pretrained(tmp_path)
    restored = ManifoldSentenceTransformer.from_pretrained(tmp_path)
    assert restored.encoder.loading_kwargs == {}
    assert restored.encoder.model_name_or_path == str(tmp_path / "encoder")
    assert torch.allclose(
        model.encode(["dog"], convert_to_tensor=True),
        restored.encode(["dog"], convert_to_tensor=True),
    )


@pytest.mark.parametrize("options", [
    {"revision": "fixed-commit"},
    {"local_files_only": True},
    {"cache_folder": "/a/custom/cache"},
    {"device": "cpu"},
    {
        "revision": "fixed-commit",
        "local_files_only": True,
        "cache_folder": "/a/custom/cache",
        "device": "cpu",
    },
])
def test_loading_options_are_forwarded_without_enabling_remote_code(
    monkeypatch, tmp_path, options,
):
    monkeypatch.setattr(model_module, "SentenceTransformer", RecordingEncoder)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2, **options)
    assert model.encoder.loading_kwargs == options
    assert "trust_remote_code" not in model.encoder.loading_kwargs
    assert model.projection.weight.device == model.encoder.device
    assert next(model.manifold.parameters()).device == model.encoder.device

    model.save_pretrained(tmp_path)
    restored = ManifoldSentenceTransformer.from_pretrained(tmp_path, **options)
    assert restored.encoder.loading_kwargs == options
    assert restored.encoder.model_name_or_path == str(tmp_path / "encoder")
    assert restored.projection.weight.device == restored.encoder.device
    assert torch.allclose(
        model.encode(["dog", "cat"], convert_to_tensor=True),
        restored.encode(["dog", "cat"], convert_to_tensor=True),
    )
    # Persist provenance only, not a dictionary of load-time paths/options.
    config = json.loads((tmp_path / "neembed_config.json").read_text())
    assert {key: config[key] for key in ("embedding_dim", "manifold", "curvature")} == {
        "embedding_dim": 2,
        "manifold": "poincare",
        "curvature": 1.0,
    }
    assert config["base_model"] == {"model_id": "fake-model", "revision": options.get("revision")}
    assert not ({"device", "cache_folder", "local_files_only", "trust_remote_code"} & config.keys())
    assert "/a/custom/cache" not in json.dumps(config)


@pytest.mark.parametrize("geometry", [
    {"manifold": "poincare", "embedding_dim": 2},
    {"manifold": "lorentz", "embedding_dim": 2},
    {"product_config": [
        {"manifold": "lorentz", "intrinsic_dim": 2},
        {"manifold": "euclidean", "intrinsic_dim": 2},
    ]},
])
def test_real_saved_encoder_loads_without_network(monkeypatch, tmp_path, geometry):
    """Use actual Sentence Transformers modules built entirely from local data."""
    encoder_path = tmp_path / "source-encoder"
    encoder = SentenceTransformer(
        modules=[models.BoW(["dog", "cat", "mammal", "animal"])], device="cpu",
    )
    encoder.save_pretrained(str(encoder_path))
    network_attempts = []

    def reject_network(*args, **kwargs):
        network_attempts.append(args)
        raise AssertionError("local loading attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    cache_folder = str(tmp_path / "empty-cache")
    model = ManifoldSentenceTransformer(
        str(encoder_path), local_files_only=True, cache_folder=cache_folder,
        device="cpu", **geometry,
    )
    before = model.encode(["dog mammal", "cat animal"], convert_to_tensor=True)
    save_path = tmp_path / "saved-model"
    model.save_pretrained(save_path)
    restored = ManifoldSentenceTransformer.from_pretrained(
        save_path, local_files_only=True, cache_folder=cache_folder, device="cpu",
    )
    after = restored.encode(["dog mammal", "cat animal"], convert_to_tensor=True)
    assert not network_attempts
    assert restored.encoder.device.type == "cpu"
    assert torch.isfinite(after).all()
    assert torch.allclose(before, after)
    assert torch.allclose(model.distance(before[0], before[1]),
                          restored.distance(after[0], after[1]))
