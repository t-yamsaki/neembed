"""Tests for local save/load round trips."""

import json
from pathlib import Path

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
