"""Compatibility tests for the v0.9 constant-curvature contract."""

from inspect import signature
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed.manifolds import get_manifold
from neembed.model import ManifoldSentenceTransformer


ROOT = Path(__file__).parents[1]


class FakeSentenceTransformer(nn.Module):
    """Tiny local encoder for constructor and persistence compatibility tests."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

        state_path = Path(model_name_or_path) / "encoder.pt"
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


def test_legacy_model_constructor_keeps_curvature_defaults() -> None:
    parameters = signature(ManifoldSentenceTransformer).parameters

    assert parameters["manifold"].default == "poincare"
    assert parameters["curvature"].default == 1.0
    assert parameters["learnable_curvature"].default is False


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_legacy_curvature_remains_magnitude_of_negative_sectional_curvature(
    manifold_name: str,
) -> None:
    public_curvature = 2.0
    manifold = get_manifold(manifold_name, curvature=public_curvature)

    if manifold_name == "poincare":
        assert float(manifold.c) == pytest.approx(public_curvature)
        sectional_curvature = -float(manifold.c)
    else:
        assert float(manifold.k) == pytest.approx(1.0 / public_curvature)
        sectional_curvature = -1.0 / float(manifold.k)

    assert sectional_curvature == pytest.approx(-public_curvature)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_legacy_model_constructor_still_reports_positive_curvature_magnitude(
    monkeypatch,
    manifold_name: str,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        curvature=2.0,
    )

    assert model.manifold_name == manifold_name
    assert model.curvature == pytest.approx(2.0)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_legacy_saved_models_keep_curvature_key_and_round_trip_semantics(
    monkeypatch,
    tmp_path: Path,
    manifold_name: str,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    torch.manual_seed(17)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        curvature=2.0,
    )
    before = model.encode(["dog", "mammal"], convert_to_tensor=True)
    save_path = tmp_path / f"saved-{manifold_name}"

    model.save_pretrained(save_path)
    config = json.loads((save_path / "neembed_config.json").read_text(encoding="utf-8"))
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)
    after = loaded.encode(["dog", "mammal"], convert_to_tensor=True)

    assert config == {
        "embedding_dim": 2,
        "manifold": manifold_name,
        "curvature": 2.0,
    }
    assert "sectional_curvature" not in config
    assert loaded.manifold_name == manifold_name
    assert loaded.curvature == pytest.approx(2.0)
    assert torch.allclose(before, after)


def test_v09_design_note_defines_unambiguous_signed_curvature_contract() -> None:
    note = (
        ROOT / "docs" / "user_guide" / "constant_curvature_semantics.rst"
    ).read_text(encoding="utf-8")

    assert "``sectional_curvature``" in note
    assert "signed sectional curvature" in note
    assert "sectional_curvature: float | None = None" in note
    assert "``sectional_curvature < 0``" in note
    assert "``sectional_curvature == 0``" in note
    assert "``sectional_curvature > 0``" in note
    assert '``manifold="euclidean"``' in note
    assert '``manifold="sphere_projection"``' in note
    assert '``manifold="stereographic"``' in note
    assert "``sectional_curvature`` must be ``None``" in note
    assert "non-default" in note
    assert "must raise ``ValueError``" in note
    assert '"sectional_curvature": 0.0' in note
    assert '"sectional_curvature": 2.0' in note
    assert '"sectional_curvature": -2.0' in note
    assert "learnable_sectional_curvature" in note
    assert "rather than silently ignore or reinterpret it" in note
