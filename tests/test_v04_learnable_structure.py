"""Regression coverage for the v0.4 learnable-structure example."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch
from torch import nn

import neembed.model as model_module


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "v04_learnable_structure.py"


class FakeSentenceTransformer(nn.Module):
    """Tiny deterministic encoder used without model downloads."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.linear = nn.Linear(3, 4, bias=False)

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = []
        for sentence in sentences:
            code_sum = sum(ord(character) for character in sentence)
            rows.append(
                [
                    float(len(sentence)) / 20.0,
                    float(code_sum % 31) / 31.0,
                    float(code_sum % 17) / 17.0,
                ]
            )
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def _load_example_module():
    spec = importlib.util.spec_from_file_location("v04_learnable_structure", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v04_example_reports_deterministic_learnable_structure_diagnostics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    example = _load_example_module()

    kwargs = {
        "epochs": 3,
        "seed": 23,
        "embedding_dim": 2,
        "learning_rate": 1e-2,
    }
    first = example.run_benchmark("fake-model", **kwargs)
    second = example.run_benchmark("fake-model", **kwargs)

    assert set(first) == {"fixed_structure", "learnable_structure"}
    fixed = first["fixed_structure"]
    learnable = first["learnable_structure"]

    for result in (fixed, learnable):
        assert math.isfinite(result["final_training_loss"])
        assert 0.0 <= result["assignment_accuracy"] <= 1.0
        assert math.isfinite(result["initial_curvature"])
        assert math.isfinite(result["final_curvature"])
        assert math.isfinite(result["curvature_delta"])
        assert math.isfinite(result["prototype_shift"])
        assert result["prototypes_valid"] is True
        assert result["finite_distances"] is True
        assert math.isfinite(result["embedding_distance"])
        assert result["intrinsic_dim"] == 2
        assert result["ambient_dim"] == 2

    assert fixed["curvature_delta"] == pytest.approx(0.0)
    assert fixed["prototype_shift"] == pytest.approx(0.0)
    assert abs(learnable["curvature_delta"]) > 0.0
    assert learnable["prototype_shift"] > 0.0

    for configuration in first:
        for key, value in first[configuration].items():
            if isinstance(value, bool):
                assert second[configuration][key] is value
            else:
                assert second[configuration][key] == pytest.approx(value)


def test_v04_example_uses_public_neembed_api() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")

    assert "from neembed import (" in source
    assert "neembed.model" not in source
    assert "geoopt.optim.RiemannianAdam" in source
    assert "stabilize=1" in source
    assert "ManifoldPrototypeHierarchyLoss" in source
