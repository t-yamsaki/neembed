"""Offline deterministic acceptance for the complete v0.10 product example."""

import json
from pathlib import Path
import runpy
import sys

import pytest
import torch
from torch import nn

import neembed.model as model_module


class FakeSentenceTransformer(nn.Module):
    """Small encoder with dropout and local persistence; never downloads models."""

    instances = []

    def __init__(self, model_name_or_path):
        super().__init__()
        self.linear = nn.Linear(5, 6, bias=False)
        self.dropout = nn.Dropout(p=0.25)
        self.forward_modes = []
        self.instances.append(self)
        state = Path(model_name_or_path) / "encoder.pt"
        if state.exists():
            self.load_state_dict(torch.load(state, weights_only=True))

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 6

    def preprocess(self, texts):
        rows = []
        for text in texts:
            code = sum(ord(c) for c in text)
            rows.append([len(text) / 20, len(text.split()) / 5,
                         sum(c.lower() in "aeiou" for c in text) / 10,
                         (code % 17) / 20, (code % 29) / 30])
        return {"features": torch.tensor(rows, dtype=torch.float32)}

    def forward(self, features):
        self.forward_modes.append(self.training)
        return {"sentence_embedding": self.dropout(self.linear(features["features"]))}

    def save_pretrained(self, output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(output_path) / "encoder.pt")


EXAMPLE = Path(__file__).parents[1] / "examples" / "v10_mixed_curvature_workflow.py"


@pytest.fixture
def namespace(monkeypatch):
    monkeypatch.setattr(FakeSentenceTransformer, "instances", [])
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    return runpy.run_path(str(EXAMPLE))


def test_v10_product_workflow_is_deterministic_and_preserves_checkpoint(namespace, tmp_path):
    run = namespace["run_example"]
    torch.manual_seed(999)
    first = run("fake-model", seed=73, output_path=tmp_path / "first")
    torch.manual_seed(123)
    second = run("fake-model", seed=73, output_path=tmp_path / "second")
    assert first == second
    assert first["projection_updated"]
    assert len(first["training_losses"]) == 2
    assert first["hierarchy_component"] == "hierarchy"
    assert first["scale_policy"] == "fixed"
    components = first["product_config"]["components"]
    assert [c["manifold"] for c in components] == ["poincare", "sphere_projection", "euclidean"]
    assert [c["name"] for c in components] == ["hierarchy", "semantic", "residual"]
    assert [c["scale"] for c in components] == [1.5, 0.75, 1.0]
    values = first["objective_diagnostics"]
    assert values["total_loss"] == pytest.approx(
        values["retrieval_loss"] + 0.3 * values["hierarchy_loss"])
    for stage in ("before", "after"):
        result = first[stage]
        assert set(result["retrieval"]) == {"mrr", "recall_at_1", "recall_at_3"}
        assert all(0 <= v <= 1 for v in result["retrieval"].values())
        assert set(result["hierarchy"]) == {
            "parent_child_radial_order_accuracy", "mean_radial_order_violation",
            "depth_radius_spearman",
        }
        report = result["component_diagnostics"]
        assert len(report["total_distance"]) == 2
        assert list(report["component_distances"]) == ["hierarchy", "semantic", "residual"]
        assert all(len(v) == 2 for v in report["component_distances"].values())
        assert all(len(ranking) == 3 for ranking in result["search_results"])
    assert first["after"] == first["save_load"]["loaded_evaluation"]
    assert all(first["save_load"][key] for key in
               ("embeddings_match", "distances_match", "component_diagnostics_match"))
    metadata = json.loads((tmp_path / "first" / "neembed_config.json").read_text())
    assert metadata["product_config"] == first["product_config"]
    # Pretraining evaluation must not leave dropout disabled during optimizer updates.
    assert True in FakeSentenceTransformer.instances[0].forward_modes
    assert False in FakeSentenceTransformer.instances[0].forward_modes
    assert all(torch.isfinite(p).all() for encoder in FakeSentenceTransformer.instances
               for p in encoder.parameters())


def test_v10_root_command_runs_with_local_fake_encoder(namespace, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(sys, "argv", [str(EXAMPLE), "--model", "fake-model",
                                      "--epochs", "1", "--output", str(tmp_path / "checkpoint")])
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    output = capsys.readouterr().out
    result, notice = output.rsplit("\nDiagnostics only:", 1)
    data = json.loads(result)
    assert len(data["training_losses"]) == 1
    assert (tmp_path / "checkpoint" / "projection.pt").exists()
    assert "not a benchmark" in notice
    assert "superiority claim" in notice
    assert "learnable scales are not supported" in notice
    assert data["save_load"]["loaded_evaluation"] == data["after"]


def test_v10_default_temporary_checkpoint_round_trip(namespace):
    result = namespace["run_example"]("fake-model", epochs=1)
    assert result["save_load"]["product_config"] == result["product_config"]
    assert result["save_load"]["loaded_evaluation"] == result["after"]


@pytest.mark.parametrize("epochs", [0, -1, True, 1.5])
def test_v10_rejects_invalid_epochs_before_encoder_construction(namespace, epochs):
    with pytest.raises(ValueError, match="epochs must be a positive integer"):
        namespace["run_example"]("fake-model", epochs=epochs)
    assert not FakeSentenceTransformer.instances


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf"), True])
def test_v10_rejects_invalid_learning_rate(namespace, rate):
    with pytest.raises(ValueError, match="learning_rate must be positive and finite"):
        namespace["run_example"]("fake-model", learning_rate=rate)
    assert not FakeSentenceTransformer.instances
