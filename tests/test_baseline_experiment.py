"""Tests for the minimal Euclidean-vs-Poincare baseline experiment."""

import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn

import neembed.model as model_module


class FakeSentenceTransformer(nn.Module):
    """Deterministic trainable encoder used without model downloads."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.linear = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.5, -0.25, 0.10],
                        [0.20, 0.40, -0.30],
                        [-0.10, 0.30, 0.50],
                        [0.25, -0.15, 0.35],
                    ]
                )
            )

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = [
            [float(len(sentence)), float(index + 1), -1.0]
            for index, sentence in enumerate(sentences)
        ]
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_tensor: bool = False,
    ):
        with torch.no_grad():
            embeddings = self(self.preprocess(list(sentences)))["sentence_embedding"]
        if convert_to_tensor:
            return embeddings
        return embeddings.numpy()


def _load_experiment_module():
    experiment_path = (
        Path(__file__).parents[1] / "experiments" / "compare_euclidean_poincare.py"
    )
    spec = importlib.util.spec_from_file_location("neembed_baseline_experiment", experiment_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_baseline_experiment_returns_directly_comparable_results(monkeypatch) -> None:
    experiment = _load_experiment_module()
    monkeypatch.setattr(experiment, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)

    result = experiment.run_experiment()

    assert result["metadata"]["metric"] == "parent_retrieval_accuracy"
    assert result["metadata"]["seed"] == 0
    assert result["metadata"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert len(result["metadata"]["train_pairs"]) == 10
    assert len(result["metadata"]["evaluation_pairs"]) == 5

    euclidean = result["results"]["euclidean_pretrained"]
    poincare = result["results"]["poincare_finetuned"]

    assert 0.0 <= euclidean["parent_retrieval_accuracy"] <= 1.0
    assert 0.0 <= poincare["parent_retrieval_accuracy"] <= 1.0
    assert euclidean["distance"] == "euclidean_l2"
    assert poincare["distance"] == "poincare_geodesic"
    assert euclidean["training_seconds"] == 0.0
    assert poincare["training_seconds"] >= 0.0
    assert torch.isfinite(torch.tensor(poincare["final_training_loss"]))
    json.dumps(result)


def test_baseline_experiment_cli_emits_json(monkeypatch, capsys) -> None:
    experiment = _load_experiment_module()
    monkeypatch.setattr(experiment, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)

    experiment.main()

    output = json.loads(capsys.readouterr().out)
    assert set(output["results"]) == {"euclidean_pretrained", "poincare_finetuned"}


def test_experiment_readme_documents_baseline_command() -> None:
    readme = (
        Path(__file__).parents[1] / "experiments" / "README.md"
    ).read_text(encoding="utf-8")

    assert "[Euclidean baseline experiment](compare_euclidean_poincare.py)" in readme
    assert "python experiments/compare_euclidean_poincare.py" in readme
