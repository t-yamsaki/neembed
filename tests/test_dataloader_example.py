"""Smoke tests for the PyTorch DataLoader training example."""

from pathlib import Path
import runpy

import torch
from torch import nn

import neembed.model as model_module


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used to execute the example without downloads."""

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
        rows = [[float(len(sentence)), 1.0, -1.0] for sentence in sentences]
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}


def test_dataloader_example_runs_multi_epoch_training(monkeypatch, capsys) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    example_path = Path(__file__).parents[1] / "examples" / "train_dataloader.py"

    runpy.run_path(str(example_path), run_name="__main__")

    output = capsys.readouterr().out.lower()
    assert "epoch 1/2 - loss:" in output
    assert "epoch 2/2 - loss:" in output
    assert "validation history:" in output
    assert "retrieval_accuracy=" in output
    assert "mean_positive_distance=" in output
    assert "mean_negative_distance=" in output
    assert "nan" not in output
    assert "inf" not in output


def test_training_docs_link_to_dataloader_example() -> None:
    training_doc = (
        Path(__file__).parents[1] / "docs" / "user_guide" / "training.rst"
    ).read_text(encoding="utf-8")

    assert "examples/train_dataloader.py" in training_doc
