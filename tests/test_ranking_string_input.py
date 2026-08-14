"""Regression test for ambiguous bare-string candidate input."""

import pytest
from torch import nn

import neembed.model as model_module
from neembed.model import ManifoldSentenceTransformer


class FakeSentenceTransformer(nn.Module):
    """Minimal encoder stub; ranking should reject before encoding."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.weight = nn.Parameter(__import__("torch").zeros(1))

    @property
    def device(self):
        return self.weight.device

    def get_embedding_dimension(self) -> int:
        return 1


def test_rank_rejects_bare_string_candidates(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    model = ManifoldSentenceTransformer("fake-model")

    with pytest.raises(ValueError, match="sequence of strings"):
        model.rank("query", "mammal")
