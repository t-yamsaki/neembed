"""Regression coverage for bounded corpus retrieval evaluation."""

import torch
from torch import nn

import neembed.evaluator as evaluator_module
from neembed.evaluator import ManifoldCorpusRetrievalEvaluator


class StreamingModel(nn.Module):
    """Small deterministic model for evaluator streaming behavior."""

    def __init__(self, points: dict[str, float]) -> None:
        super().__init__()
        self.points = points

    def encode(self, sentences: list[str]):
        self.eval()
        with torch.inference_mode():
            return torch.tensor(
                [[self.points[sentence]] for sentence in sentences],
                dtype=torch.float32,
            )

    def distance(self, a, b) -> torch.Tensor:
        return torch.abs(a - b).sum(dim=-1)


def test_corpus_evaluator_does_not_materialize_full_exact_search(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("full exact_corpus_search ranking must not be materialized")

    # The previous implementation called this module-level helper with top_k=None,
    # creating Q x C Python result objects. Adding the attribute with raising=False
    # keeps this regression effective even though the streaming implementation no
    # longer imports the public full-ranking helper.
    monkeypatch.setattr(
        evaluator_module,
        "exact_corpus_search",
        fail_if_called,
        raising=False,
    )

    points = {"q0": 0.0, "q1": 10.0}
    points.update({f"c{i}": float(i) for i in range(8)})
    model = StreamingModel(points)
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q0-id", "q1-id"],
        queries=["q0", "q1"],
        corpus_ids=[f"c{i}-id" for i in range(8)],
        corpus=[f"c{i}" for i in range(8)],
        relevance={
            "q0-id": ["c1-id", "c4-id"],
            "q1-id": ["c7-id"],
        },
        recall_at_k=(1, 3, 8),
        query_chunk_size=1,
        corpus_chunk_size=3,
    )

    metrics = evaluator()

    assert 0.0 < metrics["mrr"] <= 1.0
    assert 0.0 <= metrics["recall_at_1"] <= 1.0
    assert 0.0 <= metrics["recall_at_3"] <= 1.0
    assert metrics["recall_at_8"] == 1.0
