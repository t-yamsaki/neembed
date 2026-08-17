"""Tests for exact chunked retrieval primitives."""

import numpy as np
import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed.model import ManifoldSentenceTransformer
from neembed.retrieval import _iter_exact_geodesic_distance_blocks


class FakeSentenceTransformer(nn.Module):
    """Small trainable encoder used to avoid model downloads in retrieval tests."""

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


def _make_model(monkeypatch, manifold_name: str) -> ManifoldSentenceTransformer:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)
    torch.manual_seed(0)
    return ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
    )


def _assemble_distance_matrix(
    blocks: list[tuple[int, int, torch.Tensor]],
    *,
    num_queries: int,
    num_corpus: int,
) -> torch.Tensor:
    matrix = torch.empty(
        (num_queries, num_corpus),
        dtype=blocks[0][2].dtype,
        device=blocks[0][2].device,
    )
    for query_start, corpus_start, distances in blocks:
        query_end = query_start + distances.shape[0]
        corpus_end = corpus_start + distances.shape[1]
        matrix[query_start:query_end, corpus_start:corpus_end] = distances
    return matrix


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
@pytest.mark.parametrize(
    ("query_chunk_size", "corpus_chunk_size"),
    [(1, 1), (2, 3), (4, 2)],
)
def test_chunked_distances_match_pairwise_reference(
    monkeypatch,
    manifold_name: str,
    query_chunk_size: int,
    corpus_chunk_size: int,
) -> None:
    model = _make_model(monkeypatch, manifold_name)
    queries = model.encode(["q", "query", "long query"], convert_to_tensor=True)
    corpus = model.encode(
        ["a", "answer", "candidate", "long candidate", "final"],
        convert_to_tensor=True,
    )

    blocks = list(
        _iter_exact_geodesic_distance_blocks(
            model,
            queries,
            corpus,
            query_chunk_size=query_chunk_size,
            corpus_chunk_size=corpus_chunk_size,
        )
    )
    actual = _assemble_distance_matrix(
        blocks,
        num_queries=len(queries),
        num_corpus=len(corpus),
    )
    expected = torch.stack(
        [
            torch.stack([model.distance(query, candidate) for candidate in corpus])
            for query in queries
        ]
    )

    assert actual.shape == (3, 5)
    assert torch.isfinite(actual).all()
    assert (actual >= 0).all()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert all(block.shape[0] <= query_chunk_size for _, _, block in blocks)
    assert all(block.shape[1] <= corpus_chunk_size for _, _, block in blocks)


@pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])
def test_chunked_distances_do_not_track_gradients_or_mutate_model(
    monkeypatch,
    manifold_name: str,
) -> None:
    model = _make_model(monkeypatch, manifold_name)
    model.train()
    queries = model(["query", "another query"])
    corpus = model(["candidate", "another candidate", "third candidate"])
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    blocks = list(
        _iter_exact_geodesic_distance_blocks(
            model,
            queries,
            corpus,
            query_chunk_size=1,
            corpus_chunk_size=2,
        )
    )

    assert model.training
    assert all(not distances.requires_grad for _, _, distances in blocks)
    assert all(not torch.is_inference(distances) for _, _, distances in blocks)
    assert all(parameter.grad is None for parameter in model.parameters())
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)


@pytest.mark.parametrize(
    ("query_chunk_size", "corpus_chunk_size"),
    [(0, 1), (-1, 1), (True, 1), (1.5, 1), (1, 0), (1, False)],
)
def test_chunked_distances_reject_invalid_chunk_sizes(
    monkeypatch,
    query_chunk_size,
    corpus_chunk_size,
) -> None:
    model = _make_model(monkeypatch, "poincare")
    queries = np.zeros((2, 2), dtype=np.float32)
    corpus = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="positive integer"):
        _iter_exact_geodesic_distance_blocks(
            model,
            queries,
            corpus,
            query_chunk_size=query_chunk_size,
            corpus_chunk_size=corpus_chunk_size,
        )


def test_chunked_distances_validate_embedding_batches(monkeypatch) -> None:
    model = _make_model(monkeypatch, "poincare")

    with pytest.raises(ValueError, match="queries must contain at least one embedding"):
        _iter_exact_geodesic_distance_blocks(
            model,
            torch.empty((0, 2)),
            torch.zeros((1, 2)),
            query_chunk_size=1,
            corpus_chunk_size=1,
        )

    with pytest.raises(ValueError, match="corpus must be a 2D embedding batch"):
        _iter_exact_geodesic_distance_blocks(
            model,
            torch.zeros((1, 2)),
            torch.zeros(2),
            query_chunk_size=1,
            corpus_chunk_size=1,
        )

    with pytest.raises(ValueError, match="same width"):
        _iter_exact_geodesic_distance_blocks(
            model,
            torch.zeros((1, 2)),
            torch.zeros((1, 3)),
            query_chunk_size=1,
            corpus_chunk_size=1,
        )
