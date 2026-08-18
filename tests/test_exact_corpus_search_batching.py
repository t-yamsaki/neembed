"""Regression coverage for bounded text encoding in exact corpus search."""

import numpy as np
import torch

from neembed.retrieval import exact_corpus_search


class RecordingModel:
    """Small duck-typed search model that records encoder batch boundaries."""

    def __init__(self) -> None:
        self.encoded_batches: list[list[str]] = []

    def encode(self, sentences, *, convert_to_tensor=False):
        batch = list(sentences)
        self.encoded_batches.append(batch)
        values = np.asarray([[float(len(text))] for text in batch], dtype=np.float32)
        if convert_to_tensor:
            return torch.as_tensor(values)
        return values

    def distance(self, a, b):
        a_tensor = torch.as_tensor(a, dtype=torch.float32)
        b_tensor = torch.as_tensor(b, dtype=torch.float32)
        return torch.linalg.vector_norm(a_tensor - b_tensor, dim=-1)


def test_exact_corpus_search_batches_query_and_corpus_encoding() -> None:
    model = RecordingModel()
    queries = ["q0", "q1", "q2", "q3", "q4"]
    corpus = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "ggggggg"]

    results = exact_corpus_search(
        model,
        queries,
        corpus,
        top_k=2,
        query_chunk_size=2,
        corpus_chunk_size=3,
    )

    assert model.encoded_batches == [
        ["q0", "q1"],
        ["q2", "q3"],
        ["q4"],
        ["a", "bb", "ccc"],
        ["dddd", "eeeee", "ffffff"],
        ["ggggggg"],
    ]
    assert len(results) == len(queries)
    assert all(len(rows) == 2 for rows in results)
