"""Internal retrieval primitives built on exact manifold geodesic distance."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from neembed.model import ManifoldSentenceTransformer


def _validate_chunk_size(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_embedding_batch(value: Any, *, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a rectangular 2D embedding batch") from exc

    if tensor.numel() == 0:
        raise ValueError(f"{name} must contain at least one embedding")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D embedding batch")
    if tensor.shape[1] == 0:
        raise ValueError(f"{name} embeddings must have at least one coordinate")
    return tensor


def _iter_exact_geodesic_distance_blocks(
    model: "ManifoldSentenceTransformer",
    queries: Any,
    corpus: Any,
    *,
    query_chunk_size: int,
    corpus_chunk_size: int,
) -> Iterator[tuple[int, int, torch.Tensor]]:
    """Yield exact query-by-corpus geodesic distance blocks.

    The helper intentionally returns a block stream rather than assembling a full
    ``num_queries x num_corpus`` matrix. Each item is
    ``(query_start, corpus_start, distances)`` where ``distances`` has shape
    ``(query_block_size, corpus_block_size)``. Callers such as exact top-k search
    can therefore consume one block at a time and keep only the state they need.

    Distance calculation delegates to :meth:`ManifoldSentenceTransformer.distance`
    so Poincare/Lorentz dtype, device, Geoopt semantics, and no-grad behavior stay
    identical to the existing inference helper.
    """
    query_batch = _as_embedding_batch(queries, name="queries")
    corpus_batch = _as_embedding_batch(corpus, name="corpus")
    query_chunk_size = _validate_chunk_size(
        query_chunk_size,
        name="query_chunk_size",
    )
    corpus_chunk_size = _validate_chunk_size(
        corpus_chunk_size,
        name="corpus_chunk_size",
    )

    if query_batch.shape[1] != corpus_batch.shape[1]:
        raise ValueError("queries and corpus embeddings must have the same width")

    def blocks() -> Iterator[tuple[int, int, torch.Tensor]]:
        for query_start in range(0, query_batch.shape[0], query_chunk_size):
            query_chunk = query_batch[
                query_start : query_start + query_chunk_size
            ]
            for corpus_start in range(0, corpus_batch.shape[0], corpus_chunk_size):
                corpus_chunk = corpus_batch[
                    corpus_start : corpus_start + corpus_chunk_size
                ]
                distances = model.distance(
                    query_chunk[:, None, :],
                    corpus_chunk[None, :, :],
                )
                yield query_start, corpus_start, distances

    return blocks()
