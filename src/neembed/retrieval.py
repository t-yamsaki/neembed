"""Exact retrieval utilities built on manifold geodesic distance."""

from collections.abc import Iterator, Sequence
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


def _as_text_sequence(value: Sequence[str], *, name: str) -> list[str]:
    if isinstance(value, str):
        raise ValueError(f"{name} must be a sequence of strings, not a single string")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if not items:
        raise ValueError(f"{name} must contain at least one item")
    if any(not isinstance(item, str) for item in items):
        raise ValueError(f"{name} must contain only strings")
    return items


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


def exact_corpus_search(
    model: "ManifoldSentenceTransformer",
    queries: Sequence[str],
    corpus: Sequence[str],
    *,
    top_k: int | None = None,
    query_chunk_size: int = 32,
    corpus_chunk_size: int = 256,
) -> list[list[dict[str, str | int | float]]]:
    """Search a text corpus by exact manifold geodesic distance.

    Args:
        model: Configured manifold sentence model used for encoding and distance.
        queries: Non-empty sequence of query texts. A bare string is rejected.
        corpus: Non-empty sequence of corpus texts. A bare string is rejected.
        top_k: Number of results to retain per query. ``None`` returns the full
            corpus ranking. Integer values must be between 1 and corpus size.
        query_chunk_size: Positive query block size used for distance evaluation.
        corpus_chunk_size: Positive corpus block size used for distance evaluation.

    Returns:
        One ranked result list per query, preserving query input order. Each result
        contains ``candidate``, its original corpus ``index``, and exact geodesic
        ``distance``. Results are sorted by ascending distance; equal distances
        preserve corpus input order.

    Notes:
        This is exact search, not ANN. The query-by-corpus distance matrix is never
        materialized in full: distance blocks are streamed through
        :func:`_iter_exact_geodesic_distance_blocks`, and top-k mode retains only
        the best candidates seen for each query. Text embeddings themselves remain
        in memory for the duration of the call. No index, cache, or persistence is
        created.
    """
    query_list = _as_text_sequence(queries, name="queries")
    corpus_list = _as_text_sequence(corpus, name="corpus")
    query_chunk_size = _validate_chunk_size(
        query_chunk_size,
        name="query_chunk_size",
    )
    corpus_chunk_size = _validate_chunk_size(
        corpus_chunk_size,
        name="corpus_chunk_size",
    )

    if top_k is None:
        result_count = len(corpus_list)
    else:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= len(corpus_list)
        ):
            raise ValueError(
                "top_k must be an integer between 1 and the corpus size"
            )
        result_count = top_k

    query_embeddings = model.encode(query_list, convert_to_tensor=True)
    corpus_embeddings = model.encode(corpus_list, convert_to_tensor=True)

    retained: list[list[tuple[float, int]]] = [[] for _ in query_list]
    for query_start, corpus_start, distance_block in (
        _iter_exact_geodesic_distance_blocks(
            model,
            query_embeddings,
            corpus_embeddings,
            query_chunk_size=query_chunk_size,
            corpus_chunk_size=corpus_chunk_size,
        )
    ):
        block_values = distance_block.detach().cpu().tolist()
        for query_offset, distances in enumerate(block_values):
            query_index = query_start + query_offset
            candidates = retained[query_index]
            candidates.extend(
                (float(distance), corpus_start + corpus_offset)
                for corpus_offset, distance in enumerate(distances)
            )
            candidates.sort(key=lambda item: (item[0], item[1]))
            if len(candidates) > result_count:
                del candidates[result_count:]

    return [
        [
            {
                "candidate": corpus_list[index],
                "index": index,
                "distance": distance,
            }
            for distance, index in candidates
        ]
        for candidates in retained
    ]
