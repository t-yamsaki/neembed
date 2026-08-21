"""Offline hard-negative mining over exact manifold corpus retrieval."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from neembed.retrieval import (
    _as_text_sequence,
    _encode_text_batches,
    _iter_exact_geodesic_distance_blocks,
    _validate_chunk_size,
)

if TYPE_CHECKING:
    from neembed.model import ManifoldSentenceTransformer


def _as_unique_ids(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"{name} must be a sequence of IDs, not a string")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of IDs") from exc
    if not items:
        raise ValueError(f"{name} must contain at least one ID")
    if any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{name} must contain non-empty string IDs")
    if len(set(items)) != len(items):
        raise ValueError(f"{name} must contain unique IDs")
    return items


def _normalize_id_mapping(
    value: Mapping[str, Sequence[str]],
    *,
    name: str,
    query_ids: tuple[str, ...],
    corpus_id_set: set[str],
    require_all_queries: bool,
    require_nonempty_values: bool,
) -> dict[str, frozenset[str]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping from query IDs to corpus IDs")
    if any(not isinstance(query_id, str) or not query_id for query_id in value):
        raise ValueError(f"{name} query IDs must be non-empty strings")

    query_id_set = set(query_ids)
    mapping_query_ids = set(value)
    unknown_query_ids = sorted(mapping_query_ids - query_id_set)
    if unknown_query_ids:
        raise ValueError(f"unknown {name} query IDs: {unknown_query_ids}")
    if require_all_queries:
        missing_query_ids = sorted(query_id_set - mapping_query_ids)
        if missing_query_ids:
            raise ValueError(f"missing {name} for query IDs: {missing_query_ids}")

    normalized: dict[str, frozenset[str]] = {}
    for query_id in query_ids:
        if query_id not in value:
            normalized[query_id] = frozenset()
            continue
        corpus_ids = value[query_id]
        if isinstance(corpus_ids, str):
            raise ValueError(
                f"{name} for query {query_id!r} must be a sequence of corpus IDs"
            )
        try:
            corpus_ids = tuple(corpus_ids)
        except TypeError as exc:
            raise ValueError(
                f"{name} for query {query_id!r} must be a sequence of corpus IDs"
            ) from exc
        if require_nonempty_values and not corpus_ids:
            raise ValueError(f"{name} for query {query_id!r} must not be empty")
        if any(not isinstance(corpus_id, str) or not corpus_id for corpus_id in corpus_ids):
            raise ValueError(f"{name} corpus IDs must be non-empty strings")
        if len(set(corpus_ids)) != len(corpus_ids):
            raise ValueError(
                f"{name} for query {query_id!r} must contain unique corpus IDs"
            )
        unknown_corpus_ids = sorted(set(corpus_ids) - corpus_id_set)
        if unknown_corpus_ids:
            raise ValueError(
                f"unknown {name} corpus IDs for query {query_id!r}: {unknown_corpus_ids}"
            )
        normalized[query_id] = frozenset(corpus_ids)
    return normalized


def mine_hard_negatives(
    model: "ManifoldSentenceTransformer",
    queries: Sequence[str],
    corpus: Sequence[str],
    *,
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    positive_corpus_ids: Mapping[str, Sequence[str]],
    excluded_corpus_ids: Mapping[str, Sequence[str]] | None = None,
    num_negatives: int = 1,
    query_chunk_size: int = 32,
    corpus_chunk_size: int = 256,
) -> list[list[dict[str, str | int | float]]]:
    """Mine exact, deterministic hard negatives before training.

    The miner is deliberately offline and caller-invoked. It reuses the bounded
    text-encoding and exact geodesic distance-block path behind
    :func:`neembed.exact_corpus_search`, but filters known positives and exclusions
    before retaining nearest candidates. ``ManifoldTrainer.fit()`` is not modified.

    Args:
        model: Configured manifold sentence model used for encoding and distance.
        queries: Non-empty query/anchor texts.
        corpus: Non-empty candidate corpus texts.
        query_ids: Unique caller-owned IDs aligned with ``queries``.
        corpus_ids: Unique caller-owned IDs aligned with ``corpus``.
        positive_corpus_ids: Mapping from every query ID to one or more known
            positive corpus IDs. Known positives are never returned.
        excluded_corpus_ids: Optional additional exclusions per query. Missing
            query IDs mean no additional exclusions. When a query ID also occurs
            in ``corpus_ids``, that matching corpus item is excluded automatically
            as an explicit self item.
        num_negatives: Positive number of negatives to return per query.
        query_chunk_size: Positive query batch/block size used for encoding and
            exact distance evaluation.
        corpus_chunk_size: Positive corpus batch/block size used for encoding and
            exact distance evaluation.

    Returns:
        One list per input query. Each inner list contains exactly
        ``num_negatives`` dictionaries sorted by ascending exact geodesic distance,
        with corpus input order as the tie-breaker. Every dictionary contains
        ``corpus_id``, ``candidate``, original corpus ``index``, and ``distance``.

    Raises:
        ValueError: If IDs, mappings, chunk sizes, or ``num_negatives`` are invalid,
            or if any query has too few valid corpus items after exclusions.
    """
    query_list = _as_text_sequence(queries, name="queries")
    corpus_list = _as_text_sequence(corpus, name="corpus")
    query_ids_tuple = _as_unique_ids(query_ids, name="query_ids")
    corpus_ids_tuple = _as_unique_ids(corpus_ids, name="corpus_ids")

    if len(query_ids_tuple) != len(query_list):
        raise ValueError("query_ids and queries must contain the same number of items")
    if len(corpus_ids_tuple) != len(corpus_list):
        raise ValueError("corpus_ids and corpus must contain the same number of items")
    if isinstance(num_negatives, bool) or not isinstance(num_negatives, int) or num_negatives <= 0:
        raise ValueError("num_negatives must be a positive integer")
    query_chunk_size = _validate_chunk_size(query_chunk_size, name="query_chunk_size")
    corpus_chunk_size = _validate_chunk_size(corpus_chunk_size, name="corpus_chunk_size")

    corpus_id_set = set(corpus_ids_tuple)
    positives = _normalize_id_mapping(
        positive_corpus_ids,
        name="positive_corpus_ids",
        query_ids=query_ids_tuple,
        corpus_id_set=corpus_id_set,
        require_all_queries=True,
        require_nonempty_values=True,
    )
    extra_exclusions = _normalize_id_mapping(
        {} if excluded_corpus_ids is None else excluded_corpus_ids,
        name="excluded_corpus_ids",
        query_ids=query_ids_tuple,
        corpus_id_set=corpus_id_set,
        require_all_queries=False,
        require_nonempty_values=False,
    )

    corpus_index = {corpus_id: index for index, corpus_id in enumerate(corpus_ids_tuple)}
    excluded_indices: list[frozenset[int]] = []
    for query_id in query_ids_tuple:
        excluded_ids = set(positives[query_id]) | set(extra_exclusions[query_id])
        if query_id in corpus_id_set:
            excluded_ids.add(query_id)
        indices = frozenset(corpus_index[corpus_id] for corpus_id in excluded_ids)
        valid_count = len(corpus_ids_tuple) - len(indices)
        if valid_count < num_negatives:
            raise ValueError(
                f"query {query_id!r} has only {valid_count} valid negatives after exclusions; "
                f"requested {num_negatives}"
            )
        excluded_indices.append(indices)

    was_training = model.training
    try:
        with torch.no_grad():
            query_embeddings = _encode_text_batches(
                model,
                query_list,
                batch_size=query_chunk_size,
            )
            corpus_embeddings = _encode_text_batches(
                model,
                corpus_list,
                batch_size=corpus_chunk_size,
            )

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
                    blocked = excluded_indices[query_index]
                    candidates.extend(
                        (float(distance), corpus_start + corpus_offset)
                        for corpus_offset, distance in enumerate(distances)
                        if corpus_start + corpus_offset not in blocked
                    )
                    candidates.sort(key=lambda item: (item[0], item[1]))
                    if len(candidates) > num_negatives:
                        del candidates[num_negatives:]

            return [
                [
                    {
                        "corpus_id": corpus_ids_tuple[index],
                        "candidate": corpus_list[index],
                        "index": index,
                        "distance": distance,
                    }
                    for distance, index in candidates
                ]
                for candidates in retained
            ]
    finally:
        model.train(was_training)
