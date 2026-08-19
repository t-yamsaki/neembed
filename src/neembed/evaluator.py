"""Evaluation helpers for manifold-valued sentence embeddings."""

from collections.abc import Mapping, Sequence

import torch

from neembed.model import ManifoldSentenceTransformer
from neembed.prototypes import ManifoldPrototypes
from neembed.retrieval import exact_corpus_search


class ManifoldEmbeddingEvaluator:
    """Evaluate aligned anchor-positive pairs using manifold geodesic distance.

    Each anchor is matched against every positive candidate. The positive at the
    same index is treated as the correct retrieval target, while off-diagonal
    candidates are treated as negatives.

    Args:
        model: Manifold sentence embedding model to evaluate.
        anchors: Anchor texts. At least two aligned pairs are required.
        positives: Positive texts aligned by index with ``anchors``.
        recall_at_k: Positive integer cutoffs for Recall@K metrics. The default
            reports Recall@1. A cutoff greater than or equal to the candidate
            count has Recall@K equal to 1 because the aligned target is always
            present in the candidate pool.

    Raises:
        ValueError: If the anchor and positive counts differ, fewer than two
            aligned pairs are provided, or ``recall_at_k`` is empty, duplicated,
            or contains a non-positive integer.
    """

    def __init__(
        self,
        *,
        model: ManifoldSentenceTransformer,
        anchors: Sequence[str],
        positives: Sequence[str],
        recall_at_k: Sequence[int] = (1,),
    ) -> None:
        self.model = model
        self.anchors = list(anchors)
        self.positives = list(positives)
        self.recall_at_k = tuple(recall_at_k)

        if len(self.anchors) != len(self.positives):
            raise ValueError("anchors and positives must contain the same number of items")
        if len(self.anchors) < 2:
            raise ValueError("evaluation requires at least two aligned pairs")
        if not self.recall_at_k:
            raise ValueError("recall_at_k must contain at least one cutoff")
        if any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0
            for k in self.recall_at_k
        ):
            raise ValueError("recall_at_k values must be positive integers")
        if len(set(self.recall_at_k)) != len(self.recall_at_k):
            raise ValueError("recall_at_k values must be unique")

    def __call__(self) -> dict[str, float]:
        """Return retrieval, ranking, and aligned/off-diagonal distance metrics.

        Returns:
            A dictionary containing the existing ``retrieval_accuracy``,
            ``mean_positive_distance``, and ``mean_negative_distance`` metrics,
            plus ``mrr`` and one ``recall_at_<k>`` key per configured cutoff.

        Notes:
            Evaluation does not track gradients. Because ``model.encode()`` enters
            evaluation mode, the model's original train/eval mode is restored
            before this method returns. Candidate ranks are ordered by ascending
            manifold geodesic distance with stable index-order tie handling.
        """
        was_training = self.model.training
        try:
            with torch.no_grad():
                anchor_embeddings = self.model.encode(
                    self.anchors,
                    convert_to_tensor=True,
                )
                positive_embeddings = self.model.encode(
                    self.positives,
                    convert_to_tensor=True,
                )

                distances = self.model.distance(
                    anchor_embeddings.unsqueeze(1),
                    positive_embeddings.unsqueeze(0),
                )

                pair_count = len(self.anchors)
                targets = torch.arange(pair_count, device=distances.device)
                positive_distances = distances.diagonal()

                # Compute the exact stable rank of each aligned target without
                # materializing an N x N int64 argsort result. Candidates with a
                # smaller distance rank first; equal-distance candidates preserve
                # their original index order.
                target_distances = positive_distances.unsqueeze(1)
                closer_counts = (distances < target_distances).sum(dim=1)
                candidate_indices = torch.arange(
                    pair_count,
                    device=distances.device,
                ).unsqueeze(0)
                equal_before_counts = (
                    (distances == target_distances)
                    & (candidate_indices < targets.unsqueeze(1))
                ).sum(dim=1)
                target_ranks = closer_counts + equal_before_counts + 1

                negative_mask = ~torch.eye(
                    pair_count,
                    dtype=torch.bool,
                    device=distances.device,
                )
                negative_distances = distances[negative_mask]

                metrics = {
                    "retrieval_accuracy": float(
                        (target_ranks == 1).float().mean().item()
                    ),
                    "mean_positive_distance": float(
                        positive_distances.mean().item()
                    ),
                    "mean_negative_distance": float(
                        negative_distances.mean().item()
                    ),
                    "mrr": float(
                        target_ranks.float().reciprocal().mean().item()
                    ),
                }
                for k in self.recall_at_k:
                    metrics[f"recall_at_{k}"] = float(
                        (target_ranks <= k).float().mean().item()
                    )
                return metrics
        finally:
            self.model.train(was_training)


class ManifoldCorpusRetrievalEvaluator:
    """Evaluate exact corpus retrieval with explicit multi-positive relevance.

    This evaluator complements :class:`ManifoldEmbeddingEvaluator`: queries and
    corpus items have caller-owned IDs, and each query may map to one or more
    relevant corpus IDs. Ranking delegates to :func:`exact_corpus_search`, so it
    inherits the exact Geoopt geodesic distance, bounded text/distance batching,
    and stable corpus-index tie ordering from the v0.6 retrieval path.

    ``Recall@K`` is computed per query as the fraction of that query's relevant
    corpus IDs retrieved in the first ``K`` results, then averaged across queries.
    ``MRR`` uses the rank of the first relevant result for each query.
    """

    def __init__(
        self,
        *,
        model: ManifoldSentenceTransformer,
        query_ids: Sequence[str],
        queries: Sequence[str],
        corpus_ids: Sequence[str],
        corpus: Sequence[str],
        relevance: Mapping[str, Sequence[str]],
        recall_at_k: Sequence[int] = (1,),
        query_chunk_size: int = 32,
        corpus_chunk_size: int = 256,
    ) -> None:
        if isinstance(query_ids, str):
            raise ValueError("query_ids must be a sequence of IDs, not a string")
        if isinstance(queries, str):
            raise ValueError("queries must be a sequence of texts, not a string")
        if isinstance(corpus_ids, str):
            raise ValueError("corpus_ids must be a sequence of IDs, not a string")
        if isinstance(corpus, str):
            raise ValueError("corpus must be a sequence of texts, not a string")
        if not isinstance(relevance, Mapping):
            raise ValueError("relevance must be a mapping from query IDs to corpus IDs")

        self.model = model
        self.query_ids = tuple(query_ids)
        self.queries = list(queries)
        self.corpus_ids = tuple(corpus_ids)
        self.corpus = list(corpus)
        self.recall_at_k = tuple(recall_at_k)
        self.query_chunk_size = query_chunk_size
        self.corpus_chunk_size = corpus_chunk_size

        if not self.query_ids:
            raise ValueError("evaluation requires at least one query")
        if not self.corpus_ids:
            raise ValueError("evaluation requires at least one corpus item")
        if len(self.query_ids) != len(self.queries):
            raise ValueError("query_ids and queries must contain the same number of items")
        if len(self.corpus_ids) != len(self.corpus):
            raise ValueError("corpus_ids and corpus must contain the same number of items")
        if any(not isinstance(query_id, str) or not query_id for query_id in self.query_ids):
            raise ValueError("query_ids must be non-empty strings")
        if any(not isinstance(corpus_id, str) or not corpus_id for corpus_id in self.corpus_ids):
            raise ValueError("corpus_ids must be non-empty strings")
        if any(not isinstance(text, str) for text in self.queries):
            raise ValueError("queries must contain only strings")
        if any(not isinstance(text, str) for text in self.corpus):
            raise ValueError("corpus must contain only strings")
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("query_ids must be unique")
        if len(set(self.corpus_ids)) != len(self.corpus_ids):
            raise ValueError("corpus_ids must be unique")

        if not self.recall_at_k:
            raise ValueError("recall_at_k must contain at least one cutoff")
        if any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0
            for k in self.recall_at_k
        ):
            raise ValueError("recall_at_k values must be positive integers")
        if len(set(self.recall_at_k)) != len(self.recall_at_k):
            raise ValueError("recall_at_k values must be unique")
        for value, name in (
            (self.query_chunk_size, "query_chunk_size"),
            (self.corpus_chunk_size, "corpus_chunk_size"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if any(not isinstance(query_id, str) or not query_id for query_id in relevance):
            raise ValueError("relevance query IDs must be non-empty strings")
        query_id_set = set(self.query_ids)
        relevance_query_ids = set(relevance)
        unknown_query_ids = sorted(relevance_query_ids - query_id_set)
        if unknown_query_ids:
            raise ValueError(f"unknown relevance query IDs: {unknown_query_ids}")
        missing_query_ids = sorted(query_id_set - relevance_query_ids)
        if missing_query_ids:
            raise ValueError(f"missing relevance for query IDs: {missing_query_ids}")

        corpus_id_set = set(self.corpus_ids)
        normalized_relevance: dict[str, frozenset[str]] = {}
        for query_id in self.query_ids:
            relevant_value = relevance[query_id]
            if isinstance(relevant_value, str):
                raise ValueError(
                    f"relevance for query {query_id!r} must be a sequence of corpus IDs"
                )
            try:
                relevant_ids = tuple(relevant_value)
            except TypeError as exc:
                raise ValueError(
                    f"relevance for query {query_id!r} must be a sequence of corpus IDs"
                ) from exc
            if not relevant_ids:
                raise ValueError(f"relevance for query {query_id!r} must not be empty")
            if any(
                not isinstance(corpus_id, str) or not corpus_id
                for corpus_id in relevant_ids
            ):
                raise ValueError("relevant corpus IDs must be non-empty strings")
            if len(set(relevant_ids)) != len(relevant_ids):
                raise ValueError(
                    f"relevance for query {query_id!r} must contain unique corpus IDs"
                )
            unknown_corpus_ids = sorted(set(relevant_ids) - corpus_id_set)
            if unknown_corpus_ids:
                raise ValueError(
                    f"unknown relevant corpus IDs for query {query_id!r}: "
                    f"{unknown_corpus_ids}"
                )
            normalized_relevance[query_id] = frozenset(relevant_ids)
        self.relevance = normalized_relevance

    def __call__(self) -> dict[str, float]:
        """Return exact corpus ``mrr`` and configured multi-positive Recall@K."""
        was_training = self.model.training
        try:
            with torch.no_grad():
                rankings = exact_corpus_search(
                    self.model,
                    self.queries,
                    self.corpus,
                    top_k=None,
                    query_chunk_size=self.query_chunk_size,
                    corpus_chunk_size=self.corpus_chunk_size,
                )

                reciprocal_rank_sum = 0.0
                recall_sums = {k: 0.0 for k in self.recall_at_k}
                for query_id, ranking in zip(
                    self.query_ids,
                    rankings,
                    strict=True,
                ):
                    relevant_ids = self.relevance[query_id]
                    ranked_corpus_ids = [
                        self.corpus_ids[int(result["index"])]
                        for result in ranking
                    ]
                    first_relevant_rank = next(
                        rank
                        for rank, corpus_id in enumerate(ranked_corpus_ids, start=1)
                        if corpus_id in relevant_ids
                    )
                    reciprocal_rank_sum += 1.0 / first_relevant_rank

                    for k in self.recall_at_k:
                        retrieved_relevant = sum(
                            corpus_id in relevant_ids
                            for corpus_id in ranked_corpus_ids[:k]
                        )
                        recall_sums[k] += retrieved_relevant / len(relevant_ids)

                query_count = len(self.query_ids)
                metrics = {"mrr": reciprocal_rank_sum / query_count}
                for k in self.recall_at_k:
                    metrics[f"recall_at_{k}"] = recall_sums[k] / query_count
                return metrics
        finally:
            self.model.train(was_training)


class ManifoldPrototypeAssignmentEvaluator:
    """Evaluate nearest-prototype assignments using manifold geodesic distance.

    Prototype IDs are caller-owned metadata aligned by index with the rows of
    :class:`neembed.ManifoldPrototypes`. The evaluator does not store IDs inside
    the Geoopt prototype parameter.

    Args:
        model: Manifold sentence embedding model used to encode ``sentences``.
        prototypes: Prototype module whose rows define the assignment candidates.
        prototype_ids: Unique, non-empty string identifiers aligned to prototype
            indices.
        sentences: Non-empty sequence of texts to evaluate.
        expected_prototype_ids: Expected prototype identifier for each sentence.

    Raises:
        ValueError: If the prototypes do not use the model's manifold instance,
            prototype IDs do not align with the prototype count, IDs are invalid
            or duplicated, sentence and expected-ID counts differ, evaluation is
            empty, an expected ID is unknown, or a sequence argument is passed as
            a bare string.
    """

    def __init__(
        self,
        *,
        model: ManifoldSentenceTransformer,
        prototypes: ManifoldPrototypes,
        prototype_ids: Sequence[str],
        sentences: Sequence[str],
        expected_prototype_ids: Sequence[str],
    ) -> None:
        if prototypes.manifold is not model.manifold:
            raise ValueError("prototypes must use the model's manifold instance")
        if isinstance(prototype_ids, str):
            raise ValueError("prototype_ids must be a sequence of IDs, not a string")
        if isinstance(sentences, str):
            raise ValueError("sentences must be a sequence of texts, not a string")
        if isinstance(expected_prototype_ids, str):
            raise ValueError(
                "expected_prototype_ids must be a sequence of IDs, not a string"
            )

        self.model = model
        self.prototypes = prototypes
        self.prototype_ids = tuple(prototype_ids)
        self.sentences = list(sentences)
        self.expected_prototype_ids = tuple(expected_prototype_ids)

        if len(self.prototype_ids) != self.prototypes.num_prototypes:
            raise ValueError(
                "prototype_ids must contain one ID per prototype: "
                f"expected {self.prototypes.num_prototypes}, got {len(self.prototype_ids)}"
            )
        if any(
            not isinstance(prototype_id, str) or not prototype_id
            for prototype_id in self.prototype_ids
        ):
            raise ValueError("prototype_ids must be non-empty strings")
        if len(set(self.prototype_ids)) != len(self.prototype_ids):
            raise ValueError("prototype_ids must be unique")
        if len(self.sentences) != len(self.expected_prototype_ids):
            raise ValueError(
                "sentences and expected_prototype_ids must contain the same number of items"
            )
        if not self.sentences:
            raise ValueError("evaluation requires at least one sentence")

        self._prototype_index = {
            prototype_id: index
            for index, prototype_id in enumerate(self.prototype_ids)
        }
        unknown_ids = sorted(
            {
                prototype_id
                for prototype_id in self.expected_prototype_ids
                if prototype_id not in self._prototype_index
            }
        )
        if unknown_ids:
            raise ValueError(f"unknown expected prototype IDs: {unknown_ids}")

    def __call__(self) -> dict[str, float]:
        """Return nearest-prototype assignment accuracy and mean distance.

        ``mean_assigned_prototype_distance`` is the mean geodesic distance from
        each sentence embedding to the prototype selected by nearest-distance
        assignment, independent of whether that predicted ID matches the expected
        ID.

        Evaluation runs without gradient tracking and restores the original
        train/eval modes of both the sentence model and prototype module.
        """
        model_was_training = self.model.training
        prototypes_were_training = self.prototypes.training
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    self.sentences,
                    convert_to_tensor=True,
                )
                distances = self.prototypes(embeddings)
                predicted_indices = distances.argmin(dim=1)
                expected_indices = torch.tensor(
                    [
                        self._prototype_index[prototype_id]
                        for prototype_id in self.expected_prototype_ids
                    ],
                    device=distances.device,
                )
                assigned_distances = distances.gather(
                    1,
                    predicted_indices.unsqueeze(1),
                ).squeeze(1)

                return {
                    "assignment_accuracy": float(
                        (predicted_indices == expected_indices).float().mean().item()
                    ),
                    "mean_assigned_prototype_distance": float(
                        assigned_distances.mean().item()
                    ),
                }
        finally:
            self.model.train(model_was_training)
            self.prototypes.train(prototypes_were_training)
