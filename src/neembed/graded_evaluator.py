"""Graded exact corpus retrieval evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real

from neembed.evaluator import ManifoldCorpusRetrievalEvaluator
from neembed.model import ManifoldSentenceTransformer
from neembed.retrieval import exact_corpus_search


class ManifoldGradedCorpusRetrievalEvaluator(ManifoldCorpusRetrievalEvaluator):
    """Evaluate exact corpus retrieval with graded relevance and nDCG@K.

    This is the graded-relevance companion to
    :class:`neembed.ManifoldCorpusRetrievalEvaluator`. ``graded_relevance`` maps
    each caller-owned query ID to a mapping of caller-owned corpus IDs and
    non-negative relevance grades. Corpus IDs omitted from a query's mapping have
    grade zero.

    For MRR and Recall@K, any corpus item with grade greater than zero is treated
    as binary relevant and the existing exact corpus evaluator semantics are
    reused unchanged. Each query must therefore contain at least one positive
    grade. nDCG uses gain ``2**grade - 1`` and discount
    ``1 / log2(rank + 1)``. Equal-distance ranking ties preserve corpus input
    order because nDCG consumes :func:`neembed.exact_corpus_search` results.

    Args:
        model: Configured manifold sentence model.
        query_ids: Unique caller-owned query IDs aligned with ``queries``.
        queries: Query texts.
        corpus_ids: Unique caller-owned corpus IDs aligned with ``corpus``.
        corpus: Corpus texts.
        graded_relevance: Mapping from every query ID to ``{corpus_id: grade}``.
            Grades must be finite, real, non-negative numbers. Unlisted corpus
            IDs have grade zero; each query must include at least one grade
            greater than zero.
        recall_at_k: Positive unique cutoffs for binary Recall@K, using
            ``grade > 0`` as relevant.
        ndcg_at_k: Positive unique cutoffs for nDCG@K. A cutoff larger than the
            corpus size is evaluated over the full corpus.
        query_chunk_size: Positive query encoding/distance chunk size.
        corpus_chunk_size: Positive corpus encoding/distance chunk size.

    Notes:
        nDCG is opt-in through this graded evaluator so the existing
        :class:`neembed.ManifoldCorpusRetrievalEvaluator` output remains exactly
        backward compatible for binary/multi-positive callers. Internally, gains
        are divided by the largest gain for each query before DCG/IDCG are formed.
        This common scaling cancels in nDCG and avoids overflow for large finite
        grades while preserving the documented ``2**grade - 1`` metric.
    """

    def __init__(
        self,
        *,
        model: ManifoldSentenceTransformer,
        query_ids: Sequence[str],
        queries: Sequence[str],
        corpus_ids: Sequence[str],
        corpus: Sequence[str],
        graded_relevance: Mapping[str, Mapping[str, float]],
        recall_at_k: Sequence[int] = (1,),
        ndcg_at_k: Sequence[int] = (1,),
        query_chunk_size: int = 32,
        corpus_chunk_size: int = 256,
    ) -> None:
        if not isinstance(graded_relevance, Mapping):
            raise ValueError(
                "graded_relevance must map query IDs to corpus-grade mappings"
            )
        ndcg_cutoffs = tuple(ndcg_at_k)
        if not ndcg_cutoffs:
            raise ValueError("ndcg_at_k must contain at least one cutoff")
        if any(
            isinstance(k, bool) or not isinstance(k, int) or k <= 0
            for k in ndcg_cutoffs
        ):
            raise ValueError("ndcg_at_k values must be positive integers")
        if len(set(ndcg_cutoffs)) != len(ndcg_cutoffs):
            raise ValueError("ndcg_at_k values must be unique")

        normalized: dict[str, dict[str, float]] = {}
        binary_relevance: dict[str, list[str]] = {}
        for query_id, grade_mapping in graded_relevance.items():
            if not isinstance(grade_mapping, Mapping):
                raise ValueError(
                    f"graded relevance for query {query_id!r} must be a mapping"
                )
            query_grades: dict[str, float] = {}
            for corpus_id, grade in grade_mapping.items():
                if not isinstance(corpus_id, str) or not corpus_id:
                    raise ValueError(
                        "graded relevance corpus IDs must be non-empty strings"
                    )
                if isinstance(grade, bool) or not isinstance(grade, Real):
                    raise ValueError("graded relevance values must be real numbers")
                grade_value = float(grade)
                if not math.isfinite(grade_value) or grade_value < 0.0:
                    raise ValueError(
                        "graded relevance values must be finite and non-negative"
                    )
                query_grades[corpus_id] = grade_value

            positives = [
                corpus_id
                for corpus_id, grade_value in query_grades.items()
                if grade_value > 0.0
            ]
            if not positives:
                raise ValueError(
                    f"graded relevance for query {query_id!r} must contain at least "
                    "one positive grade"
                )
            normalized[query_id] = query_grades
            binary_relevance[query_id] = positives

        super().__init__(
            model=model,
            query_ids=query_ids,
            queries=queries,
            corpus_ids=corpus_ids,
            corpus=corpus,
            relevance=binary_relevance,
            recall_at_k=recall_at_k,
            query_chunk_size=query_chunk_size,
            corpus_chunk_size=corpus_chunk_size,
        )

        corpus_id_set = set(self.corpus_ids)
        for query_id, query_grades in normalized.items():
            unknown_corpus_ids = sorted(set(query_grades) - corpus_id_set)
            if unknown_corpus_ids:
                raise ValueError(
                    f"unknown graded corpus IDs for query {query_id!r}: "
                    f"{unknown_corpus_ids}"
                )

        self.graded_relevance = {
            query_id: dict(normalized[query_id]) for query_id in self.query_ids
        }
        self.ndcg_at_k = ndcg_cutoffs

    @staticmethod
    def _log_gain(grade: float) -> float:
        """Return log(2**grade - 1) without overflowing for finite grades."""
        if grade == 0.0:
            return -math.inf
        exponent = grade * math.log(2.0)
        if exponent < 50.0:
            return math.log(math.expm1(exponent))
        return exponent + math.log1p(-math.exp(-exponent))

    @classmethod
    def _normalized_gain(cls, grade: float, *, max_grade: float) -> float:
        """Return gain divided by the query's largest gain in log space."""
        if grade == 0.0:
            return 0.0
        if grade == max_grade:
            return 1.0
        return math.exp(cls._log_gain(grade) - cls._log_gain(max_grade))

    def __call__(self) -> dict[str, float]:
        """Return exact MRR, Recall@K, and configured nDCG@K metrics."""
        was_training = self.model.training
        try:
            metrics = super().__call__()
            result_count = min(max(self.ndcg_at_k), len(self.corpus))
            rankings = exact_corpus_search(
                self.model,
                self.queries,
                self.corpus,
                top_k=result_count,
                query_chunk_size=self.query_chunk_size,
                corpus_chunk_size=self.corpus_chunk_size,
            )

            ndcg_sums = {k: 0.0 for k in self.ndcg_at_k}
            for query_id, ranking in zip(self.query_ids, rankings, strict=True):
                grades = self.graded_relevance[query_id]
                max_grade = max(grades.values())
                normalized_gains = {
                    corpus_id: self._normalized_gain(
                        grade,
                        max_grade=max_grade,
                    )
                    for corpus_id, grade in grades.items()
                }
                ideal_gains = sorted(normalized_gains.values(), reverse=True)

                for k in self.ndcg_at_k:
                    effective_k = min(k, len(self.corpus))
                    dcg = 0.0
                    for rank, result in enumerate(ranking[:effective_k], start=1):
                        corpus_id = self.corpus_ids[int(result["index"])]
                        gain = normalized_gains.get(corpus_id, 0.0)
                        dcg += gain / math.log2(rank + 1.0)

                    idcg = sum(
                        gain / math.log2(rank + 1.0)
                        for rank, gain in enumerate(
                            ideal_gains[:effective_k],
                            start=1,
                        )
                    )
                    ndcg_sums[k] += 0.0 if idcg == 0.0 else dcg / idcg

            query_count = len(self.query_ids)
            for k in self.ndcg_at_k:
                metrics[f"ndcg_at_{k}"] = ndcg_sums[k] / query_count
            return metrics
        finally:
            self.model.train(was_training)
