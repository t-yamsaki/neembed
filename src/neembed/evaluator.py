"""Evaluation helpers for manifold-valued sentence embeddings."""

from collections.abc import Sequence

import torch

from neembed.model import ManifoldSentenceTransformer
from neembed.prototypes import ManifoldPrototypes


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


class ManifoldPrototypeAssignmentEvaluator:
    """Evaluate nearest-prototype assignments using manifold geodesic distance.

    Prototype IDs are caller-owned metadata aligned by index with the rows of
    :class:`neembed.ManifoldPrototypes`. The evaluator does not store IDs inside
    the Geoopt prototype parameter.

    Args:
        model: Manifold sentence embedding model used to encode ``sentences``.
        prototypes: Prototype module whose rows define the assignment candidates.
        prototype_ids: Unique prototype identifiers aligned to prototype indices.
        sentences: Non-empty sequence of texts to evaluate.
        expected_prototype_ids: Expected prototype identifier for each sentence.

    Raises:
        ValueError: If prototype IDs do not align with the prototype count, IDs
            are duplicated, sentence and expected-ID counts differ, evaluation is
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
