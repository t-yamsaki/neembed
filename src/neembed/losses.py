"""Losses for manifold-valued sentence embeddings."""

from collections.abc import Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn

from neembed.model import ManifoldSentenceTransformer
from neembed.prototypes import ManifoldPrototypes


class ManifoldMultipleNegativesRankingLoss(nn.Module):
    """In-batch ranking loss based on manifold geodesic distance.

    Each anchor is paired with the positive at the same batch index. All
    off-diagonal positive candidates are treated as negatives, so duplicate
    positives within one batch should be avoided.

    Args:
        model: Manifold sentence model used to encode anchors and positives.
        temperature: Positive, finite temperature used to scale distance logits.
    """

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperature must be positive and finite")

        self.model = model
        self.temperature = float(temperature)

    def forward(
        self,
        anchors: Sequence[str],
        positives: Sequence[str],
    ) -> torch.Tensor:
        """Return the contrastive loss for aligned anchor-positive pairs.

        Args:
            anchors: Batch of anchor texts.
            positives: Batch of positive texts aligned by index with ``anchors``.

        Returns:
            A scalar cross-entropy loss built from pairwise geodesic distances.
        """
        anchor_embeddings = self.model(anchors)
        positive_embeddings = self.model(positives)

        distances = self.model.manifold.dist(
            anchor_embeddings[:, None, :],
            positive_embeddings[None, :, :],
        )
        logits = -distances / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, targets)


class ManifoldPrototypeHierarchyLoss(nn.Module):
    """Attract sentences to labeled prototypes while preserving simple hierarchy.

    The objective has two terms. Sentence embeddings are attracted to their
    assigned prototype, and every declared child prototype is encouraged to be
    closer to its parent than to every other prototype by ``margin``. Both terms
    use the Geoopt manifold distance exposed by :class:`ManifoldPrototypes`.

    Args:
        model: Manifold sentence model used to encode input texts.
        prototypes: Trainable manifold prototype module associated with ``model``.
        prototype_ids: Unique non-empty string identifiers aligned by index with
            ``prototypes.prototypes``.
        parent_relations: Explicit ``(child_id, parent_id)`` pairs. Each child may
            declare at most one parent and the relations must be acyclic.
        margin: Non-negative, finite ranking margin for the hierarchy term.
        hierarchy_weight: Non-negative, finite multiplier for the hierarchy term.

    Notes:
        For a child ``c``, parent ``p``, and unrelated prototype ``n``, the
        hierarchy penalty is ``relu(margin + d(c, p) - d(c, n))``. The loss
        averages this penalty over all unrelated prototypes and declared
        relations. A relation with no unrelated prototype contributes no ranking
        penalty.
    """

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        prototypes: ManifoldPrototypes,
        prototype_ids: Sequence[str],
        parent_relations: Sequence[tuple[str, str]],
        *,
        margin: float = 0.1,
        hierarchy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if prototypes.manifold is not model.manifold:
            raise ValueError("prototypes must use the model's manifold instance")
        if margin < 0 or not math.isfinite(margin):
            raise ValueError("margin must be non-negative and finite")
        if hierarchy_weight < 0 or not math.isfinite(hierarchy_weight):
            raise ValueError("hierarchy_weight must be non-negative and finite")

        ids = tuple(prototype_ids)
        if len(ids) != prototypes.num_prototypes:
            raise ValueError(
                "prototype_ids must contain exactly one identifier per prototype"
            )
        if any(not isinstance(prototype_id, str) or not prototype_id for prototype_id in ids):
            raise ValueError("prototype_ids must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("prototype_ids must be unique")

        prototype_index = {prototype_id: index for index, prototype_id in enumerate(ids)}
        relations: list[tuple[str, str]] = []
        parent_by_child: dict[str, str] = {}
        for relation in parent_relations:
            if (
                isinstance(relation, (str, bytes))
                or not isinstance(relation, Sequence)
                or len(relation) != 2
            ):
                raise ValueError(
                    "each parent relation must be a (child_id, parent_id) pair"
                )
            child_id, parent_id = relation
            if child_id not in prototype_index or parent_id not in prototype_index:
                raise ValueError(
                    "parent relations must reference identifiers in prototype_ids"
                )
            if child_id == parent_id:
                raise ValueError("a prototype cannot be its own parent")
            if child_id in parent_by_child:
                raise ValueError("each child prototype may declare at most one parent")
            parent_by_child[child_id] = parent_id
            relations.append((child_id, parent_id))

        if not relations:
            raise ValueError("parent_relations must contain at least one relation")
        self._validate_acyclic(parent_by_child)

        self.model = model
        self.prototypes = prototypes
        self.prototype_ids = ids
        self.parent_relations = tuple(relations)
        self.margin = float(margin)
        self.hierarchy_weight = float(hierarchy_weight)
        self._prototype_index = prototype_index
        self._relation_indices = tuple(
            (prototype_index[child_id], prototype_index[parent_id])
            for child_id, parent_id in relations
        )

    @staticmethod
    def _validate_acyclic(parent_by_child: dict[str, str]) -> None:
        for start in parent_by_child:
            visited: set[str] = set()
            current = start
            while current in parent_by_child:
                if current in visited:
                    raise ValueError("parent_relations must be acyclic")
                visited.add(current)
                current = parent_by_child[current]

    def _hierarchy_loss(self) -> torch.Tensor:
        points = self.prototypes.prototypes
        pairwise_distances = self.prototypes.manifold.dist(
            points[:, None, :],
            points[None, :, :],
        )
        relation_losses: list[torch.Tensor] = []

        for child_index, parent_index in self._relation_indices:
            negative_mask = torch.ones(
                self.prototypes.num_prototypes,
                dtype=torch.bool,
                device=points.device,
            )
            negative_mask[child_index] = False
            negative_mask[parent_index] = False
            negative_distances = pairwise_distances[child_index, negative_mask]
            if negative_distances.numel() == 0:
                continue

            child_parent_distance = pairwise_distances[child_index, parent_index]
            relation_losses.append(
                F.relu(
                    self.margin + child_parent_distance - negative_distances
                ).mean()
            )

        if not relation_losses:
            return pairwise_distances.new_zeros(())
        return torch.stack(relation_losses).mean()

    def forward(
        self,
        sentences: Sequence[str],
        assignments: Sequence[str],
    ) -> torch.Tensor:
        """Return the hierarchy-aware prototype objective for one text batch.

        Args:
            sentences: Batch of texts to encode.
            assignments: Prototype identifier for each sentence, aligned by index.

        Returns:
            Scalar sentence-attraction plus hierarchy-ranking loss.
        """
        if len(sentences) == 0:
            raise ValueError("sentences must not be empty")
        if len(sentences) != len(assignments):
            raise ValueError("sentences and assignments must have the same length")

        assignment_indices: list[int] = []
        for prototype_id in assignments:
            if prototype_id not in self._prototype_index:
                raise ValueError(
                    f"unknown prototype identifier in assignments: {prototype_id!r}"
                )
            assignment_indices.append(self._prototype_index[prototype_id])

        embeddings = self.model(sentences)
        distances = self.prototypes(embeddings)
        targets = torch.tensor(
            assignment_indices,
            dtype=torch.long,
            device=distances.device,
        )
        sentence_loss = distances.gather(1, targets[:, None]).mean()
        return sentence_loss + self.hierarchy_weight * self._hierarchy_loss()
