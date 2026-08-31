"""Hierarchy structure evaluation for manifold sentence embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Literal

import torch

from neembed.hierarchy import _normalize_hierarchy_supervision
from neembed.model import ManifoldSentenceTransformer


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Return one-based average ranks with deterministic tie handling."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        value = values[order[start]]
        while end < len(order) and values[order[end]] == value:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _spearman_association(depths: Sequence[int], radii: Sequence[float]) -> float:
    """Return tie-aware Spearman association with a finite degenerate fallback."""
    if len(depths) < 2:
        return 0.0

    depth_ranks = _average_ranks([float(depth) for depth in depths])
    radius_ranks = _average_ranks(radii)
    depth_mean = sum(depth_ranks) / len(depth_ranks)
    radius_mean = sum(radius_ranks) / len(radius_ranks)
    depth_centered = [rank - depth_mean for rank in depth_ranks]
    radius_centered = [rank - radius_mean for rank in radius_ranks]
    numerator = sum(
        depth_value * radius_value
        for depth_value, radius_value in zip(depth_centered, radius_centered)
    )
    depth_scale = sum(value * value for value in depth_centered)
    radius_scale = sum(value * value for value in radius_centered)
    denominator = math.sqrt(depth_scale * radius_scale)
    if denominator == 0.0:
        return 0.0
    return max(-1.0, min(1.0, numerator / denominator))


class ManifoldHierarchyEvaluator:
    """Evaluate caller-provided hierarchy structure in manifold embeddings.

    Parent-child radial-order accuracy is the fraction of supplied edges whose
    parent has strictly smaller geodesic radius from the manifold origin than its
    child. Equal-radius parent/child ties therefore count as incorrect. Mean
    radial-order violation is ``mean(max(parent_radius - child_radius, 0))``;
    ties have zero violation magnitude even though they fail the strict accuracy
    criterion.

    When depth labels are supplied, ``depth_radius_spearman`` reports a tie-aware
    Spearman rank association between caller-provided depth and geodesic radius.
    It is a descriptive association only and does not imply a causal relationship.
    Partial depth mappings are allowed after hierarchy validation. If fewer than
    two labeled nodes are available, or either ranked variable has zero variance,
    the association is reported as ``0.0`` to keep evaluation finite and stable.

    Args:
        model: Manifold sentence embedding model to evaluate.
        node_ids: Unique caller-owned node identifiers aligned with ``texts``.
        texts: Texts whose embeddings represent the hierarchy nodes.
        parent_child_edges: Directed ``(parent_id, child_id)`` hierarchy edges.
        depths: Optional mapping from node IDs to non-negative integer depths.
        contract: ``"dag"`` for an acyclic directed graph or ``"tree"`` for an
            acyclic forest whose nodes have at most one parent.

    Notes:
        Evaluation uses ``model.manifold.dist0`` so both Poincare and Lorentz
        workflows use their native geodesic radius. Evaluation runs under
        ``torch.no_grad()`` and restores the model's original train/eval mode.
    """

    def __init__(
        self,
        *,
        model: ManifoldSentenceTransformer,
        node_ids: Sequence[str],
        texts: Sequence[str],
        parent_child_edges: Sequence[Sequence[str]],
        depths: Mapping[str, int] | None = None,
        contract: Literal["tree", "dag"] = "dag",
    ) -> None:
        if isinstance(texts, (str, bytes)):
            raise ValueError("texts must be a sequence of strings, not a string")
        normalized_texts = tuple(texts)
        if len(normalized_texts) != len(node_ids):
            raise ValueError("node_ids and texts must contain the same number of items")
        if any(not isinstance(text, str) for text in normalized_texts):
            raise ValueError("texts must contain only strings")

        hierarchy = _normalize_hierarchy_supervision(
            node_ids,
            parent_child_edges,
            depths=depths,
            contract=contract,
        )
        if not hierarchy.parent_child_edges:
            raise ValueError("hierarchy evaluation requires at least one parent-child edge")

        self.model = model
        self.node_ids = hierarchy.node_ids
        self.texts = normalized_texts
        self.parent_child_edges = hierarchy.parent_child_edges
        self.roots = hierarchy.roots
        self.depths = hierarchy.depths
        self.contract = hierarchy.contract

    def __call__(self) -> dict[str, float]:
        """Return radial hierarchy metrics for the configured supervision."""
        was_training = self.model.training
        try:
            with torch.no_grad():
                embeddings = self.model.encode(
                    self.texts,
                    convert_to_tensor=True,
                )
                radii = self.model.manifold.dist0(embeddings)
                if radii.ndim != 1 or radii.shape[0] != len(self.node_ids):
                    raise ValueError("model must return one manifold embedding per text")

                node_index = {
                    node_id: index for index, node_id in enumerate(self.node_ids)
                }
                parent_indices = torch.tensor(
                    [node_index[parent_id] for parent_id, _ in self.parent_child_edges],
                    device=radii.device,
                )
                child_indices = torch.tensor(
                    [node_index[child_id] for _, child_id in self.parent_child_edges],
                    device=radii.device,
                )
                radial_differences = radii[parent_indices] - radii[child_indices]
                metrics = {
                    "parent_child_radial_order_accuracy": float(
                        (radial_differences < 0).float().mean().item()
                    ),
                    "mean_radial_order_violation": float(
                        radial_differences.clamp_min(0).mean().item()
                    ),
                }

                if self.depths:
                    depth_values = [depth for _, depth in self.depths]
                    depth_radii = [
                        float(radii[node_index[node_id]].item())
                        for node_id, _ in self.depths
                    ]
                    metrics["depth_radius_spearman"] = _spearman_association(
                        depth_values,
                        depth_radii,
                    )
                return metrics
        finally:
            self.model.train(was_training)
