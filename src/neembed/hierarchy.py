"""Internal validation helpers for caller-owned hierarchy supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


_HierarchyContract = Literal["tree", "dag"]


@dataclass(frozen=True)
class _NormalizedHierarchySupervision:
    """Immutable, deterministic hierarchy metadata for internal consumers.

    ``parent_child_edges`` are always stored as ``(parent_id, child_id)`` pairs
    ordered by the caller's normalized node order. ``roots`` are inferred from
    the topology; caller-supplied roots, when present, are used as a consistency
    check. ``depths`` contains only explicitly supplied depth values, also in
    node order.
    """

    node_ids: tuple[str, ...]
    parent_child_edges: tuple[tuple[str, str], ...]
    roots: tuple[str, ...]
    depths: tuple[tuple[str, int], ...]
    contract: _HierarchyContract


def _normalize_hierarchy_supervision(
    node_ids: Sequence[str],
    parent_child_edges: Sequence[Sequence[str]],
    *,
    roots: Sequence[str] | None = None,
    depths: Mapping[str, int] | None = None,
    contract: _HierarchyContract = "dag",
) -> _NormalizedHierarchySupervision:
    """Validate and normalize caller-owned tree/DAG supervision.

    The helper intentionally owns no model state and performs no transitive
    closure. A ``tree`` contract means an acyclic directed forest: every node
    has at most one parent. A ``dag`` contract permits multiple parents while
    still requiring acyclicity.
    """

    if isinstance(node_ids, (str, bytes)):
        raise ValueError("node_ids must be a sequence of non-empty strings")
    normalized_nodes = tuple(node_ids)
    if not normalized_nodes:
        raise ValueError("node_ids must not be empty")
    if any(
        not isinstance(node_id, str) or not node_id
        for node_id in normalized_nodes
    ):
        raise ValueError("node_ids must contain only non-empty strings")
    if len(set(normalized_nodes)) != len(normalized_nodes):
        raise ValueError("node_ids must be unique")
    if contract not in ("tree", "dag"):
        raise ValueError("contract must be 'tree' or 'dag'")

    node_index = {node_id: index for index, node_id in enumerate(normalized_nodes)}
    seen_edges: set[tuple[str, str]] = set()
    edges: list[tuple[str, str]] = []
    parents_by_child: dict[str, set[str]] = {
        node_id: set() for node_id in normalized_nodes
    }
    children_by_parent: dict[str, list[str]] = {
        node_id: [] for node_id in normalized_nodes
    }

    if isinstance(parent_child_edges, (str, bytes)):
        raise ValueError(
            "parent_child_edges must contain (parent_id, child_id) pairs"
        )
    for edge in parent_child_edges:
        if (
            isinstance(edge, (str, bytes))
            or not isinstance(edge, Sequence)
            or len(edge) != 2
        ):
            raise ValueError(
                "each hierarchy edge must be a (parent_id, child_id) pair"
            )
        parent_id, child_id = edge
        if (
            not isinstance(parent_id, str)
            or not parent_id
            or not isinstance(child_id, str)
            or not child_id
        ):
            raise ValueError("hierarchy edge IDs must be non-empty strings")
        if parent_id not in node_index or child_id not in node_index:
            raise ValueError(
                "hierarchy edges must reference identifiers in node_ids"
            )
        if parent_id == child_id:
            raise ValueError("hierarchy edges must not contain self edges")
        normalized_edge = (parent_id, child_id)
        if normalized_edge in seen_edges:
            raise ValueError("hierarchy edges must be unique")
        seen_edges.add(normalized_edge)
        edges.append(normalized_edge)
        parents_by_child[child_id].add(parent_id)
        children_by_parent[parent_id].append(child_id)

    if contract == "tree" and any(
        len(parents) > 1 for parents in parents_by_child.values()
    ):
        raise ValueError("tree hierarchy nodes may have at most one parent")

    _validate_acyclic_hierarchy(
        normalized_nodes,
        children_by_parent,
        parents_by_child,
    )

    inferred_roots = tuple(
        node_id for node_id in normalized_nodes if not parents_by_child[node_id]
    )
    if roots is not None:
        if isinstance(roots, (str, bytes)):
            raise ValueError("roots must be a sequence of node IDs")
        supplied_roots = tuple(roots)
        if any(
            not isinstance(root, str) or not root for root in supplied_roots
        ):
            raise ValueError("roots must contain only non-empty strings")
        if any(root not in node_index for root in supplied_roots):
            raise ValueError("roots must reference identifiers in node_ids")
        if len(set(supplied_roots)) != len(supplied_roots):
            raise ValueError("roots must be unique")
        if set(supplied_roots) != set(inferred_roots):
            raise ValueError(
                "roots must match nodes with no incoming hierarchy edge"
            )

    normalized_depths: tuple[tuple[str, int], ...] = ()
    depth_mapping: dict[str, int] = {}
    if depths is not None:
        if not isinstance(depths, Mapping):
            raise ValueError("depths must map node IDs to non-negative integers")
        for node_id, depth in depths.items():
            if node_id not in node_index:
                raise ValueError("depths must reference identifiers in node_ids")
            if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
                raise ValueError("depth values must be non-negative integers")
            depth_mapping[node_id] = depth
        for parent_id, child_id in edges:
            if parent_id in depth_mapping and child_id in depth_mapping:
                if depth_mapping[parent_id] >= depth_mapping[child_id]:
                    raise ValueError(
                        "depth values must increase along hierarchy edges"
                    )
        normalized_depths = tuple(
            (node_id, depth_mapping[node_id])
            for node_id in normalized_nodes
            if node_id in depth_mapping
        )

    normalized_edges = tuple(
        sorted(
            edges,
            key=lambda edge: (node_index[edge[0]], node_index[edge[1]]),
        )
    )
    return _NormalizedHierarchySupervision(
        node_ids=normalized_nodes,
        parent_child_edges=normalized_edges,
        roots=inferred_roots,
        depths=normalized_depths,
        contract=contract,
    )


def _validate_acyclic_hierarchy(
    node_ids: tuple[str, ...],
    children_by_parent: Mapping[str, Sequence[str]],
    parents_by_child: Mapping[str, set[str]],
) -> None:
    """Reject directed cycles without materializing transitive closure."""

    remaining_parents = {
        node_id: len(parents_by_child[node_id]) for node_id in node_ids
    }
    queue = [node_id for node_id in node_ids if remaining_parents[node_id] == 0]
    visited = 0
    cursor = 0
    while cursor < len(queue):
        parent_id = queue[cursor]
        cursor += 1
        visited += 1
        for child_id in children_by_parent[parent_id]:
            remaining_parents[child_id] -= 1
            if remaining_parents[child_id] == 0:
                queue.append(child_id)

    if visited != len(node_ids):
        raise ValueError("hierarchy edges must be acyclic")
