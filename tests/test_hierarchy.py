"""Focused tests for caller-owned hierarchy supervision validation."""

import pytest

import neembed
from neembed.hierarchy import _normalize_hierarchy_supervision


def test_normalizes_tree_deterministically() -> None:
    normalized = _normalize_hierarchy_supervision(
        node_ids=("root", "left", "right", "leaf"),
        parent_child_edges=(
            ("right", "leaf"),
            ("root", "right"),
            ("root", "left"),
        ),
        roots=("root",),
        depths={"leaf": 2, "root": 0, "right": 1, "left": 1},
        contract="tree",
    )
    reordered = _normalize_hierarchy_supervision(
        node_ids=("root", "left", "right", "leaf"),
        parent_child_edges=(
            ("root", "left"),
            ("right", "leaf"),
            ("root", "right"),
        ),
        roots=("root",),
        depths={"root": 0, "left": 1, "right": 1, "leaf": 2},
        contract="tree",
    )

    assert normalized == reordered
    assert normalized.node_ids == ("root", "left", "right", "leaf")
    assert normalized.parent_child_edges == (
        ("root", "left"),
        ("root", "right"),
        ("right", "leaf"),
    )
    assert normalized.roots == ("root",)
    assert normalized.depths == (
        ("root", 0),
        ("left", 1),
        ("right", 1),
        ("leaf", 2),
    )
    assert normalized.contract == "tree"


def test_normalizes_dag_with_multiple_parents() -> None:
    normalized = _normalize_hierarchy_supervision(
        node_ids=("root-a", "root-b", "shared"),
        parent_child_edges=(("root-b", "shared"), ("root-a", "shared")),
        contract="dag",
    )

    assert normalized.parent_child_edges == (
        ("root-a", "shared"),
        ("root-b", "shared"),
    )
    assert normalized.roots == ("root-a", "root-b")
    assert normalized.depths == ()


def test_rejects_invalid_node_identifiers() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _normalize_hierarchy_supervision((), ())
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_hierarchy_supervision(("root", ""), ())
    with pytest.raises(ValueError, match="unique"):
        _normalize_hierarchy_supervision(("root", "root"), ())


def test_rejects_malformed_unknown_duplicate_and_self_edges() -> None:
    ids = ("root", "child")

    with pytest.raises(ValueError, match="parent_id, child_id"):
        _normalize_hierarchy_supervision(ids, (("root", "child", "extra"),))
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_hierarchy_supervision(ids, ((1, "child"),))
    with pytest.raises(ValueError, match="reference identifiers"):
        _normalize_hierarchy_supervision(ids, (("missing", "child"),))
    with pytest.raises(ValueError, match="self edges"):
        _normalize_hierarchy_supervision(ids, (("root", "root"),))
    with pytest.raises(ValueError, match="unique"):
        _normalize_hierarchy_supervision(
            ids,
            (("root", "child"), ("root", "child")),
        )


def test_tree_rejects_multiple_parents_but_dag_accepts_them() -> None:
    ids = ("a", "b", "child")
    edges = (("a", "child"), ("b", "child"))

    with pytest.raises(ValueError, match="at most one parent"):
        _normalize_hierarchy_supervision(ids, edges, contract="tree")

    normalized = _normalize_hierarchy_supervision(ids, edges, contract="dag")
    assert normalized.roots == ("a", "b")


@pytest.mark.parametrize("contract", ["tree", "dag"])
def test_rejects_directed_cycles(contract: str) -> None:
    with pytest.raises(ValueError, match="acyclic"):
        _normalize_hierarchy_supervision(
            ("a", "b", "c"),
            (("a", "b"), ("b", "c"), ("c", "a")),
            contract=contract,
        )


def test_validates_optional_roots_against_topology() -> None:
    ids = ("root", "child", "isolated")
    edges = (("root", "child"),)

    normalized = _normalize_hierarchy_supervision(
        ids,
        edges,
        roots=("isolated", "root"),
    )
    assert normalized.roots == ("root", "isolated")

    with pytest.raises(ValueError, match="reference identifiers"):
        _normalize_hierarchy_supervision(ids, edges, roots=("missing",))
    with pytest.raises(ValueError, match="unique"):
        _normalize_hierarchy_supervision(ids, edges, roots=("root", "root"))
    with pytest.raises(ValueError, match="no incoming"):
        _normalize_hierarchy_supervision(ids, edges, roots=("root",))


def test_validates_and_orders_optional_depths() -> None:
    ids = ("root", "mid", "leaf")
    edges = (("root", "mid"), ("mid", "leaf"))

    normalized = _normalize_hierarchy_supervision(
        ids,
        edges,
        depths={"leaf": 4, "root": 1},
    )
    assert normalized.depths == (("root", 1), ("leaf", 4))

    with pytest.raises(ValueError, match="must map"):
        _normalize_hierarchy_supervision(ids, edges, depths=[("root", 0)])
    with pytest.raises(ValueError, match="reference identifiers"):
        _normalize_hierarchy_supervision(ids, edges, depths={"missing": 0})
    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative integers"):
            _normalize_hierarchy_supervision(ids, edges, depths={"root": invalid})
    with pytest.raises(ValueError, match="increase along hierarchy edges"):
        _normalize_hierarchy_supervision(
            ids,
            edges,
            depths={"root": 1, "mid": 1},
        )


def test_rejects_unknown_contract_and_keeps_helper_internal() -> None:
    with pytest.raises(ValueError, match="tree.*dag"):
        _normalize_hierarchy_supervision(("root",), (), contract="graph")

    assert "_normalize_hierarchy_supervision" not in neembed.__all__
    assert "_NormalizedHierarchySupervision" not in neembed.__all__
