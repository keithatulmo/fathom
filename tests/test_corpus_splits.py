"""Tests for vessel-level split assignment and the split-integrity audit."""

from __future__ import annotations

from fathom.corpus.splits import assign_splits


def test_assignment_is_disjoint_and_seeds_test_and_train() -> None:
    registry = {f"V{i}" for i in range(60)}
    quiet = {f"V{i}" for i in range(30)}  # 30 quiet-tail vessels
    rows, audit = assign_splits(registry, quiet, "run1")

    # Every vessel gets exactly one split.
    assert len({row["vessel_id"] for row in rows}) == 60
    assert audit["disjoint"] is True
    assert all(not shared for shared in audit["intersections"].values())
    # Quiet-tail seeds test and train to the CA2 minimums.
    assert audit["quiet_tail_counts"]["test"] >= 12
    assert audit["quiet_tail_counts"]["train"] >= 12
    assert audit["meets_ca2"] is True
    assert audit["total_quiet_tail"] == 30
    assert audit["total_vessels"] == 60
    assert audit["meets_ca3"] is True


def test_insufficient_quiet_tail_fails_ca2() -> None:
    registry = {f"V{i}" for i in range(60)}
    quiet = {f"V{i}" for i in range(20)}  # only 20 quiet-tail, below 12+12
    _, audit = assign_splits(registry, quiet, "run1")
    assert audit["meets_ca2"] is False
    assert audit["meets_ca3"] is False  # 20 < 30 quiet-tail


def test_assignment_is_deterministic() -> None:
    registry = {f"V{i}" for i in range(40)}
    quiet = {f"V{i}" for i in range(26)}
    a, _ = assign_splits(registry, quiet, "run1")
    b, _ = assign_splits(registry, quiet, "run1")
    assert a == b
