"""Vessel-level split assignment.

SD3 fixes the vessel as the split unit: no vessel appears in more than one of train, calibration,
or test. This assigns the AIS-verified registry-grade vessels deterministically, seeding test and
train with the quiet-tail cohort so the CA2 held-out and train-side counts are met, then
spreading the general vessels evenly. The order is a hash of the vessel identity, so the assignment
is unbiased by site or time and is reproducible. A split-integrity audit records the counts and the
pairwise-disjointness proof, which is guaranteed by construction since each vessel gets one split.
"""

from __future__ import annotations

from typing import Any

from ..hashing import hash_json

# CA2 requires at least this many quiet-tail vessels held out (test) and train-side (train).
QUIET_HELDOUT_MIN = 12
QUIET_TRAINSIDE_MIN = 12
VESSELS_TOTAL_MIN = 50
QUIET_TAIL_TOTAL_MIN = 30

_SPLITS = ("train", "calibration", "test")


def _order_key(vessel_id: str) -> str:
    return hash_json(vessel_id)


def assign_splits(
    registry_vessels: set[str], quiet_tail_vessels: set[str], assign_run: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign vessels to splits and return the rows and the split-integrity audit.

    Quiet-tail vessels are ordered by identity hash and the first held-out and train-side quotas go
    to test and train so the CA2 minimums are met; the remainder seed calibration. General vessels
    are spread evenly across the three splits. Every vessel receives exactly one split.
    """
    quiet = sorted(quiet_tail_vessels & registry_vessels, key=_order_key)
    general = sorted(registry_vessels - quiet_tail_vessels, key=_order_key)

    assignments: dict[str, tuple[str, bool]] = {}
    for index, vessel in enumerate(quiet):
        if index < QUIET_HELDOUT_MIN:
            split = "test"
        elif index < QUIET_HELDOUT_MIN + QUIET_TRAINSIDE_MIN:
            split = "train"
        else:
            split = "calibration"
        assignments[vessel] = (split, True)
    for index, vessel in enumerate(general):
        assignments[vessel] = (_SPLITS[index % 3], False)

    rows = [
        {
            "vessel_id": vessel,
            "assign_run": assign_run,
            "split": split,
            "is_quiet_tail": int(is_quiet_tail),
        }
        for vessel, (split, is_quiet_tail) in sorted(assignments.items())
    ]
    return rows, build_split_audit(rows)


def build_split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the split-integrity audit: counts, quiet-tail per split, and disjointness."""
    by_split: dict[str, set[str]] = {split: set() for split in _SPLITS}
    quiet_by_split: dict[str, set[str]] = {split: set() for split in _SPLITS}
    for row in rows:
        by_split[str(row["split"])].add(str(row["vessel_id"]))
        if row["is_quiet_tail"]:
            quiet_by_split[str(row["split"])].add(str(row["vessel_id"]))

    pairs = (("train", "calibration"), ("train", "test"), ("calibration", "test"))
    intersections = {f"{a}&{b}": sorted(by_split[a] & by_split[b]) for a, b in pairs}
    disjoint = all(not shared for shared in intersections.values())

    total = sum(len(members) for members in by_split.values())
    total_quiet = sum(len(members) for members in quiet_by_split.values())
    meets_ca2 = (
        len(quiet_by_split["test"]) >= QUIET_HELDOUT_MIN
        and len(quiet_by_split["train"]) >= QUIET_TRAINSIDE_MIN
    )
    meets_ca3 = disjoint and total >= VESSELS_TOTAL_MIN and total_quiet >= QUIET_TAIL_TOTAL_MIN
    return {
        "disjoint": disjoint,
        "intersections": intersections,
        "vessel_counts": {split: len(members) for split, members in by_split.items()},
        "quiet_tail_counts": {split: len(members) for split, members in quiet_by_split.items()},
        "total_vessels": total,
        "total_quiet_tail": total_quiet,
        "meets_ca2": meets_ca2,
        "meets_ca3": meets_ca3,
    }
