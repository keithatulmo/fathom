"""The split-integrity audit.

SD3 Section 5.6 makes leakage resistance a measured property, not a promise. Every run emits an
audit recording the vessel-to-split assignment hash, proofs of empty intersection between the
train, calibration, and test vessel sets, verification of the temporal guard, the quarantine
roster, and the truth-tier composition of the evaluation set. A run whose audit does not pass is
invalid by construction, and its metrics are not reportable.

In this scaffold the corpus is synthetic and carries no acquisition timestamps, so the temporal
guard is verified against the guard the registry declares rather than against real inter-segment
intervals; that narrowing is recorded in docs/DEVIATIONS.md and closes when the real corpus lands.
"""

from __future__ import annotations

from typing import Any

from .artifacts import Registry, SplitAudit, Truth
from .hashing import hash_json

# The three splits whose vessel sets must be pairwise disjoint.
_SPLITS = ("train", "calibration", "test")

# Splits used to report and score; evaluation is tier-one only per SD3 Section 5.3.
_EVAL_SPLITS = ("calibration", "test")


def _vessel_sets(registry: Registry) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {split: set() for split in _SPLITS}
    for vessel in registry.vessels:
        if vessel.get("quarantined"):
            continue
        split = str(vessel["split"])
        if split in sets:
            sets[split].add(str(vessel["vessel_id"]))
    return sets


def build_audit(
    registry: Registry,
    truth: Truth,
    *,
    guard_hours: float,
    require_tier_one_eval: bool,
) -> SplitAudit:
    """Compute the split-integrity audit for a run's registry and truth.

    The audit passes only when the three vessel sets are pairwise disjoint, the declared guard
    meets the required guard, no quarantined vessel appears in an evaluation split, and the
    evaluation set is composed of tier-one truth alone when that rule is required.
    """
    vessel_sets = _vessel_sets(registry)

    intersections: dict[str, list[str]] = {}
    intersections_empty = True
    for i, left in enumerate(_SPLITS):
        for right in _SPLITS[i + 1 :]:
            shared = sorted(vessel_sets[left] & vessel_sets[right])
            intersections[f"{left}&{right}"] = shared
            if shared:
                intersections_empty = False

    assignment = sorted(
        (str(v["vessel_id"]), str(v["split"]), bool(v.get("quarantined", False)))
        for v in registry.vessels
    )
    assignment_hash = hash_json(assignment)

    quarantine_roster = sorted(
        str(v["vessel_id"]) for v in registry.vessels if v.get("quarantined")
    )
    quarantined_set = set(quarantine_roster)

    # Truth-tier composition of the evaluation set, and quarantine leakage into evaluation.
    eval_tiers: dict[str, int] = {}
    quarantine_leak: list[str] = []
    eval_tier_one_only = True
    for channel in truth.channels:
        if str(channel["split"]) not in _EVAL_SPLITS:
            continue
        tier = int(channel["truth_tier"])
        key = f"tier{tier}"
        eval_tiers[key] = eval_tiers.get(key, 0) + 1
        if tier != 1:
            eval_tier_one_only = False
        vessel_id = channel.get("vessel_id")
        if vessel_id is not None and str(vessel_id) in quarantined_set:
            quarantine_leak.append(str(channel["channel_id"]))

    guard_ok = float(registry.guard_hours) >= float(guard_hours)
    tier_ok = eval_tier_one_only or not require_tier_one_eval

    passed = intersections_empty and guard_ok and tier_ok and not quarantine_leak

    payload: dict[str, Any] = {
        "assignment_hash": assignment_hash,
        "vessel_counts": {split: len(members) for split, members in vessel_sets.items()},
        "intersections": intersections,
        "intersections_empty": intersections_empty,
        "guard_hours_required": float(guard_hours),
        "guard_hours_declared": float(registry.guard_hours),
        "guard_ok": guard_ok,
        "quarantine_roster": quarantine_roster,
        "quarantine_leak_into_eval": sorted(quarantine_leak),
        "eval_truth_tiers": eval_tiers,
        "eval_tier_one_only": eval_tier_one_only,
        "require_tier_one_eval": require_tier_one_eval,
        "calibration_target_events": registry.calibration_target_events,
    }
    return SplitAudit(passed=passed, payload=payload)
