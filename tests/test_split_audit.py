"""Tests for the split-integrity audit and the not-reportable path."""

from __future__ import annotations

from typing import Any

import numpy as np

from fathom.artifacts import Detections, Registry, SplitAudit, Truth
from fathom.config import ScoringConfig
from fathom.report import build_report_payload
from fathom.split import build_audit


def _registry(vessels: list[dict[str, Any]]) -> Registry:
    return Registry(vessels=tuple(vessels), guard_hours=24.0, calibration_target_events=500)


def _clean_vessels() -> list[dict[str, Any]]:
    return [
        {"vessel_id": "A", "site": "s0", "split": "train", "truth_tier": 1, "quarantined": False},
        {
            "vessel_id": "B",
            "site": "s0",
            "split": "calibration",
            "truth_tier": 1,
            "quarantined": False,
        },
        {"vessel_id": "C", "site": "s1", "split": "test", "truth_tier": 1, "quarantined": False},
        {"vessel_id": "Q", "site": "s0", "split": "train", "truth_tier": 3, "quarantined": True},
    ]


def _truth(channels: list[dict[str, Any]]) -> Truth:
    return Truth(channels=tuple(channels))


def _clean_truth() -> Truth:
    return _truth(
        [
            {
                "channel_id": "c0",
                "vessel_id": "C",
                "site": "s1",
                "split": "test",
                "present": True,
                "truth_tier": 1,
            },
            {
                "channel_id": "c1",
                "vessel_id": None,
                "site": "s1",
                "split": "test",
                "present": False,
                "truth_tier": 1,
            },
            {
                "channel_id": "c2",
                "vessel_id": "B",
                "site": "s0",
                "split": "calibration",
                "present": True,
                "truth_tier": 1,
            },
        ]
    )


def test_clean_split_passes() -> None:
    audit = build_audit(
        _registry(_clean_vessels()), _clean_truth(), guard_hours=24.0, require_tier_one_eval=True
    )
    assert audit.passed
    assert audit.payload["intersections_empty"] is True
    assert audit.payload["quarantine_roster"] == ["Q"]


def test_vessel_in_two_splits_fails() -> None:
    vessels = _clean_vessels()
    vessels[2]["vessel_id"] = "A"  # C becomes A, now A is in both train and test
    audit = build_audit(
        _registry(vessels), _clean_truth(), guard_hours=24.0, require_tier_one_eval=True
    )
    assert not audit.passed
    assert audit.payload["intersections_empty"] is False


def test_tier_two_in_eval_fails() -> None:
    channels = _clean_truth().channels
    leaked = [dict(c) for c in channels]
    leaked[0]["truth_tier"] = 2  # a tier-two label in the evaluation set
    audit = build_audit(
        _registry(_clean_vessels()), _truth(leaked), guard_hours=24.0, require_tier_one_eval=True
    )
    assert not audit.passed
    assert audit.payload["eval_tier_one_only"] is False


def test_short_guard_fails() -> None:
    audit = build_audit(
        _registry(_clean_vessels()), _clean_truth(), guard_hours=48.0, require_tier_one_eval=True
    )
    assert not audit.passed
    assert audit.payload["guard_ok"] is False


def test_quarantined_vessel_in_eval_is_flagged() -> None:
    channels = [dict(c) for c in _clean_truth().channels]
    channels[0]["vessel_id"] = "Q"  # a quarantined identity appears in the test set
    audit = build_audit(
        _registry(_clean_vessels()), _truth(channels), guard_hours=24.0, require_tier_one_eval=True
    )
    assert not audit.passed
    assert audit.payload["quarantine_leak_into_eval"] == ["c0"]


def test_failed_audit_withholds_metrics() -> None:
    # A report built on a failed audit is not reportable and carries no metric sections.
    failed = SplitAudit(passed=False, payload={"assignment_hash": "x"})
    detections = Detections(
        channel_ids=("c0",),
        confidence=np.zeros(1),
        band_energy=np.zeros(1),
        band_low_hz=200.0,
        band_high_hz=300.0,
    )
    payload = build_report_payload(detections, _clean_truth(), failed, ScoringConfig(), seed=1)
    assert payload["reportable"] is False
    assert "reject_versus_miss" not in payload
