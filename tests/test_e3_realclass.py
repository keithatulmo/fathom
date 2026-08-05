"""Tests for the WO-9 real-class detection test: frozen detectors, branch selection, verifier."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from fathom.config import E3Config
from fathom.detectors import LearnedDetector
from fathom.e3_realclass import _select_branch, run_e3_realclass

_VERIFIER = Path(__file__).resolve().parents[1] / "scripts" / "verify_e3_realclass.py"
_RATE = 1000.0


def test_learned_detector_roundtrips_frozen() -> None:
    original = LearnedDetector(
        weights=(0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4, -0.3),
        bias=-0.7,
        feature_mean=tuple(float(i) for i in range(9)),
        feature_std=tuple(1.0 + i for i in range(9)),
    )
    restored = LearnedDetector.from_dict(original.to_dict())
    assert restored == original


def _row(name: str, real: float, far: float, surrogate: float) -> dict[str, Any]:
    return {
        "name": name,
        "real_detection_rate": real,
        "operating_point_detection_rate": real,
        "floor_4hz_detection_rate": 0.0,
        "false_alarm_rate": far,
        "per_site_detection_rate": {},
        "e3_surrogate_operating_sensitivity": surrogate,
        "detection_vs_snr": [],
    }


def test_branch_learned_artifact() -> None:
    per_detector = [
        _row("learned", real=0.02, far=0.06, surrogate=0.95),
        _row("cfar", real=0.24, far=0.07, surrogate=0.0),
        _row("integration", real=0.19, far=0.06, surrogate=0.0),
    ]
    branch = _select_branch(per_detector, E3Config())
    assert branch["selected"] == "learned_artifact"
    assert branch["closes_sd5"] is False


def test_branch_learned_confirmed() -> None:
    per_detector = [
        _row("learned", real=0.80, far=0.05, surrogate=0.95),
        _row("cfar", real=0.10, far=0.07, surrogate=0.0),
        _row("integration", real=0.08, far=0.06, surrogate=0.0),
    ]
    branch = _select_branch(per_detector, E3Config())
    assert branch["selected"] == "learned_confirmed"
    assert branch["closes_sd5"] is True


def test_branch_lines_succeed() -> None:
    per_detector = [
        _row("learned", real=0.10, far=0.05, surrogate=0.95),
        _row("cfar", real=0.70, far=0.06, surrogate=0.0),
        _row("integration", real=0.20, far=0.06, surrogate=0.0),
    ]
    branch = _select_branch(per_detector, E3Config())
    assert branch["selected"] == "lines_succeed_on_real"
    assert branch["closes_sd5"] is False


def _windows(n_windows: int, seed0: int) -> list[dict[str, Any]]:
    n = int(_RATE * 32)
    out = []
    for i in range(n_windows):
        samples = np.ascontiguousarray(0.3 * np.random.default_rng(seed0 + i).standard_normal(n))
        out.append({"vessel_id": f"V{i}", "site": "siteA", "snr_db": 6.0 + i, "samples": samples})
    return out


def _frozen_learned() -> LearnedDetector:
    # A detector biased to never fire, so the real-class detection collapses to the artifact branch.
    return LearnedDetector(
        weights=tuple(0.0 for _ in range(9)),
        bias=-5.0,
        feature_mean=tuple(0.0 for _ in range(9)),
        feature_std=tuple(1.0 for _ in range(9)),
    )


def test_run_structure_and_no_absolute_level(tmp_path: Path) -> None:
    vessels = _windows(3, seed0=1)
    backgrounds = [{"site": "bg", "samples": w["samples"]} for w in _windows(3, seed0=100)]
    thresholds = {"cfar": 1e9, "integration": 1e9, "learned": 0.5}  # frozen, high so nothing fires
    payload = run_e3_realclass(
        vessels,
        backgrounds,
        _frozen_learned(),
        thresholds,
        thresholds,
        {"cfar": 0.0, "integration": 0.0, "learned": 0.95},
        {"held_out_recording_shas": ["a"], "e3_training_shas": ["b"], "disjoint": True},
        E3Config(),
    )
    assert {d["name"] for d in payload["per_detector"]} == {"cfar", "integration", "learned"}
    assert payload["branch"]["selected"] == "learned_artifact"
    assert payload["owner_certified"] == {"detection_range_r": None, "confirmer_pd": None}
    assert payload["leak_attestation"]["disjoint"] is True
    for detector in payload["per_detector"]:
        assert 0.0 <= detector["real_detection_rate"] <= 1.0
        assert "detection_vs_snr" in detector

    payload["run_id"] = "e3-realclass-test"
    payload["seed"] = 1
    lineage = tmp_path / "e3_realclass_lineage.json"
    lineage.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(_VERIFIER), str(lineage)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AUDIT OK" in result.stdout
