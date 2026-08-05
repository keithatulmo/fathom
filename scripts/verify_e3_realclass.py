#!/usr/bin/env python3
"""Audit the committed WO-9 real-class branch from committed state alone.

WO-9 settles the SD5 close by running the frozen E3 detectors against the real held-out quiet class,
and requires the selected branch to be auditable from committed state without the ledger or R2. This
script reads only the committed real-class lineage and re-derives the branch and the
leak-disjointness attestation from the per-detector numbers it carries, so a third party can confirm
the SD5 close or its reopening from the recorded real-class detection rates. It uses the standard
library only, so it runs without the project environment, PyTorch, or R2, as the E1, E2, and E3
verifiers do.

The checks are: a detector "detects" the real class when its detection rate reaches the success
floor and exceeds twice its real-background false-alarm rate; the branch is a line success if a
line detector detects, else a learned confirmation if the learned detector detects, else an
artifact; the SD5-close flag is set only on the learned confirmation; and the recorded
disjointness attestation matches the committed test and training sha lists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_DEFAULT = Path(__file__).resolve().parent.parent / "docs" / "e3_realclass_lineage.json"


def _by_name(lineage: dict[str, Any], name: str) -> dict[str, Any]:
    return next(d for d in lineage["per_detector"] if d["name"] == name)


def _detects(row: dict[str, Any], floor: float) -> bool:
    return bool(
        float(row["real_detection_rate"]) >= floor
        and float(row["real_detection_rate"]) > 2.0 * float(row["false_alarm_rate"])
    )


def _rederive(lineage: dict[str, Any]) -> tuple[str, bool]:
    floor = float(lineage["success_floor"])
    learned = _by_name(lineage, "learned")
    cfar = _by_name(lineage, "cfar")
    integration = _by_name(lineage, "integration")
    if _detects(cfar, floor) or _detects(integration, floor):
        return "lines_succeed_on_real", False
    if _detects(learned, floor):
        return "learned_confirmed", True
    return "learned_artifact", False


def _check(lineage: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    branch, closes = _rederive(lineage)
    recorded = lineage["branch"]
    if str(recorded["selected"]) != branch:
        problems.append(
            f"branch recorded {recorded['selected']!r} but the numbers select {branch!r}"
        )
    if bool(recorded["closes_sd5"]) != closes:
        problems.append(
            f"closes_sd5 recorded {recorded['closes_sd5']} but the branch gives {closes}"
        )
    attestation = lineage["leak_attestation"]
    held = set(attestation["held_out_recording_shas"])
    training = set(attestation["e3_training_shas"])
    if bool(attestation["disjoint"]) != held.isdisjoint(training):
        problems.append(
            "disjointness attestation does not match the committed test and training sha lists"
        )
    if not attestation["disjoint"]:
        problems.append(
            "leak: the test recordings are not disjoint from the E3 training backgrounds"
        )
    return problems


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
    lineage = json.loads(path.read_text(encoding="utf-8"))
    problems = _check(lineage)

    print(f"lineage: {path}")
    print(f"run id:  {lineage.get('run_id')}  (seed {lineage.get('seed')})")
    print(f"front end: {lineage.get('front_end')}  |  cohort: {lineage.get('cohort')}")
    print("detector    real-detect  op-point  4Hz-floor  FA(real bg)  E3-surrogate")
    for detector in lineage["per_detector"]:
        print(
            f"  {detector['name']:9s}  {detector['real_detection_rate']:.2f}         "
            f"{detector['operating_point_detection_rate']:.2f}      "
            f"{detector['floor_4hz_detection_rate']:.2f}       "
            f"{detector['false_alarm_rate']:.2f}         "
            f"{detector['e3_surrogate_operating_sensitivity']:.2f}"
        )
    print(
        f"\nbranch: {lineage['branch']['selected']}  closes_sd5={lineage['branch']['closes_sd5']}"
    )
    print(f"leak disjoint: {lineage['leak_attestation']['disjoint']}")
    if problems:
        print("\nAUDIT FAILED: the committed branch does not follow from the committed numbers:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAUDIT OK: the committed branch and attestation follow from the committed numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
