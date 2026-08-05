#!/usr/bin/env python3
"""Audit the committed E3 detection selection from committed state alone.

WO-8 requires the SD5 close to be auditable from committed state without the ledger or R2. This
script reads only the committed E3 lineage and re-derives the winner and the flip clause from the
per-candidate numbers it carries, so a third party can confirm the selected detector follows from
recorded operating-point sensitivity and per-site false-alarm rates without rebuilding from the
corpus. It uses the standard library only, so it runs without the project environment, PyTorch, or
credentials, the same discipline as the E1 and E2 verifiers.

The checks are: each candidate's false-alarm-stability flag equals its per-site false-alarm rates
all within tolerance of the target; the selection maximizes operating-point sensitivity among the
false-alarm-stable candidates, subject to the implementation-risk trade that the learned detector is
chosen over hand-built integration only when it beats it by the material margin; and the recorded
selection, flip branch, and SD5-close verdict match the re-derivation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_DEFAULT = Path(__file__).resolve().parent.parent / "docs" / "e3_detection_lineage.json"


def _fa_stable(candidate: dict[str, Any], lineage: dict[str, Any]) -> bool:
    target = float(lineage["target_far"])
    tolerance = float(lineage["far_tolerance"])
    lo, hi = target / tolerance, target * tolerance
    return all(lo <= float(v) <= hi for v in candidate["per_site_far"].values())


def _sens(candidate: dict[str, Any]) -> float:
    return float(candidate["operating_point_sensitivity"])


def _rederive(lineage: dict[str, Any]) -> tuple[str | None, bool, str]:
    """Return the re-derived (selected_detector, flip_fired, flip_branch) from the numbers."""
    candidates = lineage["candidates"]
    stable = [c for c in candidates if _fa_stable(c, lineage)]
    if not stable:
        return None, True, "no_stable_candidate"

    overall_best = max(candidates, key=_sens)
    sensitivity_winner = max(stable, key=_sens)
    stability_flip = overall_best["name"] != sensitivity_winner["name"]

    hand_built = [c for c in stable if c["name"] != "learned"]
    best_hand = max(hand_built, key=_sens) if hand_built else None
    margin = float(lineage.get("learned_material_margin", 0.1))
    risk_trade = bool(
        sensitivity_winner["name"] == "learned"
        and best_hand is not None
        and _sens(sensitivity_winner) - _sens(best_hand) < margin
    )
    selected = best_hand if (risk_trade and best_hand is not None) else sensitivity_winner
    if risk_trade:
        branch = "implementation_risk_trade"
    elif stability_flip:
        branch = "false_alarm_stability"
    else:
        branch = "none"
    assert selected is not None
    return selected["name"], bool(stability_flip or risk_trade), branch


def _check(lineage: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for candidate in lineage["candidates"]:
        if bool(candidate["false_alarm_stable"]) != _fa_stable(candidate, lineage):
            problems.append(
                f"candidate {candidate['name']}: recorded false_alarm_stable "
                f"{candidate['false_alarm_stable']} but its per-site false-alarm rates give the "
                "opposite"
            )
    name, fired, branch = _rederive(lineage)
    recorded = lineage["selected"]["name"] if lineage.get("selected") else None
    if name != recorded:
        problems.append(f"selected {recorded!r} but the numbers select {name!r}")
    flip = lineage["flip_clause"]
    if bool(flip["fired"]) != fired or str(flip["branch"]) != branch:
        problems.append(
            f"flip recorded fired={flip['fired']} branch={flip['branch']!r} but the numbers give "
            f"fired={fired} branch={branch!r}"
        )
    closes = recorded is not None
    if bool(lineage["verdict"]["closes_sd5_on_measurement"]) != closes:
        problems.append(
            f"verdict closes_sd5_on_measurement={lineage['verdict']['closes_sd5_on_measurement']} "
            f"but a selection is {'present' if closes else 'absent'}"
        )
    return problems


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
    lineage = json.loads(path.read_text(encoding="utf-8"))
    problems = _check(lineage)

    selected = lineage.get("selected")
    flip = lineage["flip_clause"]
    print(f"lineage: {path}")
    print(f"run id:  {lineage.get('run_id')}  (seed {lineage.get('seed')})")
    print(f"front end: {lineage.get('front_end')}")
    if selected:
        print(
            f"selected: {selected['name']}  operating-point sensitivity="
            f"{selected['operating_point_sensitivity']:.2f}  4Hz-floor="
            f"{selected['floor_4hz_sensitivity']:.2f}  FA-stable={selected['false_alarm_stable']}"
        )
    print(f"flip: fired={flip['fired']} branch={flip['branch']!r}")
    print(f"closes SD5 on measurement: {lineage['verdict']['closes_sd5_on_measurement']}")
    if problems:
        print("\nAUDIT FAILED: the committed selection does not follow from the committed numbers:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAUDIT OK: the committed selection and flip follow from the committed numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
