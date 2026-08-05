#!/usr/bin/env python3
"""Audit the committed E2 front-end selection from committed state alone.

WO-6 requires the SD4 close to be auditable from committed state without the ledger or R2. This
script reads only the committed E2 lineage and re-derives the selection and the flip clause from the
per-configuration scores it carries, so a third party can confirm the selected configuration follows
from the recorded separability, flatness, and constraint numbers without rebuilding from the corpus.
It uses the standard library only, so it runs without the project environment, PyTorch, or R2
credentials, the same discipline as the E1 verifier.

The checks are: each configuration's candidate flag equals its compute, determinism, and flatness
checks; the selection follows the SD4 memo rule (a Hann config reaching the separability floor with
flatness is preferred, else the multitaper arm, else the ordered flip clause); and the recorded
selection, flip branch, and SD4-close verdict match the re-derivation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_DEFAULT = Path(__file__).resolve().parent.parent / "docs" / "e2_front_end_lineage.json"


def _candidate(config: dict[str, Any]) -> bool:
    constraints = config["constraints"]
    return bool(
        constraints["compute_cleared"]
        and constraints["determinism_cleared"]
        and config["flatness"]["flatness_holds"]
    )


def _sep(config: dict[str, Any]) -> float:
    return float(config["separability"]["separability_db"])


def _best(configs: list[dict[str, Any]]) -> dict[str, Any]:
    return max(configs, key=_sep)


def _rederive(lineage: dict[str, Any]) -> tuple[str | None, bool, int]:
    """Return the re-derived (selected_key, flip_fired, flip_branch) from the per-config numbers."""
    per_config = lineage["per_config"]
    floor = float(lineage["separability_floor_db"])
    candidates = [c for c in per_config if _candidate(c)]
    if not candidates:
        holds_any = any(c["flatness"]["flatness_holds"] for c in per_config)
        return None, True, (0 if holds_any else 3)
    hann_adequate = [c for c in candidates if c["params"]["taper"] == "hann" and _sep(c) >= floor]
    if hann_adequate:
        return _best(hann_adequate)["key"], False, 0
    mt_adequate = [
        c for c in candidates if c["params"]["taper"] == "multitaper" and _sep(c) >= floor
    ]
    if mt_adequate:
        return _best(mt_adequate)["key"], True, 1
    return _best(candidates)["key"], True, 2


def _check(lineage: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for config in lineage["per_config"]:
        if bool(config["candidate"]) != _candidate(config):
            problems.append(
                f"config {config['key']}: recorded candidate {config['candidate']} but its "
                "compute, determinism, and flatness checks give the opposite"
            )
    key, fired, branch = _rederive(lineage)
    recorded_key = lineage["selected"]["key"] if lineage.get("selected") else None
    if key != recorded_key:
        problems.append(f"selected {recorded_key!r} but the numbers select {key!r}")
    flip = lineage["flip_clause"]
    if bool(flip["fired"]) != fired or int(flip["branch"]) != branch:
        problems.append(
            f"flip recorded fired={flip['fired']} branch={flip['branch']} but the numbers give "
            f"fired={fired} branch={branch}"
        )
    closes = bool(recorded_key is not None and not fired)
    if bool(lineage["verdict"]["closes_sd4_on_measurement"]) != closes:
        problems.append(
            f"verdict closes_sd4_on_measurement={lineage['verdict']['closes_sd4_on_measurement']} "
            f"but selection and flip give {closes}"
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
    print(
        f"configs: {len(lineage['per_config'])}  candidates: "
        f"{sum(1 for c in lineage['per_config'] if c['candidate'])}"
    )
    if selected:
        print(
            f"selected: {selected['key']}  "
            f"separability={selected['separability']['separability_db']:.2f} dB"
        )
    print(f"flip: fired={flip['fired']} branch={flip['branch']}")
    print(f"closes SD4 on measurement: {lineage['verdict']['closes_sd4_on_measurement']}")
    if problems:
        print("\nAUDIT FAILED: the committed selection does not follow from the committed numbers:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        "\nAUDIT OK: the committed selection and flip follow from the committed per-config numbers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
