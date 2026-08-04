#!/usr/bin/env python3
"""Audit the committed E1 real-data verdict from committed state alone.

WO-4 requires the result to be auditable from committed state without the ledger or R2. This script
reads only the committed lineage JSON and re-derives the verdict from the per-rung numbers it
carries, so a third party can confirm the pass-or-fail follows from the recorded data without
rebuilding anything from the object store. It uses the standard library only, so it runs without the
project environment, PyTorch, or R2 credentials.

The checks are: each rung's inside-null flag equals its Wasserstein test median being within the
rung's null tolerance; the operating band is the rungs whose detectability intervals overlap; and
the overall verdict is that realism holds inside the null across every operating rung. It exits
non-zero if the committed verdict does not follow from the committed numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_DEFAULT = Path(__file__).resolve().parent.parent / "docs" / "e1_realdata_lineage.json"


def _check(lineage: dict[str, Any]) -> list[str]:
    """Return a list of discrepancies; empty means the committed verdict is self-consistent."""
    problems: list[str] = []
    per_rung = lineage["per_rung"]

    for rung in per_rung:
        snr = rung["snr_db"]
        recomputed_inside = rung["test_median"] <= rung["null_tolerance"]
        if bool(recomputed_inside) != bool(rung["rejector_inside_null"]):
            problems.append(
                f"rung {snr} dB: recorded inside-null {rung['rejector_inside_null']} but "
                f"test_median {rung['test_median']} vs tolerance {rung['null_tolerance']} gives "
                f"{recomputed_inside}"
            )

    operating = [k for k, r in enumerate(per_rung) if r["detectability_overlap"]]
    if operating != list(lineage["operating_indices"]):
        problems.append(
            f"operating band {operating} does not match recorded {lineage['operating_indices']}"
        )

    recomputed_pass = bool(operating) and all(
        per_rung[k]["rejector_inside_null"] for k in operating
    )
    if recomputed_pass != bool(lineage["verdict"]["passed"]):
        problems.append(
            f"verdict passed={lineage['verdict']['passed']} but recomputing over the operating "
            f"band gives {recomputed_pass}"
        )
    if len(operating) != lineage["verdict"]["operating_rungs"]:
        problems.append(
            f"operating_rungs {lineage['verdict']['operating_rungs']} but the band has "
            f"{len(operating)} rungs"
        )
    return problems


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
    lineage = json.loads(path.read_text(encoding="utf-8"))
    problems = _check(lineage)

    verdict = lineage["verdict"]
    print(f"lineage: {path}")
    print(f"run id:  {lineage.get('run_id')}  (seed {lineage.get('seed')})")
    print(
        f"distance: {lineage['distance']}  tolerance: {lineage['tolerance_method']} "
        f"@ p{lineage['null_percentile']}"
    )
    print(
        f"verdict: {'PASS' if verdict['passed'] else 'FAIL'} "
        f"({verdict['operating_rungs']} operating rungs of {verdict['rungs_total']})"
    )
    if problems:
        print("\nAUDIT FAILED: the committed verdict does not follow from the committed numbers:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nAUDIT OK: the committed verdict follows from the committed per-rung numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
