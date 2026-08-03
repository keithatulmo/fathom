"""The Fathom command-line interface.

SD10 Section 6.6 fixes two entry points on one mechanism. ``fathom run <manifest>`` executes a run
and writes its scored report with full lineage. ``fathom repro <run-id>`` re-executes a recorded
run from the ledger and verifies every output hash, reporting any mismatch as a failure rather than
a warning. ``fathom demo`` is a declared stub that states it is not yet implemented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifest import Manifest
from .runner import Runner


def find_repo_root(start: Path) -> Path:
    """Return the nearest ancestor of ``start`` containing a pyproject.toml, or ``start``."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def _cmd_run(manifest_path: Path, repo_root: Path) -> int:
    manifest = Manifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    runner = Runner(repo_root)
    try:
        result = runner.run(manifest)
    finally:
        runner.close()

    print(f"run {result.run_id}")
    print(f"  reportable: {result.reportable}")
    if result.report_path is not None:
        document = json.loads(result.report_path.read_text(encoding="utf-8"))
        report = document["report"]
        if result.reportable:
            point = report["reject_versus_miss"]["operating_point"]
            print(
                f"  miss rate: {point['miss_rate']:.4f} "
                f"(exact upper {point['miss_rate_upper']:.4f} at alpha {point['alpha']})"
            )
            print(f"  rejection fraction: {point['rejection_fraction']:.4f}")
            print(f"  calibration ECE: {report['calibration']['ece']:.4f}")
            print(f"  coverage fraction: {report['coverage']['fraction']:.4f}")
        else:
            print(f"  reason: {report.get('reason', 'unknown')}")
        print(f"  report: {result.report_path}")
    return 0 if result.reportable else 2


def _cmd_repro(run_id: str, repo_root: Path) -> int:
    runner = Runner(repo_root)
    try:
        result = runner.reproduce(run_id)
    finally:
        runner.close()

    if result.ok:
        print(f"reproduced {run_id}: checked {result.checked} artifacts, 0 mismatches")
        return 0
    print(f"reproduction FAILED for {run_id}: {len(result.mismatches)} mismatch(es)")
    for mismatch in result.mismatches:
        print(
            f"  {mismatch.node_id}:{mismatch.output} "
            f"expected {mismatch.expected[:16]} got {mismatch.actual[:16]}"
        )
    return 1


def _cmd_demo() -> int:
    print("fathom demo is not yet implemented.")
    print("It is a declared stub reserved for the assembled proof demonstration in later work.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the Fathom CLI."""
    parser = argparse.ArgumentParser(prog="fathom", description="Fathom evaluation harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Execute a manifest end to end and score it.")
    run.add_argument("manifest", type=Path, help="Path to the run manifest (JSON).")

    repro = sub.add_parser("repro", help="Re-execute a recorded run and verify output hashes.")
    repro.add_argument("run_id", help="Identifier of the run to reproduce.")

    sub.add_parser("demo", help="Declared stub; not yet implemented.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``fathom`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = find_repo_root(Path.cwd())

    if args.command == "run":
        return _cmd_run(args.manifest.resolve(), repo_root)
    if args.command == "repro":
        return _cmd_repro(args.run_id, repo_root)
    if args.command == "demo":
        return _cmd_demo()
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
