"""The Fathom command-line interface.

SD10 Section 6.6 fixes two evaluation entry points on one mechanism. ``fathom run <manifest>``
executes a run and writes its scored report with full lineage. ``fathom repro <run-id>`` re-executes
a recorded run from the ledger and verifies every output hash, reporting any mismatch as a failure
rather than a warning. ``fathom demo`` is a declared stub that states it is not yet implemented.

The corpus commands implement the WO-2 data thread. ``fathom acquire`` acquires a source's objects
into the object store with manifests and ledger rows, ``fathom corpus-audit`` generates the SD1
audit report from the ledger, and ``fathom corpus-probe`` lists a few keys from a public source to
verify reachability. Object-store credentials are read from the environment only, never the repo.
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


# -- evaluation commands --------------------------------------------------------------------


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


# -- corpus commands ------------------------------------------------------------------------


def _open_store(args: argparse.Namespace) -> object:
    from .corpus.objectstore import LocalObjectStore, R2ObjectStore

    if getattr(args, "r2", False):
        return R2ObjectStore.from_env()
    if getattr(args, "local", None):
        return LocalObjectStore(Path(args.local))
    raise SystemExit("specify a destination: --r2 or --local DIR")


def _family_prior_bytes(ledger: object, family: int) -> int:
    from .corpus.sources import FIRST_WAVE

    family_of = {spec.source_id: spec.family for spec in FIRST_WAVE}
    total = 0
    for obj in ledger.get_corpus_objects():  # type: ignore[attr-defined]
        if family_of.get(str(obj["source_id"])) == family:
            total += int(obj["byte_count"])
    return total


def _cmd_acquire(args: argparse.Namespace, repo_root: Path) -> int:
    from .corpus.acquire import acquire_batch, plan_anon_s3_items
    from .corpus.fetch import anon_s3_list
    from .corpus.sources import AccessClass, family_cap_gb, source_by_id
    from .ledger import Ledger

    spec = source_by_id(args.source_id)
    store = _open_store(args)
    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    scratch = repo_root / ".fathom" / "scratch"

    try:
        if args.key:
            keys = list(args.key)
        elif spec.access_class == AccessClass.ANON_S3 and spec.s3_bucket:
            keys = anon_s3_list(
                spec.s3_bucket,
                args.prefix or "",
                spec.s3_region or "us-east-1",
                max_keys=args.limit,
            )
        else:
            raise SystemExit(
                f"source {spec.source_id!r} needs explicit --key(s); only anon-S3 sources list"
            )
        if not keys:
            print(f"no keys found for {spec.source_id} (prefix {args.prefix!r})")
            return 1

        items = plan_anon_s3_items(spec, keys)
        cap_bytes = None if args.no_cap else int(family_cap_gb(spec.family) * 1e9)
        prior = 0 if args.no_cap else _family_prior_bytes(ledger, spec.family)
        result = acquire_batch(
            spec,
            items,
            store,  # type: ignore[arg-type]
            scratch,
            ledger=ledger,
            cap_bytes=cap_bytes,
            prior_bytes=prior,
            deep_verify=not args.no_deep_verify,
        )
    finally:
        ledger.close()

    print(f"acquired batch {result.manifest.batch_id} for {spec.source_id}")
    print(f"  uploaded: {result.uploaded}  skipped-existing: {result.skipped_existing}")
    print(f"  bytes stored this batch: {result.bytes_stored:,}")
    if result.cap_reached:
        print(f"  CAP REACHED for family {spec.family}: held {len(result.held_for_cap)} object(s)")
    print(f"  manifest objects: {len(result.manifest.objects)}")
    return 0


def _cmd_corpus_audit(args: argparse.Namespace, repo_root: Path) -> int:
    from .corpus.audit import build_audit_report
    from .ledger import Ledger

    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    try:
        objects = ledger.get_corpus_objects()
    finally:
        ledger.close()

    report = build_audit_report(objects)
    out_dir = repo_root / ".fathom" / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_report.md").write_text(report.markdown, encoding="utf-8")
    (out_dir / "audit_table.json").write_text(
        json.dumps(report.table, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(report.markdown)
    print(f"\nwrote {out_dir / 'audit_report.md'} and {out_dir / 'audit_table.json'}")
    return 0


def _cmd_corpus_probe(args: argparse.Namespace) -> int:
    from .corpus.fetch import anon_s3_list
    from .corpus.sources import AccessClass, source_by_id

    spec = source_by_id(args.source_id)
    if spec.access_class != AccessClass.ANON_S3 or not spec.s3_bucket:
        print(f"{spec.source_id} is not an anonymous-S3 source; nothing to probe unattended")
        return 1
    keys = anon_s3_list(
        spec.s3_bucket, args.prefix or "", spec.s3_region or "us-east-1", max_keys=args.limit
    )
    print(
        f"{spec.source_id}: s3://{spec.s3_bucket} reachable, {len(keys)} key(s) under prefix "
        f"{args.prefix!r}:"
    )
    for key in keys:
        print(f"  {key}")
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

    acquire = sub.add_parser("acquire", help="Acquire a source's objects into the object store.")
    acquire.add_argument("source_id", help="First-wave source identifier.")
    acquire.add_argument("--key", action="append", help="Explicit object key; repeatable.")
    acquire.add_argument("--prefix", default="", help="Key prefix to list (anon-S3 sources).")
    acquire.add_argument("--limit", type=int, default=1, help="Max keys to list.")
    acquire.add_argument("--r2", action="store_true", help="Destination is the R2 bucket (env).")
    acquire.add_argument("--local", help="Destination is a local directory (offline).")
    acquire.add_argument("--no-deep-verify", action="store_true", help="Skip re-hash verification.")
    acquire.add_argument("--no-cap", action="store_true", help="Ignore the family volume cap.")

    audit = sub.add_parser("corpus-audit", help="Generate the SD1 corpus audit report.")
    audit.add_argument("--r2", action="store_true", help="(reserved) publish to R2.")
    audit.add_argument("--local", help="(reserved) publish to a local directory.")

    probe = sub.add_parser("corpus-probe", help="List a few keys from a public source (read-only).")
    probe.add_argument("source_id", help="First-wave source identifier.")
    probe.add_argument("--prefix", default="", help="Key prefix to list.")
    probe.add_argument("--limit", type=int, default=10, help="Max keys to list.")
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
    if args.command == "acquire":
        return _cmd_acquire(args, repo_root)
    if args.command == "corpus-audit":
        return _cmd_corpus_audit(args, repo_root)
    if args.command == "corpus-probe":
        return _cmd_corpus_probe(args)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
