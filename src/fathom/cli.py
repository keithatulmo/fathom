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
import secrets
import sys
from datetime import UTC, datetime
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
    from .corpus.planners import PlanningError, plan_items
    from .corpus.sources import family_cap_gb, source_by_id
    from .ledger import Ledger

    spec = source_by_id(args.source_id)
    store = _open_store(args)
    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    scratch = repo_root / ".fathom" / "scratch"

    try:
        if args.key:
            # Explicit S3 keys for an anonymous-S3 source, bypassing the lister.
            items = plan_anon_s3_items(spec, list(args.key))
        else:
            try:
                items = plan_items(
                    spec,
                    prefix=args.prefix or "",
                    limit=args.limit,
                    urls=args.url,
                    index_url=args.index_url,
                    ais_start=args.date_start,
                    ais_end=args.date_end,
                    onc_location=args.onc_location,
                    onc_device_category=args.onc_device_category,
                    onc_extension=args.onc_extension,
                )
            except PlanningError as exc:
                raise SystemExit(str(exc)) from exc
        if not items:
            print(f"no objects found for {spec.source_id} with the given parameters")
            return 1

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
        counts = ledger.registry_vessel_counts_by_source()
    finally:
        ledger.close()

    report = build_audit_report(objects, vessel_counts=counts or None)
    out_dir = repo_root / ".fathom" / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_report.md").write_text(report.markdown, encoding="utf-8")
    (out_dir / "audit_table.json").write_text(
        json.dumps(report.table, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(report.markdown)
    print(f"\nwrote {out_dir / 'audit_report.md'} and {out_dir / 'audit_table.json'}")
    return 0


def _cmd_corpus_adequacy(args: argparse.Namespace, repo_root: Path) -> int:
    from .corpus.adequacy import score_adequacy
    from .ledger import Ledger

    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    try:
        objects = ledger.get_corpus_objects()
        presences = ledger.get_vessel_presences()
    finally:
        ledger.close()

    report = score_adequacy(objects, presences)
    out_dir = repo_root / ".fathom" / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "adequacy_scorecard.md").write_text(report.markdown, encoding="utf-8")
    (out_dir / "adequacy_table.json").write_text(
        json.dumps(report.table, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(report.markdown)
    print(f"\nSD1 close met: {report.sd1_close_met}")
    if report.blocking_failures:
        print(f"blocking (binding, not passing): {', '.join(report.blocking_failures)}")
    return 0 if report.sd1_close_met else 3


def _cmd_corpus_truth(args: argparse.Namespace, repo_root: Path) -> int:
    from .corpus.ais import AISRecord
    from .corpus.truth import (
        SITE_COORDS,
        ais_date_key,
        bbox_for,
        correlate_recording,
        read_ais_records,
        recording_date_key,
        recording_window,
    )
    from .ledger import Ledger

    if args.site not in SITE_COORDS:
        raise SystemExit(f"unknown site {args.site!r}; known sites: {', '.join(SITE_COORDS)}")
    lat, lon = SITE_COORDS[args.site]
    radius_m = args.radius_km * 1000.0
    bbox = bbox_for(lat, lon, radius_m)
    # A fresh run identifier so this correlation supersedes any earlier one under append-only.
    corr_run = f"corr-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"

    store = _open_store(args)
    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    try:
        objects = ledger.get_corpus_objects()
        # Load each AIS day once, filtered to the site's bounding box to keep memory bounded.
        ais_by_date: dict[str, tuple[list[AISRecord], str]] = {}
        for obj in objects:
            if obj["source_id"] != args.ais_source:
                continue
            key = ais_date_key(str(obj["origin_url"]))
            if key is None:
                continue
            blob = store.get_bytes(str(obj["raw_key"]))  # type: ignore[attr-defined]
            ais_by_date[key] = (read_ais_records(blob, bbox), str(obj["sha256"]))
            print(f"  loaded AIS {key}: {len(ais_by_date[key][0])} records within box")

        recorded = 0
        for obj in objects:
            if obj["source_id"] != args.source or obj["site"] != args.site:
                continue
            duration = obj["duration_s"]
            window = recording_window(
                str(obj["origin_url"]),
                None if duration is None else float(duration),  # type: ignore[arg-type]
            )
            key = recording_date_key(str(obj["origin_url"]))
            if window is None or key is None or key not in ais_by_date:
                continue
            ais_records, ais_sha = ais_by_date[key]
            rows = correlate_recording(
                corr_run=corr_run,
                recording_sha=str(obj["sha256"]),
                site=args.site,
                lat=lat,
                lon=lon,
                start_s=window[0],
                end_s=window[1],
                radius_m=radius_m,
                ais_records=ais_records,
                ais_source_sha=ais_sha,
            )
            for row in rows:
                ledger.insert_vessel_presence(record=row)
            recorded += len(rows)

        counts = ledger.registry_vessel_counts_by_source()
        quiet = ledger.quiet_tail_vessel_counts_by_source()
    finally:
        ledger.close()

    print(f"correlation run {corr_run}")
    print(f"  correlated {recorded} vessel-presence rows for {args.source} at {args.site}")
    print(f"  distinct registry-grade vessels by source: {counts}")
    print(f"  distinct quiet-tail vessels by source: {quiet}")
    return 0


def _cmd_corpus_probe(args: argparse.Namespace) -> int:
    from .corpus.fetch import anon_s3_list
    from .corpus.gcs import gcs_list
    from .corpus.sources import AccessClass, source_by_id

    spec = source_by_id(args.source_id)
    if spec.s3_bucket is None:
        print(f"{spec.source_id} has no listable bucket; provide explicit --url objects to acquire")
        return 1
    if spec.access_class == AccessClass.ANON_S3:
        keys = anon_s3_list(
            spec.s3_bucket, args.prefix or "", spec.s3_region or "us-east-1", max_keys=args.limit
        )
        origin = f"s3://{spec.s3_bucket}"
    elif spec.access_class == AccessClass.GCS_HTTPS:
        keys = gcs_list(spec.s3_bucket, args.prefix or "", max_keys=args.limit)
        origin = f"gs://{spec.s3_bucket}"
    else:
        print(f"{spec.source_id} ({spec.access_class}) is not listable unattended")
        return 1
    print(f"{spec.source_id}: {origin} reachable, {len(keys)} key(s) under prefix {args.prefix!r}:")
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
    acquire.add_argument("--key", action="append", help="Explicit S3 object key; repeatable.")
    acquire.add_argument("--url", action="append", help="Explicit object URL; repeatable.")
    acquire.add_argument("--prefix", default="", help="Key prefix to list (anon-S3 or GCS).")
    acquire.add_argument("--limit", type=int, default=1, help="Max objects to list.")
    acquire.add_argument("--index-url", help="OOI archive directory index URL to enumerate.")
    acquire.add_argument("--date-start", help="Range start (YYYY-MM-DD) for AIS or ONC.")
    acquire.add_argument("--date-end", help="Range end (YYYY-MM-DD) for AIS or ONC.")
    acquire.add_argument("--onc-location", help="ONC location code.")
    acquire.add_argument("--onc-device-category", help="ONC device category code.")
    acquire.add_argument("--onc-extension", help="Restrict ONC files to an extension, e.g. flac.")
    acquire.add_argument("--r2", action="store_true", help="Destination is the R2 bucket (env).")
    acquire.add_argument("--local", help="Destination is a local directory (offline).")
    acquire.add_argument("--no-deep-verify", action="store_true", help="Skip re-hash verification.")
    acquire.add_argument("--no-cap", action="store_true", help="Ignore the family volume cap.")

    audit = sub.add_parser("corpus-audit", help="Generate the SD1 corpus audit report.")
    audit.add_argument("--r2", action="store_true", help="(reserved) publish to R2.")
    audit.add_argument("--local", help="(reserved) publish to a local directory.")

    sub.add_parser(
        "corpus-adequacy", help="Score the corpus against the CA1-CA7 adequacy criteria."
    )

    truth = sub.add_parser(
        "corpus-truth", help="Correlate AIS to recordings into tier-one vessel truth."
    )
    truth.add_argument("--source", default="mbari_pacific_sound_2khz", help="Acoustic source id.")
    truth.add_argument("--ais-source", default="marinecadastre_ais", help="AIS source id.")
    truth.add_argument("--site", default="mars_monterey_bay", help="Site key with known coords.")
    truth.add_argument("--radius-km", type=float, default=20.0, help="Correlation radius (km).")
    truth.add_argument("--r2", action="store_true", help="Read AIS from the R2 bucket (env).")
    truth.add_argument("--local", help="Read AIS from a local directory.")

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
    if args.command == "corpus-adequacy":
        return _cmd_corpus_adequacy(args, repo_root)
    if args.command == "corpus-truth":
        return _cmd_corpus_truth(args, repo_root)
    if args.command == "corpus-probe":
        return _cmd_corpus_probe(args)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
