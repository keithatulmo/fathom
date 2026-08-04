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


def _cmd_fetch_quiet_tail(args: argparse.Namespace, repo_root: Path) -> int:
    from .corpus.acquire import AcquisitionItem, acquire_batch
    from .corpus.quiet_tail import (
        QuietTailFile,
        covered_vessels,
        select_files,
        select_files_by_closeness,
    )
    from .corpus.sources import FIRST_WAVE, family_cap_gb, source_by_id
    from .ledger import Ledger

    store = _open_store(args)
    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    scratch = repo_root / ".fathom" / "scratch"
    try:
        objects = ledger.get_corpus_objects()
        object_shas = {str(o["sha256"]) for o in objects}
        presences = ledger.get_vessel_presences()
        # Optionally exclude vessels already audio-backed, so a top-up fetch pursues new ones.
        audible = (
            {str(p["vessel_id"]) for p in presences if p.get("audio_quiet_tail") == 1}
            if args.exclude_audible
            else set()
        )
        # Window-only quiet-tail passages: flagged quiet, but the recording is a registered window
        # rather than a downloaded object, so the vessel has no audio behind it yet. Track the
        # closest-approach range so files with near (audible) passages can be prioritised.
        vessels_by_recording: dict[str, set[str]] = {}
        range_by_recording: dict[str, float] = {}
        for p in presences:
            if p.get("quiet_tail") != 1:
                continue
            rec = str(p["recording_sha"])
            vid = str(p["vessel_id"])
            if rec in object_shas or vid in audible:
                continue
            vessels_by_recording.setdefault(rec, set()).add(vid)
            rng = float(p["closest_range_m"])  # type: ignore[arg-type]
            range_by_recording[rec] = min(range_by_recording.get(rec, rng), rng)

        # Resolve each window's origin URL, source, site, and rate so a file can be fetched.
        candidates: list[QuietTailFile] = []
        for spec in FIRST_WAVE:
            for win in ledger.get_recording_windows(spec.source_id):
                wid = str(win["window_id"])
                if wid not in vessels_by_recording:
                    continue
                candidates.append(
                    QuietTailFile(
                        source_id=spec.source_id,
                        origin_url=str(win["origin_url"]),
                        site=str(win["site"]),
                        sample_rate_hz=(
                            None
                            if win["sample_rate_hz"] is None
                            else float(win["sample_rate_hz"])  # type: ignore[arg-type]
                        ),
                        vessels=frozenset(vessels_by_recording[wid]),
                        min_range_m=range_by_recording[wid],
                    )
                )

        if args.prioritize == "closest":
            chosen = select_files_by_closeness(
                candidates, args.target_vessels, max_files=args.max_files
            )
        else:
            chosen = select_files(candidates, args.target_vessels, max_files=args.max_files)
        print(
            f"selected {len(chosen)} file(s) covering {len(covered_vessels(chosen))} distinct "
            f"quiet-tail vessels (target {args.target_vessels}, by {args.prioritize}) from "
            f"{len(candidates)} candidates"
        )
        if not chosen:
            return 1

        by_source: dict[str, list[QuietTailFile]] = {}
        for f in chosen:
            by_source.setdefault(f.source_id, []).append(f)

        total_uploaded = 0
        total_bytes = 0
        for source_id, files in sorted(by_source.items()):
            spec = source_by_id(source_id)
            items = [
                AcquisitionItem(
                    origin_url=f.origin_url,
                    url=f.origin_url,
                    site=f.site,
                    sample_rate_hz=f.sample_rate_hz,
                )
                for f in files
            ]
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
            )
            total_uploaded += result.uploaded
            total_bytes += result.bytes_stored
            note = "  CAP REACHED" if result.cap_reached else ""
            print(
                f"  {source_id}: uploaded {result.uploaded}, "
                f"skipped {result.skipped_existing}, {result.bytes_stored:,} bytes{note}"
            )
    finally:
        ledger.close()
    print(f"fetched {total_uploaded} quiet-tail recording(s), {total_bytes / 1e9:.2f} GB stored")
    print("next: re-run corpus-truth to correlate the fetched audio into audio-backed presences")
    return 0


def _cmd_audio_lineage(args: argparse.Namespace, repo_root: Path) -> int:
    from .ledger import Ledger

    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    try:
        objects = {str(o["sha256"]): o for o in ledger.get_corpus_objects()}
        audio_shas = {
            sha
            for sha, o in objects.items()
            if o["duration_s"] is not None and o["sample_rate_hz"] is not None
        }
        splits = {str(s["vessel_id"]): str(s["split"]) for s in ledger.get_vessel_splits()}
        best: dict[str, dict[str, object]] = {}
        best_snr: dict[str, float] = {}
        for p in ledger.get_vessel_presences():
            if p.get("quiet_tail") != 1:
                continue
            rec = str(p["recording_sha"])
            raw_snr = p.get("audio_snr_db")
            if rec not in audio_shas or raw_snr is None:
                continue
            snr = round(float(raw_snr), 3)  # type: ignore[arg-type]
            if snr < args.snr_db:
                continue
            vessel = str(p["vessel_id"])
            if vessel not in best_snr or snr > best_snr[vessel]:
                best_snr[vessel] = snr
                o = objects[rec]
                best[vessel] = {
                    "vessel_id": vessel,
                    "name": p.get("name"),
                    "backing_object_sha256": rec,
                    "source_id": o["source_id"],
                    "site": p.get("site"),
                    "sample_rate_hz": o["sample_rate_hz"],
                    "duration_s": o["duration_s"],
                    "closest_range_m": p.get("closest_range_m"),
                    "audio_snr_db": snr,
                    "corr_run": p.get("corr_run"),
                    "split": splits.get(vessel),
                }
        rows = sorted(
            best.values(), key=lambda r: (-best_snr[str(r["vessel_id"])], str(r["vessel_id"]))
        )
        doc = {
            "snr_threshold_db": args.snr_db,
            "audio_backed_quiet_tail_vessels": len(rows),
            "note": "Each vessel's audio-backed quiet-tail passage: the stored object that backs "
            "it, its sample rate and duration, and the measured in-band SNR. Regenerate with "
            "`fathom audio-lineage`; the object shas resolve against the R2 store.",
            "vessels": rows,
        }
        out_path = (repo_root / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        ledger.close()
    print(f"wrote audio-backing lineage for {len(rows)} vessels to {out_path}")
    return 0


def _cmd_capture_audio(args: argparse.Namespace, repo_root: Path) -> int:
    import io

    import soundfile as sf

    from .ledger import Ledger

    store = _open_store(args)
    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    recorded = 0
    failed = 0
    try:
        for obj in ledger.get_corpus_objects():
            media = obj.get("media_type")
            if not (isinstance(media, str) and media.startswith("audio/")):
                continue
            # Only measure objects whose sample rate was not captured at acquisition (ADEON, ONC);
            # the rest already carry a recorded rate on the object row.
            if obj.get("sample_rate_hz") is not None:
                continue
            try:
                info = sf.info(io.BytesIO(store.get_bytes(str(obj["raw_key"]))))  # type: ignore[attr-defined]
            except Exception:
                failed += 1
                continue
            ledger.insert_object_audio(
                sha256=str(obj["sha256"]),
                sample_rate_hz=float(info.samplerate),
                duration_s=float(info.duration),
                channels=int(info.channels),
            )
            recorded += 1
        total = ledger.count_object_audio()
    finally:
        ledger.close()
    print(f"measured audio headers for {recorded} object(s), {failed} undecodable (total {total})")
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
        splits = ledger.get_vessel_splits()
        window_rates = ledger.get_window_source_rates()
        measured_rates = ledger.measured_band_source_rates()
    finally:
        ledger.close()

    report = score_adequacy(
        objects,
        presences,
        splits=splits or None,
        window_source_rates=window_rates or None,
        measured_source_rates=measured_rates or None,
        snr_threshold_db=args.snr_db,
    )
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


def _cmd_register_windows(args: argparse.Namespace, repo_root: Path) -> int:
    from .corpus.gcs import gcs_list, gcs_object_url
    from .corpus.sources import AccessClass, source_by_id
    from .corpus.truth import recording_date_key, recording_window
    from .hashing import sha256_hex
    from .ledger import Ledger

    spec = source_by_id(args.source_id)
    if spec.access_class != AccessClass.GCS_HTTPS or not spec.s3_bucket:
        raise SystemExit(f"{spec.source_id} is not a GCS source; window registration needs listing")
    keys = gcs_list(spec.s3_bucket, args.prefix or "", max_keys=args.limit)
    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    registered = 0
    try:
        for key in keys:
            if not key.lower().endswith((".flac", ".wav")):
                continue
            origin_url = gcs_object_url(spec.s3_bucket, key)
            # Filter to a date range so the same window can be registered across sites, letting one
            # (US-wide) AIS day serve every site.
            if args.date_start or args.date_end:
                date_key = recording_date_key(origin_url)
                iso = None if date_key is None else date_key.replace("_", "-")
                if iso is None or (args.date_start and iso < args.date_start):
                    continue
                if args.date_end and iso > args.date_end:
                    continue
            window = recording_window(origin_url, args.window_seconds)
            if window is None:
                continue
            ledger.insert_recording_window(
                record={
                    "window_id": sha256_hex(origin_url.encode()),
                    "source_id": spec.source_id,
                    "origin_url": origin_url,
                    "site": spec.site,
                    "sample_rate_hz": spec.sample_rate_hz,
                    "start_epoch_s": window[0],
                    "end_epoch_s": window[1],
                }
            )
            registered += 1
        total = ledger.count_recording_windows(spec.source_id)
    finally:
        ledger.close()
    print(f"registered {registered} recording windows for {spec.source_id} (total {total})")
    return 0


def _cmd_assign_splits(args: argparse.Namespace, repo_root: Path) -> int:
    from .corpus.quiet_tail import audio_backed_vessels
    from .corpus.splits import assign_splits
    from .corpus.truth import QUIET_TAIL_SNR_DB_MIN
    from .ledger import Ledger

    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    try:
        objects = ledger.get_corpus_objects()
        audio_shas = {
            str(o["sha256"])
            for o in objects
            if o["duration_s"] is not None and o["sample_rate_hz"] is not None
        }
        presences = ledger.get_vessel_presences()
        registry = {str(p["vessel_id"]) for p in presences if p["registry_grade"] == 1}
        # The quiet-tail split seeds from the audio-backed cohort, derived from persisted state:
        # only vessels whose closest approach is a stored audio object audible in-band validate the
        # surrogate (SD2, E1). Same derivation the scorer uses, so the split and audit agree.
        quiet = audio_backed_vessels(presences, audio_shas, QUIET_TAIL_SNR_DB_MIN)
        assign_run = f"assign-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"
        rows, audit = assign_splits(registry, quiet, assign_run)
        for row in rows:
            ledger.insert_vessel_split(record=row)
    finally:
        ledger.close()

    out_dir = repo_root / ".fathom" / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"assigned {len(rows)} vessels to splits (run {assign_run})")
    print(f"  vessel counts:     {audit['vessel_counts']}")
    print(f"  quiet-tail counts: {audit['quiet_tail_counts']}")
    print(
        f"  disjoint: {audit['disjoint']}  meets CA2: {audit['meets_ca2']}  "
        f"meets CA3: {audit['meets_ca3']}"
    )
    return 0


def _cmd_corpus_truth(args: argparse.Namespace, repo_root: Path) -> int:
    import math

    from .corpus.ais import AISRecord
    from .corpus.snr import measure_inband_snr
    from .corpus.truth import (
        QUIET_TAIL_SNR_DB_MIN,
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
    # Share a run identifier across sites (pass the same --corr-run for each) so one correlation
    # pass spans every site; otherwise a fresh identifier supersedes prior runs under append-only.
    corr_run = (
        args.corr_run
        or f"corr-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(2)}"
    )

    store = _open_store(args)
    ledger = Ledger(repo_root / ".fathom" / "ledger.db")
    try:
        objects = ledger.get_corpus_objects()
        # Recordings to correlate come from real acquired objects (audio present, so its raw key is
        # carried for the SNR measurement) and from registered windows (audio not downloaded). A
        # window whose audio has since been fetched is superseded by its object and is not
        # correlated again as an audio-less recording.
        recordings: list[tuple[str, str, float, float, str | None]] = []
        object_origins: set[str] = set()
        for obj in objects:
            if obj["source_id"] != args.source or obj["site"] != args.site:
                continue
            duration = obj["duration_s"]
            window = recording_window(
                str(obj["origin_url"]),
                None if duration is None else float(duration),  # type: ignore[arg-type]
            )
            if window is not None:
                object_origins.add(str(obj["origin_url"]))
                recordings.append(
                    (str(obj["sha256"]), str(obj["origin_url"]), *window, str(obj["raw_key"]))
                )
        # In audio-only mode only the downloaded objects are correlated, to measure SNR and add
        # audio-backed presences to an existing run without re-parsing every window's AIS day.
        windows = [] if args.audio_only else ledger.get_recording_windows(args.source)
        for win in windows:
            if win["site"] == args.site and str(win["origin_url"]) not in object_origins:
                recordings.append(
                    (
                        str(win["window_id"]),
                        str(win["origin_url"]),
                        float(win["start_epoch_s"]),  # type: ignore[arg-type]
                        float(win["end_epoch_s"]),  # type: ignore[arg-type]
                        None,
                    )
                )

        # Load only the AIS days these recordings need, filtered to the site's bounding box.
        needed = {recording_date_key(origin) for _, origin, _, _, _ in recordings} - {None}
        ais_by_date: dict[str, tuple[list[AISRecord], str]] = {}
        for obj in objects:
            if obj["source_id"] != args.ais_source:
                continue
            key = ais_date_key(str(obj["origin_url"]))
            if key is None or key not in needed:
                continue
            blob = store.get_bytes(str(obj["raw_key"]))  # type: ignore[attr-defined]
            ais_by_date[key] = (read_ais_records(blob, bbox), str(obj["sha256"]))
            print(f"  loaded AIS {key}: {len(ais_by_date[key][0])} records within box")

        recorded = 0
        snr_measured = 0
        for recording_id, origin_url, start_s, end_s, raw_key in recordings:
            key = recording_date_key(origin_url)
            if key is None or key not in ais_by_date:
                continue
            ais_records, ais_sha = ais_by_date[key]
            rows = correlate_recording(
                corr_run=corr_run,
                recording_sha=recording_id,
                site=args.site,
                lat=lat,
                lon=lon,
                start_s=start_s,
                end_s=end_s,
                radius_m=radius_m,
                ais_records=ais_records,
                ais_source_sha=ais_sha,
            )
            # For a recording whose audio is present, measure the in-band SNR at each quiet-tail
            # passage's closest approach: the vessel counts as audio-backed quiet-tail only when it
            # is audible, not merely kinematically slow and close. The blob is fetched once per
            # recording and only when a quiet-tail passage is present.
            if raw_key is not None and any(row["quiet_tail"] for row in rows):
                blob = store.get_bytes(raw_key)  # type: ignore[attr-defined]
                for row in rows:
                    if not row["quiet_tail"] or row["closest_time_s"] is None:
                        continue
                    offset_s = float(row["closest_time_s"]) - start_s  # type: ignore[arg-type]
                    result = measure_inband_snr(blob, offset_s)
                    audible = (
                        math.isfinite(result.snr_db) and result.snr_db >= QUIET_TAIL_SNR_DB_MIN
                    )
                    row["audio_snr_db"] = None if math.isnan(result.snr_db) else result.snr_db
                    row["audio_quiet_tail"] = int(audible)
                    snr_measured += 1
            for row in rows:
                ledger.insert_vessel_presence(record=row)
            recorded += len(rows)

        counts = ledger.registry_vessel_counts_by_source()
        quiet = ledger.quiet_tail_vessel_counts_by_source()
        audio_quiet = ledger.audio_quiet_tail_vessel_counts_by_source()
    finally:
        ledger.close()

    print(f"correlation run {corr_run}")
    print(f"  correlated {recorded} vessel-presence rows for {args.source} at {args.site}")
    print(f"  measured in-band SNR for {snr_measured} quiet-tail passage(s)")
    print(f"  distinct registry-grade vessels by source: {counts}")
    print(f"  distinct quiet-tail vessels by source: {quiet}")
    print(f"  distinct AUDIO-BACKED quiet-tail vessels by source: {audio_quiet}")
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

    fetchqt = sub.add_parser(
        "fetch-quiet-tail",
        help="Download audio for the kinematically-flagged quiet-tail passages (bounded set).",
    )
    fetchqt.add_argument(
        "--target-vessels", type=int, default=60, help="Distinct quiet-tail vessels to cover."
    )
    fetchqt.add_argument(
        "--max-files", type=int, default=80, help="File-count budget for the fetch."
    )
    fetchqt.add_argument(
        "--prioritize",
        choices=("count", "closest"),
        default="count",
        help="Select files by distinct-vessel count (default) or by closest passage first.",
    )
    fetchqt.add_argument(
        "--exclude-audible",
        action="store_true",
        help="Skip vessels already audio-backed, so a top-up fetch pursues new ones.",
    )
    fetchqt.add_argument("--r2", action="store_true", help="Destination is the R2 bucket (env).")
    fetchqt.add_argument("--local", help="Destination is a local directory (offline).")
    fetchqt.add_argument("--no-cap", action="store_true", help="Ignore the family volume cap.")

    capaudio = sub.add_parser(
        "capture-audio", help="Measure and persist sample rate/duration from stored audio headers."
    )
    capaudio.add_argument("--r2", action="store_true", help="Read audio from the R2 bucket (env).")
    capaudio.add_argument("--local", help="Read audio from a local directory.")

    lineage = sub.add_parser(
        "audio-lineage", help="Export the audio-backed quiet-tail cohort's lineage as JSON."
    )
    lineage.add_argument(
        "--snr-db", type=float, default=6.0, help="Audibility bar (in-band SNR dB)."
    )
    lineage.add_argument(
        "--out", default="docs/audio_backing_lineage.json", help="Output path (repo-relative)."
    )

    audit = sub.add_parser("corpus-audit", help="Generate the SD1 corpus audit report.")
    audit.add_argument("--r2", action="store_true", help="(reserved) publish to R2.")
    audit.add_argument("--local", help="(reserved) publish to a local directory.")

    adequacy = sub.add_parser(
        "corpus-adequacy", help="Score the corpus against the CA1-CA7 adequacy criteria."
    )
    adequacy.add_argument(
        "--snr-db",
        type=float,
        default=6.0,
        help="Audibility bar (in-band SNR dB) an audio-backed quiet-tail vessel must clear.",
    )

    sub.add_parser("assign-splits", help="Assign vessels to train/calibration/test splits (SD3).")

    regwin = sub.add_parser(
        "register-windows", help="Register recording windows for correlation without audio."
    )
    regwin.add_argument("source_id", help="GCS acoustic source identifier.")
    regwin.add_argument("--prefix", default="", help="Audio object prefix to list.")
    regwin.add_argument("--limit", type=int, default=2000, help="Max files to list.")
    regwin.add_argument("--date-start", help="Register windows on/after this date (YYYY-MM-DD).")
    regwin.add_argument("--date-end", help="Register windows on/before this date (YYYY-MM-DD).")
    regwin.add_argument(
        "--window-seconds", type=float, default=21600.0, help="Window duration (default 6 h)."
    )

    truth = sub.add_parser(
        "corpus-truth", help="Correlate AIS to recordings into tier-one vessel truth."
    )
    truth.add_argument("--source", default="mbari_pacific_sound_2khz", help="Acoustic source id.")
    truth.add_argument("--ais-source", default="marinecadastre_ais", help="AIS source id.")
    truth.add_argument("--site", default="mars_monterey_bay", help="Site key with known coords.")
    truth.add_argument("--radius-km", type=float, default=20.0, help="Correlation radius (km).")
    truth.add_argument("--corr-run", help="Shared correlation run id, to span sites in one pass.")
    truth.add_argument(
        "--audio-only",
        action="store_true",
        help="Correlate only downloaded audio objects (measure SNR), not registered windows.",
    )
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
    if args.command == "fetch-quiet-tail":
        return _cmd_fetch_quiet_tail(args, repo_root)
    if args.command == "capture-audio":
        return _cmd_capture_audio(args, repo_root)
    if args.command == "audio-lineage":
        return _cmd_audio_lineage(args, repo_root)
    if args.command == "corpus-audit":
        return _cmd_corpus_audit(args, repo_root)
    if args.command == "corpus-adequacy":
        return _cmd_corpus_adequacy(args, repo_root)
    if args.command == "assign-splits":
        return _cmd_assign_splits(args, repo_root)
    if args.command == "register-windows":
        return _cmd_register_windows(args, repo_root)
    if args.command == "corpus-truth":
        return _cmd_corpus_truth(args, repo_root)
    if args.command == "corpus-probe":
        return _cmd_corpus_probe(args)
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
