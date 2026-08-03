"""The acquisition driver.

This is where a source's objects become stored, recorded corpus. For each object the driver fetches
to scratch while hashing, stores it under its content-addressed ``raw/`` key only if it is not
already present, verifies the stored bytes, records it in the batch manifest and, when a ledger is
supplied, in the corpus ledger with its license and truth condition. Re-running is idempotent
because a present content hash is skipped (AC4). A per-family volume cap is respected: the object
that would cross the cap is not stored and the near-miss is reported rather than silently exceeded
(AC3). A source whose dataset is quarantined is never acquired, and every stored object carries the
mechanical training-eligibility flag that enforces the license rule (AC2).
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..ledger import Ledger
from .fetch import FetchedFile, fetch_anon_s3, fetch_https
from .manifest import AcquisitionManifest, ObjectRecord
from .objectstore import ObjectStore, manifest_key, raw_key
from .sources import SourceSpec

# Objects at or below this size are re-read and re-hashed after upload for end-to-end verification;
# larger objects are verified by stored size, since a full re-download would double the bandwidth.
DEEP_VERIFY_MAX_BYTES = 64 << 20


class AcquisitionError(RuntimeError):
    """Raised when acquisition cannot proceed, for example a quarantined dataset."""


class VerificationError(RuntimeError):
    """Raised when a stored object fails its post-upload verification."""


@dataclass(frozen=True)
class AcquisitionItem:
    """One object to acquire: an HTTPS URL or an anonymous-S3 locator, plus per-object truth."""

    origin_url: str
    url: str | None = None
    s3_bucket: str | None = None
    s3_key: str | None = None
    s3_region: str | None = None
    truth_condition: str | None = None
    site: str | None = None
    instrument: str | None = None
    sample_rate_hz: float | None = None
    band_low_hz: float | None = None
    band_high_hz: float | None = None
    band_partial: bool = False


@dataclass
class AcquisitionResult:
    """The outcome of acquiring a batch: what was stored, skipped, and held back for the cap."""

    manifest: AcquisitionManifest
    uploaded: int = 0
    skipped_existing: int = 0
    bytes_stored: int = 0
    cap_reached: bool = False
    held_for_cap: list[str] = field(default_factory=list)


def plan_anon_s3_items(spec: SourceSpec, keys: list[str]) -> list[AcquisitionItem]:
    """Build acquisition items for an anonymous-S3 source from a list of object keys."""
    if spec.s3_bucket is None:
        raise AcquisitionError(f"source {spec.source_id!r} has no S3 bucket configured")
    region = spec.s3_region or "us-east-1"
    return [
        AcquisitionItem(
            origin_url=f"s3://{spec.s3_bucket}/{key}",
            s3_bucket=spec.s3_bucket,
            s3_key=key,
            s3_region=region,
        )
        for key in keys
    ]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _make_batch_id(source_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{source_id}-{stamp}-{secrets.token_hex(2)}"


def _fetch(item: AcquisitionItem, scratch_dir: Path) -> FetchedFile:
    if item.s3_bucket is not None and item.s3_key is not None:
        region = item.s3_region or "us-east-1"
        return fetch_anon_s3(item.s3_bucket, item.s3_key, region, scratch_dir)
    return fetch_https(item.url or item.origin_url, scratch_dir)


def _verify(store: ObjectStore, key: str, fetched: FetchedFile, *, deep: bool) -> None:
    stored = store.size(key)
    if stored != fetched.byte_count:
        raise VerificationError(
            f"stored size {stored} does not match fetched size {fetched.byte_count} for {key}"
        )
    if deep and fetched.byte_count <= DEEP_VERIFY_MAX_BYTES:
        import hashlib

        digest = hashlib.sha256(store.get_bytes(key)).hexdigest()
        if digest != fetched.sha256:
            raise VerificationError(
                f"stored bytes for {key} hash to {digest}, expected {fetched.sha256}"
            )


def _record(item: AcquisitionItem, spec: SourceSpec, fetched: FetchedFile) -> ObjectRecord:
    return ObjectRecord(
        source_id=spec.source_id,
        origin_url=item.origin_url,
        retrieved_at=_now(),
        byte_count=fetched.byte_count,
        sha256=fetched.sha256,
        raw_key=raw_key(fetched.sha256),
        license_class=spec.license_class,
        license_evidence_url=spec.license_evidence_url,
        truth_condition=item.truth_condition or spec.truth_condition_default,
        training_eligible=spec.training_eligible,
        media_type=fetched.media_type,
        site=item.site or spec.site,
        instrument=item.instrument or spec.instrument,
        sample_rate_hz=item.sample_rate_hz or spec.sample_rate_hz,
        band_low_hz=item.band_low_hz or spec.band_low_hz,
        band_high_hz=item.band_high_hz or spec.band_high_hz,
        band_partial=item.band_partial or spec.band_partial,
    )


def _ledger_record(record: ObjectRecord) -> dict[str, object]:
    return {
        "sha256": record.sha256,
        "source_id": record.source_id,
        "origin_url": record.origin_url,
        "retrieved_at": record.retrieved_at,
        "byte_count": record.byte_count,
        "raw_key": record.raw_key,
        "license_class": record.license_class.value,
        "license_evidence_url": record.license_evidence_url,
        "truth_condition": record.truth_condition,
        "training_eligible": int(record.training_eligible),
        "media_type": record.media_type,
        "site": record.site,
        "instrument": record.instrument,
        "sample_rate_hz": record.sample_rate_hz,
        "band_low_hz": record.band_low_hz,
        "band_high_hz": record.band_high_hz,
        "band_partial": int(record.band_partial),
        "duration_s": record.duration_s,
    }


def acquire_batch(
    spec: SourceSpec,
    items: list[AcquisitionItem],
    store: ObjectStore,
    scratch_dir: Path,
    *,
    ledger: Ledger | None = None,
    cap_bytes: int | None = None,
    prior_bytes: int = 0,
    deep_verify: bool = True,
) -> AcquisitionResult:
    """Acquire a batch of objects for one source into the store, manifest, and ledger.

    ``cap_bytes`` is the family cap and ``prior_bytes`` the bytes already stored for the family in
    earlier batches, so the driver can respect the cap across batches. The object that would cross
    the cap is held back and named in the result rather than stored.
    """
    if spec.quarantined_dataset:
        raise AcquisitionError(
            f"source {spec.source_id!r} is a quarantined dataset and is not acquired under WO-2"
        )
    if ledger is not None:
        ledger.insert_corpus_source(
            source_id=spec.source_id,
            name=spec.name,
            license_=spec.license_class.value,
            truth_condition=spec.truth_condition_default,
        )

    batch_id = _make_batch_id(spec.source_id)
    records: list[ObjectRecord] = []
    result_stub = AcquisitionResult(
        manifest=AcquisitionManifest(
            batch_id=batch_id, source_id=spec.source_id, family=spec.family, created_at=_now()
        )
    )
    stored_family_bytes = prior_bytes

    for item in items:
        fetched = _fetch(item, scratch_dir)
        key = raw_key(fetched.sha256)
        try:
            if store.exists(key):
                # Idempotent: the content is already stored, so store nothing new (AC4). The
                # manifest and ledger record are still (re-)asserted, and those inserts ignore
                # duplicates, so a re-run converges without growing the store.
                record = _record(item, spec, fetched)
                records.append(record)
                if ledger is not None:
                    ledger.insert_corpus_object(batch_id=batch_id, record=_ledger_record(record))
                result_stub.skipped_existing += 1
                continue

            if cap_bytes is not None and stored_family_bytes + fetched.byte_count > cap_bytes:
                # Respect the family cap: do not store the object that would cross it (AC3).
                result_stub.cap_reached = True
                result_stub.held_for_cap.append(item.origin_url)
                break

            store.put_file(key, fetched.path)
            _verify(store, key, fetched, deep=deep_verify)
            record = _record(item, spec, fetched)
            records.append(record)
            if ledger is not None:
                ledger.insert_corpus_object(batch_id=batch_id, record=_ledger_record(record))
            result_stub.uploaded += 1
            result_stub.bytes_stored += fetched.byte_count
            stored_family_bytes += fetched.byte_count
        finally:
            fetched.path.unlink(missing_ok=True)

    manifest = AcquisitionManifest(
        batch_id=batch_id,
        source_id=spec.source_id,
        family=spec.family,
        created_at=_now(),
        objects=tuple(records),
        notes=spec.notes,
    )
    store.put_bytes(manifest_key(batch_id), manifest.to_json_bytes())
    result_stub.manifest = manifest
    return result_stub
