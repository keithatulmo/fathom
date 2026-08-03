"""Tests for the acquisition driver: storage, verification, idempotency, caps, and quarantine."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fathom.corpus.acquire import AcquisitionError, AcquisitionItem, acquire_batch
from fathom.corpus.objectstore import LocalObjectStore, raw_key
from fathom.corpus.sources import AccessClass, LicenseClass, SourceSpec
from fathom.ledger import Ledger


def _source(**overrides: object) -> SourceSpec:
    base: dict[str, object] = {
        "source_id": "test_pub",
        "name": "Test public source",
        "family": 2,
        "role": "nominal clutter site",
        "license_class": LicenseClass.PUBLIC,
        "license_evidence_url": "https://example.org/license",
        "access_class": AccessClass.HTTPS,
        "truth_condition_default": "tier1_ais_correlated",
        "cap_gb": 1.0,
        "site": "test_site",
        "instrument": "test_hydrophone",
        "sample_rate_hz": 2000.0,
        "band_low_hz": 10.0,
        "band_high_hz": 300.0,
    }
    base.update(overrides)
    return SourceSpec(**base)


def _item(path: Path) -> AcquisitionItem:
    return AcquisitionItem(origin_url=path.as_uri(), url=path.as_uri())


def test_acquire_stores_verifies_and_records(tmp_path: Path) -> None:
    data = b"synthetic acoustic bytes " * 200
    src = tmp_path / "obj.bin"
    src.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()

    store = LocalObjectStore(tmp_path / "store")
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        result = acquire_batch(_source(), [_item(src)], store, tmp_path / "scratch", ledger=ledger)
        assert result.uploaded == 1
        assert result.bytes_stored == len(data)
        assert store.exists(raw_key(sha))
        assert ledger.corpus_object_exists(sha)

        record = result.manifest.objects[0]
        assert record.sha256 == sha
        assert record.byte_count == len(data)
        assert record.training_eligible is True
        assert record.site == "test_site"
        assert record.raw_key == raw_key(sha)
    finally:
        ledger.close()


def test_reacquire_is_idempotent(tmp_path: Path) -> None:
    data = b"idempotent payload"
    src = tmp_path / "obj.bin"
    src.write_bytes(data)
    store = LocalObjectStore(tmp_path / "store")
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        first = acquire_batch(_source(), [_item(src)], store, tmp_path / "scratch", ledger=ledger)
        second = acquire_batch(_source(), [_item(src)], store, tmp_path / "scratch", ledger=ledger)
        assert first.uploaded == 1
        assert second.uploaded == 0
        assert second.skipped_existing == 1
        # Exactly one raw object exists despite two acquisitions.
        assert len(store.list("raw/")) == 1
    finally:
        ledger.close()


def test_family_cap_holds_back_crossing_object(tmp_path: Path) -> None:
    data = b"x" * 1000
    src = tmp_path / "obj.bin"
    src.write_bytes(data)
    store = LocalObjectStore(tmp_path / "store")
    result = acquire_batch(
        _source(),
        [_item(src)],
        store,
        tmp_path / "scratch",
        cap_bytes=len(data) - 1,
    )
    assert result.uploaded == 0
    assert result.cap_reached is True
    assert result.held_for_cap == [src.as_uri()]
    assert store.list("raw/") == []


def test_research_only_license_is_not_training_eligible(tmp_path: Path) -> None:
    data = b"research only payload"
    src = tmp_path / "obj.bin"
    src.write_bytes(data)
    store = LocalObjectStore(tmp_path / "store")
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        spec = _source(license_class=LicenseClass.RESEARCH_ONLY)
        result = acquire_batch(spec, [_item(src)], store, tmp_path / "scratch", ledger=ledger)
        assert result.manifest.objects[0].training_eligible is False
        rows = ledger.get_corpus_objects()
        assert rows[0]["training_eligible"] == 0
    finally:
        ledger.close()


def test_acquire_records_audio_duration(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "clip.wav"
    # 2000 frames at 1000 Hz is two seconds; the header carries the duration.
    sf.write(str(wav), np.zeros(2000, dtype="float32"), 1000)
    store = LocalObjectStore(tmp_path / "store")
    item = AcquisitionItem(origin_url=wav.as_uri(), url=wav.as_uri())
    result = acquire_batch(_source(), [item], store, tmp_path / "scratch")
    assert result.manifest.objects[0].duration_s == pytest.approx(2.0)


def test_quarantined_dataset_is_not_acquired(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "store")
    spec = _source(quarantined_dataset=True)
    with pytest.raises(AcquisitionError, match="quarantined"):
        acquire_batch(spec, [], store, tmp_path / "scratch")


def test_deep_verify_detects_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A store that silently drops content must fail verification rather than record a bad object.
    data = b"payload to corrupt"
    src = tmp_path / "obj.bin"
    src.write_bytes(data)
    store = LocalObjectStore(tmp_path / "store")

    original_put = store.put_file

    def truncating_put(key: str, path: Path) -> None:
        original_put(key, path)
        # Corrupt the stored object so size no longer matches.
        (tmp_path / "store" / key).write_bytes(b"short")

    monkeypatch.setattr(store, "put_file", truncating_put)
    from fathom.corpus.acquire import VerificationError

    with pytest.raises(VerificationError):
        acquire_batch(_source(), [_item(src)], store, tmp_path / "scratch")
