"""Tests for the local object store and the corpus ledger tables."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fathom.corpus.objectstore import LocalObjectStore, derived_key, manifest_key, raw_key
from fathom.ledger import Ledger


def test_key_helpers() -> None:
    digest = "abcd" + "0" * 60
    assert raw_key(digest) == f"raw/ab/{digest}"
    assert derived_key(digest) == f"derived/ab/{digest}"
    assert manifest_key("batch1") == "manifests/batch1.json"


def test_local_store_roundtrip_and_list(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "store")
    store.put_bytes("raw/aa/one", b"first")
    store.put_bytes("raw/bb/two", b"second")
    assert store.exists("raw/aa/one")
    assert store.size("raw/aa/one") == 5
    assert store.get_bytes("raw/bb/two") == b"second"
    assert store.list("raw/") == ["raw/aa/one", "raw/bb/two"]
    assert store.list("manifests/") == []


def test_put_file_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"file bytes")
    store = LocalObjectStore(tmp_path / "store")
    store.put_file("raw/cc/three", source)
    assert store.get_bytes("raw/cc/three") == b"file bytes"


def _record(sha: str, *, training_eligible: int = 1) -> dict[str, object]:
    return {
        "sha256": sha,
        "source_id": "mbari_pacific_sound_2khz",
        "origin_url": "s3://pacific-sound-2khz/x.wav",
        "retrieved_at": "2026-08-03T00:00:00+00:00",
        "byte_count": 100,
        "raw_key": raw_key(sha),
        "license_class": "cc_by",
        "license_evidence_url": "https://example.org",
        "truth_condition": "tier1_ais_correlated",
        "training_eligible": training_eligible,
        "media_type": "audio/wav",
        "site": "mars_monterey_bay",
        "instrument": "cabled_hydrophone",
        "sample_rate_hz": 2000.0,
        "band_low_hz": 10.0,
        "band_high_hz": 300.0,
        "band_partial": 0,
        "duration_s": 3600.0,
    }


def test_corpus_object_insert_and_query(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        sha = "a" * 64
        ledger.insert_corpus_object(batch_id="b1", record=_record(sha))
        assert ledger.corpus_object_exists(sha)
        # Idempotent re-insert.
        ledger.insert_corpus_object(batch_id="b1", record=_record(sha))
        rows = ledger.get_corpus_objects()
        assert len(rows) == 1
        assert rows[0]["training_eligible"] == 1
        assert rows[0]["batch_id"] == "b1"
    finally:
        ledger.close()


def test_corpus_object_is_append_only(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        ledger.insert_corpus_object(batch_id="b1", record=_record("b" * 64))
        with pytest.raises(sqlite3.Error):
            ledger._conn.execute("UPDATE corpus_objects SET byte_count = 1")
        with pytest.raises(sqlite3.Error):
            ledger._conn.execute("DELETE FROM corpus_objects")
    finally:
        ledger.close()


def test_corpus_derivative_insert(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        parent = "c" * 64
        ledger.insert_corpus_object(batch_id="b1", record=_record(parent))
        ledger.insert_corpus_derivative(
            record={
                "sha256": "d" * 64,
                "parent_sha256": parent,
                "derived_key": derived_key("d" * 64),
                "operation": "decimate",
                "params_json": '{"sample_rate_out": 1000.0}',
                "byte_count": 50,
                "sample_rate_hz": 1000.0,
                "band_low_hz": 3.0,
                "band_high_hz": 300.0,
            }
        )
        row = ledger._conn.execute(
            "SELECT operation, parent_sha256 FROM corpus_derivatives WHERE sha256 = ?",
            ("d" * 64,),
        ).fetchone()
        assert row == ("decimate", parent)
    finally:
        ledger.close()
