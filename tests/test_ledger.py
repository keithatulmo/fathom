"""Tests for the append-only lineage ledger."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fathom.ledger import Ledger


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.db")


def test_run_status_is_derived_from_latest_event(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.insert_run(
        run_id="r1",
        manifest_hash="m",
        code_version="c",
        env_fingerprint="e",
        seed=1,
        thread_limit=1,
    )
    assert ledger.run_status("r1") == "started"
    ledger.append_run_status("r1", "completed")
    assert ledger.run_status("r1") == "completed"


def test_update_is_forbidden(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.insert_run(
        run_id="r1",
        manifest_hash="m",
        code_version="c",
        env_fingerprint="e",
        seed=1,
        thread_limit=1,
    )
    with pytest.raises(sqlite3.Error):
        ledger._conn.execute("UPDATE runs SET seed = 2 WHERE run_id = 'r1'")


def test_delete_is_forbidden(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.insert_run(
        run_id="r1",
        manifest_hash="m",
        code_version="c",
        env_fingerprint="e",
        seed=1,
        thread_limit=1,
    )
    with pytest.raises(sqlite3.Error):
        ledger._conn.execute("DELETE FROM runs WHERE run_id = 'r1'")


def test_artifact_insert_is_idempotent(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    for _ in range(2):
        ledger.insert_artifact(
            artifact_hash="h1",
            kind="gram",
            producing_stage="front_end",
            stage_version="1",
            code_version="c",
            config_hash="cfg",
            env_fingerprint="e",
            seed=1,
            parents=("p1",),
        )
    assert ledger.artifact_exists("h1")


def test_cache_roundtrip(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    assert ledger.cache_get("k") is None
    ledger.cache_put("k", {"out": "h"})
    assert ledger.cache_get("k") == {"out": "h"}


def test_corpus_source_owner_certified_fields_are_null(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.insert_corpus_source(
        source_id="s1", name="synthetic", license_="internal", truth_condition="tier1_verified"
    )
    row = ledger._conn.execute(
        "SELECT detection_range_r, confirmer_pd FROM corpus_sources WHERE source_id = 's1'"
    ).fetchone()
    assert row == (None, None)
