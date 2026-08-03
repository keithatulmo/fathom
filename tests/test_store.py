"""Tests for the content-addressed artifact store."""

from __future__ import annotations

from pathlib import Path

import pytest

from fathom.hashing import sha256_hex
from fathom.store import ContentAddressedStore, StoreCorruptionError


def test_put_get_roundtrip(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    blob = b"fathom acceptance blob"
    digest = store.put(blob)
    assert digest == sha256_hex(blob)
    assert store.has(digest)
    assert store.get(digest) == blob


def test_put_is_write_once_and_idempotent(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    blob = b"identical bytes"
    first = store.put(blob)
    second = store.put(blob)
    assert first == second


def test_missing_blob_raises(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    with pytest.raises(KeyError):
        store.get("0" * 64)


def test_corruption_is_detected(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "store")
    digest = store.put(b"original")
    # Tamper with the stored bytes under the same hash-named path.
    path = tmp_path / "store" / digest[:2] / digest
    path.write_bytes(b"tampered")
    with pytest.raises(StoreCorruptionError):
        store.get(digest)
