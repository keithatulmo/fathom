"""Canonical hashing utilities.

Every identity in Fathom is a SHA-256 over a canonical byte serialization, so that two
byte-identical values hash identically and any difference is visible as a different hash.
Canonical JSON fixes key order, disallows non-finite floats, and uses compact separators, so
the same logical object always produces the same bytes regardless of construction order.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Container magic for the length-prefixed serialization of a store mapping (see below).
_MAP_MAGIC = b"FZMAP1\n"


def sha256_hex(blob: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of the given bytes."""
    return hashlib.sha256(blob).hexdigest()


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize an object to canonical JSON bytes.

    Keys are sorted, separators are compact, non-ASCII is preserved, and non-finite floats
    are rejected so that no ``NaN`` or ``Infinity`` token ever enters a hashed artifact.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def hash_json(obj: Any) -> str:
    """Return the SHA-256 hex digest of an object's canonical JSON serialization."""
    return sha256_hex(canonical_json_bytes(obj))


def encode_mapping(mapping: dict[str, bytes]) -> bytes:
    """Serialize a string-to-bytes mapping into one deterministic, self-describing blob.

    Keys are emitted in sorted order with explicit length prefixes, so the encoding depends
    only on the mapping's contents and never on iteration order or filesystem metadata. This
    is how a Zarr store's key-to-chunk mapping becomes a single content-addressed artifact.
    """
    out = bytearray(_MAP_MAGIC)
    keys = sorted(mapping)
    out += len(keys).to_bytes(8, "big")
    for key in keys:
        key_bytes = key.encode("utf-8")
        value = mapping[key]
        out += len(key_bytes).to_bytes(8, "big")
        out += key_bytes
        out += len(value).to_bytes(8, "big")
        out += value
    return bytes(out)


def decode_mapping(blob: bytes) -> dict[str, bytes]:
    """Invert :func:`encode_mapping`, reconstructing the original string-to-bytes mapping."""
    if not blob.startswith(_MAP_MAGIC):
        raise ValueError("blob is not a Fathom mapping container")
    offset = len(_MAP_MAGIC)
    count = int.from_bytes(blob[offset : offset + 8], "big")
    offset += 8
    mapping: dict[str, bytes] = {}
    for _ in range(count):
        key_len = int.from_bytes(blob[offset : offset + 8], "big")
        offset += 8
        key = blob[offset : offset + key_len].decode("utf-8")
        offset += key_len
        value_len = int.from_bytes(blob[offset : offset + 8], "big")
        offset += 8
        value = blob[offset : offset + value_len]
        offset += value_len
        mapping[key] = value
    return mapping
