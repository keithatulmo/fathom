"""Tests for canonical hashing and deterministic artifact serialization."""

from __future__ import annotations

import numpy as np
import pytest

from fathom.hashing import (
    canonical_json_bytes,
    decode_mapping,
    encode_mapping,
    hash_json,
    sha256_hex,
)
from fathom.serialize import (
    decode_array,
    decode_bundle,
    encode_array,
    encode_bundle,
)


def test_canonical_json_is_key_order_independent() -> None:
    a = {"z": 1, "a": 2, "m": {"y": 3, "x": 4}}
    b = {"a": 2, "m": {"x": 4, "y": 3}, "z": 1}
    assert canonical_json_bytes(a) == canonical_json_bytes(b)
    assert hash_json(a) == hash_json(b)


def test_canonical_json_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"x": float("nan")})


def test_mapping_roundtrip_and_determinism() -> None:
    mapping = {"b": b"\x00\x01", "a": b"hello", "c": b""}
    blob1 = encode_mapping(mapping)
    blob2 = encode_mapping(dict(reversed(list(mapping.items()))))
    assert blob1 == blob2  # sorted keys make encoding order-independent
    assert decode_mapping(blob1) == mapping


def test_array_roundtrip_and_determinism() -> None:
    array = (np.arange(24, dtype=np.float64).reshape(2, 3, 4) * 0.25) - 1.0
    blob1 = encode_array(array)
    blob2 = encode_array(array)
    assert sha256_hex(blob1) == sha256_hex(blob2)
    restored = decode_array(blob1)
    assert np.array_equal(restored, array)
    assert restored.dtype == array.dtype


def test_array_roundtrip_integer_dtype() -> None:
    array = np.array([[1, 2], [3, 4]], dtype=np.int64)
    restored = decode_array(encode_array(array))
    assert np.array_equal(restored, array)
    assert restored.dtype == np.int64


def test_bundle_roundtrip_and_determinism() -> None:
    meta = {"units": "power", "nfft": 256}
    arrays = {"gram": np.ones((2, 3), dtype=np.float64), "freqs": np.arange(3, dtype=np.float64)}
    blob1 = encode_bundle(meta, arrays)
    blob2 = encode_bundle(meta, arrays)
    assert sha256_hex(blob1) == sha256_hex(blob2)
    out_meta, out_arrays = decode_bundle(blob1)
    assert out_meta == meta
    assert np.array_equal(out_arrays["gram"], arrays["gram"])
    assert np.array_equal(out_arrays["freqs"], arrays["freqs"])
