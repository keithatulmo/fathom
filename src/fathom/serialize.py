"""Artifact serialization to deterministic, content-addressable blobs.

Two artifact encodings exist. Derived arrays are stored as Zarr version 3 chunked arrays per
SD10 Section 6.3, reduced to a single blob by serializing the store's key-to-chunk mapping with
:func:`fathom.hashing.encode_mapping`. Structured artifacts are stored as canonical JSON. Both
encodings are byte-deterministic, so an artifact's SHA-256 depends only on its contents and the
double-execution hash-equality check of the acceptance path is meaningful.

Compression is deliberately disabled for array artifacts in this scaffold, so that the stored
bytes are exactly the little-endian array payload plus fixed JSON metadata with no codec state
to vary. This is recorded as a deviation in docs/DEVIATIONS.md.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import zarr
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

from .hashing import canonical_json_bytes, decode_mapping, encode_mapping

_ARRAY_KEY = "array"


async def _dump_store(store: MemoryStore) -> dict[str, bytes]:
    prototype = default_buffer_prototype()
    mapping: dict[str, bytes] = {}
    async for key in store.list():
        value = await store.get(key, prototype=prototype)
        if value is None:  # pragma: no cover - a listed key always resolves
            continue
        mapping[key] = bytes(value.to_bytes())
    return mapping


async def _load_store(mapping: dict[str, bytes]) -> MemoryStore:
    prototype = default_buffer_prototype()
    store = MemoryStore()
    for key, value in mapping.items():
        await store.set(key, prototype.buffer.from_bytes(value))
    return store


def encode_array(array: npt.NDArray[Any]) -> bytes:
    """Encode a NumPy array as a single deterministic Zarr v3 artifact blob."""
    contiguous = np.ascontiguousarray(array)
    store = MemoryStore()
    chunks = tuple(dim if dim > 0 else 1 for dim in contiguous.shape)
    z = zarr.create_array(
        store=store,
        name=_ARRAY_KEY,
        shape=contiguous.shape,
        dtype=contiguous.dtype,
        chunks=chunks,
        compressors=None,
    )
    z[...] = contiguous
    mapping = asyncio.run(_dump_store(store))
    return encode_mapping(mapping)


def decode_array(blob: bytes) -> npt.NDArray[Any]:
    """Decode a Zarr v3 artifact blob back into a NumPy array."""
    mapping = decode_mapping(blob)
    store = asyncio.run(_load_store(mapping))
    z = zarr.open_array(store=store, path=_ARRAY_KEY, mode="r")
    return np.asarray(z[...])


def encode_json(obj: Any) -> bytes:
    """Encode a JSON-serializable object as canonical JSON bytes."""
    return canonical_json_bytes(obj)


def decode_json(blob: bytes) -> Any:
    """Decode canonical JSON bytes back into a Python object."""
    import json

    return json.loads(blob.decode("utf-8"))


def encode_bundle(meta: dict[str, Any], arrays: Mapping[str, npt.NDArray[Any]]) -> bytes:
    """Encode an artifact as one blob bundling canonical-JSON metadata and named Zarr arrays.

    Every artifact is a single content-addressed blob. A bundle carries a ``meta.json`` entry and
    zero or more arrays under ``arrays/<name>``, each itself a deterministic Zarr v3 blob, all
    folded into one deterministic mapping. This lets a typed artifact hold several arrays and its
    scalar metadata while keeping exactly one hash.
    """
    mapping: dict[str, bytes] = {"meta.json": canonical_json_bytes(meta)}
    for name, array in arrays.items():
        mapping[f"arrays/{name}"] = encode_array(array)
    return encode_mapping(mapping)


def decode_bundle(blob: bytes) -> tuple[dict[str, Any], dict[str, npt.NDArray[Any]]]:
    """Invert :func:`encode_bundle`, returning the metadata and the named arrays."""
    import json

    mapping = decode_mapping(blob)
    meta: dict[str, Any] = json.loads(mapping["meta.json"].decode("utf-8"))
    arrays: dict[str, npt.NDArray[Any]] = {}
    prefix = "arrays/"
    for key, value in mapping.items():
        if key.startswith(prefix):
            arrays[key[len(prefix) :]] = decode_array(value)
    return meta, arrays
