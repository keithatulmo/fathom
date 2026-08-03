"""The content-addressed artifact store.

Artifacts are opaque blobs keyed by the SHA-256 of their contents, per SD10 Section 6.5. The
store is write-once: putting a blob whose hash already exists is a no-op, because identical
contents have identical identity. Reads verify the hash on the way out, so silent corruption of
the store surfaces as an error rather than as a wrong number downstream.
"""

from __future__ import annotations

import os
from pathlib import Path

from .hashing import sha256_hex


class StoreCorruptionError(RuntimeError):
    """Raised when a stored blob's bytes do not match the hash under which it is filed."""


class ContentAddressedStore:
    """A directory of blobs addressed by their hexadecimal SHA-256 digest."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the root directory of the store."""
        return self._root

    def _path_for(self, digest: str) -> Path:
        # Fan out on the first two hex characters so no single directory grows without bound.
        return self._root / digest[:2] / digest

    def put(self, blob: bytes) -> str:
        """Store a blob and return its content hash, writing only if not already present."""
        digest = sha256_hex(blob)
        path = self._path_for(digest)
        if path.exists():
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename, so a reader never sees a partial blob.
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(blob)
        os.replace(tmp, path)
        return digest

    def has(self, digest: str) -> bool:
        """Return whether a blob with the given hash is present."""
        return self._path_for(digest).exists()

    def get(self, digest: str) -> bytes:
        """Return the blob for the given hash, verifying integrity on the way out."""
        path = self._path_for(digest)
        if not path.exists():
            raise KeyError(digest)
        blob = path.read_bytes()
        actual = sha256_hex(blob)
        if actual != digest:
            raise StoreCorruptionError(f"blob filed under {digest} hashes to {actual}")
        return blob
