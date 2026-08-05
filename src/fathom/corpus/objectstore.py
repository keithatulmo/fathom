"""S3-compatible object storage for the corpus bucket.

SD1/WO-2 Section 2 fixes the bucket layout: ``raw/`` holds immutable originals content-addressed by
SHA-256 and sharded by hash prefix, ``derived/`` holds content-addressed derivatives, and
``manifests/`` holds one JSON manifest per acquisition batch. Because every object's key is the
hash of its contents, re-running any acquisition is idempotent by construction: an object that is
already present is detected and skipped rather than stored again.

Two backends implement one interface. The local backend is a filesystem directory used for offline
tests and dry runs. The R2 backend is a boto3 S3 client pointed at the owner-provided endpoint,
configured entirely from environment variables so that no credential ever enters the repository.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

# Environment variables that carry the R2 wiring. Values are read only from the process
# environment supplied by the owner's session and are never written to the repository (AC6).
ENV_ENDPOINT = "FATHOM_R2_ENDPOINT"
ENV_ACCESS_KEY = "FATHOM_R2_ACCESS_KEY_ID"
ENV_SECRET_KEY = "FATHOM_R2_SECRET_ACCESS_KEY"
ENV_BUCKET = "FATHOM_R2_BUCKET"


def raw_key(digest: str) -> str:
    """Return the ``raw/`` object key for a content hash, sharded by its first two characters."""
    return f"raw/{digest[:2]}/{digest}"


def derived_key(digest: str) -> str:
    """Return the ``derived/`` key for a content hash, sharded by its first two characters."""
    return f"derived/{digest[:2]}/{digest}"


def manifest_key(batch_id: str) -> str:
    """Return the ``manifests/`` object key for an acquisition batch."""
    return f"manifests/{batch_id}.json"


class ObjectStore(ABC):
    """A content-addressed object store with a small, backend-neutral interface."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return whether an object with the given key is present."""

    @abstractmethod
    def size(self, key: str) -> int | None:
        """Return the byte size of an object, or None if it is absent."""

    @abstractmethod
    def put_file(self, key: str, path: Path) -> None:
        """Upload a local file to the given key."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None:
        """Store raw bytes at the given key."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Return the bytes stored at the given key."""

    @abstractmethod
    def get_head(self, key: str, n_bytes: int) -> bytes:
        """Return the first ``n_bytes`` of an object, a range read for streamable formats.

        A streamable codec such as FLAC decodes its leading frames from the head of the file, so a
        window near the start can be extracted without downloading a multi-hour object in full.
        """

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """Return the keys present under the given prefix, sorted."""


class LocalObjectStore(ObjectStore):
    """A filesystem-backed object store for offline tests and dry runs."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._root / key

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def size(self, key: str) -> int | None:
        path = self._path(key)
        return path.stat().st_size if path.is_file() else None

    def put_file(self, key: str, path: Path) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(path, tmp)
        os.replace(tmp, target)

    def put_bytes(self, key: str, data: bytes) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)

    def get_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def get_head(self, key: str, n_bytes: int) -> bytes:
        with self._path(key).open("rb") as handle:
            return handle.read(n_bytes)

    def list(self, prefix: str) -> list[str]:
        base = self._root
        keys: list[str] = []
        for path in base.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                rel = path.relative_to(base).as_posix()
                if rel.startswith(prefix):
                    keys.append(rel)
        return sorted(keys)


class R2ObjectStore(ObjectStore):
    """An R2 (S3-compatible) object store configured from environment variables."""

    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str) -> None:
        import boto3
        from botocore.config import Config

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 5, "mode": "standard"},
            ),
        )

    @classmethod
    def from_env(cls) -> R2ObjectStore:
        """Construct the R2 store from the FATHOM_R2_* environment variables.

        A missing variable raises a clear error naming what the owner must provision, so the
        credential path fails loudly rather than silently writing to the wrong place.
        """
        missing = [
            name
            for name in (ENV_ENDPOINT, ENV_ACCESS_KEY, ENV_SECRET_KEY, ENV_BUCKET)
            if not os.environ.get(name)
        ]
        if missing:
            raise RuntimeError(
                "missing R2 environment variables: "
                + ", ".join(missing)
                + "; provision them in the session before acquiring"
            )
        return cls(
            endpoint=os.environ[ENV_ENDPOINT],
            access_key=os.environ[ENV_ACCESS_KEY],
            secret_key=os.environ[ENV_SECRET_KEY],
            bucket=os.environ[ENV_BUCKET],
        )

    def exists(self, key: str) -> bool:
        return self.size(key) is not None

    def size(self, key: str) -> int | None:
        from botocore.exceptions import ClientError

        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return int(response["ContentLength"])

    def put_file(self, key: str, path: Path) -> None:
        self._client.upload_file(str(path), self._bucket, key)

    def put_bytes(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get_bytes(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        body: bytes = response["Body"].read()
        return body

    def get_head(self, key: str, n_bytes: int) -> bytes:
        response = self._client.get_object(
            Bucket=self._bucket, Key=key, Range=f"bytes=0-{max(0, n_bytes - 1)}"
        )
        body: bytes = response["Body"].read()
        return body

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                keys.append(str(item["Key"]))
        return sorted(keys)
