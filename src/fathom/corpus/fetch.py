"""Source fetchers.

Sources are read by fetching named objects, not by crawling: WO-2 Section 3a notes that the OOI
archive's robots.txt disallows crawlers but does not govern a data client retrieving specific files
by path. HTTPS and public-cloud sources are fetched over HTTPS; anonymous public S3 sources such as
MBARI Pacific Sound are fetched with unsigned requests. Every fetch streams to a scratch file while
computing the SHA-256 and byte count, so a large object never needs to be held in memory and its
content hash is known before it is stored.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20  # one mebibyte
_USER_AGENT = "fathom-corpus-acquire/0.1"
_TIMEOUT_S = 120

_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".zip": "application/zip",
    ".csv": "text/csv",
    ".nc": "application/x-netcdf",
    ".json": "application/json",
    ".txt": "text/plain",
}


@dataclass(frozen=True)
class FetchedFile:
    """A fetched object on local scratch disk, with its content hash and size."""

    path: Path
    sha256: str
    byte_count: int
    media_type: str | None


def guess_media_type(name: str) -> str | None:
    """Return a media type guessed from a file name's extension, or None."""
    suffix = Path(name).suffix.lower()
    return _MEDIA_TYPES.get(suffix)


def _finish(path: Path, hasher: hashlib._Hash, size: int, name: str) -> FetchedFile:
    return FetchedFile(
        path=path,
        sha256=hasher.hexdigest(),
        byte_count=size,
        media_type=guess_media_type(name),
    )


def fetch_https(url: str, scratch_dir: Path) -> FetchedFile:
    """Fetch an object over HTTPS, streaming to scratch while hashing."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    target = scratch_dir / (hashlib.sha256(url.encode()).hexdigest() + Path(url).suffix)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    hasher = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response, target.open("wb") as out:
        while True:
            chunk = response.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    return _finish(target, hasher, size, url)


def anon_s3_client(region: str) -> Any:
    """Return a boto3 S3 client that makes unsigned (anonymous) requests."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", region_name=region, config=Config(signature_version=UNSIGNED))


def anon_s3_list(bucket: str, prefix: str, region: str, *, max_keys: int = 1000) -> list[str]:
    """List keys under a prefix in an anonymous public S3 bucket, up to ``max_keys``."""
    client = anon_s3_client(region)
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            keys.append(str(item["Key"]))
            if len(keys) >= max_keys:
                return keys
    return keys


def fetch_anon_s3(bucket: str, key: str, region: str, scratch_dir: Path) -> FetchedFile:
    """Fetch an object from an anonymous public S3 bucket, streaming to scratch while hashing."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    client = anon_s3_client(region)
    target = scratch_dir / (
        hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest() + Path(key).suffix
    )
    hasher = hashlib.sha256()
    size = 0
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    with target.open("wb") as out:
        while True:
            chunk = body.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    return _finish(target, hasher, size, key)
