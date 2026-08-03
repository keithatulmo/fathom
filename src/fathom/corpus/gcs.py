"""Google Cloud Storage listing for the NCEI passive-acoustic archive.

WO-2 Section 3a fixes the NCEI archive, which serves ADEON and SanctSound raw data and annotations,
as public through the Google Cloud Storage bucket ``noaa-passive-bioacoustic`` over anonymous HTTPS.
This module lists object names under a prefix through the storage JSON API and builds the anonymous
download URL for each object, so those sources are acquired by the same driver as any other. No
credential is involved, because the bucket is public.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from ._http import http_get_json

_LIST_ENDPOINT = "https://storage.googleapis.com/storage/v1/b/{bucket}/o"
_OBJECT_ENDPOINT = "https://storage.googleapis.com/{bucket}/{name}"


def parse_gcs_listing(payload: dict[str, Any]) -> tuple[list[str], str | None]:
    """Return the object names and the next page token from a storage JSON API page."""
    names = [str(item["name"]) for item in payload.get("items", [])]
    token = payload.get("nextPageToken")
    return names, (str(token) if token is not None else None)


def gcs_object_url(bucket: str, name: str) -> str:
    """Return the anonymous HTTPS download URL for an object in a public GCS bucket."""
    return _OBJECT_ENDPOINT.format(bucket=bucket, name=urllib.parse.quote(name))


def gcs_list(bucket: str, prefix: str, max_keys: int = 1000) -> list[str]:
    """List object names under a prefix in a public GCS bucket, up to ``max_keys``.

    Listing pages through the storage JSON API anonymously, which a public bucket permits, and
    stops as soon as ``max_keys`` names have been collected so a large prefix is bounded.
    """
    names: list[str] = []
    page_token: str | None = None
    while len(names) < max_keys:
        query = {"prefix": prefix, "maxResults": str(min(1000, max_keys - len(names)))}
        if page_token:
            query["pageToken"] = page_token
        url = _LIST_ENDPOINT.format(bucket=bucket) + "?" + urllib.parse.urlencode(query)
        payload = http_get_json(url)
        page_names, page_token = parse_gcs_listing(payload)
        names.extend(page_names)
        if not page_token or not page_names:
            break
    return names[:max_keys]
