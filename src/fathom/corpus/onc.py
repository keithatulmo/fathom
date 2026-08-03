"""Ocean Networks Canada archive-file access.

WO-2 Section 3a records ONC as a credentialed source: its Oceans 3.0 platform requires a free
account and an API token for programmatic download. This module lists archived files for a location
and device category over a date range and builds the download URL for each. The token is read from
the environment and appears only in the transient fetch URL; the provenance URL recorded in the
manifest and ledger is token-free, so the credential never enters a stored record.
"""

from __future__ import annotations

import os
import urllib.parse
from typing import Any

from ._http import http_get_json

ENV_ONC_TOKEN = "FATHOM_ONC_TOKEN"
_BASE = "https://data.oceannetworks.ca/api/archivefiles"


def onc_token_from_env() -> str:
    """Return the ONC API token from the environment, raising if it is not provisioned."""
    token = os.environ.get(ENV_ONC_TOKEN)
    if not token:
        raise RuntimeError(
            f"{ENV_ONC_TOKEN} is not set; provision the ONC API token before acquiring ONC data"
        )
    return token


def parse_archivefiles(payload: dict[str, Any]) -> list[str]:
    """Return the list of archived file names from an ONC archivefiles response."""
    files = payload.get("files", [])
    names: list[str] = []
    for entry in files:
        if isinstance(entry, str):
            names.append(entry)
        elif isinstance(entry, dict) and "filename" in entry:
            names.append(str(entry["filename"]))
    return names


def onc_origin_url(filename: str) -> str:
    """Return the token-free canonical URL recorded as provenance for an ONC file."""
    query = urllib.parse.urlencode({"method": "getFile", "filename": filename})
    return f"{_BASE}?{query}"


def onc_download_url(filename: str, token: str) -> str:
    """Return the tokened download URL used to fetch an ONC file (never recorded or printed)."""
    query = urllib.parse.urlencode({"method": "getFile", "filename": filename, "token": token})
    return f"{_BASE}?{query}"


def onc_list_archive_files(
    *,
    location_code: str,
    device_category_code: str,
    date_from: str,
    date_to: str,
    token: str,
    extension: str | None = None,
) -> list[str]:
    """List ONC archived file names for a location and device category over a date range.

    The archive returns every product for the window, not audio alone, so an optional extension
    restricts the result to audio (for example ``wav``). Filtering is done on the returned file
    names rather than through the API's extension parameter, which rejects the request; it is
    matched case-insensitively against the file suffix.
    """
    params = {
        "method": "getListByLocation",
        "token": token,
        "locationCode": location_code,
        "deviceCategoryCode": device_category_code,
        "dateFrom": date_from,
        "dateTo": date_to,
    }
    url = f"{_BASE}?{urllib.parse.urlencode(params)}"
    names = parse_archivefiles(http_get_json(url))
    if extension is not None:
        suffix = "." + extension.lower().lstrip(".")
        names = [name for name in names if name.lower().endswith(suffix)]
    return names
