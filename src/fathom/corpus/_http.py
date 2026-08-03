"""Small shared HTTP helpers for the source enumerators."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

_USER_AGENT = "fathom-corpus-acquire/0.1"
_TIMEOUT_S = 120


def http_get_json(url: str) -> dict[str, Any]:
    """Fetch a URL and parse its body as JSON."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return payload
