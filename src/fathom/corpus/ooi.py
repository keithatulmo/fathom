"""OOI raw-archive directory reading.

WO-2 Section 3a fixes the OOI broadband raw archive as an open Apache HTTPS directory whose
robots.txt disallows crawlers but does not govern a data client fetching named files by path. This
module reads a single named directory index and returns the file URLs it lists, which is a bounded
retrieval of a named path rather than a recursive crawl of the tree; the caller derives which
year-and-month directory to read from the documented structure. The low-frequency channels reached
through IRIS/EarthScope FDSN are a separate source and are not handled here.
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request

_USER_AGENT = "fathom-corpus-acquire/0.1"
_TIMEOUT_S = 120
_HREF = re.compile(r'href="([^"?][^"]*)"', re.IGNORECASE)


def parse_apache_index(html: str, base_url: str) -> list[str]:
    """Return absolute file URLs from an Apache autoindex page.

    Sort links (which carry a query string), the parent-directory link, and subdirectory links
    (which end in a slash) are excluded, so the result is the set of files directly listed in the
    named directory. Relative hrefs are resolved against the directory URL.
    """
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    files: list[str] = []
    seen: set[str] = set()
    for href in _HREF.findall(html):
        if href.startswith("/") or href.startswith("..") or href.endswith("/"):
            continue
        if href.startswith("?") or "://" in href and not href.startswith(base_url):
            continue
        absolute = urllib.parse.urljoin(base_url, href)
        if absolute not in seen:
            seen.add(absolute)
            files.append(absolute)
    return files


def ooi_list_index(index_url: str) -> list[str]:
    """Fetch a named OOI archive directory index and return the file URLs it lists."""
    request = urllib.request.Request(index_url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
        html = response.read().decode("utf-8", errors="replace")
    return parse_apache_index(html, index_url)
