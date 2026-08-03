"""AIS-to-recording correlation producing tier-one vessel-presence truth.

WO-2 Family three correlates the acquired AIS, filtered to the dates and regions overlapping the
acoustic recordings, into tier-one vessel-presence truth with registry-grade identity. This module
reads AIS records from the daily zips, derives each recording's time window and site position, and
runs the correlation of :mod:`fathom.corpus.ais` over the union. The result is recorded as vessel
presence in the ledger, and the distinct registry-grade vessel counts feed the SD1 audit and the
rule-of-three arithmetic that depends on how many present targets the corpus can actually supply.
"""

from __future__ import annotations

import csv
import io
import math
import re
import zipfile
from datetime import UTC, datetime

from ..hashing import hash_json
from .ais import AISRecord, RecordingWindow, correlate, parse_marinecadastre_row

# Known site coordinates (latitude, longitude). Extend as sites are added to the corpus.
SITE_COORDS: dict[str, tuple[float, float]] = {
    "mars_monterey_bay": (36.7128, -122.186),
}

BBox = tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)

_MBARI_STAMP = re.compile(r"MARS-(\d{8})T(\d{6})Z")
_AIS_DATE = re.compile(r"AIS_(\d{4})_(\d{2})_(\d{2})")


def bbox_for(lat: float, lon: float, radius_m: float) -> BBox:
    """Return a latitude/longitude bounding box enclosing a radius around a point.

    The box is a cheap pre-filter that discards the vast majority of AIS rows before the exact
    great-circle test in the correlation, so a national daily file reduces to a site's traffic.
    """
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(0.01, math.cos(math.radians(lat))))
    return (lat - dlat, lat + dlat, lon - dlon, lon + dlon)


def _in_bbox(record: AISRecord, bbox: BBox) -> bool:
    return bbox[0] <= record.lat <= bbox[1] and bbox[2] <= record.lon <= bbox[3]


def read_ais_records(blob: bytes, bbox: BBox | None = None) -> list[AISRecord]:
    """Read AIS records from a MarineCadastre daily zip, optionally pre-filtered to a bounding box.

    Rows that cannot be parsed (missing columns, malformed coordinates) are skipped rather than
    failing the whole file, because a single day of national AIS routinely carries irregular rows.
    """
    records: list[AISRecord] = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            with archive.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
                for row in reader:
                    try:
                        record = parse_marinecadastre_row(row)
                    except (KeyError, ValueError):
                        continue
                    if bbox is None or _in_bbox(record, bbox):
                        records.append(record)
    return records


def mbari_recording_window(origin_url: str, duration_s: float | None) -> tuple[float, float] | None:
    """Return a MBARI recording's (start, end) epoch seconds from its file name, or None."""
    match = _MBARI_STAMP.search(origin_url)
    if match is None:
        return None
    stamp = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    start = stamp.timestamp()
    end = start + (duration_s if duration_s is not None else 86400.0)
    return start, end


def ais_date_key(origin_url: str) -> str | None:
    """Return the ``YYYY_MM_DD`` key of a MarineCadastre AIS object, or None."""
    match = _AIS_DATE.search(origin_url)
    return None if match is None else f"{match.group(1)}_{match.group(2)}_{match.group(3)}"


def mbari_date_key(origin_url: str) -> str | None:
    """Return the ``YYYY_MM_DD`` key of a MBARI recording, matching the AIS key format."""
    match = _MBARI_STAMP.search(origin_url)
    if match is None:
        return None
    day = match.group(1)
    return f"{day[0:4]}_{day[4:6]}_{day[6:8]}"


def presence_id(recording_sha: str, vessel_id: str) -> str:
    """Return the stable identifier of a vessel's presence in a recording."""
    return hash_json({"recording": recording_sha, "vessel": vessel_id})


def correlate_recording(
    *,
    recording_sha: str,
    site: str,
    lat: float,
    lon: float,
    start_s: float,
    end_s: float,
    radius_m: float,
    ais_records: list[AISRecord],
    ais_source_sha: str | None,
) -> list[dict[str, object]]:
    """Correlate one recording against AIS records, returning vessel-presence ledger rows."""
    window = RecordingWindow(
        site=site, lat=lat, lon=lon, start_epoch_s=start_s, end_epoch_s=end_s, radius_m=radius_m
    )
    rows: list[dict[str, object]] = []
    for presence in correlate(ais_records, window):
        rows.append(
            {
                "presence_id": presence_id(recording_sha, presence.vessel_id),
                "recording_sha": recording_sha,
                "site": site,
                "vessel_id": presence.vessel_id,
                "mmsi": presence.mmsi,
                "imo": presence.imo,
                "name": presence.name,
                "first_seen_s": presence.first_seen_s,
                "last_seen_s": presence.last_seen_s,
                "closest_range_m": presence.closest_range_m,
                "registry_grade": int(presence.registry_grade),
                "truth_tier": presence.truth_tier,
                "ais_source_sha": ais_source_sha,
            }
        )
    return rows
