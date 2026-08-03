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
from .ais import AISRecord, RecordingWindow, correlate, haversine_m, parse_marinecadastre_row

# Known site coordinates (latitude, longitude), from each site's deployment metadata. Extend as
# sites are added. SanctSound coordinates are read from the deployment's metadata JSON.
SITE_COORDS: dict[str, tuple[float, float]] = {
    "mars_monterey_bay": (36.7128, -122.186),
    "stellwagen_sb01": (42.43668, -70.546655),
    "monterey_mb01": (36.798, -121.976),
    "channel_islands_ci01": (34.0438, -120.0811),
}

BBox = tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)

# A UTC timestamp of the form YYYYMMDDTHHMMSSZ, embedded in both MBARI and SanctSound file names.
_TIMESTAMP = re.compile(r"(\d{8})T(\d{6})Z")
_AIS_DATE = re.compile(r"AIS_(\d{4})_(\d{2})_(\d{2})")

# Provisional quiet-tail thresholds: a passage counts as quiet-tail when it is close, slow, and
# isolated. These are engineering defaults in the spirit of the adequacy note and firm up with data.
QUIET_TAIL_RANGE_M = 5000.0
QUIET_TAIL_SOG_KN = 5.0
QUIET_TAIL_ISOLATION_MAX = 3
# Isolation is masking at the moment of closest approach: how many other vessels are within
# acoustic range in a short window around it. A distant vessel or one present at another time does
# not mask the passage, so isolation is measured here rather than over the whole time in the box.
ISOLATION_WINDOW_S = 900.0
ISOLATION_RANGE_M = 10000.0


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


def recording_window(origin_url: str, duration_s: float | None) -> tuple[float, float] | None:
    """Return a recording's (start, end) epoch seconds from the timestamp in its file name.

    Works for any source whose file name embeds a ``YYYYMMDDTHHMMSSZ`` UTC timestamp, which covers
    both MBARI and SanctSound. The end is the start plus the recorded duration, or a day when the
    duration is unknown, so same-day AIS still bounds the correlation.
    """
    match = _TIMESTAMP.search(origin_url)
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


def recording_date_key(origin_url: str) -> str | None:
    """Return the ``YYYY_MM_DD`` key of a recording, matching the AIS key format."""
    match = _TIMESTAMP.search(origin_url)
    if match is None:
        return None
    day = match.group(1)
    return f"{day[0:4]}_{day[4:6]}_{day[6:8]}"


def presence_id(corr_run: str, recording_sha: str, vessel_id: str) -> str:
    """Return the identifier of a vessel's presence in a recording under a correlation run.

    The correlation run identifier is part of the key so that a re-correlation, for example after
    the quiet-tail classifier or its thresholds change, writes a fresh superseding set of rows
    rather than being ignored by the append-only ledger.
    """
    return hash_json({"run": corr_run, "recording": recording_sha, "vessel": vessel_id})


def correlate_recording(
    *,
    corr_run: str,
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
    presences = correlate(ais_records, window)
    rows: list[dict[str, object]] = []
    for presence in presences:
        # Isolation is the count of other vessels acoustically nearby at the moment of closest
        # approach: within the masking range and a short time window. A quiet-tail passage is
        # close, slow, and isolated, the scarce, unmasked resource the note names.
        masking = {
            record.mmsi
            for record in ais_records
            if record.mmsi != presence.mmsi
            and abs(record.epoch_s - presence.closest_time_s) <= ISOLATION_WINDOW_S
            and haversine_m(record.lat, record.lon, lat, lon) <= ISOLATION_RANGE_M
        }
        isolation = len(masking)
        is_close = presence.closest_range_m <= QUIET_TAIL_RANGE_M
        is_slow = presence.min_sog is not None and presence.min_sog <= QUIET_TAIL_SOG_KN
        is_isolated = isolation <= QUIET_TAIL_ISOLATION_MAX
        quiet_tail = int(presence.registry_grade and is_close and is_slow and is_isolated)
        rows.append(
            {
                "presence_id": presence_id(corr_run, recording_sha, presence.vessel_id),
                "corr_run": corr_run,
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
                "min_sog": presence.min_sog,
                "isolation": isolation,
                "quiet_tail": quiet_tail,
                "ais_source_sha": ais_source_sha,
            }
        )
    return rows
