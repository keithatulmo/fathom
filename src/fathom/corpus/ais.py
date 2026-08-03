"""AIS correlation into vessel-presence truth.

WO-2 Family three correlates MarineCadastre AIS, filtered to the dates and regions overlapping the
acoustic recordings, into tier-one vessel-presence truth with registry-grade identity per the SD3
vessel registry. SD3 Section 5.1 requires MMSI to be reconciled against IMO number and name where
available, because MMSI alone is reused and transferred; a contact that cannot be resolved to
registry grade is not tier-one truth. This module computes, for a recording window, which vessels
were within range and whether each resolves to registry grade, and it is written fresh from those
definitions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

_EARTH_RADIUS_M = 6_371_000.0
_MARINECADASTRE_TEMPLATE = (
    "https://chs.coast.noaa.gov/htdata/CMSP/AISDataHandler/"
    "{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
)


def marinecadastre_daily_urls(start: str, end: str) -> list[str]:
    """Return the MarineCadastre daily AIS zip URLs for an inclusive date range.

    Dates are ISO ``YYYY-MM-DD``. The caller filters to the dates overlapping the acoustic
    recordings rather than pulling whole years, per WO-2 Section 3a, so this builds only the days
    that matter. The URLs are the bulk htdata endpoint, not the interactive clip-and-ship page.
    """
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    if last < first:
        raise ValueError("end date precedes start date")
    urls: list[str] = []
    current = first
    while current <= last:
        urls.append(
            _MARINECADASTRE_TEMPLATE.format(year=current.year, month=current.month, day=current.day)
        )
        current += timedelta(days=1)
    return urls


@dataclass(frozen=True)
class AISRecord:
    """One AIS position report."""

    mmsi: str
    epoch_s: float
    lat: float
    lon: float
    imo: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class RecordingWindow:
    """A recording's site, time window, and correlation radius."""

    site: str
    lat: float
    lon: float
    start_epoch_s: float
    end_epoch_s: float
    radius_m: float


@dataclass(frozen=True)
class VesselPresence:
    """A vessel's presence in a recording window, with its resolved identity and truth tier."""

    vessel_id: str
    mmsi: str
    imo: str | None
    name: str | None
    first_seen_s: float
    last_seen_s: float
    closest_range_m: float
    registry_grade: bool
    truth_tier: int


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in metres between two latitude/longitude points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _registry_identity(mmsi: str, imo: str | None, name: str | None) -> tuple[str, bool]:
    """Resolve a vessel identity and whether it reaches registry grade.

    Registry grade requires an MMSI reconciled against an IMO number or a name, since MMSI alone is
    reused across hulls. The identity prefers IMO, then MMSI-with-name, and falls back to MMSI.
    """
    has_name = name is not None and name.strip() != ""
    registry_grade = imo is not None or has_name
    if imo is not None:
        vessel_id = f"IMO{imo}"
    elif has_name:
        assert name is not None
        vessel_id = f"MMSI{mmsi}-{name.strip().upper().replace(' ', '_')}"
    else:
        vessel_id = f"MMSI{mmsi}"
    return vessel_id, registry_grade


def correlate(records: list[AISRecord], window: RecordingWindow) -> list[VesselPresence]:
    """Return the vessels present within the recording window, ordered by closest range.

    A record contributes if its timestamp lies within the window and its position is within the
    correlation radius. Records are grouped by MMSI, the identity is reconciled to registry grade
    where possible, and registry-grade presence is tier-one truth while an unresolved contact is
    tier three and never usable as target-side evaluation truth.
    """
    by_mmsi: dict[str, list[AISRecord]] = {}
    for record in records:
        if not window.start_epoch_s <= record.epoch_s <= window.end_epoch_s:
            continue
        if haversine_m(record.lat, record.lon, window.lat, window.lon) > window.radius_m:
            continue
        by_mmsi.setdefault(record.mmsi, []).append(record)

    presences: list[VesselPresence] = []
    for mmsi, hits in by_mmsi.items():
        imo = next((r.imo for r in hits if r.imo), None)
        name = next((r.name for r in hits if r.name), None)
        vessel_id, registry_grade = _registry_identity(mmsi, imo, name)
        ranges = [haversine_m(r.lat, r.lon, window.lat, window.lon) for r in hits]
        presences.append(
            VesselPresence(
                vessel_id=vessel_id,
                mmsi=mmsi,
                imo=imo,
                name=name,
                first_seen_s=min(r.epoch_s for r in hits),
                last_seen_s=max(r.epoch_s for r in hits),
                closest_range_m=min(ranges),
                registry_grade=registry_grade,
                truth_tier=1 if registry_grade else 3,
            )
        )
    return sorted(presences, key=lambda p: p.closest_range_m)


def parse_marinecadastre_row(row: dict[str, str]) -> AISRecord:
    """Parse a MarineCadastre AIS CSV row into an :class:`AISRecord`.

    The bulk CSVs carry ``MMSI``, ``BaseDateTime`` (ISO 8601, UTC), ``LAT``, ``LON``, and, when
    known, ``IMO`` and ``VesselName``. Empty IMO or name fields resolve to ``None`` so identity
    reconciliation can tell registry-grade contacts from bare MMSI.
    """
    imo = row.get("IMO", "").strip() or None
    name = row.get("VesselName", "").strip() or None
    if imo is not None and imo.upper().startswith("IMO"):
        imo = imo[3:].strip() or None
    return AISRecord(
        mmsi=row["MMSI"].strip(),
        epoch_s=datetime.fromisoformat(row["BaseDateTime"]).timestamp(),
        lat=float(row["LAT"]),
        lon=float(row["LON"]),
        imo=imo,
        name=name,
    )
