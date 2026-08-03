"""Tests for AIS correlation into vessel-presence truth."""

from __future__ import annotations

import pytest

from fathom.corpus.ais import (
    AISRecord,
    RecordingWindow,
    correlate,
    haversine_m,
    parse_marinecadastre_row,
)


def test_haversine_known_distances() -> None:
    assert haversine_m(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)
    # One degree of longitude at the equator is about 111.3 km.
    assert haversine_m(0.0, 0.0, 0.0, 1.0) == pytest.approx(111_195.0, rel=1e-3)


def _window() -> RecordingWindow:
    return RecordingWindow(
        site="test", lat=36.7, lon=-122.0, start_epoch_s=100.0, end_epoch_s=200.0, radius_m=5000.0
    )


def test_registry_grade_vessel_is_tier_one() -> None:
    records = [
        AISRecord(mmsi="111", epoch_s=150.0, lat=36.7, lon=-122.0, imo="9000001", name="ALPHA"),
    ]
    presences = correlate(records, _window())
    assert len(presences) == 1
    presence = presences[0]
    assert presence.registry_grade is True
    assert presence.truth_tier == 1
    assert presence.vessel_id == "IMO9000001"


def test_bare_mmsi_is_tier_three() -> None:
    records = [AISRecord(mmsi="222", epoch_s=150.0, lat=36.70, lon=-122.0)]
    presences = correlate(records, _window())
    assert presences[0].registry_grade is False
    assert presences[0].truth_tier == 3
    assert presences[0].vessel_id == "MMSI222"


def test_out_of_window_or_range_excluded() -> None:
    records = [
        AISRecord(mmsi="1", epoch_s=50.0, lat=36.7, lon=-122.0, imo="1"),  # before window
        AISRecord(mmsi="2", epoch_s=150.0, lat=40.0, lon=-122.0, imo="2"),  # far away
    ]
    assert correlate(records, _window()) == []


def test_presence_summarizes_interval_and_closest_range() -> None:
    records = [
        AISRecord(mmsi="9", epoch_s=120.0, lat=36.72, lon=-122.0, name="BETA"),
        AISRecord(mmsi="9", epoch_s=180.0, lat=36.70, lon=-122.0, name="BETA"),
    ]
    presence = correlate(records, _window())[0]
    assert presence.first_seen_s == 120.0
    assert presence.last_seen_s == 180.0
    assert presence.closest_range_m == pytest.approx(0.0, abs=1.0)


def test_parse_marinecadastre_row() -> None:
    row = {
        "MMSI": "338123456",
        "BaseDateTime": "2020-06-01T12:00:00",
        "LAT": "36.7",
        "LON": "-122.0",
        "IMO": "IMO9111222",
        "VesselName": "GAMMA",
    }
    record = parse_marinecadastre_row(row)
    assert record.mmsi == "338123456"
    assert record.imo == "9111222"
    assert record.name == "GAMMA"
    assert record.lat == pytest.approx(36.7)


def test_parse_marinecadastre_blank_identity() -> None:
    row = {
        "MMSI": "1",
        "BaseDateTime": "2020-06-01T00:00:00",
        "LAT": "0",
        "LON": "0",
        "IMO": "",
        "VesselName": "",
    }
    record = parse_marinecadastre_row(row)
    assert record.imo is None
    assert record.name is None
