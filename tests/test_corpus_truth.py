"""Tests for AIS-to-recording correlation and vessel-presence truth."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from fathom.corpus.truth import (
    bbox_for,
    correlate_recording,
    read_ais_records,
    recording_date_key,
    recording_window,
)
from fathom.ledger import Ledger

_CSV_HEADER = "MMSI,BaseDateTime,LAT,LON,IMO,VesselName"
_ROWS = [
    "111,2015-07-28T12:00:00,36.71,-122.18,IMO9000001,ALPHA",  # near site, registry-grade
    "222,2015-07-28T12:00:00,40.00,-122.00,IMO9000002,BETA",  # far outside the box
    "333,2015-07-28T12:00:00,36.72,-122.19,,",  # near site, bare MMSI (tier three)
]


def _ais_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("AIS_2015_07_28.csv", "\n".join([_CSV_HEADER, *_ROWS]))
    return buffer.getvalue()


def _site() -> tuple[float, float]:
    return 36.7128, -122.186


def test_bbox_filter_excludes_distant_vessels() -> None:
    lat, lon = _site()
    bbox = bbox_for(lat, lon, 20_000.0)
    records = read_ais_records(_ais_zip(), bbox)
    mmsis = {r.mmsi for r in records}
    assert mmsis == {"111", "333"}  # the vessel at 40N is outside the box


def test_recording_window_from_filename() -> None:
    url = "s3://pacific-sound-2khz/2015/07/MARS-20150728T000000Z-2kHz.wav"
    window = recording_window(url, 86400.0)
    assert window is not None
    start, end = window
    assert start == datetime(2015, 7, 28, tzinfo=UTC).timestamp()
    assert end - start == 86400.0
    assert recording_date_key(url) == "2015_07_28"


def test_correlation_produces_registry_and_bare_presences() -> None:
    lat, lon = _site()
    records = read_ais_records(_ais_zip(), bbox_for(lat, lon, 20_000.0))
    window = recording_window(
        "s3://pacific-sound-2khz/2015/07/MARS-20150728T000000Z-2kHz.wav", 86400.0
    )
    assert window is not None
    rows = correlate_recording(
        corr_run="test-run",
        recording_sha="r" * 64,
        site="mars_monterey_bay",
        lat=lat,
        lon=lon,
        start_s=window[0],
        end_s=window[1],
        radius_m=20_000.0,
        ais_records=records,
        ais_source_sha="a" * 64,
    )
    by_vessel = {row["vessel_id"]: row for row in rows}
    assert by_vessel["IMO9000001"]["registry_grade"] == 1
    assert by_vessel["IMO9000001"]["truth_tier"] == 1
    assert by_vessel["MMSI333"]["registry_grade"] == 0
    assert by_vessel["MMSI333"]["truth_tier"] == 3


def test_quiet_tail_classification() -> None:
    from fathom.corpus.ais import AISRecord
    from fathom.corpus.truth import correlate_recording

    lat, lon = _site()
    # One close, slow, isolated vessel (quiet-tail) and one far, fast vessel (not).
    records = [
        AISRecord(mmsi="1", epoch_s=100.0, lat=36.713, lon=-122.186, imo="9001", sog=1.5),
        AISRecord(mmsi="2", epoch_s=100.0, lat=36.90, lon=-122.0, imo="9002", sog=18.0),
    ]
    rows = correlate_recording(
        corr_run="test-run",
        recording_sha="r" * 64,
        site="mars_monterey_bay",
        lat=lat,
        lon=lon,
        start_s=0.0,
        end_s=200.0,
        radius_m=30_000.0,
        ais_records=records,
        ais_source_sha="a" * 64,
    )
    by_vessel = {row["vessel_id"]: row for row in rows}
    assert by_vessel["IMO9001"]["quiet_tail"] == 1  # close, slow (1.5 kn), isolated
    assert by_vessel["IMO9001"]["min_sog"] == 1.5
    assert by_vessel["IMO9002"]["quiet_tail"] == 0  # far and fast


def test_window_only_recording_counts(tmp_path: Path) -> None:
    # A vessel-presence row keyed to a registered window (no audio) still resolves its source.
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        window_id = "w" * 64
        ledger.insert_recording_window(
            record={
                "window_id": window_id,
                "source_id": "sanctsound_sb01",
                "origin_url": "https://storage.googleapis.com/x/SB01_20190128T233033Z.flac",
                "site": "stellwagen_sb01",
                "sample_rate_hz": 48000.0,
                "start_epoch_s": 0.0,
                "end_epoch_s": 21600.0,
            }
        )
        ledger.insert_vessel_presence(
            record={
                "presence_id": "p1",
                "corr_run": "r1",
                "recording_sha": window_id,
                "site": "stellwagen_sb01",
                "vessel_id": "IMO1",
                "mmsi": "x",
                "imo": "1",
                "name": None,
                "first_seen_s": 0.0,
                "last_seen_s": 1.0,
                "closest_range_m": 100.0,
                "registry_grade": 1,
                "truth_tier": 1,
                "quiet_tail": 1,
                "ais_source_sha": "a" * 64,
            }
        )
        assert ledger.registry_vessel_counts_by_source() == {"sanctsound_sb01": 1}
        assert ledger.quiet_tail_vessel_counts_by_source() == {"sanctsound_sb01": 1}
        assert ledger.count_recording_windows("sanctsound_sb01") == 1
    finally:
        ledger.close()


def test_ledger_registry_vessel_counts(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.db")
    try:
        recording = "r" * 64
        ledger.insert_corpus_object(
            batch_id="b1",
            record={
                "sha256": recording,
                "source_id": "mbari_pacific_sound_2khz",
                "origin_url": "s3://pacific-sound-2khz/2015/07/MARS-20150728T000000Z-2kHz.wav",
                "retrieved_at": "2026-08-03T00:00:00+00:00",
                "byte_count": 1,
                "raw_key": f"raw/rr/{recording}",
                "license_class": "cc_by",
                "license_evidence_url": "https://example.org",
                "truth_condition": "tier1_ais_correlated",
                "training_eligible": 1,
                "media_type": "audio/wav",
                "site": "mars_monterey_bay",
                "instrument": "cabled_hydrophone",
                "sample_rate_hz": 2000.0,
                "band_low_hz": None,
                "band_high_hz": None,
                "band_partial": 0,
                "duration_s": 86400.0,
            },
        )
        for vessel, grade in (("IMO1", 1), ("IMO2", 1), ("MMSI3", 0)):
            ledger.insert_vessel_presence(
                record={
                    "presence_id": f"{recording}:{vessel}",
                    "corr_run": "run1",
                    "recording_sha": recording,
                    "site": "mars_monterey_bay",
                    "vessel_id": vessel,
                    "mmsi": "x",
                    "imo": None,
                    "name": None,
                    "first_seen_s": 0.0,
                    "last_seen_s": 1.0,
                    "closest_range_m": 100.0,
                    "registry_grade": grade,
                    "truth_tier": 1 if grade else 3,
                    "ais_source_sha": "a" * 64,
                }
            )
        # Two distinct registry-grade vessels; the bare-MMSI contact is excluded.
        assert ledger.registry_vessel_counts_by_source() == {"mbari_pacific_sound_2khz": 2}
        assert ledger.count_vessel_presence() == 3
    finally:
        ledger.close()
