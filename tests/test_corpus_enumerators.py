"""Tests for the source enumerators and the per-source planner (offline)."""

from __future__ import annotations

import pytest

from fathom.corpus.ais import marinecadastre_daily_urls
from fathom.corpus.gcs import gcs_object_url, parse_gcs_listing
from fathom.corpus.onc import onc_download_url, onc_origin_url, parse_archivefiles
from fathom.corpus.ooi import parse_apache_index
from fathom.corpus.planners import PlanningError, plan_items
from fathom.corpus.sources import AccessClass, LicenseClass, SourceSpec, source_by_id

# -- GCS -------------------------------------------------------------------------------------


def test_parse_gcs_listing() -> None:
    payload = {
        "items": [{"name": "ADEON/a.wav"}, {"name": "ADEON/b.flac"}],
        "nextPageToken": "tok2",
    }
    names, token = parse_gcs_listing(payload)
    assert names == ["ADEON/a.wav", "ADEON/b.flac"]
    assert token == "tok2"


def test_parse_gcs_listing_last_page() -> None:
    names, token = parse_gcs_listing({"items": [{"name": "x"}]})
    assert names == ["x"]
    assert token is None


def test_gcs_object_url_quotes_name() -> None:
    url = gcs_object_url("noaa-passive-bioacoustic", "SanctSound/site 01/file.wav")
    assert (
        url
        == "https://storage.googleapis.com/noaa-passive-bioacoustic/SanctSound/site%2001/file.wav"
    )


# -- OOI Apache index ------------------------------------------------------------------------


def test_parse_apache_index_returns_files_only() -> None:
    html = """
    <html><body>
    <a href="?C=N;O=D">Name</a>
    <a href="/files/">Parent Directory</a>
    <a href="2015/">2015/</a>
    <a href="OO-HYVM1--2015-07-28T00.00.00.mseed">file1</a>
    <a href="OO-HYVM1--2015-07-28T01.00.00.mseed">file2</a>
    </body></html>
    """
    base = "https://rawdata.oceanobservatories.org/files/RS01SLBS/dir/"
    files = parse_apache_index(html, base)
    assert files == [
        base + "OO-HYVM1--2015-07-28T00.00.00.mseed",
        base + "OO-HYVM1--2015-07-28T01.00.00.mseed",
    ]


# -- MarineCadastre AIS URLs -----------------------------------------------------------------


def test_marinecadastre_daily_urls() -> None:
    urls = marinecadastre_daily_urls("2020-06-01", "2020-06-03")
    assert urls == [
        "https://chs.coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/AIS_2020_06_01.zip",
        "https://chs.coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/AIS_2020_06_02.zip",
        "https://chs.coast.noaa.gov/htdata/CMSP/AISDataHandler/2020/AIS_2020_06_03.zip",
    ]


def test_marinecadastre_reversed_range_raises() -> None:
    with pytest.raises(ValueError):
        marinecadastre_daily_urls("2020-06-03", "2020-06-01")


# -- ONC -------------------------------------------------------------------------------------


def test_parse_archivefiles_strings_and_dicts() -> None:
    assert parse_archivefiles({"files": ["A.wav", "B.wav"]}) == ["A.wav", "B.wav"]
    assert parse_archivefiles({"files": [{"filename": "C.wav"}]}) == ["C.wav"]
    assert parse_archivefiles({}) == []


def test_onc_origin_url_is_token_free() -> None:
    origin = onc_origin_url("ICLISTENHF1234_20200601.wav")
    assert "token" not in origin
    assert "filename=ICLISTENHF1234_20200601.wav" in origin


def test_onc_download_url_carries_token() -> None:
    download = onc_download_url("f.wav", "SECRET-TOKEN")
    assert "token=SECRET-TOKEN" in download


# -- planner dispatch ------------------------------------------------------------------------


def test_plan_explicit_urls() -> None:
    spec = source_by_id("dclde_baleen")
    items = plan_items(
        spec, urls=["https://example.org/set/a.wav", "https://example.org/set/b.wav"]
    )
    assert [i.url for i in items] == [
        "https://example.org/set/a.wav",
        "https://example.org/set/b.wav",
    ]
    assert items[0].origin_url == items[0].url


def test_plan_marinecadastre_by_date_range() -> None:
    spec = source_by_id("marinecadastre_ais")
    items = plan_items(spec, ais_start="2020-06-01", ais_end="2020-06-02")
    assert len(items) == 2
    assert (items[0].url or "").endswith("AIS_2020_06_01.zip")


def test_plan_https_without_params_raises() -> None:
    spec = source_by_id("marinecadastre_ais")
    with pytest.raises(PlanningError):
        plan_items(spec)


def test_plan_unsupported_access_class_raises() -> None:
    spec = SourceSpec(
        source_id="fdsn_src",
        name="FDSN source",
        family=1,
        role="low-frequency",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://example.org",
        access_class=AccessClass.FDSN,
        truth_condition_default="tier1_verified_quiet",
        cap_gb=1.0,
    )
    with pytest.raises(PlanningError):
        plan_items(spec)
