"""Tests for the corpus audit report generator."""

from __future__ import annotations

from typing import Any

from fathom.corpus.audit import build_audit_report


def _object(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "source_id": "mbari_pacific_sound_2khz",
        "byte_count": 2_000_000_000,
        "duration_s": 3600.0,
        "site": "mars_monterey_bay",
        "band_low_hz": 10.0,
        "band_high_hz": 300.0,
        "band_partial": 0,
        "truth_condition": "tier1_ais_correlated",
        "training_eligible": 1,
    }
    base.update(overrides)
    return base


def test_report_summarizes_source() -> None:
    report = build_audit_report([_object(), _object()], generated_at="2026-08-03T00:00:00+00:00")
    rows = report.table["per_source"]
    assert len(rows) == 1
    row = rows[0]
    assert row["source_id"] == "mbari_pacific_sound_2khz"
    assert row["family"] == 2
    assert row["objects"] == 2
    assert row["hours"] == 2.0
    assert row["truth_tiers"] == {1: 2, 2: 0, 3: 0}
    assert "# Fathom corpus audit report" in report.markdown


def test_quiet_anchor_ranking_prefers_family_one() -> None:
    objects = [
        _object(
            source_id="ooi_slope_base_broadband",
            site="slope_base",
            duration_s=7200.0,
            truth_condition="tier1_verified_quiet",
        ),
        _object(
            source_id="ooi_axial_base_broadband",
            site="axial_base",
            duration_s=3600.0,
            truth_condition="tier1_verified_quiet",
        ),
    ]
    report = build_audit_report(objects, generated_at="2026-08-03T00:00:00+00:00")
    ranking = report.table["quiet_anchor_ranking"]
    assert [candidate["source_id"] for candidate in ranking] == [
        "ooi_slope_base_broadband",
        "ooi_axial_base_broadband",
    ]


def test_gaps_flag_missing_vessel_truth_and_busy_site() -> None:
    report = build_audit_report([_object()], generated_at="2026-08-03T00:00:00+00:00")
    criteria = {gap["criterion"] for gap in report.table["gaps"]}
    assert "truth quality" in criteria  # no vessel counts supplied
    assert "site diversity" in criteria  # no busy regime acquired


def test_not_acquired_lists_deferred_and_quarantined() -> None:
    report = build_audit_report([_object()], generated_at="2026-08-03T00:00:00+00:00")
    not_acquired = {entry["source_id"] for entry in report.table["not_acquired"]}
    assert "onc_strait_of_georgia" in not_acquired
    assert "deepship" in not_acquired
    assert "shipsear" in not_acquired


def test_empty_corpus_reports_volume_gap() -> None:
    report = build_audit_report([], generated_at="2026-08-03T00:00:00+00:00")
    assert report.table["per_source"] == []
    assert any(gap["criterion"] == "volume" for gap in report.table["gaps"])
