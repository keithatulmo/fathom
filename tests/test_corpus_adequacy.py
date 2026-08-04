"""Tests for corpus adequacy scoring against CA1-CA7."""

from __future__ import annotations

from typing import Any

from fathom.corpus.adequacy import score_adequacy

_WHEN = "2026-08-03T00:00:00+00:00"


def _obj(
    source_id: str,
    site: str,
    sample_rate: float | None = None,
    *,
    sha: str | None = None,
    duration: float | None = None,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "site": site,
        "sample_rate_hz": sample_rate,
        "truth_condition": "tier1_ais_correlated",
        "sha256": sha or f"obj-{source_id}-{site}",
        "duration_s": duration,
    }


def _pres(
    vessel_id: str,
    rng: float = 15000.0,
    site: str = "mars_monterey_bay",
    quiet_tail: int = 0,
    recording_sha: str | None = None,
    audio_snr_db: float | None = None,
) -> dict[str, Any]:
    return {
        "vessel_id": vessel_id,
        "registry_grade": 1,
        "site": site,
        "closest_range_m": rng,
        "quiet_tail": quiet_tail,
        "recording_sha": recording_sha or f"window-{vessel_id}",
        "audio_snr_db": audio_snr_db,
    }


def _audio_pair(vessel_id: str, snr: float, site: str = "stellwagen_sb01") -> tuple[dict, dict]:
    """A stored audio object plus a quiet-tail presence backed by it, audible at the given SNR."""
    sha = f"audio-{vessel_id}"
    obj = _obj("sanctsound_sb01", site, 48000.0, sha=sha, duration=21600.0)
    pres = _pres(
        vessel_id, rng=1000.0, site=site, quiet_tail=1, recording_sha=sha, audio_snr_db=snr
    )
    return obj, pres


def _three_sites() -> list[dict[str, Any]]:
    return [
        _obj("mbari_pacific_sound_2khz", "mars_monterey_bay", 2000.0, duration=3600.0),
        _obj("adeon_ncei", "adeon_atlantic"),
        _obj("onc_strait_of_georgia", "strait_of_georgia"),
    ]


def _verdicts(report: Any) -> dict[str, str]:
    return {c.id: c.verdict for c in report.criteria}


def test_thin_corpus_matches_current_state() -> None:
    # 21 distant registry-grade vessels at one site, three sites present, no biologics.
    presences = [_pres(f"V{i}") for i in range(21)]
    report = score_adequacy(_three_sites(), presences, generated_at=_WHEN)
    verdicts = _verdicts(report)
    assert verdicts["CA5"] == "pass"  # three sites spanning the regimes
    assert verdicts["CA6"] == "marginal"  # proof site band known, others not
    assert verdicts["CA2"] == "fail"  # no audio-backed passages
    assert verdicts["CA3"] == "fail"  # 21 < 50
    assert verdicts["CA4"] == "fail"
    assert verdicts["CA7"] == "fail"  # only 21 verified tier-one vessels
    assert report.sd1_close_met is False
    assert "CA2" in report.blocking_failures
    assert "CA5" not in report.blocking_failures


def test_ca5_requires_all_three_regimes() -> None:
    objects = [
        _obj("mbari_pacific_sound_2khz", "mars_monterey_bay", 2000.0, duration=3600.0),
        _obj("adeon_ncei", "adeon_atlantic"),
    ]  # quiet and nominal only, no busy
    report = score_adequacy(objects, [], generated_at=_WHEN)
    assert _verdicts(report)["CA5"] == "fail"


def test_audio_backed_cohort_promotes_quiet_criteria_to_marginal() -> None:
    # 35 audio-backed quiet-tail vessels (each a stored object with SNR above the bar): CA3 count
    # met (marginal, not pass) and CA2 count met (marginal), pending the disjoint split assignment.
    objects = list(_three_sites())
    presences: list[dict[str, Any]] = []
    for i in range(35):
        obj, pres = _audio_pair(f"Q{i}", snr=8.0)
        objects.append(obj)
        presences.append(pres)
    presences += [_pres(f"F{i}", rng=15000.0) for i in range(20)]
    report = score_adequacy(objects, presences, generated_at=_WHEN)
    verdicts = _verdicts(report)
    assert verdicts["CA2"] == "marginal"
    assert verdicts["CA3"] == "marginal"
    assert report.sd1_close_met is False
    assert report.table["measured"]["distinct_audio_backed_quiet_tail_vessels"] == 35


def test_kinematic_quiet_without_audio_does_not_satisfy_ca2_ca3() -> None:
    # 35 vessels flagged quiet-tail from AIS kinematics but with no stored audio object behind them
    # (recording is a window): the review's failure mode. They are candidates only, so CA2/CA3 fail.
    presences = [_pres(f"Q{i}", rng=1000.0, quiet_tail=1) for i in range(35)]
    presences += [_pres(f"F{i}", rng=15000.0) for i in range(20)]
    report = score_adequacy(_three_sites(), presences, generated_at=_WHEN)
    verdicts = _verdicts(report)
    assert verdicts["CA2"] == "fail"
    assert verdicts["CA3"] == "fail"
    assert report.sd1_close_met is False
    assert report.table["measured"]["distinct_kinematic_quiet_tail_candidates"] == 35
    assert report.table["measured"]["distinct_audio_backed_quiet_tail_vessels"] == 0


def test_audio_backing_requires_snr_above_the_bar() -> None:
    # A stored audio object is present, but the passage is below the audibility bar, so it does not
    # count. The scorer reads only persisted SNR and the bar, so the count is auditable.
    objects = list(_three_sites())
    presences: list[dict[str, Any]] = []
    for i in range(30):
        obj, pres = _audio_pair(f"Q{i}", snr=4.0)  # below the 6 dB default
        objects.append(obj)
        presences.append(pres)
    report = score_adequacy(objects, presences, generated_at=_WHEN)
    assert report.table["measured"]["distinct_audio_backed_quiet_tail_vessels"] == 0
    # Lowering the bar admits them, since the SNR is persisted and the threshold is a parameter.
    report_low = score_adequacy(objects, presences, snr_threshold_db=3.0, generated_at=_WHEN)
    assert report_low.table["measured"]["distinct_audio_backed_quiet_tail_vessels"] == 30


def test_measured_rate_covers_ca6_for_present_audio() -> None:
    # ADEON and ONC carry decodable audio (a duration) but no rate on the object row.
    objects = [
        _obj("mbari_pacific_sound_2khz", "mars_monterey_bay", 2000.0, duration=3600.0),
        _obj("adeon_ncei", "adeon_atlantic", duration=539.0),
        _obj("onc_strait_of_georgia", "strait_of_georgia", duration=300.0),
    ]
    presences = [_pres(f"V{i}") for i in range(21)]
    # Without a measured rate, both present-audio sources are flagged as uncaptured.
    base = score_adequacy(objects, presences, generated_at=_WHEN)
    assert set(base.table["measured"]["audio_sources_missing_sample_rate"]) == {
        "adeon_ncei",
        "onc_strait_of_georgia",
    }
    # A rate measured from the stored header credits their band coverage and clears the gap.
    report = score_adequacy(
        objects,
        presences,
        measured_source_rates={"adeon_ncei": 16000.0, "onc_strait_of_georgia": 64000.0},
        generated_at=_WHEN,
    )
    assert _verdicts(report)["CA6"] == "pass"
    assert report.table["measured"]["audio_sources_missing_sample_rate"] == []


def test_biologics_absent_makes_ca1_marginal() -> None:
    report = score_adequacy(_three_sites(), [_pres("V0")], generated_at=_WHEN)
    assert _verdicts(report)["CA1"] == "marginal"
