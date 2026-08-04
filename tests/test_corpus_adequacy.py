"""Tests for corpus adequacy scoring against CA1-CA7."""

from __future__ import annotations

from typing import Any

from fathom.corpus.adequacy import score_adequacy

_WHEN = "2026-08-03T00:00:00+00:00"


def _obj(source_id: str, site: str, sample_rate: float | None = None) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "site": site,
        "sample_rate_hz": sample_rate,
        "truth_condition": "tier1_ais_correlated",
    }


def _pres(
    vessel_id: str,
    rng: float = 15000.0,
    site: str = "mars_monterey_bay",
    quiet_tail: int = 0,
    audio_quiet_tail: int = 0,
) -> dict[str, Any]:
    return {
        "vessel_id": vessel_id,
        "registry_grade": 1,
        "site": site,
        "closest_range_m": rng,
        "quiet_tail": quiet_tail,
        "audio_quiet_tail": audio_quiet_tail,
    }


def _three_sites() -> list[dict[str, Any]]:
    return [
        _obj("mbari_pacific_sound_2khz", "mars_monterey_bay", 2000.0),
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
    assert verdicts["CA2"] == "fail"  # no close passages, below 24
    assert verdicts["CA3"] == "fail"  # 21 < 50
    assert verdicts["CA4"] == "fail"
    assert verdicts["CA7"] == "fail"  # only 21 verified tier-one vessels
    assert report.sd1_close_met is False
    assert "CA2" in report.blocking_failures
    assert "CA5" not in report.blocking_failures


def test_ca5_requires_all_three_regimes() -> None:
    objects = [
        _obj("mbari_pacific_sound_2khz", "mars_monterey_bay", 2000.0),
        _obj("adeon_ncei", "adeon_atlantic"),
    ]  # quiet and nominal only, no busy
    report = score_adequacy(objects, [], generated_at=_WHEN)
    assert _verdicts(report)["CA5"] == "fail"


def test_audio_backed_quiet_tail_cohort_promotes_quiet_criteria_to_marginal() -> None:
    # 55 registry-grade vessels, 35 audio-backed quiet-tail: CA3 counts met (marginal, not pass) and
    # CA2 count met (marginal), because the disjoint split assignment is still pending.
    presences = [_pres(f"Q{i}", rng=1000.0, quiet_tail=1, audio_quiet_tail=1) for i in range(35)]
    presences += [_pres(f"F{i}", rng=15000.0) for i in range(20)]
    report = score_adequacy(_three_sites(), presences, generated_at=_WHEN)
    verdicts = _verdicts(report)
    assert verdicts["CA2"] == "marginal"
    assert verdicts["CA3"] == "marginal"
    # Marginal on a binding criterion still blocks the SD1 close.
    assert report.sd1_close_met is False


def test_kinematic_quiet_without_audio_does_not_satisfy_ca2_ca3() -> None:
    # 35 vessels flagged quiet-tail from AIS kinematics but with no audio behind them: the review's
    # failure mode. Without audio backing they are candidates only, so CA2 and CA3 still fail.
    presences = [_pres(f"Q{i}", rng=1000.0, quiet_tail=1, audio_quiet_tail=0) for i in range(35)]
    presences += [_pres(f"F{i}", rng=15000.0) for i in range(20)]
    report = score_adequacy(_three_sites(), presences, generated_at=_WHEN)
    verdicts = _verdicts(report)
    assert verdicts["CA2"] == "fail"
    assert verdicts["CA3"] == "fail"
    assert report.sd1_close_met is False
    # The kinematic candidates are still reported, so the attrition is visible.
    assert report.table["measured"]["distinct_kinematic_quiet_tail_candidates"] == 35
    assert report.table["measured"]["distinct_audio_backed_quiet_tail_vessels"] == 0


def test_biologics_absent_makes_ca1_marginal() -> None:
    report = score_adequacy(_three_sites(), [_pres("V0")], generated_at=_WHEN)
    assert _verdicts(report)["CA1"] == "marginal"
