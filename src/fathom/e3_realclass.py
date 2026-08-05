"""The WO-9 real-class detection test that settles the SD5 close.

E3 closed its bake-off on the learned detector at 0.95, but every positive it saw was the closed
hybrid surrogate, and the winning detector reads a broadband distributional lift rather than the
lines, the fitted, boundary-member part of the surrogate that no experiment has compared to a
real quiet vessel. This module runs the three detectors, frozen exactly as E3 committed them: the
learned detector with its committed weights and each detector with its E3 threshold, against the
held-out quiet vessel class, so the detection rate is measured on real targets rather than synthetic
ones. Nothing is retrained and nothing is retuned. Real quiet-vessel windows are the positives and
real background windows the negatives; each is processed through the closed SD4 front end, cut into
dwell subwindows, and scored, and the detection rate on the vessels and the false-alarm rate on the
background are read at the frozen E3 threshold, per site, against the vessels' measured signal-to-
noise, and at the four-hertz floor. The branch the numbers select is reported, not argued: a learned
detector confirmed on the real class closes SD5, a learned failure reopens SD2 and E3, and a line
success reopens the surrogate's broadband balance. It asserts no absolute level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .config import E3Config
from .detectors import (
    LearnedDetector,
    cfar_peak_score,
    comb_features,
    harmonic_integration_score,
)
from .frontend import closed_front_end, lofar_gram
from .normalize import normalize_gram

_DETECTORS = ("cfar", "integration", "learned")


def _full_band(config: E3Config) -> tuple[float, float]:
    return (config.band_low_hz, config.band_high_hz)


def _floor_band(config: E3Config) -> tuple[float, float]:
    return (config.floor_band_low_hz, config.floor_band_high_hz)


def _subwindow_scores(
    samples: npt.NDArray[np.float64], learned: LearnedDetector, config: E3Config
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Return per-subwindow full-band and floor-band scores for the three frozen detectors."""
    params = closed_front_end()
    rate = config.analysis_rate_hz
    full = _full_band(config)
    floor = _floor_band(config)
    subwindow_n = int(round(config.subwindow_seconds * rate))
    count = max(
        1, min(config.max_subwindows_per_background, samples.shape[0] // max(1, subwindow_n))
    )
    full_scores: dict[str, list[float]] = {d: [] for d in _DETECTORS}
    floor_scores: dict[str, list[float]] = {d: [] for d in _DETECTORS}
    for i in range(count):
        window = np.ascontiguousarray(samples[i * subwindow_n : (i + 1) * subwindow_n])
        if window.shape[0] < subwindow_n // 2:
            continue
        gram, freqs = lofar_gram(window, rate, params)
        normalized, nf = normalize_gram(gram, freqs, full, params)
        full_scores["cfar"].append(cfar_peak_score(normalized, nf, full))
        full_scores["integration"].append(
            harmonic_integration_score(normalized, nf, full, config.dwell_frames)
        )
        full_scores["learned"].append(
            learned.score(comb_features(normalized, nf, full, config.dwell_frames))
        )
        floor_scores["cfar"].append(cfar_peak_score(normalized, nf, floor))
        floor_scores["integration"].append(
            harmonic_integration_score(normalized, nf, floor, config.dwell_frames)
        )
        floor_scores["learned"].append(
            learned.score(comb_features(normalized, nf, floor, config.dwell_frames))
        )
    return full_scores, floor_scores


def _rate(scores: list[float], threshold: float) -> float:
    return float(np.mean(np.array(scores) >= threshold)) if scores else 0.0


def run_e3_realclass(
    vessel_windows: list[dict[str, Any]],
    background_windows: list[dict[str, Any]],
    learned: LearnedDetector,
    thresholds: dict[str, float],
    floor_thresholds: dict[str, float],
    surrogate_reference: dict[str, float],
    leak_attestation: dict[str, Any],
    config: E3Config,
) -> dict[str, Any]:
    """Run the three frozen detectors against the real quiet class; return the committed payload."""
    if len(vessel_windows) < 2 or len(background_windows) < 2:
        raise ValueError("real-class test needs at least two vessels and two backgrounds")

    # Per-vessel detection: the fraction of a vessel's subwindows that clear the frozen threshold.
    per_vessel: dict[str, list[dict[str, Any]]] = {d: [] for d in _DETECTORS}
    for entry in vessel_windows:
        full_scores, floor_scores = _subwindow_scores(
            np.asarray(entry["samples"], dtype=np.float64), learned, config
        )
        for detector in _DETECTORS:
            per_vessel[detector].append(
                {
                    "vessel_id": entry["vessel_id"],
                    "site": entry.get("site"),
                    "snr_db": float(entry["snr_db"]),
                    "detection_rate": _rate(full_scores[detector], thresholds[detector]),
                    "floor_detection_rate": _rate(
                        floor_scores[detector], floor_thresholds[detector]
                    ),
                }
            )

    # False-alarm rate on real background, pooled over all background subwindows, at the threshold.
    background_scores: dict[str, list[float]] = {d: [] for d in _DETECTORS}
    for entry in background_windows:
        full_scores, _ = _subwindow_scores(
            np.asarray(entry["samples"], dtype=np.float64), learned, config
        )
        for detector in _DETECTORS:
            background_scores[detector].extend(full_scores[detector])

    per_detector: list[dict[str, Any]] = []
    op_lo, op_hi = config.operating_snr_db - 2.0, config.operating_snr_db + 2.0
    for detector in _DETECTORS:
        rows = per_vessel[detector]
        overall = float(np.mean([r["detection_rate"] for r in rows]))
        operating = [r["detection_rate"] for r in rows if op_lo <= r["snr_db"] <= op_hi]
        per_site: dict[str, float] = {}
        for site in sorted({str(r["site"]) for r in rows}):
            site_rows = [r["detection_rate"] for r in rows if str(r["site"]) == site]
            per_site[site] = float(np.mean(site_rows))
        per_detector.append(
            {
                "name": detector,
                "real_detection_rate": overall,
                "operating_point_detection_rate": (
                    float(np.mean(operating)) if operating else float("nan")
                ),
                "floor_4hz_detection_rate": float(
                    np.mean([r["floor_detection_rate"] for r in rows])
                ),
                "false_alarm_rate": _rate(background_scores[detector], thresholds[detector]),
                "per_site_detection_rate": per_site,
                "e3_surrogate_operating_sensitivity": surrogate_reference.get(
                    detector, float("nan")
                ),
                "detection_vs_snr": sorted(
                    (
                        {
                            "vessel_id": r["vessel_id"],
                            "snr_db": r["snr_db"],
                            "detection_rate": r["detection_rate"],
                        }
                        for r in rows
                    ),
                    key=lambda r: float(r["snr_db"]),
                ),
            }
        )

    branch = _select_branch(per_detector, config)
    return {
        "front_end": closed_front_end().key(),
        "operating_snr_db": config.operating_snr_db,
        "success_floor": config.realclass_success_floor,
        "band_hz": list(_full_band(config)),
        "floor_band_hz": list(_floor_band(config)),
        "analysis_rate_hz": config.analysis_rate_hz,
        "e3_thresholds": thresholds,
        "e3_floor_thresholds": floor_thresholds,
        "leak_attestation": leak_attestation,
        "cohort": {
            "n_vessels": len(vessel_windows),
            "n_backgrounds": len(background_windows),
            "snr_range_db": [
                min(float(v["snr_db"]) for v in vessel_windows),
                max(float(v["snr_db"]) for v in vessel_windows),
            ],
        },
        "per_detector": per_detector,
        "branch": branch,
        "owner_certified": config.owner_certified.model_dump(mode="json"),
    }


def _by_name(per_detector: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(d for d in per_detector if d["name"] == name)


def _select_branch(per_detector: list[dict[str, Any]], config: E3Config) -> dict[str, Any]:
    """Select the SD5 fork branch from the real-class detection rates, retuning nothing.

    The branches are ordered so a surrogate defect is surfaced before a close: if a line detector
    detects the real vessels where it failed on the surrogate, the surrogate's lines are too weak
    relative to its broadband and SD2's balance reopens; else if the learned detector detects the
    vessels, its E3 win is a real capability and SD5 closes; else the learned win was a surrogate
    artifact and SD2 and E3 reopen. A detector "detects" when its rate reaches the success
    floor and clears its own real-background false-alarm rate.
    """
    floor = config.realclass_success_floor
    learned = _by_name(per_detector, "learned")
    cfar = _by_name(per_detector, "cfar")
    integration = _by_name(per_detector, "integration")

    def detects(row: dict[str, Any]) -> bool:
        return bool(
            row["real_detection_rate"] >= floor
            and row["real_detection_rate"] > 2.0 * row["false_alarm_rate"]
        )

    lines_detect = detects(cfar) or detects(integration)
    learned_detects = detects(learned)

    if lines_detect:
        best_line = max((cfar, integration), key=lambda d: d["real_detection_rate"])
        return {
            "selected": "lines_succeed_on_real",
            "closes_sd5": False,
            "reason": (
                f"A line detector ({best_line['name']!r}) detects the real quiet vessels at "
                f"{best_line['real_detection_rate']:.2f} where it failed on the surrogate "
                f"({best_line['e3_surrogate_operating_sensitivity']:.2f}); the surrogate lines are "
                "too weak relative to its broadband, so SD2's broadband balance reopens."
            ),
        }
    if learned_detects:
        return {
            "selected": "learned_confirmed",
            "closes_sd5": True,
            "reason": (
                f"The learned detector detects the real quiet vessels at "
                f"{learned['real_detection_rate']:.2f} (surrogate {surrogate_str(learned)}) while "
                "line detectors fail, so the surrogate is realistic on the detection axis and the "
                "learned detector is a real capability; SD5 closes on it."
            ),
        }
    return {
        "selected": "learned_artifact",
        "closes_sd5": False,
        "reason": (
            f"The learned detector detects the real quiet vessels at only "
            f"{learned['real_detection_rate']:.2f} against its surrogate "
            f"{surrogate_str(learned)}, so its E3 win was a surrogate-broadband artifact; "
            "SD2's broadband balance and E3 reopen."
        ),
    }


def surrogate_str(row: dict[str, Any]) -> str:
    value = row["e3_surrogate_operating_sensitivity"]
    return f"{value:.2f}" if value == value else "n/a"  # value != value is True only for NaN
