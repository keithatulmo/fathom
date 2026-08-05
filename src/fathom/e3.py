"""The E3 detection bake-off that closes SD5 by measurement.

WO-8 runs three narrowband detectors (:mod:`fathom.detectors`) on the output of the closed SD4 front
end (:func:`fathom.frontend.closed_front_end`), so the bake-off measures the detector and nothing
else. The closed SD2 hybrid surrogate is injected across a signal-to-noise ladder onto real
background from the quiet, nominal, and busy sites, through the closed front end; each detector
reduces every trial to a graded soft score, one cross-site threshold per detector is set at the
fixed false-alarm rate, and sensitivity is the detection rate on the injected targets at that
threshold, read as a curve against signal-to-noise and reported at the operating point and at the
four-hertz shaft floor. False-alarm stability is whether the one threshold holds the false-alarm
rate across the three sites. The learned detector trains on train-side backgrounds only, disjoint
from the eval backgrounds it is scored on, which is the leak rule. The operating point and threshold
are set
once across sites, never per site. This module holds the pure bake-off over already-loaded
backgrounds, so it is testable apart from the ledger and the object store, and asserts no level.
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
from .determinism import derive_seed
from .e2 import synth_hybrid_probe
from .frontend import closed_front_end, lofar_gram
from .normalize import normalize_gram
from .surrogate.fit import SurrogateDistributions
from .surrogate.level import inject_at_snr

_DETECTORS = ("cfar", "integration", "learned")
_IMPLEMENTATION_RISK = {"cfar": "low", "integration": "moderate", "learned": "high"}


def _full_band(config: E3Config) -> tuple[float, float]:
    return (config.band_low_hz, config.band_high_hz)


def _floor_band(config: E3Config) -> tuple[float, float]:
    return (config.floor_band_low_hz, config.floor_band_high_hz)


def _normalized(
    signal: npt.NDArray[np.float64], config: E3Config
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Run the closed SD4 front end and normalizer on a signal, returning the in-band gram and
    freqs."""
    params = closed_front_end()
    gram, freqs = lofar_gram(signal, config.analysis_rate_hz, params)
    return normalize_gram(gram, freqs, _full_band(config), params)


def _scores(
    normalized: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    band: tuple[float, float],
    config: E3Config,
    learned: LearnedDetector,
) -> dict[str, float]:
    """Return the three detectors' soft scores for one normalized gram over the given band."""
    return {
        "cfar": cfar_peak_score(normalized, freqs, band),
        "integration": harmonic_integration_score(normalized, freqs, band, config.dwell_frames),
        "learned": learned.score(comb_features(normalized, freqs, band, config.dwell_frames)),
    }


def _subwindows(
    signal: npt.NDArray[np.float64], subwindow_n: int, max_windows: int
) -> list[npt.NDArray[np.float64]]:
    """Slice a signal into non-overlapping subwindows of the dwell length, capped in count."""
    if subwindow_n <= 0:
        return [signal]
    count = min(max_windows, signal.shape[0] // subwindow_n)
    return [
        np.ascontiguousarray(signal[i * subwindow_n : (i + 1) * subwindow_n], dtype=np.float64)
        for i in range(max(1, count))
    ]


def _all_subwindows(
    backgrounds_by_regime: dict[str, list[npt.NDArray[np.float64]]], config: E3Config
) -> dict[str, list[npt.NDArray[np.float64]]]:
    subwindow_n = int(round(config.subwindow_seconds * config.analysis_rate_hz))
    out: dict[str, list[npt.NDArray[np.float64]]] = {}
    for regime, backgrounds in backgrounds_by_regime.items():
        windows: list[npt.NDArray[np.float64]] = []
        for background in backgrounds:
            windows.extend(
                _subwindows(background, subwindow_n, config.max_subwindows_per_background)
            )
        out[regime] = windows
    return out


def _inject(
    subwindow: npt.NDArray[np.float64],
    distributions: SurrogateDistributions,
    config: E3Config,
    snr_db: float,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Inject one closed hybrid surrogate onto a background subwindow at the given
    signal-to-noise."""
    signal, _ = synth_hybrid_probe(distributions, config, subwindow.shape[0], seed)
    injected, _ = inject_at_snr(
        signal, subwindow, _full_band(config), config.analysis_rate_hz, snr_db
    )
    return injected


def _train_learned(
    train_windows: dict[str, list[npt.NDArray[np.float64]]],
    distributions: SurrogateDistributions,
    config: E3Config,
    seed: int,
) -> LearnedDetector:
    """Train the learned detector on train-side backgrounds only, injected as labeled targets.

    Each train subwindow yields a negative example from the background and a positive example from
    the
    same background with the closed hybrid injected at the operating point, so the detector learns
    to
    separate an injected quiet target from real background under the leak rule, never seeing the
    eval
    backgrounds it is later scored on.
    """
    band = _full_band(config)
    features: list[npt.NDArray[np.float64]] = []
    labels: list[float] = []
    for regime, windows in sorted(train_windows.items()):
        for index, window in enumerate(windows):
            negative, negative_freqs = _normalized(window, config)
            features.append(comb_features(negative, negative_freqs, band, config.dwell_frames))
            labels.append(0.0)
            injected = _inject(
                window,
                distributions,
                config,
                config.operating_snr_db,
                derive_seed(seed, f"train-{regime}-{index}"),
            )
            positive, positive_freqs = _normalized(injected, config)
            features.append(comb_features(positive, positive_freqs, band, config.dwell_frames))
            labels.append(1.0)
    return LearnedDetector.fit(np.array(features), np.array(labels))


def _pearson(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    if x.shape[0] < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_e3_bakeoff(
    train_backgrounds_by_regime: dict[str, list[npt.NDArray[np.float64]]],
    eval_backgrounds_by_regime: dict[str, list[npt.NDArray[np.float64]]],
    site_names_by_regime: dict[str, list[str]],
    distributions: SurrogateDistributions,
    config: E3Config,
    seed: int,
) -> dict[str, Any]:
    """Run the full E3 bake-off and return the committed-state payload with the selection and
    flip."""
    regimes = [r for r in ("quiet", "nominal", "busy") if eval_backgrounds_by_regime.get(r)]
    if len(regimes) < 2:
        raise ValueError("E3 needs at least two site regimes with eval background")
    full = _full_band(config)
    floor = _floor_band(config)
    ladder = list(config.snr_ladder_db)

    train_windows = _all_subwindows(train_backgrounds_by_regime, config)
    eval_windows = _all_subwindows(eval_backgrounds_by_regime, config)
    learned = _train_learned(train_windows, distributions, config, derive_seed(seed, "train"))

    # Background scores per detector and site, full band and floor band, computed once per
    # subwindow.
    bg_full: dict[str, dict[str, list[float]]] = {d: {r: [] for r in regimes} for d in _DETECTORS}
    bg_floor: dict[str, dict[str, list[float]]] = {d: {r: [] for r in regimes} for d in _DETECTORS}
    for regime in regimes:
        for window in eval_windows[regime]:
            normalized, freqs = _normalized(window, config)
            full_scores = _scores(normalized, freqs, full, config, learned)
            floor_scores = _scores(normalized, freqs, floor, config, learned)
            for detector in _DETECTORS:
                bg_full[detector][regime].append(full_scores[detector])
                bg_floor[detector][regime].append(floor_scores[detector])

    # One cross-site threshold per detector at the fixed false-alarm rate, on the pooled background.
    threshold_full = {
        d: float(
            np.quantile(np.concatenate([bg_full[d][r] for r in regimes]), 1.0 - config.target_far)
        )
        for d in _DETECTORS
    }
    threshold_floor = {
        d: float(
            np.quantile(np.concatenate([bg_floor[d][r] for r in regimes]), 1.0 - config.target_far)
        )
        for d in _DETECTORS
    }

    # Detection scores per detector, site, and rung; the target is one hybrid per subwindow swept in
    # level, so the sensitivity curve is the same target seen at each signal-to-noise.
    detection_full: dict[str, dict[str, dict[float, list[float]]]] = {
        d: {r: {snr: [] for snr in ladder} for r in regimes} for d in _DETECTORS
    }
    detection_floor_op: dict[str, dict[str, list[float]]] = {
        d: {r: [] for r in regimes} for d in _DETECTORS
    }
    for regime in regimes:
        for index, window in enumerate(eval_windows[regime]):
            base = synth_hybrid_probe(
                distributions, config, window.shape[0], derive_seed(seed, f"eval-{regime}-{index}")
            )[0]
            for snr in ladder:
                injected, _ = inject_at_snr(base, window, full, config.analysis_rate_hz, snr)
                normalized, freqs = _normalized(injected, config)
                full_scores = _scores(normalized, freqs, full, config, learned)
                for detector in _DETECTORS:
                    detection_full[detector][regime][snr].append(full_scores[detector])
                if snr == config.operating_snr_db:
                    floor_scores = _scores(normalized, freqs, floor, config, learned)
                    for detector in _DETECTORS:
                        detection_floor_op[detector][regime].append(floor_scores[detector])

    candidates = _assemble_candidates(
        config,
        regimes,
        ladder,
        bg_full,
        threshold_full,
        threshold_floor,
        detection_full,
        detection_floor_op,
    )
    selected, runner_ups, flip = _select_and_flip(candidates, config)

    return {
        "operating_snr_db": config.operating_snr_db,
        "snr_ladder_db": ladder,
        "target_far": config.target_far,
        "far_tolerance": config.far_tolerance,
        "learned_material_margin": config.learned_material_margin,
        "band_hz": list(full),
        "floor_band_hz": list(floor),
        "analysis_rate_hz": config.analysis_rate_hz,
        "front_end": closed_front_end().key(),
        "corpus_slice": {
            regime: {
                "eval_backgrounds": len(eval_backgrounds_by_regime.get(regime, [])),
                "train_backgrounds": len(train_backgrounds_by_regime.get(regime, [])),
                "eval_subwindows": len(eval_windows.get(regime, [])),
                "sites": site_names_by_regime.get(regime, []),
            }
            for regime in regimes
        },
        "injection_set": {
            "probe": "closed_sd2_hybrid_surrogate",
            "fit_provenance": distributions.provenance,
        },
        "learned_detector": learned.to_dict(),
        "leak_rule": "learned detector trained on train-side backgrounds only, disjoint from eval",
        "candidates": candidates,
        "selected": selected,
        "runner_ups": runner_ups,
        "flip_clause": flip,
        "verdict": {
            "selected_detector": selected["name"] if selected else None,
            "closes_sd5_on_measurement": bool(selected is not None),
            "note": (
                "SD5 closes on the detector that maximizes operating-point sensitivity subject to "
                "holding false-alarm stability across sites. A candidate that wins sensitivity but "
                "fails cross-site false-alarm stability does not close SD5 on sensitivity alone; "
                "the selection falls to the one that holds both (flip). Provisional until the SD6 "
                "rejector consumes the detector's soft scores."
            ),
        },
        "owner_certified": config.owner_certified.model_dump(mode="json"),
    }


def _assemble_candidates(
    config: E3Config,
    regimes: list[str],
    ladder: list[float],
    bg_full: dict[str, dict[str, list[float]]],
    threshold_full: dict[str, float],
    threshold_floor: dict[str, float],
    detection_full: dict[str, dict[str, dict[float, list[float]]]],
    detection_floor_op: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    """Reduce the raw scores to the per-candidate record the lineage and the selection consume."""
    window_seconds = closed_front_end().window_length_s
    lo = config.target_far / config.far_tolerance
    hi = config.target_far * config.far_tolerance
    candidates: list[dict[str, Any]] = []
    for detector in _DETECTORS:
        t_full = threshold_full[detector]
        per_site_far = {
            r: float(np.mean(np.array(bg_full[detector][r]) >= t_full)) for r in regimes
        }
        fa_values = list(per_site_far.values())
        fa_stable = bool(all(lo <= v <= hi for v in fa_values))

        sensitivity_curve: dict[str, dict[str, float]] = {}
        op_per_site: dict[str, float] = {}
        cal_scores: list[float] = []
        cal_snrs: list[float] = []
        for regime in regimes:
            curve = {}
            for snr in ladder:
                scores = np.array(detection_full[detector][regime][snr])
                rate = float(np.mean(scores >= t_full)) if scores.size else 0.0
                curve[f"{snr:g}"] = rate
                cal_scores.extend(scores.tolist())
                cal_snrs.extend([snr] * scores.shape[0])
                if snr == config.operating_snr_db:
                    op_per_site[regime] = rate
            sensitivity_curve[regime] = curve
        operating_sensitivity = float(np.mean(list(op_per_site.values()))) if op_per_site else 0.0

        floor_scores = np.concatenate([detection_floor_op[detector][r] for r in regimes])
        floor_sensitivity = float(np.mean(floor_scores >= threshold_floor[detector]))

        candidates.append(
            {
                "name": detector,
                "operating_point_sensitivity": operating_sensitivity,
                "operating_point_sensitivity_per_site": op_per_site,
                "floor_4hz_sensitivity": floor_sensitivity,
                "sensitivity_curve": sensitivity_curve,
                "per_site_far": per_site_far,
                "false_alarm_stable": fa_stable,
                "threshold": t_full,
                "floor_threshold": threshold_floor[detector],
                "soft_score": {
                    "graded": True,
                    "snr_calibration": _pearson(np.array(cal_scores), np.array(cal_snrs)),
                },
                "latency_s": (window_seconds if detector == "cfar" else config.subwindow_seconds),
                "implementation_risk": _IMPLEMENTATION_RISK[detector],
            }
        )
    return candidates


def _select_and_flip(
    candidates: list[dict[str, Any]], config: E3Config
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """Select the detector maximizing operating-point sensitivity subject to false-alarm stability.

    Two conditions can move the selection off the raw sensitivity argmax and are reported rather
    than
    resolved silently. If the sensitivity winner fails cross-site false-alarm stability, the
    selection
    falls to the best stable candidate (the false-alarm-stability flip). And because the learned
    detector carries the highest implementation risk, if it wins sensitivity while stable but does
    not
    beat the best hand-built stable detector by the material margin, integration is selected and the
    learned detector's ceiling is the runner-up (the implementation-risk trade of the SD5 memo).
    """

    def sens(c: dict[str, Any]) -> float:
        return float(c["operating_point_sensitivity"])

    stable = [c for c in candidates if c["false_alarm_stable"]]
    if not stable:
        return (
            None,
            [],
            {
                "fired": True,
                "branch": "no_stable_candidate",
                "reason": (
                    "No detector held the false-alarm rate across sites under one threshold; none "
                    "closes SD5, because an unstable false-alarm rate breaks the queue bound the "
                    "rejection stage depends on."
                ),
            },
        )

    overall_best = max(candidates, key=sens)
    sensitivity_winner = max(stable, key=sens)
    stability_flip = bool(overall_best["name"] != sensitivity_winner["name"])

    hand_built = [c for c in stable if c["name"] != "learned"]
    best_hand = max(hand_built, key=sens) if hand_built else None
    risk_trade = bool(
        sensitivity_winner["name"] == "learned"
        and best_hand is not None
        and sens(sensitivity_winner) - sens(best_hand) < config.learned_material_margin
    )
    selected = best_hand if (risk_trade and best_hand is not None) else sensitivity_winner

    if risk_trade and best_hand is not None:
        branch = "implementation_risk_trade"
        reason = (
            f"The learned detector wins operating-point sensitivity "
            f"({sens(sensitivity_winner):.2f}) while stable, but does not beat hand-built "
            f"{best_hand['name']!r} ({sens(best_hand):.2f}) by the material margin "
            f"{config.learned_material_margin:g}, so {best_hand['name']!r} is selected and the "
            "learned ceiling is the runner-up (SD5 memo implementation-risk trade)."
        )
    elif stability_flip:
        branch = "false_alarm_stability"
        reason = (
            f"The sensitivity winner {overall_best['name']!r} failed cross-site false-alarm "
            f"stability, so the selection falls to {selected['name']!r}, which holds both "
            "(SD5 flip)."
        )
    else:
        branch = "none"
        reason = (
            f"{selected['name']!r} maximizes operating-point sensitivity while holding false-alarm "
            "stability across sites"
            + (
                "; the learned detector is selected with its implementation-risk cost named."
                if selected["name"] == "learned"
                else "."
            )
        )
    runner_ups = sorted(
        (c for c in candidates if c["name"] != selected["name"]), key=sens, reverse=True
    )
    fired = bool(stability_flip or risk_trade)
    return selected, runner_ups, {"fired": fired, "branch": branch, "reason": reason}
