"""The E1 real-data run that closes SD2.

WO-4 executes the deferred WO-3 AC7. The surrogate line statistics are fit to real train-side
quiet-tail audio, replacing the first-principles priors, with the leak guard enforcing train-only
reads. The fitted surrogate is then swept across the signal-to-noise ladder against the real
held-out quiet-tail class through the identical reference front end, detector, and reference
rejector. The two required refinements are made here: the two-sample distance is the tail-sensitive
Wasserstein distance rather than the fixture Kolmogorov-Smirnov statistic, because the operating
point lives in the low-miss tail, and the realism tolerance is the null distribution of that
distance between random halves of the held-out class rather than a chosen number, so the surrogate
must fall
inside the class's own variability. This module holds the pure run over already-loaded segments, so
it is testable apart from the ledger and the object store. It asserts no absolute level.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .config import E1RealDataConfig
from .determinism import derive_seed
from .e1 import (
    bootstrap_rate_interval,
    channel_responses,
    detection_rate,
    held_out_null_and_test,
    intervals_overlap,
    ks_two_sample,
    null_verdict,
    wasserstein_distance,
)
from .surrogate.extract import fit_from_segments
from .surrogate.fit import SurrogateDistributions, draw_machinery
from .surrogate.kinematics import KinematicParams, evaluate_track
from .surrogate.level import inject_at_snr
from .surrogate.lines import build_lines
from .surrogate.propagation import PropagationParams
from .surrogate.synth import synthesize

_DISTANCE_REASONING = (
    "Wasserstein integrates the difference of the empirical distribution functions across the "
    "support, so it registers the tail-mass differences the operating point depends on, unlike the "
    "Kolmogorov-Smirnov supremum used on fixtures (WO-4 Section 4). KS is reported alongside for "
    "continuity, but the verdict rests on the Wasserstein-inside-null test."
)


def _responses(
    channel: npt.NDArray[np.float64],
    rate: float,
    config: E1RealDataConfig,
    band: tuple[float, float],
) -> tuple[float, float]:
    confidence, rejector = channel_responses(
        channel[np.newaxis, :], rate, config.nfft, config.hop, band, config.energy_scale
    )
    return float(confidence[0]), float(rejector[0])


def run_e1_realdata(
    train_segments: list[dict[str, Any]],
    held_out_segments: list[dict[str, Any]],
    backgrounds: list[npt.NDArray[np.float64]],
    config: E1RealDataConfig,
    seed: int,
) -> tuple[dict[str, Any], SurrogateDistributions]:
    """Fit from train-side audio and compare the surrogate to the real held-out class, per rung."""
    band = (config.band_low_hz, config.band_high_hz)
    rate = config.analysis_rate_hz
    if len(backgrounds) < 2 or len(held_out_segments) < 2:
        raise ValueError("E1 real-data needs at least two backgrounds and two held-out segments")
    distributions = fit_from_segments(train_segments, band)
    held_signals = [np.asarray(s["samples"], dtype=np.float64) for s in held_out_segments]
    propagation = PropagationParams(
        absorption_coeff=config.absorption_coeff,
        multipath_depth=config.multipath_depth,
        multipath_spacing_hz=config.multipath_spacing_hz,
        spreading_exponent=config.spreading_exponent,
    )

    # The held-out class is processed as-is through the identical front end, detector, and rejector,
    # per SD2 Section 7: these are real recordings, not re-injected, so their response is a fixed
    # distribution independent of the surrogate's swept level. Re-injecting them would over-amplify
    # each recording's own broadband ambient and misstate the real class's detectability.
    held_conf_a = np.empty(len(held_signals), dtype=np.float64)
    held_rej_a = np.empty(len(held_signals), dtype=np.float64)
    for i, held_signal in enumerate(held_signals):
        confidence, rejector = _responses(held_signal, rate, config, band)
        held_conf_a[i] = confidence
        held_rej_a[i] = rejector
    held_rate = detection_rate(held_conf_a, config.threshold)
    held_ci = bootstrap_rate_interval(
        held_conf_a,
        config.threshold,
        config.bootstrap_resamples,
        config.bootstrap_alpha,
        derive_seed(seed, "bs-held"),
    )

    per_rung: list[dict[str, Any]] = []
    ladder = list(config.snr_ladder_db)
    for k, snr in enumerate(ladder):
        surrogate_conf: list[float] = []
        surrogate_rej: list[float] = []
        for j in range(config.surrogate_draws):
            background = backgrounds[j % len(backgrounds)]
            n = background.shape[0]
            machinery = draw_machinery(distributions, derive_seed(seed, f"draw-{k}-{j}"))
            kinematics = KinematicParams(
                bearing0_deg=config.bearing0_deg,
                closest_proxy=config.closest_proxy,
                speed_proxy=config.speed_proxy,
                t_cpa_s=config.t_cpa_fraction * n / rate,
                doppler_peak=config.doppler_peak,
            )
            track = evaluate_track(kinematics, np.arange(n, dtype=np.float64) / rate)
            signal = synthesize(
                build_lines(machinery),
                track,
                propagation,
                rate,
                n / rate,
                derive_seed(seed, f"synth-{k}-{j}"),
            )
            injected, _ = inject_at_snr(signal, background, band, rate, snr)
            confidence, rejector = _responses(injected, rate, config, band)
            surrogate_conf.append(confidence)
            surrogate_rej.append(rejector)

        surr_conf_a = np.array(surrogate_conf)
        surr_rej_a = np.array(surrogate_rej)
        surr_ci = bootstrap_rate_interval(
            surr_conf_a,
            config.threshold,
            config.bootstrap_resamples,
            config.bootstrap_alpha,
            derive_seed(seed, f"bs-s-{k}"),
        )
        null, test = held_out_null_and_test(
            surr_rej_a, held_rej_a, config.null_resamples, derive_seed(seed, f"null-{k}")
        )
        verdict = null_verdict(null, test, config.null_percentile)
        per_rung.append(
            {
                "snr_db": snr,
                "surrogate": {
                    "detect_rate": detection_rate(surr_conf_a, config.threshold),
                    "ci": list(surr_ci),
                },
                "held_out": {"detect_rate": held_rate, "ci": list(held_ci)},
                "detectability_overlap": intervals_overlap(surr_ci, held_ci),
                "wasserstein": wasserstein_distance(surr_rej_a, held_rej_a),
                "ks": ks_two_sample(surr_rej_a, held_rej_a),
                "null_tolerance": verdict["tolerance"],
                "test_median": verdict["test_median"],
                "null_median": verdict["null_median"],
                "rejector_inside_null": verdict["passed"],
            }
        )

    # Operating band: the rungs where the surrogate is as detectable as the real class, its
    # detectability interval overlapping the held-out's, which are the signal-to-noise ratios where
    # the realism comparison is on-point. Realism passes when the surrogate's rejector distribution
    # falls inside the held-out null across that band.
    operating = [k for k in range(len(ladder)) if per_rung[k]["detectability_overlap"]]
    overall = bool(operating) and all(per_rung[k]["rejector_inside_null"] for k in operating)
    payload = {
        "snr_ladder_db": ladder,
        "per_rung": per_rung,
        "operating_indices": operating,
        "held_out_response": {"detect_rate": held_rate, "ci": list(held_ci), "processed": "as_is"},
        "distance": "wasserstein",
        "distance_reasoning": _DISTANCE_REASONING,
        "tolerance_method": "held_out_null_distribution",
        "null_percentile": config.null_percentile,
        "null_resamples": config.null_resamples,
        "fit_provenance": distributions.provenance,
        "train_vessel_count": distributions.train_vessel_count,
        "train_vessel_ids": list(distributions.train_vessel_ids),
        "held_out_vessel_count": len(held_out_segments),
        "verdict": {
            "passed": overall,
            "operating_rungs": len(operating),
            "rungs_total": len(ladder),
            "note": (
                "Pass closes SD2 on measurement; a fail is the SD2 Section 8 flip (recalibrate "
                "against other train-side quiet vessels, never move the tolerance). Provisional "
                "under the placeholder line-prominence rejector until SD6."
            ),
        },
        "owner_certified": config.owner_certified.model_dump(mode="json"),
    }
    return payload, distributions
