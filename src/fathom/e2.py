"""The E2 front-end parameter sweep that closes SD4 by measurement.

WO-6 runs E2 over the sweepable linear split-window front end (:mod:`fathom.frontend`,
:mod:`fathom.normalize`). Tonal separability, the primary criterion at weight 40, is scored as a
known-truth recovery: the closed SD2 hybrid surrogate is injected at one operating-point
signal-to-noise onto real background from each site, and because the injected line frequencies are
ground truth, separability is the recovered line prominence of the injected comb after each
configuration. Background flatness, the second criterion at weight 30, is the site-invariance of the
normalized background: the spread across bins and across the quiet, nominal, and busy sites, and
whether one threshold holds a fixed false-alarm rate across every site. The operating-point
signal-to-noise and the threshold are set once across the site set, never per site. Compute cost and
bit-level determinism are constraints each configuration clears rather than scored quantities. This
module holds the pure sweep over already-loaded backgrounds, so it is testable apart from the ledger
and the object store, and it asserts no absolute level.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from .config import E2Config
from .determinism import derive_seed
from .frontend import FrontEndParams, compute_cost_proxy, lofar_gram
from .normalize import normalize_gram
from .surrogate.fit import SurrogateDistributions, draw_broadband, draw_machinery
from .surrogate.kinematics import KinematicParams, evaluate_track
from .surrogate.level import inject_at_snr
from .surrogate.lines import build_lines
from .surrogate.propagation import PropagationParams
from .surrogate.synth import add_broadband, synthesize, synthesize_continuum

_FLOOR = 1e-30
# Reference front end WO-3 used and E1 ran on, kept for comparison: a 256-point Hann gram at the
# 1 kHz analysis rate with a median-over-frequency normalizer.
_REFERENCE_NFFT = 256
_REFERENCE_HOP = 128


def _band(config: E2Config) -> tuple[float, float]:
    return (config.band_low_hz, config.band_high_hz)


def build_grid(config: E2Config) -> list[FrontEndParams]:
    """Enumerate the front-end configurations in the sweep grid, in a stable order."""
    grid: list[FrontEndParams] = []
    for window_length_s in config.window_lengths_s:
        for overlap in config.overlaps:
            for integration in config.integration_counts:
                for taper in config.tapers:
                    for normalizer in config.normalizers:
                        grid.append(
                            FrontEndParams(
                                window_length_s=window_length_s,
                                overlap=overlap,
                                integration_count=integration,
                                taper=taper,
                                n_tapers=config.multitaper_n_tapers,
                                nw=config.multitaper_nw,
                                normalizer=normalizer,
                                sw_half_window_hz=config.sw_half_window_hz,
                                sw_guard_hz=config.sw_guard_hz,
                                sw_truncation=config.sw_truncation,
                                tm_window_frames=config.tm_window_frames,
                            )
                        )
    return grid


class HybridProbeConfig(Protocol):
    """The configuration fields the hybrid-surrogate probe needs, shared by E2 and E3.

    Declaring it structurally lets E3 reuse the probe with its own configuration without importing
    or
    constructing an E2 configuration, so the injection truth is one code path across both bake-offs.
    The members are read-only properties so a frozen pydantic configuration satisfies the protocol.
    """

    @property
    def analysis_rate_hz(self) -> float: ...
    @property
    def band_low_hz(self) -> float: ...
    @property
    def band_high_hz(self) -> float: ...
    @property
    def bearing0_deg(self) -> float: ...
    @property
    def closest_proxy(self) -> float: ...
    @property
    def speed_proxy(self) -> float: ...
    @property
    def doppler_peak(self) -> float: ...
    @property
    def t_cpa_fraction(self) -> float: ...
    @property
    def absorption_coeff(self) -> float: ...
    @property
    def multipath_depth(self) -> float: ...
    @property
    def multipath_spacing_hz(self) -> float: ...
    @property
    def spreading_exponent(self) -> float: ...


def synth_hybrid_probe(
    distributions: SurrogateDistributions, config: HybridProbeConfig, n: int, seed: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Synthesize one closed hybrid surrogate; return the signal and its ground-truth line set."""
    rate = config.analysis_rate_hz
    band = (config.band_low_hz, config.band_high_hz)
    machinery = draw_machinery(distributions, derive_seed(seed, "m"))
    lines = build_lines(machinery)
    propagation = PropagationParams(
        absorption_coeff=config.absorption_coeff,
        multipath_depth=config.multipath_depth,
        multipath_spacing_hz=config.multipath_spacing_hz,
        spreading_exponent=config.spreading_exponent,
    )
    kinematics = KinematicParams(
        bearing0_deg=config.bearing0_deg,
        closest_proxy=config.closest_proxy,
        speed_proxy=config.speed_proxy,
        t_cpa_s=config.t_cpa_fraction * n / rate,
        doppler_peak=config.doppler_peak,
    )
    track = evaluate_track(kinematics, np.arange(n, dtype=np.float64) / rate)
    line_signal = synthesize(lines, track, propagation, rate, n / rate, derive_seed(seed, "s"))
    broadband = draw_broadband(distributions, derive_seed(seed, "b"))
    continuum = synthesize_continuum(
        n, rate, band, broadband.continuum_exponent, derive_seed(seed, "c")
    )
    signal = add_broadband(line_signal, continuum, band, rate, broadband.lbr_db)
    return signal, np.asarray(lines.freqs, dtype=np.float64)


# The known-truth recovery tolerates the line inside this frequency window, matching the surrogate's
# slow wander and Doppler shift. It is stated in hertz, not bins, so a coarse configuration cannot
# inflate its score by letting a wide bin's neighbourhood span a whole comb interval.
_RECOVERY_TOLERANCE_HZ = 0.5


def recovered_prominence(
    normalized: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    line_freqs: npt.NDArray[np.float64],
    band: tuple[float, float],
) -> float:
    """Median recovered prominence of the known in-band lines, in decibels over the flat background.

    For each ground-truth line inside the band, the recovered prominence is the largest normalized
    value over frames within a fixed frequency tolerance of the line, tolerating its slow
    wander and leakage without letting a coarse configuration's wide bins reach across a comb
    interval. The score is the median across the comb's lines, so losing a few harmonics into the
    noise degrades it rather than a single lucky line carrying it.
    """
    in_band_lines = line_freqs[(line_freqs >= band[0]) & (line_freqs <= band[1])]
    if in_band_lines.size == 0 or freqs.size == 0:
        return float("nan")
    bin_hz = float(freqs[1] - freqs[0]) if freqs.shape[0] >= 2 else 1.0
    reach = int(np.floor(_RECOVERY_TOLERANCE_HZ / bin_hz)) if bin_hz > 0 else 0
    # The between-line floor this configuration actually achieves: the median normalized level over
    # the bins not within tolerance of any known line. Measuring the line against this floor
    # rather than against unity makes separability a detection index fair across normalizers,
    # so a normalizer that lifts the whole in-band region uniformly earns no free prominence.
    near_line = np.zeros(freqs.shape[0], dtype=bool)
    for line in in_band_lines:
        near_line |= np.abs(freqs - line) <= _RECOVERY_TOLERANCE_HZ
    off_line = normalized[~near_line]
    ambient = float(np.median(off_line)) if off_line.size else float(np.median(normalized))
    ambient = max(ambient, _FLOOR)
    recovered = []
    for line in in_band_lines:
        centre = int(np.argmin(np.abs(freqs - line)))
        lo = max(0, centre - reach)
        hi = min(freqs.shape[0], centre + reach + 1)
        recovered.append(float(np.max(normalized[lo:hi])) / ambient)
    ratio = float(np.median(recovered))
    return 10.0 * float(np.log10(max(ratio, _FLOOR)))


def separability_for_config(
    params: FrontEndParams,
    injected_by_regime: dict[str, list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]],
    config: E2Config,
) -> dict[str, Any]:
    """Score separability for one config as the recovered comb prominence, per site and overall."""
    rate = config.analysis_rate_hz
    band = _band(config)
    per_regime: dict[str, float] = {}
    everything: list[float] = []
    for regime, injected in injected_by_regime.items():
        values = []
        for signal, line_freqs in injected:
            gram, freqs = lofar_gram(signal, rate, params)
            normalized, nf = normalize_gram(gram, freqs, band, params)
            values.append(recovered_prominence(normalized, nf, line_freqs, band))
        finite = [v for v in values if np.isfinite(v)]
        per_regime[regime] = float(np.median(finite)) if finite else float("nan")
        everything.extend(finite)
    overall = float(np.median(everything)) if everything else float("nan")
    return {"separability_db": overall, "per_regime_db": per_regime}


def flatness_for_config(
    params: FrontEndParams,
    backgrounds_by_regime: dict[str, list[npt.NDArray[np.float64]]],
    config: E2Config,
) -> dict[str, Any]:
    """Score flatness as site-invariance of the normalized background and one-threshold FAR.

    The normalized background of each site is pooled, one threshold is set across the whole site set
    at the target false-alarm quantile, and the realized rate is measured per site. A flat
    normalizer yields a site-invariant false-alarm rate; flatness holds when every site's rate sits
    within tolerance of the target, the single-threshold-across-sites property the criterion
    names. The cross-bin coefficient of variation of the background is recorded alongside.
    """
    rate = config.analysis_rate_hz
    band = _band(config)
    per_regime_cells: dict[str, npt.NDArray[np.float64]] = {}
    cv_values: list[float] = []
    for regime, backgrounds in backgrounds_by_regime.items():
        cells = []
        for background in backgrounds:
            gram, freqs = lofar_gram(background, rate, params)
            normalized, _ = normalize_gram(gram, freqs, band, params)
            cells.append(normalized.ravel())
            per_bin = normalized.mean(axis=1)
            cv_values.append(float(np.std(per_bin) / max(float(np.mean(per_bin)), _FLOOR)))
        per_regime_cells[regime] = np.concatenate(cells)
    pooled = np.concatenate(list(per_regime_cells.values()))
    threshold = float(np.quantile(pooled, 1.0 - config.target_far))
    per_regime_far = {
        regime: float(np.mean(cells > threshold)) for regime, cells in per_regime_cells.items()
    }
    far_values = list(per_regime_far.values())
    far_spread = float(max(far_values) - min(far_values)) if far_values else 1.0
    lo = config.target_far / config.far_tolerance
    hi = config.target_far * config.far_tolerance
    flatness_holds = bool(all(lo <= v <= hi for v in far_values))
    flatness_score = max(0.0, 1.0 - far_spread / max(config.target_far, _FLOOR))
    return {
        "flatness_score": flatness_score,
        "flatness_holds": flatness_holds,
        "single_threshold": threshold,
        "per_regime_far": per_regime_far,
        "far_spread": far_spread,
        "background_cv_median": float(np.median(cv_values)) if cv_values else float("nan"),
    }


def constraints_for_config(
    params: FrontEndParams, config: E2Config, duration_s: float
) -> dict[str, Any]:
    """Record the compute-cost and bit-level-determinism constraint checks for one configuration."""
    rate = config.analysis_rate_hz
    cost = compute_cost_proxy(params, rate, duration_s)
    probe = _determinism_probe(int(round(min(duration_s, 8.0) * rate)))
    gram_a, _ = lofar_gram(probe, rate, params)
    gram_b, _ = lofar_gram(probe, rate, params)
    return {
        "compute_cost_proxy": cost,
        "compute_cleared": bool(cost <= config.compute_budget_proxy),
        "determinism_cleared": bool(np.array_equal(gram_a, gram_b)),
    }


def _determinism_probe(n: int) -> npt.NDArray[np.float64]:
    """A fixed deterministic probe signal for the determinism check, no randomness involved."""
    t = np.arange(max(n, 64), dtype=np.float64) / 1000.0
    return np.ascontiguousarray(
        np.sin(2.0 * np.pi * 12.0 * t) + 0.5 * np.sin(2.0 * np.pi * 47.0 * t), dtype=np.float64
    )


def _reference_normalized(
    signal: npt.NDArray[np.float64], rate: float, band: tuple[float, float]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The WO-3 reference: a 256-point Hann gram with a median-over-frequency normalizer."""
    from .stages.dsp import _hann, stft_power

    window = _hann(_REFERENCE_NFFT)
    freqs = np.fft.rfftfreq(_REFERENCE_NFFT, d=1.0 / rate).astype(np.float64)
    gram = stft_power(signal, _REFERENCE_NFFT, _REFERENCE_HOP, window)
    median = np.median(gram, axis=0, keepdims=True)
    normalized = gram / np.maximum(median, _FLOOR)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return normalized[in_band], freqs[in_band]


def reference_baseline(
    injected_by_regime: dict[str, list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]],
    backgrounds_by_regime: dict[str, list[npt.NDArray[np.float64]]],
    config: E2Config,
) -> dict[str, Any]:
    """Score the WO-3 reference front end on the same probes, so the sweep result has a baseline."""
    rate = config.analysis_rate_hz
    band = _band(config)
    sep: list[float] = []
    for injected in injected_by_regime.values():
        for signal, line_freqs in injected:
            normalized, nf = _reference_normalized(signal, rate, band)
            value = recovered_prominence(normalized, nf, line_freqs, band)
            if np.isfinite(value):
                sep.append(value)
    per_regime_cells: dict[str, npt.NDArray[np.float64]] = {}
    for regime, backgrounds in backgrounds_by_regime.items():
        cells = [
            _reference_normalized(background, rate, band)[0].ravel() for background in backgrounds
        ]
        per_regime_cells[regime] = np.concatenate(cells)
    pooled = np.concatenate(list(per_regime_cells.values()))
    threshold = float(np.quantile(pooled, 1.0 - config.target_far))
    far = {r: float(np.mean(c > threshold)) for r, c in per_regime_cells.items()}
    lo = config.target_far / config.far_tolerance
    hi = config.target_far * config.far_tolerance
    return {
        "front_end": "wo3_reference_hann256_median_over_frequency",
        "separability_db": float(np.median(sep)) if sep else float("nan"),
        "per_regime_far": far,
        "far_spread": float(max(far.values()) - min(far.values())) if far else float("nan"),
        "flatness_holds": bool(all(lo <= v <= hi for v in far.values())) if far else False,
    }


def band_endpoint_finding(
    params: FrontEndParams,
    injected_by_regime: dict[str, list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]],
    config: E2Config,
) -> dict[str, Any]:
    """Report whether the 4 and 150 Hz band endpoints hold on the corpus grams under the selection.

    The recovered prominence of the known lines is split into a low decade near the 4 Hz floor, a
    mid band, and a high edge near 150 Hz, so a materially weaker endpoint stands out as a candidate
    for a band adjustment. This is feedback to the design-assumptions band entry the owner rules on,
    not a change made here.
    """
    rate = config.analysis_rate_hz
    band = _band(config)
    low: list[float] = []
    mid: list[float] = []
    high: list[float] = []
    for injected in injected_by_regime.values():
        for signal, line_freqs in injected:
            gram, freqs = lofar_gram(signal, rate, params)
            normalized, nf = normalize_gram(gram, freqs, band, params)
            for edge, lo, hi in (
                ("low", band[0], 10.0),
                ("mid", 10.0, 120.0),
                ("high", 120.0, band[1]),
            ):
                sub = line_freqs[(line_freqs >= lo) & (line_freqs <= hi)]
                value = (
                    recovered_prominence(normalized, nf, sub, (lo, hi))
                    if sub.size
                    else float("nan")
                )
                if np.isfinite(value):
                    {"low": low, "mid": mid, "high": high}[edge].append(value)
    low_db = float(np.median(low)) if low else float("nan")
    mid_db = float(np.median(mid)) if mid else float("nan")
    high_db = float(np.median(high)) if high else float("nan")
    # An endpoint "holds" if its recovered prominence is within 3 dB of the mid band.
    low_holds = bool(np.isfinite(low_db) and np.isfinite(mid_db) and low_db >= mid_db - 3.0)
    high_holds = bool(np.isfinite(high_db) and np.isfinite(mid_db) and high_db >= mid_db - 3.0)
    return {
        "low_4hz_recovered_db": low_db,
        "mid_recovered_db": mid_db,
        "high_150hz_recovered_db": high_db,
        "low_endpoint_holds": low_holds,
        "high_endpoint_holds": high_holds,
        "note": (
            "Feedback to the design-assumptions band entry; the owner rules on any band change. "
            "An endpoint holds when its recovered comb prominence is within 3 dB of the mid band."
        ),
    }


def operating_separability(config_row: dict[str, Any], operating_point_regime: str) -> float:
    """The separability at the operating-point site, the quantity the WO-7 objective maximizes.

    The proof runs at the quiet-site operating point, so the selection reads separability at that
    site rather than the site-average that equal-weights a quiet, a nominal, and a busy site (SD4
    close v1.0). It falls back to the site-averaged value only if the per-site number is absent.
    """
    per_regime = config_row["separability"].get("per_regime_db", {})
    value = per_regime.get(operating_point_regime)
    if value is None:
        return float(config_row["separability"]["separability_db"])
    return float(value)


def _select_and_flip(
    per_config: list[dict[str, Any]], config: E2Config
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Select the closing configuration and evaluate the ordered SD4 flip clause.

    The objective is quiet-site operating-point separability subject to flatness (WO-7): the proof
    runs at the quiet-site operating point, so the selection maximizes separability there, not the
    site-average. The selection then follows the SD4 memo: Hann is the default and a config is
    adequate when its operating-point separability reaches the floor with cross-site flatness
    at bounded compute and determinism. If a Hann config is adequate, the best such Hann config is
    selected and the flip does not fire. If no single-taper config is adequate, the multitaper is
    adopted (branch one). If even multitaper falls short, the dwell is flagged for revisit before
    resolution (branch two), and the best candidate is selected provisionally. If no config
    holds cross-site flatness at all, that is a nonstationarity routed to the data thread (branch
    three) and nothing is selected.
    """
    candidates = [c for c in per_config if c["candidate"]]
    floor = config.separability_floor_db
    regime = config.operating_point_regime

    def best(cands: list[dict[str, Any]]) -> dict[str, Any]:
        return max(cands, key=lambda c: operating_separability(c, regime))

    def reaches(c: dict[str, Any]) -> bool:
        return bool(operating_separability(c, regime) >= floor)

    if not candidates:
        holds_any = any(c["flatness"]["flatness_holds"] for c in per_config)
        branch = 3 if not holds_any else 0
        return None, {
            "fired": True,
            "branch": branch,
            "reason": (
                "No configuration held cross-site flatness at bounded compute and determinism; a "
                "cross-site background nonstationarity is routed to the data thread as a site and "
                "corpus finding, never cured by per-site tuning (SD4 flip branch three)."
            ),
        }
    hann = [c for c in candidates if c["params"]["taper"] == "hann"]
    hann_adequate = [c for c in hann if reaches(c)]
    if hann_adequate:
        return best(hann_adequate), {
            "fired": False,
            "branch": 0,
            "reason": "A single-taper Hann configuration reaches operating-point separability with "
            "cross-site flatness; the multitaper arm is not needed.",
        }
    multitaper = [c for c in candidates if c["params"]["taper"] == "multitaper"]
    mt_adequate = [c for c in multitaper if reaches(c)]
    if mt_adequate:
        return best(mt_adequate), {
            "fired": True,
            "branch": 1,
            "reason": "No single-taper configuration reached operating-point separability with "
            "flatness; the multitaper taper is adopted as the default and pays its compute cost "
            "(SD4 flip branch one).",
        }
    return best(candidates), {
        "fired": True,
        "branch": 2,
        "reason": "Neither single-taper nor multitaper reached the separability floor "
        "with flatness; the dwell is flagged for revisit against the measured line "
        "coherence before the resolution (SD4 flip branch two). The best available candidate is "
        "reported provisionally.",
    }


def run_e2_sweep(
    backgrounds_by_regime: dict[str, list[npt.NDArray[np.float64]]],
    site_names_by_regime: dict[str, list[str]],
    distributions: SurrogateDistributions,
    config: E2Config,
    seed: int,
) -> dict[str, Any]:
    """Run the full E2 sweep and return the committed-state payload with the selection and flip."""
    rate = config.analysis_rate_hz
    band = _band(config)
    regimes = [r for r in ("quiet", "nominal", "busy") if backgrounds_by_regime.get(r)]
    if len(regimes) < 2:
        raise ValueError("E2 needs at least two site regimes with background")

    # Precompute the injected known-truth probes once; injection is config-independent, so every
    # configuration scores the same set of injected signals, which also keeps the sweep affordable.
    injected_by_regime: dict[
        str, list[tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]
    ] = {}
    for regime in regimes:
        injected = []
        for b_index, background in enumerate(backgrounds_by_regime[regime]):
            n = background.shape[0]
            for draw in range(config.injection_draws):
                sig, line_freqs = synth_hybrid_probe(
                    distributions, config, n, derive_seed(seed, f"{regime}-{b_index}-{draw}")
                )
                injected_signal, _ = inject_at_snr(
                    sig, background, band, rate, config.operating_snr_db
                )
                injected.append((injected_signal, line_freqs))
        injected_by_regime[regime] = injected

    durations = [b.shape[0] / rate for bs in backgrounds_by_regime.values() for b in bs]
    duration_s = float(np.median(durations))

    per_config: list[dict[str, Any]] = []
    for params in build_grid(config):
        separability = separability_for_config(params, injected_by_regime, config)
        flatness = flatness_for_config(params, backgrounds_by_regime, config)
        constraints = constraints_for_config(params, config, duration_s)
        candidate = bool(
            constraints["compute_cleared"]
            and constraints["determinism_cleared"]
            and flatness["flatness_holds"]
        )
        per_config.append(
            {
                "key": params.key(),
                "params": params.to_dict(),
                "separability": separability,
                "flatness": flatness,
                "constraints": constraints,
                "candidate": candidate,
            }
        )

    selected, flip = _select_and_flip(per_config, config)
    ranked = sorted(
        (c for c in per_config if c["candidate"]),
        key=lambda c: operating_separability(c, config.operating_point_regime),
        reverse=True,
    )
    runner_ups = [
        {
            "key": c["key"],
            "operating_separability_db": operating_separability(c, config.operating_point_regime),
            "per_regime_db": c["separability"]["per_regime_db"],
            "site_averaged_separability_db": c["separability"]["separability_db"],
            "flatness_score": c["flatness"]["flatness_score"],
        }
        for c in ranked
        if selected is None or c["key"] != selected["key"]
    ][:4]
    baseline = reference_baseline(injected_by_regime, backgrounds_by_regime, config)
    endpoints = (
        band_endpoint_finding(FrontEndParams(**selected["params"]), injected_by_regime, config)
        if selected
        else {"note": "no configuration selected; endpoint finding not computed"}
    )

    return {
        "operating_snr_db": config.operating_snr_db,
        "target_far": config.target_far,
        "far_tolerance": config.far_tolerance,
        "separability_floor_db": config.separability_floor_db,
        "selection_objective": "operating_point_separability_subject_to_flatness",
        "operating_point_regime": config.operating_point_regime,
        "band_hz": list(band),
        "analysis_rate_hz": rate,
        "corpus_slice": {
            regime: {
                "n_backgrounds": len(backgrounds_by_regime[regime]),
                "sites": site_names_by_regime.get(regime, []),
            }
            for regime in regimes
        },
        "injection_set": {
            "probe": "closed_sd2_hybrid_surrogate",
            "draws_per_background": config.injection_draws,
            "fit_provenance": distributions.provenance,
        },
        "grid_axes": {
            "window_lengths_s": list(config.window_lengths_s),
            "overlaps": list(config.overlaps),
            "integration_counts": list(config.integration_counts),
            "tapers": list(config.tapers),
            "normalizers": list(config.normalizers),
        },
        "compute_budget_proxy": config.compute_budget_proxy,
        "per_config": per_config,
        "selected": selected,
        "runner_ups": runner_ups,
        "reference_baseline": baseline,
        "flip_clause": flip,
        "band_endpoint_finding": endpoints,
        "verdict": {
            "selected_key": selected["key"] if selected else None,
            "closes_sd4_on_measurement": bool(selected is not None and not flip["fired"]),
            "note": (
                "SD4 closes on the selection when a Hann default reaches operating-point "
                "(quiet-site) separability with cross-site flatness; a fired flip is reported "
                "for E&P to rule on. The selection is the quiet-site operating-point argmax under "
                "flatness (WO-7), not the site-averaged argmax; temporal-median stays the "
                "site-averaged runner-up with its per-site numbers. Provisional under the "
                "placeholder detector until SD5; E3 on this closed front end confirms it and is "
                "the one measured basis to revisit the normalizer."
            ),
        },
        "owner_certified": config.owner_certified.model_dump(mode="json"),
    }
