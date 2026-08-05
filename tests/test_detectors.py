"""Tests for the three E3 detection candidates (SD5 / WO-8)."""

from __future__ import annotations

import numpy as np

from fathom.detectors import (
    LearnedDetector,
    cfar_peak_score,
    comb_features,
    harmonic_integration_score,
)

_BAND = (4.0, 150.0)


def _freqs(n: int) -> np.ndarray:
    return np.linspace(4.0, 150.0, n)


def _gram(n_bins: int, n_frames: int, line_bins: tuple[int, ...], line_level: float) -> np.ndarray:
    gen = np.random.default_rng(0)
    gram = 1.0 + 0.3 * np.abs(gen.standard_normal((n_bins, n_frames)))
    for b in line_bins:
        gram[b, :] += line_level  # a persistent line across all frames
    return np.ascontiguousarray(gram, dtype=np.float64)


def test_cfar_is_the_strongest_in_band_cell() -> None:
    freqs = _freqs(60)
    gram = _gram(60, 15, line_bins=(20,), line_level=8.0)
    mask = (freqs >= _BAND[0]) & (freqs <= _BAND[1])
    assert cfar_peak_score(gram, freqs, _BAND) == float(gram[mask].max())


def test_integration_scores_a_harmonic_comb_above_noise() -> None:
    # A persistent harmonic comb (a fundamental and its integer multiples) scores above pure noise,
    # which is what the shaft-comb scan exploits.
    n_bins = 300
    freqs = np.linspace(4.0, 150.0, n_bins)
    comb_bins = tuple(int(np.argmin(np.abs(freqs - 6.0 * h))) for h in range(1, 9))
    with_comb = _gram(n_bins, 20, line_bins=comb_bins, line_level=5.0)
    noise_only = _gram(n_bins, 20, line_bins=(), line_level=0.0)
    got = harmonic_integration_score(with_comb, freqs, _BAND, dwell_frames=12)
    baseline = harmonic_integration_score(noise_only, freqs, _BAND, dwell_frames=12)
    assert got > 1.5 * baseline


def test_band_restriction_isolates_the_floor() -> None:
    freqs = _freqs(80)
    # A strong line high in the band must not affect a score restricted to the 4-12 Hz floor.
    gram = _gram(80, 15, line_bins=(70,), line_level=20.0)
    full = cfar_peak_score(gram, freqs, _BAND)
    floor = cfar_peak_score(gram, freqs, (4.0, 12.0))
    assert full > floor


def test_learned_detector_is_deterministic_and_graded() -> None:
    gen = np.random.default_rng(1)
    features = gen.standard_normal(
        (40, len(comb_features(_gram(60, 10, (), 0.0), _freqs(60), _BAND, 8)))
    )
    labels = np.array([float(i % 2) for i in range(40)])
    a = LearnedDetector.fit(features, labels)
    b = LearnedDetector.fit(features, labels)
    assert a.weights == b.weights and a.bias == b.bias
    score = a.score(features[0])
    assert 0.0 <= score <= 1.0


def test_integration_beats_cfar_at_low_snr() -> None:
    # A mini bake-off through the closed front end: at a low injected level, temporal-plus-line
    # integration detects the persistent comb more often than single-frame peak-picking.
    import json

    from fathom.config import E3Config
    from fathom.determinism import derive_seed
    from fathom.e2 import synth_hybrid_probe
    from fathom.frontend import closed_front_end, lofar_gram
    from fathom.normalize import normalize_gram
    from fathom.surrogate.fit import SurrogateDistributions
    from fathom.surrogate.level import inject_at_snr

    lineage = json.loads((__import__("pathlib").Path("docs/e1_realdata_lineage.json")).read_text())
    dist = SurrogateDistributions.from_dict(lineage["fitted_distributions"])
    config = E3Config()
    params = closed_front_end()
    rate = 1000.0
    n = int(rate * 8)

    def normalized(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gram, freqs = lofar_gram(sig, rate, params)
        return normalize_gram(gram, freqs, _BAND, params)

    cfar_bg: list[float] = []
    cfar_tg: list[float] = []
    int_bg: list[float] = []
    int_tg: list[float] = []
    for j in range(30):
        background = 0.3 * np.random.default_rng(j).standard_normal(n)
        signal, _ = synth_hybrid_probe(dist, config, n, derive_seed(j, "probe"))
        injected, _ = inject_at_snr(signal, background, _BAND, rate, 3.0)
        for arr, cf, hi in ((background, cfar_bg, int_bg), (injected, cfar_tg, int_tg)):
            ng, nf = normalized(np.ascontiguousarray(arr))
            cf.append(cfar_peak_score(ng, nf, _BAND))
            hi.append(harmonic_integration_score(ng, nf, _BAND, config.dwell_frames))

    def rate_at(bg: list[float], tg: list[float]) -> float:
        threshold = float(np.quantile(bg, 0.9))
        return float(np.mean(np.array(tg) >= threshold))

    assert rate_at(int_bg, int_tg) > rate_at(cfar_bg, cfar_tg)
