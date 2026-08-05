"""The three narrowband detection candidates for the E3 bake-off (SD5 / WO-8).

SD5 frames the detection layer as a bake-off, not a pre-selection: three candidates run on the
output of the closed SD4 front end (the linear Hann split-window gram at half-hertz resolution,
whose background is already flattened per frame), so E3 measures the detector and nothing else. Each
candidate reduces the normalized gram to one graded soft score per trial, higher when the injected
comb is present, from which a detection is a threshold crossing. The candidates differ in how much
they integrate the weak, persistent, harmonically organized signal:

  * ``cfar_peak_score`` is constant-false-alarm-rate peak-picking with no dwell, the strongest
    normalized cell in the band. It is the maturity and false-alarm-control baseline, weakest at the
    operating point because it throws away temporal and harmonic integration.
  * ``harmonic_integration_score`` integrates each line along its track over a dwell and sums across
    the strongest comb lines, trading latency for sensitivity.
  * :class:`LearnedDetector` is a bounded logistic regression on fixed comb-agnostic features,
    trained train-side only with the closed hybrid injected as labeled targets, the sensitivity
    ceiling whose cross-site false-alarm stability the bake-off tests.

Every score is computed on the normalized (dimensionless) gram, so no absolute level is expressible,
and every routine is bit-level deterministic given its input. A band argument restricts the score to
a sub-band, which E3 uses to report sensitivity at the four-hertz shaft floor SD4 flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_FLOOR = 1e-30
# The known-truth line is integrated within this frequency tolerance of its nominal position,
# matching
# the surrogate's slow wander and Doppler, the same tolerance the E2 recovery used.
_LINE_TOLERANCE_HZ = 0.5


def _in_band(freqs: npt.NDArray[np.float64], band: tuple[float, float]) -> npt.NDArray[np.bool_]:
    return (freqs >= band[0]) & (freqs <= band[1])


def cfar_peak_score(
    normalized: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    band: tuple[float, float],
) -> float:
    """Constant-false-alarm-rate peak-pick: the strongest normalized cell in the band, no
    integration.

    On the already-flattened gram a fixed threshold on the peak holds the false-alarm rate, so
    this is
    the classical peak detector. It does not integrate over time or across the comb, which is why
    it is
    the least sensitive on a weak persistent line and the tightest on false-alarm control.
    """
    mask = _in_band(freqs, band)
    sub = normalized[mask]
    return float(sub.max()) if sub.size else 0.0


def _dwell_integrate(track: npt.NDArray[np.float64], dwell_frames: int) -> float:
    """Incoherent integration of a per-frame line track over the dwell: the mean over its frames.

    A persistent line keeps a high mean over the dwell while noise averages toward the flattened
    background level, so integrating raises the line's detectability the way a single frame cannot.
    The mean, not a best-window maximum, is used so the integration does not reintroduce the
    single-frame selection bias it exists to remove.
    """
    n = track.shape[0]
    if n == 0:
        return 0.0
    dwell = max(1, min(dwell_frames, n))
    return float(track[:dwell].mean())


def harmonic_integration_score(
    normalized: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    band: tuple[float, float],
    dwell_frames: int,
    shaft_hz_range: tuple[float, float] = (2.0, 10.0),
    n_harmonics: int = 10,
    n_hypotheses: int = 80,
) -> float:
    """Line and harmonic integration over a scanned shaft comb, dwell-integrated (SD5 memo spec).

    Each in-band bin is first integrated over the dwell, its mean over the dwell frames, the line
    integration that lifts a persistent line while transient noise averages toward the flattened
    background. The detector then scans a grid of shaft-rate hypotheses across the machinery,
    sums the dwell-integrated value at the first several harmonics of each, and takes the best
    comb, so it exploits both the persistence and the harmonic structure of the target. It is not
    handed the target's shaft rate, so the scan keeps it honest, and it reads harmonic structure
    rather than a blind top-line sum, so an unstructured ambient tonal that is not part of a comb
    does not raise the statistic the way a genuine harmonic comb does.
    """
    if normalized.shape[1] == 0:
        return 0.0
    dwell = max(1, min(dwell_frames, normalized.shape[1]))
    per_bin = normalized[:, :dwell].mean(axis=1)
    hypotheses = np.linspace(shaft_hz_range[0], shaft_hz_range[1], n_hypotheses)
    best = 0.0
    for fundamental in hypotheses:
        total = 0.0
        for harmonic in range(1, n_harmonics + 1):
            frequency = fundamental * harmonic
            if frequency < band[0] or frequency > band[1]:
                continue
            near = np.abs(freqs - frequency) <= _LINE_TOLERANCE_HZ
            if np.any(near):
                total += float(per_bin[near].max())
        best = max(best, total)
    return float(best)


# The fixed, comb-agnostic feature set the learned detector consumes. It captures peak strength, the
# tail of the normalized distribution, and blind temporal-and-harmonic integration (the strongest
# and
# the summed top integrated bins), so the learned detector can weight integration without being told
# the exact comb positions, which keeps it a fair learned baseline rather than an oracle.
FEATURE_NAMES = (
    "peak_max",
    "p99",
    "p95",
    "mean",
    "std",
    "integrated_max",
    "integrated_top5_sum",
    "frac_above_2",
    "frac_above_4",
)


def comb_features(
    normalized: npt.NDArray[np.float64],
    freqs: npt.NDArray[np.float64],
    band: tuple[float, float],
    dwell_frames: int,
) -> npt.NDArray[np.float64]:
    """Return the fixed-length comb-agnostic feature vector for the learned detector."""
    mask = _in_band(freqs, band)
    sub = normalized[mask]
    if sub.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    flat = sub.ravel()
    integrated = np.array([_dwell_integrate(sub[b], dwell_frames) for b in range(sub.shape[0])])
    top5 = np.sort(integrated)[-5:] if integrated.size >= 5 else integrated
    return np.array(
        [
            float(sub.max()),
            float(np.quantile(flat, 0.99)),
            float(np.quantile(flat, 0.95)),
            float(flat.mean()),
            float(flat.std()),
            float(integrated.max()) if integrated.size else 0.0,
            float(top5.sum()),
            float(np.mean(flat > 2.0)),
            float(np.mean(flat > 4.0)),
        ],
        dtype=np.float64,
    )


def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))), dtype=np.float64)


@dataclass(frozen=True)
class LearnedDetector:
    """A bounded logistic-regression detector on the fixed comb-agnostic features.

    It is trained by deterministic gradient descent from a zero initialisation on standardized
    features, so the fit is bit-level reproducible and carries no randomness. The soft score is the
    logistic probability, a graded value in the unit interval the tracker and rejector can
    calibrate.
    The weights and the standardisation are recorded so the trained detector regenerates from
    committed state.
    """

    weights: tuple[float, ...]
    bias: float
    feature_mean: tuple[float, ...]
    feature_std: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        features: npt.NDArray[np.float64],
        labels: npt.NDArray[np.float64],
        iterations: int = 500,
        learning_rate: float = 0.2,
        l2: float = 1e-2,
    ) -> LearnedDetector:
        """Fit the logistic regression by deterministic gradient descent on standardized
        features."""
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-9
        standardized = (features - mean) / std
        weights = np.zeros(standardized.shape[1], dtype=np.float64)
        bias = 0.0
        n = float(max(1, labels.shape[0]))
        for _ in range(iterations):
            prediction = _sigmoid(standardized @ weights + bias)
            error = prediction - labels
            weights -= learning_rate * (standardized.T @ error / n + l2 * weights)
            bias -= learning_rate * float(error.mean())
        return cls(
            weights=tuple(float(w) for w in weights),
            bias=float(bias),
            feature_mean=tuple(float(m) for m in mean),
            feature_std=tuple(float(s) for s in std),
        )

    def score(self, features: npt.NDArray[np.float64]) -> float:
        """Return the graded soft score in the unit interval for one feature vector."""
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        std = np.asarray(self.feature_std, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        standardized = (np.asarray(features, dtype=np.float64) - mean) / std
        return float(_sigmoid(standardized @ weights + self.bias))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable record of the trained detector for the lineage."""
        return {
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "bias": self.bias,
            "feature_mean": list(self.feature_mean),
            "feature_std": list(self.feature_std),
        }
