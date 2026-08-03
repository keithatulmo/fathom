"""Configuration models for stages and scoring.

SD10 Section 6.4 sends configuration across process boundaries through pydantic v2 models. Each
stage owns a frozen model that forbids unknown fields, so a mistyped manifest key fails loudly
rather than being silently ignored. The provisional defaults SD3 marks as engineering choices,
the fifteen-bin ECE, the thousand-resample bootstrap, the twenty-four-hour guard, and the
five-hundred-event calibration sizing, live here in configuration and are recorded in lineage,
never buried in code.

The owner-certified figures are named fields that carry no value in this unclassified, synthetic
repository: detection range r and confirmer P_d are typed to admit only ``None`` and a validator
rejects any attempt to populate them.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

# Provisional engineering defaults from SD3, confidence "moderate", recorded here not in code.
DEFAULT_ECE_BINS = 15
DEFAULT_ECE_MIN_BIN_COUNT = 1
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_GUARD_HOURS = 24.0
DEFAULT_CALIBRATION_TARGET_EVENTS = 500
DEFAULT_ALPHA = 0.05


class _Frozen(BaseModel):
    """Base for frozen configuration models that forbid unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class OwnerCertifiedFigures(_Frozen):
    """Named owner-certified figures that carry no value in this repository.

    Detection range r and confirmer probability of detection P_d are owner-certified and never
    asserted in unclassified, synthetic work. Both fields are typed to admit only ``None``, so
    the type system itself enforces that they carry no value; any populated value is a validation
    error rather than a silently accepted figure.
    """

    detection_range_r: None = None
    confirmer_pd: None = None


class FixtureConfig(_Frozen):
    """Configuration of the seeded synthetic fixture generator.

    The generator produces present-target channels carrying an in-band tonal above noise and
    clutter channels carrying noise alone, across a small set of vessels assigned to the train,
    calibration, and test splits, with one quarantined vessel to exercise the roster. Truth is
    known by construction, so no binary data is committed and evaluation truth is tier one.
    """

    vessels_per_split: int = 3
    recordings_per_present_vessel: int = 10
    clutter_channels: int = 12
    sites: int = 2
    sample_rate: float = 2048.0
    duration_s: float = 1.0
    target_band_hz: tuple[float, float] = (200.0, 300.0)
    target_tonal_hz: float = 250.0
    target_amplitude: float = 1.0
    noise_amplitude: float = 0.25
    quarantined_vessels: int = 1
    guard_hours: float = DEFAULT_GUARD_HOURS
    calibration_target_events: int = DEFAULT_CALIBRATION_TARGET_EVENTS


class FrontEndConfig(_Frozen):
    """Configuration of the band-limited spectral front end.

    A short-time Fourier transform with a Hann window produces a power gram, optionally flattened
    by dividing each frequency bin by its median over time. This is a placeholder front end for
    the acceptance path; the real front end is selected later at experiment E2 under SD4.
    """

    nfft: int = 256
    hop: int = 128
    window: str = "hann"
    normalizer: str = "none"


class DetectionConfig(_Frozen):
    """Configuration of the trivial band-limited energy-threshold detector.

    The test statistic is the ratio of mean in-band power to mean out-of-band power, a crude
    scale-invariant signal-to-noise proxy, mapped to a confidence in the unit interval by a
    saturating exponential above unit ratio. A present target's in-band tonal drives the ratio
    well above one and yields a confidence near one; a clutter channel's ratio sits near one and
    yields a confidence near zero. This detector is deliberately trivial and holds no learned
    parameters; it exists only to exercise the harness end to end.
    """

    band_low_hz: float = 200.0
    band_high_hz: float = 300.0
    energy_scale: float = 3.0


class ScoringConfig(_Frozen):
    """Configuration of the scoring stage that assembles the report.

    The provisional SD3 defaults are the operating-point alpha for the exact miss bound, the
    equal-mass ECE binning, and the cluster-bootstrap resample count. The owner-certified figures
    are carried and remain unset.
    """

    operating_threshold: float = 0.5
    miss_alpha: float = DEFAULT_ALPHA
    ece_bins: int = DEFAULT_ECE_BINS
    ece_min_bin_count: int = DEFAULT_ECE_MIN_BIN_COUNT
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    bootstrap_alpha: float = DEFAULT_ALPHA
    owner_certified: OwnerCertifiedFigures = OwnerCertifiedFigures()


class AuditConfig(_Frozen):
    """Configuration of the split-integrity audit.

    The guard interval and the tier-one-only evaluation rule are the enforceable parts of the
    split discipline SD3 Section 5.6 requires the harness to check on every run.
    """

    guard_hours: float = DEFAULT_GUARD_HOURS
    require_tier_one_eval: bool = True
