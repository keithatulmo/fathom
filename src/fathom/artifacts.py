"""Typed stage contracts.

SD10 Section 6.4 fixes stage boundaries as frozen dataclasses whose array fields carry explicit
schemas of shape, dtype, and units, validated at the seams. Each artifact here is exactly one
content-addressed blob: it serializes to a bundle of canonical-JSON metadata and named Zarr
arrays, and it validates its arrays against their schemas on construction, which is the seam.
Validation honors the global toggle, so it is on in continuous integration and debug and off in
hot loops.

No owner-certified value appears in any of these contracts, their defaults, or their comments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from .schema import ArraySchema, f8
from .serialize import decode_bundle, encode_bundle


@dataclass(frozen=True)
class AudioCorpus:
    """A synthetic acoustic corpus: per-channel waveform samples and their sample rate."""

    KIND: ClassVar[str] = "audio_corpus"
    SAMPLES: ClassVar[ArraySchema] = f8((None, None), "amplitude")

    samples: npt.NDArray[np.float64]  # shape [channels, samples]
    sample_rate: float
    channel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        self.SAMPLES.validate(self.samples, name="samples")

    def to_blob(self) -> bytes:
        return encode_bundle(
            {
                "sample_rate": self.sample_rate,
                "channel_ids": list(self.channel_ids),
                "units": self.SAMPLES.units,
            },
            {"samples": self.samples},
        )

    @classmethod
    def from_blob(cls, blob: bytes) -> AudioCorpus:
        meta, arrays = decode_bundle(blob)
        return cls(
            samples=np.asarray(arrays["samples"], dtype=np.float64),
            sample_rate=float(meta["sample_rate"]),
            channel_ids=tuple(meta["channel_ids"]),
        )


@dataclass(frozen=True)
class Registry:
    """The vessel-identity registry: split, site, and truth tier per vessel, and quarantine.

    This is the corpus-ledger table SD3 Section 5.1 describes, at the scope the synthetic
    acceptance corpus exercises. The split unit for vessel-attributed data is the vessel, and a
    quarantined vessel is usable only as unattributed background, never in evaluation.
    """

    KIND: ClassVar[str] = "registry"

    vessels: tuple[dict[str, Any], ...]
    guard_hours: float
    calibration_target_events: int

    def to_blob(self) -> bytes:
        return encode_bundle(
            {
                "vessels": [dict(v) for v in self.vessels],
                "guard_hours": self.guard_hours,
                "calibration_target_events": self.calibration_target_events,
            },
            {},
        )

    @classmethod
    def from_blob(cls, blob: bytes) -> Registry:
        meta, _ = decode_bundle(blob)
        return cls(
            vessels=tuple(dict(v) for v in meta["vessels"]),
            guard_hours=float(meta["guard_hours"]),
            calibration_target_events=int(meta["calibration_target_events"]),
        )


@dataclass(frozen=True)
class Truth:
    """Per-channel ground truth: presence, vessel identity, site, split, and truth tier.

    Only tier-one channels are admissible to a reported metric, per SD3 Section 5.3; the tier is
    carried here so the scoring stage and the split-integrity audit can enforce that directly.
    """

    KIND: ClassVar[str] = "truth"

    channels: tuple[dict[str, Any], ...]

    def to_blob(self) -> bytes:
        return encode_bundle({"channels": [dict(c) for c in self.channels]}, {})

    @classmethod
    def from_blob(cls, blob: bytes) -> Truth:
        meta, _ = decode_bundle(blob)
        return cls(channels=tuple(dict(c) for c in meta["channels"]))


@dataclass(frozen=True)
class Gram:
    """A spectral gram per channel: the band-limited front-end output."""

    KIND: ClassVar[str] = "gram"
    GRAM: ClassVar[ArraySchema] = f8((None, None, None), "power")
    FREQS: ClassVar[ArraySchema] = f8((None,), "hertz")

    gram: npt.NDArray[np.float64]  # shape [channels, freq_bins, frames]
    freqs: npt.NDArray[np.float64]  # shape [freq_bins]
    channel_ids: tuple[str, ...]
    sample_rate: float
    nfft: int
    hop: int

    def __post_init__(self) -> None:
        self.GRAM.validate(self.gram, name="gram")
        self.FREQS.validate(self.freqs, name="freqs")
        if len(self.channel_ids) != self.gram.shape[0]:
            raise ValueError("channel_ids length must equal the gram's channel axis")

    def to_blob(self) -> bytes:
        return encode_bundle(
            {
                "channel_ids": list(self.channel_ids),
                "sample_rate": self.sample_rate,
                "nfft": self.nfft,
                "hop": self.hop,
                "units": self.GRAM.units,
            },
            {"gram": self.gram, "freqs": self.freqs},
        )

    @classmethod
    def from_blob(cls, blob: bytes) -> Gram:
        meta, arrays = decode_bundle(blob)
        return cls(
            gram=np.asarray(arrays["gram"], dtype=np.float64),
            freqs=np.asarray(arrays["freqs"], dtype=np.float64),
            channel_ids=tuple(meta["channel_ids"]),
            sample_rate=float(meta["sample_rate"]),
            nfft=int(meta["nfft"]),
            hop=int(meta["hop"]),
        )


@dataclass(frozen=True)
class Detections:
    """Per-channel detector output: an in-band energy and a confidence in the unit interval."""

    KIND: ClassVar[str] = "detections"
    CONFIDENCE: ClassVar[ArraySchema] = f8((None,), "probability")
    BAND_ENERGY: ClassVar[ArraySchema] = f8((None,), "power")

    channel_ids: tuple[str, ...]
    confidence: npt.NDArray[np.float64]  # shape [channels]
    band_energy: npt.NDArray[np.float64]  # shape [channels]
    band_low_hz: float
    band_high_hz: float

    def __post_init__(self) -> None:
        self.CONFIDENCE.validate(self.confidence, name="confidence")
        self.BAND_ENERGY.validate(self.band_energy, name="band_energy")

    def to_blob(self) -> bytes:
        return encode_bundle(
            {
                "channel_ids": list(self.channel_ids),
                "band_low_hz": self.band_low_hz,
                "band_high_hz": self.band_high_hz,
            },
            {"confidence": self.confidence, "band_energy": self.band_energy},
        )

    @classmethod
    def from_blob(cls, blob: bytes) -> Detections:
        meta, arrays = decode_bundle(blob)
        return cls(
            channel_ids=tuple(meta["channel_ids"]),
            confidence=np.asarray(arrays["confidence"], dtype=np.float64),
            band_energy=np.asarray(arrays["band_energy"], dtype=np.float64),
            band_low_hz=float(meta["band_low_hz"]),
            band_high_hz=float(meta["band_high_hz"]),
        )


@dataclass(frozen=True)
class SplitAudit:
    """The split-integrity audit artifact of SD3 Section 5.6.

    A run whose audit does not pass is invalid by construction, and its metrics are not
    reportable. The payload records the assignment hash, the empty-intersection proofs, the guard
    verification, the quarantine roster, and the truth-tier composition of the evaluation set.
    """

    KIND: ClassVar[str] = "split_audit"

    passed: bool
    payload: dict[str, Any] = field(default_factory=dict)

    def to_blob(self) -> bytes:
        return encode_bundle({"passed": self.passed, "payload": self.payload}, {})

    @classmethod
    def from_blob(cls, blob: bytes) -> SplitAudit:
        meta, _ = decode_bundle(blob)
        return cls(passed=bool(meta["passed"]), payload=dict(meta["payload"]))


@dataclass(frozen=True)
class SurrogateDist:
    """Fitted surrogate line-statistic distributions derived from train-side vessels only.

    The payload is the serialised :class:`fathom.surrogate.fit.SurrogateDistributions`, carrying the
    first-principles structural ranges, the train-side quiet fraction, and the train roster, so the
    train-side-only derivation of SD3 Section 5.4 is auditable from the artifact itself.
    """

    KIND: ClassVar[str] = "surrogate_dist"

    payload: dict[str, Any]

    def to_blob(self) -> bytes:
        return encode_bundle({"payload": self.payload}, {})

    @classmethod
    def from_blob(cls, blob: bytes) -> SurrogateDist:
        meta, _ = decode_bundle(blob)
        return cls(payload=dict(meta["payload"]))


@dataclass(frozen=True)
class SurrogateParams:
    """The per-channel surrogate injection record: every parameter and seed, for regeneration.

    Each record names the injected channel, its drawn machinery, kinematic, and propagation
    parameters, the seed, the requested signal-to-noise ratio, and the achieved ratio re-measured
    after injection. No absolute level appears; the level is stated only as a signal-to-noise ratio.
    """

    KIND: ClassVar[str] = "surrogate_params"

    records: tuple[dict[str, Any], ...]

    def to_blob(self) -> bytes:
        return encode_bundle({"records": [dict(r) for r in self.records]}, {})

    @classmethod
    def from_blob(cls, blob: bytes) -> SurrogateParams:
        meta, _ = decode_bundle(blob)
        return cls(records=tuple(dict(r) for r in meta["records"]))


@dataclass(frozen=True)
class E1Report:
    """The E1 realism report: the two response distributions, their comparison, and the verdict."""

    KIND: ClassVar[str] = "e1_report"

    payload: dict[str, Any]

    def to_blob(self) -> bytes:
        return encode_bundle({"payload": self.payload}, {})

    @classmethod
    def from_blob(cls, blob: bytes) -> E1Report:
        meta, _ = decode_bundle(blob)
        return cls(payload=dict(meta["payload"]))


@dataclass(frozen=True)
class Report:
    """The assembled scored report of the acceptance path."""

    KIND: ClassVar[str] = "report"

    payload: dict[str, Any]

    def to_blob(self) -> bytes:
        return encode_bundle({"payload": self.payload}, {})

    @classmethod
    def from_blob(cls, blob: bytes) -> Report:
        meta, _ = decode_bundle(blob)
        return cls(payload=dict(meta["payload"]))
