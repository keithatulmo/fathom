"""Acquisition manifest schema.

WO-2 Section 2 fixes what every stored object records: its origin URL, retrieval timestamp, byte
count, SHA-256, license record with evidence link, truth condition, site, instrument, sample rate,
and band coverage. One JSON manifest is written per acquisition batch, and the manifests are
authoritative in manifest-first mode until ledger rows are backfilled. The mechanical training
eligibility flag travels with each object so the license rule of AC2 is enforceable downstream.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .sources import LicenseClass


class ObjectRecord(BaseModel):
    """The manifest record of one acquired object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    origin_url: str
    retrieved_at: str
    byte_count: int
    sha256: str
    raw_key: str
    license_class: LicenseClass
    license_evidence_url: str
    truth_condition: str
    training_eligible: bool
    media_type: str | None = None
    site: str | None = None
    instrument: str | None = None
    sample_rate_hz: float | None = None
    band_low_hz: float | None = None
    band_high_hz: float | None = None
    band_partial: bool = False
    duration_s: float | None = None


class AcquisitionManifest(BaseModel):
    """One acquisition batch: its source, creation time, and object records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: str
    source_id: str
    family: int
    created_at: str
    objects: tuple[ObjectRecord, ...] = ()
    notes: str = ""

    @property
    def total_bytes(self) -> int:
        """Return the total byte count across the batch's objects."""
        return sum(record.byte_count for record in self.objects)

    def to_json_bytes(self) -> bytes:
        """Serialize the manifest to indented JSON bytes for storage under ``manifests/``."""
        return self.model_dump_json(indent=2).encode("utf-8")

    @classmethod
    def from_json_bytes(cls, blob: bytes) -> AcquisitionManifest:
        """Parse a manifest from its JSON serialization."""
        return cls.model_validate_json(blob.decode("utf-8"))
