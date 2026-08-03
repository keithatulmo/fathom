"""Tests for the source register, license eligibility, and the manifest schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.corpus.manifest import AcquisitionManifest, ObjectRecord
from fathom.corpus.sources import (
    FIRST_WAVE,
    AccessClass,
    LicenseClass,
    SourceSpec,
    family_cap_gb,
    source_by_id,
    training_eligible,
)


def test_training_eligibility_by_license() -> None:
    assert training_eligible(LicenseClass.CC_BY)
    assert training_eligible(LicenseClass.PUBLIC)
    assert training_eligible(LicenseClass.PUBLIC_DOMAIN)
    assert not training_eligible(LicenseClass.RESEARCH_ONLY)
    assert not training_eligible(LicenseClass.UNKNOWN)


def test_quarantined_datasets_are_never_training_eligible() -> None:
    for spec in FIRST_WAVE:
        if spec.quarantined_dataset:
            assert spec.training_eligible is False
            assert spec.cap_gb == 0.0


def test_deepship_and_shipsear_are_quarantined() -> None:
    assert source_by_id("deepship").quarantined_dataset is True
    assert source_by_id("shipsear").quarantined_dataset is True


def test_mbari_is_anon_s3_cc_by_and_eligible() -> None:
    spec = source_by_id("mbari_pacific_sound_2khz")
    assert spec.access_class == AccessClass.ANON_S3
    assert spec.license_class == LicenseClass.CC_BY
    assert spec.training_eligible is True
    assert spec.s3_bucket == "pacific-sound-2khz"


def test_family_caps() -> None:
    assert family_cap_gb(1) == 150.0
    assert family_cap_gb(2) == 150.0
    assert family_cap_gb(3) == 50.0
    assert family_cap_gb(4) == 50.0


def test_unknown_source_raises() -> None:
    with pytest.raises(KeyError):
        source_by_id("does_not_exist")


def test_source_spec_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SourceSpec(
            source_id="x",
            name="x",
            family=1,
            role="r",
            license_class=LicenseClass.PUBLIC,
            license_evidence_url="u",
            access_class=AccessClass.HTTPS,
            truth_condition_default="tier1",
            cap_gb=1.0,
            bogus_field=True,  # type: ignore[call-arg]
        )


def test_manifest_roundtrip_and_total_bytes() -> None:
    record = ObjectRecord(
        source_id="mbari_pacific_sound_2khz",
        origin_url="s3://pacific-sound-2khz/x.wav",
        retrieved_at="2026-08-03T00:00:00+00:00",
        byte_count=1234,
        sha256="a" * 64,
        raw_key="raw/aa/" + "a" * 64,
        license_class=LicenseClass.CC_BY,
        license_evidence_url="https://example.org",
        truth_condition="tier1_ais_correlated",
        training_eligible=True,
    )
    manifest = AcquisitionManifest(
        batch_id="b1",
        source_id="mbari_pacific_sound_2khz",
        family=2,
        created_at="2026-08-03T00:00:00+00:00",
        objects=(record, record),
    )
    restored = AcquisitionManifest.from_json_bytes(manifest.to_json_bytes())
    assert restored == manifest
    assert restored.total_bytes == 2468
