"""The first-wave source register and the license-eligibility rule.

WO-2 Section 3 fixes the first-wave sources, their families and roles, and per-family volume caps,
and Section 3a fixes the verified access methods and license classes as of 2026-08-02. This module
encodes that register as validated data, so acquisition is driven from a reviewable table rather
than from ad hoc code. The license rule is mechanical: a source whose license class is research-only
or unknown is not eligible for any training-designated partition and is quarantined until the owner
rules, per WO-2 Section 1 and AC2. Whether a quarantined source is usable even as evaluation
reference is an owner decision, not an engineering default.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class LicenseClass(StrEnum):
    """License classes that determine training eligibility."""

    CC_BY = "cc_by"
    PUBLIC_DOMAIN = "public_domain"
    PUBLIC = "public"
    RESEARCH_ONLY = "research_only"
    UNKNOWN = "unknown"


class AccessClass(StrEnum):
    """How a source is reached, which determines whether acquisition is unattended."""

    ANON_S3 = "anon_s3"
    HTTPS = "https"
    GCS_HTTPS = "gcs_https"
    FDSN = "fdsn"
    CREDENTIALED = "credentialed"


# License classes that permit use in a training-designated partition. Research-only and unknown are
# excluded and therefore quarantined until the owner rules.
_TRAINING_ELIGIBLE_LICENSES = frozenset(
    {LicenseClass.CC_BY, LicenseClass.PUBLIC_DOMAIN, LicenseClass.PUBLIC}
)


def training_eligible(license_class: LicenseClass) -> bool:
    """Return whether a license class permits use in a training-designated partition."""
    return license_class in _TRAINING_ELIGIBLE_LICENSES


class SourceSpec(BaseModel):
    """A first-wave source: its family, role, license, access method, and volume cap."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    name: str
    family: int
    role: str
    license_class: LicenseClass
    license_evidence_url: str
    access_class: AccessClass
    truth_condition_default: str
    cap_gb: float
    site: str | None = None
    instrument: str | None = None
    sample_rate_hz: float | None = None
    band_low_hz: float | None = None
    band_high_hz: float | None = None
    band_partial: bool = False
    s3_bucket: str | None = None
    s3_region: str | None = None
    deferred: bool = False
    quarantined_dataset: bool = False
    notes: str = ""

    @property
    def training_eligible(self) -> bool:
        """Whether objects from this source may enter a training-designated partition."""
        return training_eligible(self.license_class) and not self.quarantined_dataset


# The first-wave register, per WO-2 Sections 3 and 3a. Caps are per-family engineering bounds on
# this wave, not corpus ceilings. Bands and rates left None are recorded per object at acquisition.
FIRST_WAVE: tuple[SourceSpec, ...] = (
    # Family one: quiet-anchor candidates, cap 150 GB total across the family.
    SourceSpec(
        source_id="ooi_slope_base_broadband",
        name="OOI Regional Cabled Array, Slope Base broadband hydrophone",
        family=1,
        role="quiet-site background; quiet-anchor candidate",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://oceanobservatories.org/data/",
        access_class=AccessClass.HTTPS,
        truth_condition_default="tier1_verified_quiet",
        cap_gb=75.0,
        site="slope_base",
        instrument="broadband_hydrophone",
        notes="Fetch named files by archive path; robots.txt disallows crawling, not data clients.",
    ),
    SourceSpec(
        source_id="ooi_axial_base_broadband",
        name="OOI Regional Cabled Array, Axial Base broadband hydrophone",
        family=1,
        role="quiet-site background; quiet-anchor candidate",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://oceanobservatories.org/data/",
        access_class=AccessClass.HTTPS,
        truth_condition_default="tier1_verified_quiet",
        cap_gb=75.0,
        site="axial_base",
        instrument="broadband_hydrophone",
        notes="Low-frequency channels reach via IRIS/EarthScope FDSN if band is insufficient.",
    ),
    SourceSpec(
        source_id="adeon_ncei",
        name="ADEON deep-water Atlantic lander sites via the NCEI passive acoustic archive",
        family=1,
        role="quiet-site background; off-season windows",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://www.ncei.noaa.gov/products/passive-acoustic-data",
        access_class=AccessClass.GCS_HTTPS,
        truth_condition_default="tier1_verified_quiet",
        cap_gb=50.0,
        site="adeon_atlantic",
        instrument="lander_hydrophone",
        s3_bucket="noaa-passive-bioacoustic",
        notes="Public GCS bucket, no account; ADEON lives under prefix 'adeon/' (verified live).",
    ),
    # Family two: clutter-rich sites, cap 150 GB total.
    SourceSpec(
        source_id="mbari_pacific_sound_2khz",
        name="MBARI MARS cabled hydrophone, Pacific Sound 2 kHz",
        family=2,
        role="moderate-traffic clutter site; whale-season biologic load",
        license_class=LicenseClass.CC_BY,
        license_evidence_url="https://www.mbari.org/project/pacific-ocean-sound-recordings/",
        access_class=AccessClass.ANON_S3,
        truth_condition_default="tier1_ais_correlated",
        cap_gb=100.0,
        site="mars_monterey_bay",
        instrument="cabled_hydrophone",
        sample_rate_hz=2000.0,
        s3_bucket="pacific-sound-2khz",
        s3_region="us-west-2",
        notes="Anonymous public S3, CC-BY 4.0; recommended first pull to prove the path.",
    ),
    SourceSpec(
        source_id="sanctsound_ncei",
        name="SanctSound raw data via the NCEI public bucket",
        family=2,
        role="near-lane sanctuary clutter sites, in season",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://sanctsound.ioos.us/",
        access_class=AccessClass.GCS_HTTPS,
        truth_condition_default="tier1_ais_correlated",
        cap_gb=50.0,
        site="sanctsound_multi",
        instrument="moored_hydrophone",
        s3_bucket="noaa-passive-bioacoustic",
        notes="Public GCS bucket; SanctSound lives under prefix 'sanctsound/' (verified live).",
    ),
    SourceSpec(
        source_id="onc_strait_of_georgia",
        name="Ocean Networks Canada hydrophones, incl. the busy Strait of Georgia",
        family=2,
        role="busy clutter site",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://www.oceannetworks.ca/",
        access_class=AccessClass.CREDENTIALED,
        truth_condition_default="tier1_ais_correlated",
        cap_gb=40.0,
        site="strait_of_georgia",
        instrument="cabled_hydrophone",
        notes="Oceans 3.0 archive-file API; needs FATHOM_ONC_TOKEN. Location SCVIP or SEVIP "
        "(Strait of Georgia, verified live).",
    ),
    # Family three: vessel-truth substrate, cap 50 GB plus negligible AIS volume.
    SourceSpec(
        source_id="marinecadastre_ais",
        name="MarineCadastre AIS for United States waters (htdata bulk archive)",
        family=3,
        role="tier-one vessel-presence truth with registry-grade identity",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://marinecadastre.gov/ais/",
        access_class=AccessClass.HTTPS,
        truth_condition_default="tier1_ais_correlated",
        cap_gb=50.0,
        notes="Filter daily zips to dates and regions overlapping the acoustic recordings.",
    ),
    SourceSpec(
        source_id="deepship",
        name="DeepShip labeled vessel dataset",
        family=3,
        role="quarantined; no license, email-to-author only",
        license_class=LicenseClass.UNKNOWN,
        license_evidence_url="https://github.com/irfankamboh/DeepShip",
        access_class=AccessClass.CREDENTIALED,
        truth_condition_default="tier3_asserted",
        cap_gb=0.0,
        quarantined_dataset=True,
        notes="Not acquired: quarantined from training; evaluation use is an owner ruling.",
    ),
    SourceSpec(
        source_id="shipsear",
        name="ShipsEar labeled vessel dataset",
        family=3,
        role="quarantined pending written terms",
        license_class=LicenseClass.RESEARCH_ONLY,
        license_evidence_url="https://underwaternoise.atlanttic.uvigo.es/shipsear/",
        access_class=AccessClass.CREDENTIALED,
        truth_condition_default="tier3_asserted",
        cap_gb=0.0,
        quarantined_dataset=True,
        notes="Not acquired: research-only; quarantined from training pending owner ruling.",
    ),
    # Family four: labeled biologics, cap 50 GB.
    SourceSpec(
        source_id="sanctsound_annotations",
        name="SanctSound annotation products alongside raw data",
        family=4,
        role="tier-one biologic labels for the rejection layer's clutter classes",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://sanctsound.ioos.us/",
        access_class=AccessClass.GCS_HTTPS,
        truth_condition_default="tier1_curated_biologic",
        cap_gb=25.0,
        site="sanctsound_multi",
        instrument="annotation_product",
        s3_bucket="noaa-passive-bioacoustic",
    ),
    SourceSpec(
        source_id="dclde_baleen",
        name="DCLDE workshop low-frequency baleen sets",
        family=4,
        role="tier-one biologic labels, low-frequency baleen",
        license_class=LicenseClass.PUBLIC,
        license_evidence_url="https://www.soest.hawaii.edu/ore/dclde/",
        access_class=AccessClass.GCS_HTTPS,
        truth_condition_default="tier1_curated_biologic",
        cap_gb=25.0,
        instrument="annotation_product",
        s3_bucket="noaa-passive-bioacoustic",
        notes="In the NCEI bucket under prefix 'dclde/'; per-set license recorded at acquisition.",
    ),
)


def source_by_id(source_id: str) -> SourceSpec:
    """Return the first-wave source spec with the given identifier."""
    for spec in FIRST_WAVE:
        if spec.source_id == source_id:
            return spec
    raise KeyError(f"unknown source {source_id!r}")


def family_cap_gb(family: int) -> float:
    """Return the total volume cap in gigabytes for a source family, per WO-2 Section 3."""
    caps = {1: 150.0, 2: 150.0, 3: 50.0, 4: 50.0}
    return caps[family]
