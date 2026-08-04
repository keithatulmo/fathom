"""Corpus adequacy scoring against the CA1-CA7 criteria.

This module scores the corpus against Fathom_corpus_adequacy_criteria_v0.1, which decomposes "is the
corpus big enough" into a line-by-line checklist counted in distinct exchangeable units rather than
volume. Each criterion is scored pass, marginal, fail, or not-yet-measurable against the quantities
the ledger records, so SD1 closes on a reproducible checklist rather than a judgment. Thresholds are
the note's proposed engineering defaults and are provisional; any that would gate the SD1 close is
brought to the owner with its reasoning rather than applied as certified.

The scorer is honest about its limits: quiet-tail classification needs vessel kinematics (speed and
isolation) not yet recorded, splits are not yet assigned, and per-object band coverage is derived
from sample rate where recorded. Where a quantity cannot be measured, the verdict says so and names
the growth axis, per Section 6 of the note.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from .sources import FIRST_WAVE
from .splits import build_split_audit

# Proposed pass thresholds from the adequacy note (Section 4). Provisional engineering defaults.
QUIET_HELDOUT_MIN = 12
QUIET_TRAINSIDE_MIN = 12
VESSELS_TOTAL_MIN = 50
QUIET_TAIL_TOTAL_MIN = 30
CALIBRATION_EVENTS_MIN = 500
SITES_MIN = 3
WORKING_BAND_HZ = (4.0, 150.0)
REQUIRED_REGIMES = ("quiet", "nominal", "busy")

# A provisional proxy for the "close" part of a quiet-tail passage; quiet-tail also needs slow and
# isolated, which are not yet recorded, so this is an upper bound on candidates, not a count.
CLOSE_PASSAGE_M = 5000.0

# Proposed placement of each site on the clutter-intake curve. Not yet measured from the curve.
SITE_REGIMES: dict[str, str] = {
    "adeon_atlantic": "quiet",
    "mars_monterey_bay": "nominal",
    "strait_of_georgia": "busy",
}

PASS = "pass"
MARGINAL = "marginal"
FAIL = "fail"
NOT_MEASURABLE = "not_measurable"

# Binding criteria per Section 6; a fail on any of these does not close SD1.
_BINDING = {"CA2", "CA3", "CA4", "CA6", "CA7"}


@dataclass(frozen=True)
class CriterionScore:
    """One scored adequacy criterion."""

    id: str
    consumer: str
    measured: str
    threshold: str
    binding: str
    verdict: str
    directive: str


@dataclass(frozen=True)
class AdequacyReport:
    """The scored adequacy checklist and the SD1 close assessment."""

    generated_at: str
    criteria: tuple[CriterionScore, ...]
    sd1_close_met: bool
    blocking_failures: tuple[str, ...]
    markdown: str
    table: dict[str, Any]


def _tier(truth_condition: str) -> int:
    for tier in (1, 2, 3):
        if truth_condition.startswith(f"tier{tier}"):
            return tier
    return 3


def score_adequacy(
    objects: list[dict[str, Any]],
    presences: list[dict[str, Any]],
    *,
    splits: list[dict[str, Any]] | None = None,
    window_source_rates: dict[str, float] | None = None,
    generated_at: str | None = None,
) -> AdequacyReport:
    """Score the corpus against CA1-CA7 from objects, presences, and split assignment."""
    generated = generated_at or datetime.now(UTC).isoformat()
    family = {spec.source_id: spec.family for spec in FIRST_WAVE}
    split_audit = build_split_audit(splits) if splits else None

    # Distinct registry-grade (AIS-verified tier-one) vessels, total and per site, plus the
    # kinematically classified quiet-tail cohort and the coarser close-passage proxy.
    registry = {p["vessel_id"] for p in presences if int(p["registry_grade"]) == 1}
    per_site: dict[str, set[str]] = defaultdict(set)
    close_vessels: set[str] = set()
    quiet_tail_vessels: set[str] = set()
    for p in presences:
        if int(p["registry_grade"]) != 1:
            continue
        per_site[str(p["site"])].add(str(p["vessel_id"]))
        if p["closest_range_m"] is not None and float(p["closest_range_m"]) <= CLOSE_PASSAGE_M:
            close_vessels.add(str(p["vessel_id"]))
        if p.get("quiet_tail") == 1:
            quiet_tail_vessels.add(str(p["vessel_id"]))

    # Sites come from acquired audio objects and from correlated presences, so window-only
    # near-shore sites (whose audio is not downloaded) still count toward site diversity.
    sites = sorted(
        {str(o["site"]) for o in objects if o["site"]}
        | {str(p["site"]) for p in presences if p.get("site")}
    )
    regimes_present = {r for site in sites if (r := SITE_REGIMES.get(site)) is not None}
    biologic_objects = sum(1 for o in objects if family.get(str(o["source_id"])) == 4)
    # Band coverage is known where a source's sample rate reaches the top of the working band,
    # whether from an acquired audio object or a registered window (whose audio is not downloaded).
    band_sources = {
        str(o["source_id"])
        for o in objects
        if o["sample_rate_hz"] is not None
        and float(o["sample_rate_hz"]) / 2.0 >= WORKING_BAND_HZ[1]
    }
    for source_id, rate in (window_source_rates or {}).items():
        if rate / 2.0 >= WORKING_BAND_HZ[1]:
            band_sources.add(source_id)
    eval_tiers: dict[int, int] = defaultdict(int)
    for o in objects:
        eval_tiers[_tier(str(o["truth_condition"]))] += 1

    criteria = (
        _ca1(biologic_objects, len(presences), len(registry), len(per_site)),
        _ca2(len(registry), len(quiet_tail_vessels), len(close_vessels), split_audit),
        _ca3(len(registry), len(quiet_tail_vessels), split_audit),
        _ca4(len(registry)),
        _ca5(sites, regimes_present),
        _ca6(band_sources),
        _ca7(len(registry), eval_tiers),
    )

    blocking = tuple(c.id for c in criteria if c.id in _BINDING and c.verdict in (FAIL, MARGINAL))
    sd1_met = not blocking
    table = {
        "generated_at": generated,
        "criteria": [asdict(c) for c in criteria],
        "sd1_close_met": sd1_met,
        "blocking_failures": list(blocking),
        "measured": {
            "distinct_registry_vessels_total": len(registry),
            "distinct_quiet_tail_vessels": len(quiet_tail_vessels),
            "registry_vessels_by_site": {s: len(v) for s, v in per_site.items()},
            "close_passage_vessels_within_5km": len(close_vessels),
            "sites": sites,
            "regimes_present": sorted(regimes_present),
            "biologic_objects": biologic_objects,
            "band_known_sources": sorted(band_sources),
        },
    }
    return AdequacyReport(
        generated_at=generated,
        criteria=criteria,
        sd1_close_met=sd1_met,
        blocking_failures=blocking,
        markdown=_render(table, criteria),
        table=table,
    )


def _ca1(biologics: int, transits: int, ship_vessels: int, correlated_sites: int) -> CriterionScore:
    verdict = PASS if (biologics > 0 and correlated_sites >= 2) else MARGINAL
    return CriterionScore(
        id="CA1",
        consumer="Clutter background for rejection (SD6, E4)",
        measured=f"{biologics} biologic objects; {ship_vessels} ship vessels / {transits} transits "
        f"across {correlated_sites} correlated site(s), 1 season",
        threshold="Taxonomic diversity: several biologic classes and ship types, >=2 seasons",
        binding="rarely (non-blocking)",
        verdict=verdict,
        directive="Acquire family-four biologics and correlate a second season and site; record "
        "ship type. Non-blocking per Section 6, but the biologic side is currently empty.",
    )


def _ca2(
    registry_total: int, quiet_tail: int, close: int, split_audit: dict[str, Any] | None
) -> CriterionScore:
    need = QUIET_HELDOUT_MIN + QUIET_TRAINSIDE_MIN
    if split_audit is not None and split_audit["meets_ca2"]:
        verdict = PASS
    elif quiet_tail >= need:
        # The count is met, but held-out and train-side are not yet split disjointly.
        verdict = MARGINAL
    else:
        verdict = FAIL
    split_note = ""
    if split_audit is not None:
        held = split_audit["quiet_tail_counts"]["test"]
        train = split_audit["quiet_tail_counts"]["train"]
        split_note = f"; split held-out {held}, train-side {train}"
    return CriterionScore(
        id="CA2",
        consumer="Quiet-tail proxy vessels for surrogate validation (SD2, E1)",
        measured=f"{quiet_tail} quiet-tail vessels (close, slow, isolated); {close} close-only "
        f"proxy; {registry_total} registry-grade total{split_note}",
        threshold=f">={QUIET_HELDOUT_MIN} held-out and >={QUIET_TRAINSIDE_MIN} train-side disjoint",
        binding="binding",
        verdict=verdict,
        directive=f"Grow the quiet-tail cohort to >={need} disjoint vessels (have {quiet_tail}) by "
        "correlating more low-traffic passages, then assign the held-out and train-side split.",
    )


def _ca3(
    registry_total: int, quiet_tail: int, split_audit: dict[str, Any] | None
) -> CriterionScore:
    if split_audit is not None and split_audit["meets_ca3"]:
        verdict = PASS
    elif registry_total < VESSELS_TOTAL_MIN or quiet_tail < QUIET_TAIL_TOTAL_MIN:
        verdict = FAIL
    else:
        # Both counts met; disjoint split assignment and its proof are still pending.
        verdict = MARGINAL
    return CriterionScore(
        id="CA3",
        consumer="Vessel-level splits (SD3)",
        measured=f"{registry_total} distinct vessels total (need {VESSELS_TOTAL_MIN}); quiet-tail "
        f"{quiet_tail} (need {QUIET_TAIL_TOTAL_MIN}); no vessel-level splits assigned yet",
        threshold=f">={VESSELS_TOTAL_MIN} vessels total, >={QUIET_TAIL_TOTAL_MIN} quiet-tail total",
        binding="binding on the quiet subset",
        verdict=verdict,
        directive="Grow the general cohort toward 50 and the quiet-tail cohort toward 30, then "
        "assign pairwise-disjoint vessel-level splits and emit the disjointness proof.",
    )


def _ca4(exchangeable_vessels: int) -> CriterionScore:
    # The exchangeable unit for association/class calibration is the vessel (SD3 cluster bootstrap),
    # so the count that matters is distinct vessels, not raw passages.
    verdict = PASS if exchangeable_vessels >= CALIBRATION_EVENTS_MIN else FAIL
    return CriterionScore(
        id="CA4",
        consumer="Calibration set for conformal outputs (SD8, E6, A2)",
        measured=f"{exchangeable_vessels} exchangeable vessels for association/class calibration "
        f"(need ~{CALIBRATION_EVENTS_MIN}); existence/rejection draw from abundant clutter",
        threshold=f"~{CALIBRATION_EVENTS_MIN} exchangeable events per calibrated output",
        binding="partly (association/class outputs)",
        verdict=verdict,
        directive="The association and class outputs draw on distinct vessels and are short; grow "
        "the same cohort as CA2/CA3. Existence and rejection calibration pass on abundant clutter.",
    )


def _ca5(sites: list[str], regimes_present: set[str]) -> CriterionScore:
    covered = all(regime in regimes_present for regime in REQUIRED_REGIMES)
    verdict = PASS if (len(sites) >= SITES_MIN and covered) else FAIL
    return CriterionScore(
        id="CA5",
        consumer="Sites and clutter regimes (A1 staging, E6 shift)",
        measured=f"{len(sites)} sites {sites}; regimes present {sorted(regimes_present)}",
        threshold=f">={SITES_MIN} sites spanning quiet, nominal, busy with a quiet anchor",
        binding="floor",
        verdict=verdict,
        directive="Floor met by site count and span. Caveat: regime placement is proposed, not yet "
        "measured on the clutter-intake curve, and the quiet anchor's quietness is unverified.",
    )


def _ca6(band_sources: set[str]) -> CriterionScore:
    covers_proof = "mbari_pacific_sound_2khz" in band_sources
    # The working band is covered at the proof site and at least one other (near-shore) site where
    # the quiet-tail surrogate data lives; sources whose rate is unrecorded are a recorded partial.
    if covers_proof and len(band_sources) >= 2:
        verdict = PASS
    elif covers_proof:
        verdict = MARGINAL
    else:
        verdict = FAIL
    return CriterionScore(
        id="CA6",
        consumer="In-band content for the front end (SD4, E2)",
        measured=f"working band {WORKING_BAND_HZ[0]}-{WORKING_BAND_HZ[1]} Hz confirmed for "
        f"{sorted(band_sources)} via sample rate; other sources' rate not recorded",
        threshold=f"working band covered at the quiet and proof sites ({WORKING_BAND_HZ[0]}-"
        f"{WORKING_BAND_HZ[1]} Hz)",
        binding="binding if absent",
        verdict=verdict,
        directive="Record each object's sample rate at acquisition (or via decimation) so "
        "band coverage is quantified at the quiet site, not only the proof site.",
    )


def _ca7(registry_total: int, tiers: dict[int, int]) -> CriterionScore:
    total = sum(tiers.values())
    tier1_fraction = tiers.get(1, 0) / total if total else 0.0
    meets_counts = registry_total >= (QUIET_HELDOUT_MIN + QUIET_TRAINSIDE_MIN)
    verdict = PASS if (tier1_fraction >= 1.0 and meets_counts) else FAIL
    return CriterionScore(
        id="CA7",
        consumer="Truth quality (SD3 tiers)",
        measured=f"tier-one label fraction {tier1_fraction:.2f}; but only {registry_total} vessels "
        f"are AIS-verified tier-one, the rest are source-default labels",
        threshold="evaluation and held-out sets tier-one only, and the tier-one vessel count meets "
        "CA2 and CA3",
        binding="binding",
        verdict=verdict,
        directive="Verify tier-one by AIS correlation rather than source default, and grow the "
        "verified tier-one vessel count to satisfy CA2 and CA3.",
    )


def _render(table: dict[str, Any], criteria: tuple[CriterionScore, ...]) -> str:
    lines: list[str] = []
    lines.append("# Fathom corpus adequacy scorecard")
    lines.append("")
    lines.append(
        f"Scored {table['generated_at']} against Fathom_corpus_adequacy_criteria_v0.1. Adequacy is "
        "counted in distinct exchangeable units, not volume. Thresholds are the note's provisional "
        "engineering defaults; any that gates the SD1 close is brought to the owner with reasoning."
    )
    lines.append("")
    lines.append("| ID | Consumer | Measured | Threshold | Binding | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    for c in criteria:
        lines.append(
            f"| {c.id} | {c.consumer} | {c.measured} | {c.threshold} | {c.binding} | "
            f"**{c.verdict}** |"
        )
    lines.append("")
    lines.append("## Directives (Section 6: a fail is a growth axis, not a redesign)")
    lines.append("")
    for c in criteria:
        if c.verdict != PASS:
            lines.append(f"- **{c.id}** ({c.verdict}): {c.directive}")
    lines.append("")
    lines.append("## SD1 close condition")
    lines.append("")
    if table["sd1_close_met"]:
        lines.append("Every binding criterion passes; SD1 may close on this checklist.")
    else:
        lines.append(
            "SD1 does not close: binding criteria "
            + ", ".join(table["blocking_failures"])
            + " are not passing. These converge on one axis, distinct AIS-verified quiet-tail "
            "registry-grade vessels, so growth is directed there: correlate more AIS days, "
            "seasons, and sites; build quiet-tail kinematic classification; and acquire biologics "
            "for CA1. Alternatively the owner accepts a stated, reasoned narrowing of the proof "
            "claim to what the corpus supports, recorded rather than absorbed silently."
        )
    lines.append("")
    return "\n".join(lines)
