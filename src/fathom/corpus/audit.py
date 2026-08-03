"""The corpus audit report generator.

WO-2 Section 5 fixes the deliverable that closes SD1: a per-source and cross-source audit, in
markdown with a machine-readable companion table, structured so each ratified SD1 criterion can be
scored from it. Per source it reports hours by site, band coverage against the working band, license
class with its evidence link, truth-tier composition, and distinct registry-grade vessel counts.
Across sources it reports the quiet-anchor candidate ranking, coverage of the clutter-intake curve's
quiet, nominal, and busy regimes, and a gap list stating what the first wave failed to secure. The
report states what was not acquired and why, so acquisition is scored honestly rather than by the
absence of evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .sources import FIRST_WAVE, SourceSpec

WORKING_BAND_HZ = (3.0, 300.0)


@dataclass(frozen=True)
class AuditReport:
    """The audit as rendered markdown and a machine-readable companion structure."""

    markdown: str
    table: dict[str, Any]
    generated_at: str


def _tier_of(truth_condition: str) -> int:
    for tier in (1, 2, 3):
        if truth_condition.startswith(f"tier{tier}"):
            return tier
    return 3


def _aggregate(
    objects: list[dict[str, Any]], vessel_counts: dict[str, int] | None
) -> dict[str, dict[str, Any]]:
    per_source: dict[str, dict[str, Any]] = {}
    for obj in objects:
        source_id = str(obj["source_id"])
        agg = per_source.setdefault(
            source_id,
            {
                "source_id": source_id,
                "objects": 0,
                "bytes": 0,
                "hours": 0.0,
                "hours_known": True,
                "sites": set(),
                "band_low": None,
                "band_high": None,
                "band_partial": False,
                "tiers": {1: 0, 2: 0, 3: 0},
                "training_eligible": True,
            },
        )
        agg["objects"] += 1
        agg["bytes"] += int(obj["byte_count"])
        duration = obj.get("duration_s")
        if duration is None:
            agg["hours_known"] = False
        else:
            agg["hours"] += float(duration) / 3600.0
        if obj.get("site"):
            agg["sites"].add(str(obj["site"]))
        low, high = obj.get("band_low_hz"), obj.get("band_high_hz")
        if low is not None:
            agg["band_low"] = low if agg["band_low"] is None else min(agg["band_low"], low)
        if high is not None:
            agg["band_high"] = high if agg["band_high"] is None else max(agg["band_high"], high)
        agg["band_partial"] = agg["band_partial"] or bool(obj.get("band_partial"))
        agg["tiers"][_tier_of(str(obj["truth_condition"]))] += 1
        agg["training_eligible"] = agg["training_eligible"] and bool(obj.get("training_eligible"))

    for source_id, agg in per_source.items():
        agg["sites"] = sorted(agg["sites"])
        agg["registry_vessels"] = None if vessel_counts is None else vessel_counts.get(source_id)
    return per_source


def _rank_quiet_anchors(
    per_source: dict[str, dict[str, Any]], specs: dict[str, SourceSpec]
) -> list[dict[str, Any]]:
    candidates = []
    for source_id, agg in per_source.items():
        spec = specs.get(source_id)
        if spec is None or spec.family != 1:
            continue
        tier1 = agg["tiers"][1]
        total = max(1, sum(agg["tiers"].values()))
        score = agg["hours"] * (tier1 / total)
        candidates.append(
            {
                "source_id": source_id,
                "sites": agg["sites"],
                "hours": round(agg["hours"], 2),
                "tier1_fraction": round(tier1 / total, 3),
                "score": round(score, 3),
                "curve_placement": "quiet",
            }
        )
    return sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)


def build_audit_report(
    objects: list[dict[str, Any]],
    *,
    sources: tuple[SourceSpec, ...] = FIRST_WAVE,
    vessel_counts: dict[str, int] | None = None,
    generated_at: str | None = None,
) -> AuditReport:
    """Assemble the corpus audit report from recorded corpus objects and the source register."""
    generated = generated_at or datetime.now(UTC).isoformat()
    specs = {spec.source_id: spec for spec in sources}
    per_source = _aggregate(objects, vessel_counts)
    quiet_ranking = _rank_quiet_anchors(per_source, specs)

    families_present = {specs[s].family for s in per_source if s in specs}
    regimes = {
        "quiet": any(specs[s].family == 1 for s in per_source if s in specs),
        "nominal": any(specs[s].family == 2 for s in per_source if s in specs),
        "busy": any(
            s in per_source and not specs[s].deferred
            for s in specs
            if specs[s].role.startswith("busy")
        ),
    }

    not_acquired = [
        {
            "source_id": spec.source_id,
            "reason": "deferred pending credential" if spec.deferred else "quarantined dataset",
            "role": spec.role,
        }
        for spec in sources
        if (spec.deferred or spec.quarantined_dataset) and spec.source_id not in per_source
    ]

    gaps = _gap_list(per_source, specs, regimes, vessel_counts)

    table = {
        "generated_at": generated,
        "working_band_hz": list(WORKING_BAND_HZ),
        "per_source": [_source_row(agg, specs) for agg in per_source.values()],
        "quiet_anchor_ranking": quiet_ranking,
        "curve_regime_coverage": regimes,
        "families_present": sorted(families_present),
        "not_acquired": not_acquired,
        "gaps": gaps,
    }
    markdown = _render_markdown(table, specs)
    return AuditReport(markdown=markdown, table=table, generated_at=generated)


def _source_row(agg: dict[str, Any], specs: dict[str, SourceSpec]) -> dict[str, Any]:
    spec = specs.get(agg["source_id"])
    return {
        "source_id": agg["source_id"],
        "family": spec.family if spec else None,
        "sites": agg["sites"],
        "objects": agg["objects"],
        "gigabytes": round(agg["bytes"] / 1e9, 4),
        "hours": round(agg["hours"], 3) if agg["hours_known"] else None,
        "band_low_hz": agg["band_low"],
        "band_high_hz": agg["band_high"],
        "band_partial": agg["band_partial"],
        "license_class": spec.license_class.value if spec else None,
        "license_evidence_url": spec.license_evidence_url if spec else None,
        "truth_tiers": agg["tiers"],
        "training_eligible": agg["training_eligible"],
        "registry_vessels": agg["registry_vessels"],
    }


def _gap_list(
    per_source: dict[str, dict[str, Any]],
    specs: dict[str, SourceSpec],
    regimes: dict[str, bool],
    vessel_counts: dict[str, int] | None,
) -> list[dict[str, str]]:
    gaps: list[dict[str, str]] = []
    if not regimes["busy"]:
        gaps.append(
            {
                "criterion": "site diversity",
                "gap": "No busy-regime site was acquired; the Strait of Georgia site is deferred "
                "pending the Ocean Networks Canada account and token.",
            }
        )
    if vessel_counts is None or not vessel_counts:
        gaps.append(
            {
                "criterion": "truth quality",
                "gap": "No AIS-correlated registry-grade vessel counts are recorded yet, so "
                "target-side truth volume for the reject-versus-miss denominator is unquantified.",
            }
        )
    if any(agg["band_partial"] for agg in per_source.values()):
        gaps.append(
            {
                "criterion": "in-band content fit",
                "gap": "At least one source has partial band coverage below the 300 Hz top of the "
                "working band; low-frequency channels may be needed via FDSN.",
            }
        )
    if not per_source:
        gaps.append(
            {
                "criterion": "volume",
                "gap": "No objects were acquired in this wave; the corpus is empty.",
            }
        )
    return gaps


def _render_markdown(table: dict[str, Any], specs: dict[str, SourceSpec]) -> str:
    lines: list[str] = []
    lines.append("# Fathom corpus audit report (first wave)")
    lines.append("")
    lines.append(
        f"This report audits the first-wave corpus acquisition as of {table['generated_at']}, "
        f"against the working band {WORKING_BAND_HZ[0]} to {WORKING_BAND_HZ[1]} Hz. It is the "
        "deliverable that closes SD1 by memo on the ratified criteria."
    )
    lines.append("")
    lines.append("## Per-source summary")
    lines.append("")
    lines.append(
        "| Source | Fam | Sites | Objects | GB | Hours | Band (Hz) | Partial | License | "
        "Tier1/2/3 | Train | Vessels |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in table["per_source"]:
        tiers = row["truth_tiers"]
        band = f"{row['band_low_hz']}-{row['band_high_hz']}"
        hours = "n/a" if row["hours"] is None else f"{row['hours']}"
        vessels = "n/a" if row["registry_vessels"] is None else str(row["registry_vessels"])
        lines.append(
            f"| {row['source_id']} | {row['family']} | {', '.join(row['sites']) or '-'} | "
            f"{row['objects']} | {row['gigabytes']} | {hours} | {band} | {row['band_partial']} | "
            f"{row['license_class']} | {tiers[1]}/{tiers[2]}/{tiers[3]} | "
            f"{row['training_eligible']} | {vessels} |"
        )
    lines.append("")
    lines.append("## Quiet-anchor candidate ranking")
    lines.append("")
    if table["quiet_anchor_ranking"]:
        for rank, candidate in enumerate(table["quiet_anchor_ranking"], start=1):
            lines.append(
                f"{rank}. {candidate['source_id']} at sites "
                f"{', '.join(candidate['sites']) or '-'}, "
                f"{candidate['hours']} hours, tier-one fraction {candidate['tier1_fraction']}, "
                f"placed in the {candidate['curve_placement']} regime of the clutter-intake curve."
            )
    else:
        lines.append("No quiet-anchor candidates were acquired in this wave.")
    lines.append("")
    lines.append("## Clutter-intake curve regime coverage")
    lines.append("")
    for regime, present in table["curve_regime_coverage"].items():
        lines.append(f"- The {regime} regime is {'covered' if present else 'not covered'}.")
    lines.append("")
    lines.append("## What was not acquired, and why")
    lines.append("")
    if table["not_acquired"]:
        for entry in table["not_acquired"]:
            lines.append(f"- {entry['source_id']}: {entry['reason']} ({entry['role']}).")
    else:
        lines.append("- Every registered non-deferred source was acquired in this wave.")
    lines.append("")
    lines.append("## Gap list against the SD1 criteria")
    lines.append("")
    if table["gaps"]:
        for gap in table["gaps"]:
            lines.append(f"- **{gap['criterion']}**: {gap['gap']}")
    else:
        lines.append("- No gaps were identified against the SD1 criteria in this wave.")
    lines.append("")
    return "\n".join(lines)
