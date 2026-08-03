"""Assembly of the scored report payload.

The report gathers the SD3 metrics into one structure: the reject-versus-miss operating point and
curve with the exact miss bound, the calibration table and reliability data, the custody
placeholders the trivial detector cannot fill, the coverage accounting, and the cluster-bootstrap
intervals. It also carries the split-integrity summary and the reportable flag, because a run whose
audit failed has metrics that are not reportable per SD3 Section 5.6.

The payload is pure: it contains no run identifier and no timestamp, so its content hash depends
only on the data and the double-execution hash-equality check remains meaningful. Run identity is
recorded in the ledger and in the human-readable report file the runner writes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any

import numpy as np
import numpy.typing as npt

from .artifacts import Detections, SplitAudit, Truth
from .config import ScoringConfig
from .metrics import (
    cluster_bootstrap,
    coverage_from_mask,
    expected_calibration_error,
    operating_point,
    reject_miss_curve,
)

_EVAL_SPLIT = "test"


def _point_to_dict(point: Any) -> dict[str, Any]:
    data = asdict(point)
    return {
        key: (float(value) if isinstance(value, float) else value) for key, value in data.items()
    }


def _mean(values: npt.NDArray[np.float64]) -> float:
    return float(np.mean(values))


def build_report_payload(
    detections: Detections,
    truth: Truth,
    audit: SplitAudit,
    config: ScoringConfig,
    seed: int,
) -> dict[str, Any]:
    """Assemble the pure report payload from a run's detections, truth, and audit."""
    confidence_by_channel = dict(zip(detections.channel_ids, detections.confidence, strict=True))

    owner_certified = {
        "detection_range_r": config.owner_certified.detection_range_r,
        "confirmer_pd": config.owner_certified.confirmer_pd,
    }
    audit_summary = {
        "passed": audit.passed,
        "assignment_hash": audit.payload.get("assignment_hash"),
        "intersections_empty": audit.payload.get("intersections_empty"),
        "guard_ok": audit.payload.get("guard_ok"),
        "eval_truth_tiers": audit.payload.get("eval_truth_tiers"),
        "quarantine_roster": audit.payload.get("quarantine_roster"),
    }

    if not audit.passed:
        # A failed audit invalidates the run; metrics are withheld rather than reported.
        return {
            "reportable": False,
            "reason": "split-integrity audit failed; metrics are not reportable per SD3 5.6",
            "audit": audit_summary,
            "owner_certified": owner_certified,
        }

    eval_channels = sorted(
        (
            channel
            for channel in truth.channels
            if str(channel["split"]) == _EVAL_SPLIT and int(channel["truth_tier"]) == 1
        ),
        key=lambda channel: str(channel["channel_id"]),
    )
    present = np.array([bool(channel["present"]) for channel in eval_channels], dtype=np.bool_)
    scores = np.array(
        [float(confidence_by_channel[str(channel["channel_id"])]) for channel in eval_channels],
        dtype=np.float64,
    )

    point = operating_point(
        present, scores, config.operating_threshold, site=_EVAL_SPLIT, alpha=config.miss_alpha
    )
    curve = reject_miss_curve(present, scores, alpha=config.miss_alpha)

    calibration = expected_calibration_error(
        scores, present, n_bins=config.ece_bins, min_bin_count=config.ece_min_bin_count
    )

    coverage = coverage_from_mask(np.ones(len(detections.channel_ids), dtype=np.bool_))

    uncertainty = _bootstrap_section(eval_channels, confidence_by_channel, config, seed)

    return {
        "reportable": True,
        "audit": audit_summary,
        "evaluation": {
            "split": _EVAL_SPLIT,
            "truth_tier": 1,
            "present_trials": point.present_trials,
            "clutter_trials": point.clutter_trials,
        },
        "reject_versus_miss": {
            "operating_point": _point_to_dict(point),
            "curve": [_point_to_dict(p) for p in curve],
        },
        "calibration": {
            "ece": calibration.ece,
            "n_bins_effective": calibration.n_bins_effective,
            "reliability": [asdict(b) for b in calibration.bins],
        },
        "custody": {
            "status": "not_exercised",
            "note": (
                "The trivial acceptance detector produces no tracks, so the custody set is "
                "exercised by unit tests here and lands with the tracker in a later work order."
            ),
            "label_consistency": None,
            "swaps": None,
            "losses": None,
            "time_to_reacquire": None,
        },
        "coverage": asdict(coverage),
        "uncertainty": uncertainty,
        "config": {
            "operating_threshold": config.operating_threshold,
            "miss_alpha": config.miss_alpha,
            "ece_bins": config.ece_bins,
            "ece_min_bin_count": config.ece_min_bin_count,
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_alpha": config.bootstrap_alpha,
        },
        "owner_certified": owner_certified,
    }


def _bootstrap_section(
    eval_channels: list[dict[str, Any]],
    confidence_by_channel: dict[str, np.float64],
    config: ScoringConfig,
    seed: int,
) -> dict[str, Any]:
    """Cluster-bootstrap the present-target miss rate and mean confidence, grouped by vessel."""
    miss_by_vessel: dict[str, list[float]] = defaultdict(list)
    conf_by_vessel: dict[str, list[float]] = defaultdict(list)
    for channel in eval_channels:
        if not channel["present"]:
            continue
        vessel = str(channel["vessel_id"])
        confidence = float(confidence_by_channel[str(channel["channel_id"])])
        miss_by_vessel[vessel].append(0.0 if confidence >= config.operating_threshold else 1.0)
        conf_by_vessel[vessel].append(confidence)

    if not miss_by_vessel:
        return {"miss_rate": None, "present_confidence": None}

    miss_units = [np.array(values, dtype=np.float64) for values in miss_by_vessel.values()]
    conf_units = [np.array(values, dtype=np.float64) for values in conf_by_vessel.values()]
    miss_ci = cluster_bootstrap(
        miss_units,
        _mean,
        seed=seed,
        n_resamples=config.bootstrap_resamples,
        alpha=config.bootstrap_alpha,
    )
    conf_ci = cluster_bootstrap(
        conf_units,
        _mean,
        seed=seed,
        n_resamples=config.bootstrap_resamples,
        alpha=config.bootstrap_alpha,
    )
    return {
        "exchangeable_unit": "vessel",
        "miss_rate": asdict(miss_ci),
        "present_confidence": asdict(conf_ci),
    }
