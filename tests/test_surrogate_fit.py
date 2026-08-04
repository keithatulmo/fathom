"""Tests for train-only parameter fitting and the injection-leak guard (SD3 Section 5.4)."""

from __future__ import annotations

import pytest

from fathom.surrogate.fit import (
    ForbiddenSplitError,
    draw_machinery,
    fit_distributions,
    select_train_vessels,
)
from fathom.surrogate.lines import build_lines


def _vessel(vessel_id: str, split: str, **extra: object) -> dict[str, object]:
    return {"vessel_id": vessel_id, "split": split, "quarantined": False, **extra}


def test_fit_reads_train_and_records_roster() -> None:
    train = [_vessel(f"T{i}", "train", is_quiet_tail=(i % 2 == 0)) for i in range(4)]
    dist = fit_distributions(train)
    assert dist.train_vessel_count == 4
    assert dist.train_vessel_ids == ("T0", "T1", "T2", "T3")
    # Two of four flagged quiet-tail, so the quiet fraction is read from the train roster.
    assert dist.quiet_fraction == pytest.approx(0.5)
    assert dist.provenance["structural_ranges"] == "first_principles"


def test_fitting_a_held_out_vessel_raises() -> None:
    # The leak guard: a test-split (held-out) vessel must raise rather than be read (AC3).
    with pytest.raises(ForbiddenSplitError):
        fit_distributions([_vessel("H0", "test")])
    with pytest.raises(ForbiddenSplitError):
        fit_distributions([_vessel("C0", "calibration")])
    # Even a single non-train vessel mixed into a train batch raises.
    with pytest.raises(ForbiddenSplitError):
        fit_distributions([_vessel("T0", "train"), _vessel("H0", "test")])


def test_select_train_vessels_excludes_other_splits_and_quarantine() -> None:
    vessels = [
        _vessel("T0", "train"),
        _vessel("C0", "calibration"),
        _vessel("H0", "test"),
        _vessel("Q0", "train", quarantined=True),
    ]
    selected = select_train_vessels(vessels)
    assert [v["vessel_id"] for v in selected] == ["T0"]
    # And what select returns is safe to fit.
    fit_distributions(selected)


def test_draw_machinery_is_deterministic_and_buildable() -> None:
    dist = fit_distributions([_vessel("T0", "train")])
    a = draw_machinery(dist, seed=5)
    b = draw_machinery(dist, seed=5)
    assert a == b
    # The drawn parameters build a non-empty line set.
    assert build_lines(a).freqs.shape[0] > 0
