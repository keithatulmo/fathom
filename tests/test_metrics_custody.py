"""Hand-computed tests for the custody metric set."""

from __future__ import annotations

import numpy as np
import pytest

from fathom.metrics import label_consistency, loss_count, swap_count, time_to_reacquire


def test_label_consistency_majority_fraction() -> None:
    assert label_consistency(np.array([2, 2, 3, 2])) == pytest.approx(0.75)


def test_label_consistency_ignores_unheld_frames() -> None:
    # The -1 frame is not held, so consistency is measured over the three held frames.
    assert label_consistency(np.array([2, 2, -1, 2])) == pytest.approx(1.0)


def test_label_consistency_never_held_raises() -> None:
    with pytest.raises(ValueError):
        label_consistency(np.array([-1, -1]))


def test_swaps_only_between_held_frames() -> None:
    # Transitions: 0->0, 0->1 (swap), 1->1, 1->-1 (loss), -1->2 (reacquire).
    assigned = np.array([0, 0, 1, 1, -1, 2])
    assert swap_count(assigned) == 1
    assert loss_count(assigned) == 1


def test_no_swaps_or_losses_when_stable() -> None:
    assigned = np.array([5, 5, 5])
    assert swap_count(assigned) == 0
    assert loss_count(assigned) == 0


def test_time_to_reacquire_after_fade() -> None:
    present = np.array([True, False, True, True])
    associated = np.array([True, False, False, True])
    # Reappearance at index 2; first association at index 3; time-to-reacquire is 1.
    assert time_to_reacquire(present, associated) == [1]


def test_time_to_reacquire_censors_unrecovered() -> None:
    present = np.array([True, False, True])
    associated = np.array([True, False, False])
    # The target reappears but is never re-associated within the record, so it is censored.
    assert time_to_reacquire(present, associated) == []


def test_time_to_reacquire_immediate() -> None:
    present = np.array([True, False, True, True])
    associated = np.array([False, False, True, True])
    assert time_to_reacquire(present, associated) == [0]
