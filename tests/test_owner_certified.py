"""Tests that the owner-certified figures are named fields carrying no value."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.config import OwnerCertifiedFigures, ScoringConfig


def test_defaults_are_none() -> None:
    figures = OwnerCertifiedFigures()
    assert figures.detection_range_r is None
    assert figures.confirmer_pd is None


def test_populating_detection_range_raises() -> None:
    # Detection range r is owner-certified and never asserted; the type admits only None.
    with pytest.raises(ValidationError):
        OwnerCertifiedFigures(detection_range_r=12000.0)


def test_populating_confirmer_pd_raises() -> None:
    with pytest.raises(ValidationError):
        OwnerCertifiedFigures(confirmer_pd=0.9)


def test_scoring_config_carries_unset_figures() -> None:
    dumped = ScoringConfig().model_dump(mode="json")
    assert dumped["owner_certified"] == {"detection_range_r": None, "confirmer_pd": None}
