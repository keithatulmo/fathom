"""Tests that the owner-certified figures are named fields carrying no value."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fathom.config import (
    E1Config,
    FitConfig,
    InjectionConfig,
    OwnerCertifiedFigures,
    ScoringConfig,
)

# Substrings that would name an absolute source level or an owner-certified figure in an interface.
_FORBIDDEN_FIELD_SUBSTRINGS = (
    "source_level",
    "absolute_level",
    "db_re",
    "spl",
    "range_r",
    "confirmer",
)


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


def test_injector_configs_carry_unset_figures() -> None:
    # AC5: the owner-certified guard extends to the injector's fit, injection, and E1 stages.
    for config in (FitConfig(), InjectionConfig(), E1Config()):
        dumped = config.model_dump(mode="json")
        assert dumped["owner_certified"] == {"detection_range_r": None, "confirmer_pd": None}


def test_injector_interface_expresses_level_only_as_a_ratio() -> None:
    # The injected level is a signal-to-noise ratio; no absolute source level is expressible.
    assert "snr_db" in InjectionConfig.model_fields
    for config_cls in (FitConfig, InjectionConfig, E1Config):
        for name in config_cls.model_fields:
            lowered = name.lower()
            assert not any(bad in lowered for bad in _FORBIDDEN_FIELD_SUBSTRINGS), name


def test_machinery_and_records_hold_no_absolute_level() -> None:
    # The drawn machinery parameters and the injection record name no absolute or source level.
    from fathom.surrogate.fit import draw_machinery, fit_distributions

    dist = fit_distributions([{"vessel_id": "T0", "split": "train", "quarantined": False}])
    machinery = draw_machinery(dist, seed=1).to_dict()
    for key in machinery:
        assert not any(bad in key.lower() for bad in _FORBIDDEN_FIELD_SUBSTRINGS), key
    # Levels appear only as amplitudes (dimensionless family weights), never as absolute figures.
    assert "shaft_amp" in machinery and "source_level" not in machinery
