"""Tests for array-schema validation and the validation toggle."""

from __future__ import annotations

import numpy as np
import pytest

from fathom.schema import SchemaError, f8, validation_disabled, validation_enabled


def test_conforming_array_passes() -> None:
    schema = f8((None, 3), "power")
    schema.validate(np.ones((4, 3), dtype=np.float64))


def test_wrong_rank_raises() -> None:
    schema = f8((None, 3), "power")
    with pytest.raises(SchemaError):
        schema.validate(np.ones(3, dtype=np.float64))


def test_wrong_fixed_axis_raises() -> None:
    schema = f8((None, 3), "power")
    with pytest.raises(SchemaError):
        schema.validate(np.ones((4, 5), dtype=np.float64))


def test_wrong_dtype_raises() -> None:
    schema = f8((2,), "power")
    with pytest.raises(SchemaError):
        schema.validate(np.ones(2, dtype=np.int64))


def test_non_finite_raises() -> None:
    schema = f8((2,), "power")
    with pytest.raises(SchemaError):
        schema.validate(np.array([1.0, np.inf]))


def test_validation_can_be_disabled() -> None:
    schema = f8((2,), "power")
    assert validation_enabled()
    with validation_disabled():
        assert not validation_enabled()
        # A non-conforming array passes silently while validation is off (the hot-loop path).
        schema.validate(np.ones((5, 5), dtype=np.int64))
    assert validation_enabled()
