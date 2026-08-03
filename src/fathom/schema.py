"""Typed array schemas and seam validation.

SD10 Section 6.4 fixes the typed interface style: stage boundaries carry array fields whose
schemas state shape, dtype, and units, and those schemas are validated at the seams, with
validation on in continuous integration and debug runs and off in hot loops. A reviewer should
be able to read a stage's contract in one screen, and a violated contract should fail loudly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Validation is on by default and can be turned off for hot loops. The environment variable
# lets continuous integration and debug sessions force it on explicitly and lets a benchmark
# turn it off without code changes; the default already satisfies the "on in CI and debug" rule.
_VALIDATION_ENABLED = os.environ.get("FATHOM_VALIDATE", "1") != "0"


def validation_enabled() -> bool:
    """Return whether seam validation is currently enabled."""
    return _VALIDATION_ENABLED


@contextmanager
def validation_disabled() -> Iterator[None]:
    """Temporarily disable seam validation, for use inside measured hot loops."""
    global _VALIDATION_ENABLED
    previous = _VALIDATION_ENABLED
    _VALIDATION_ENABLED = False
    try:
        yield
    finally:
        _VALIDATION_ENABLED = previous


@contextmanager
def validation_forced() -> Iterator[None]:
    """Temporarily force seam validation on, regardless of the ambient setting."""
    global _VALIDATION_ENABLED
    previous = _VALIDATION_ENABLED
    _VALIDATION_ENABLED = True
    try:
        yield
    finally:
        _VALIDATION_ENABLED = previous


class SchemaError(ValueError):
    """Raised when an array violates its declared schema at a stage seam."""


@dataclass(frozen=True)
class ArraySchema:
    """A frozen declaration of an array's shape, dtype, and units.

    Shape entries are either a fixed non-negative integer or ``None`` for a free dimension, so
    a schema can pin the rank and the fixed axes while leaving batch- or time-like axes open.
    The units string carries no arithmetic; it exists so that a seam mismatch in meaning, not
    only in shape, is greppable in the contract and reviewable in one screen.
    """

    shape: tuple[int | None, ...]
    dtype: np.dtype[np.generic]
    units: str

    def validate(self, array: npt.NDArray[np.generic], *, name: str = "array") -> None:
        """Raise :class:`SchemaError` if the array does not conform to this schema.

        Validation is skipped when seam validation is disabled, which is the hot-loop path.
        """
        if not _VALIDATION_ENABLED:
            return
        if array.ndim != len(self.shape):
            raise SchemaError(
                f"{name}: expected rank {len(self.shape)}, got rank {array.ndim} "
                f"with shape {array.shape}"
            )
        for axis, (expected, actual) in enumerate(zip(self.shape, array.shape, strict=True)):
            if expected is not None and expected != actual:
                raise SchemaError(
                    f"{name}: axis {axis} expected size {expected}, got {actual} "
                    f"(full shape {array.shape})"
                )
        if array.dtype != self.dtype:
            raise SchemaError(f"{name}: expected dtype {self.dtype}, got {array.dtype}")
        if not np.all(np.isfinite(array)):
            raise SchemaError(f"{name}: array contains non-finite values")


def f8(shape: tuple[int | None, ...], units: str) -> ArraySchema:
    """Construct a float64 :class:`ArraySchema` with the given shape and units."""
    return ArraySchema(shape=shape, dtype=np.dtype(np.float64), units=units)
