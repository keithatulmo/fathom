"""Stage implementations, imported here so that importing the package populates the registry."""

from __future__ import annotations

from . import (  # noqa: F401  (registration side effects)
    audit,
    detection,
    dsp,
    e1,
    fixtures,
    scoring,
    surrogate_fit,
    surrogate_injection,
)
from .base import Stage, get_stage, register, registered_stages

__all__ = ["Stage", "get_stage", "register", "registered_stages"]
