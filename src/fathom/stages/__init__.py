"""Stage implementations, imported here so that importing the package populates the registry."""

from __future__ import annotations

from . import audit, detection, dsp, fixtures, scoring  # noqa: F401  (registration side effects)
from .base import Stage, get_stage, register, registered_stages

__all__ = ["Stage", "get_stage", "register", "registered_stages"]
