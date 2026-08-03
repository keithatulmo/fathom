"""The stage abstraction and registry.

A stage is a pure, versioned transform from named input blobs to named output blobs. Stages own
their typed contracts: each deserializes its inputs into the frozen artifact dataclasses, does
its work, and returns serialized outputs, so the runner stays thin and type-agnostic and the
lineage records only hashes. A stage is deterministic given its inputs, its configuration, and
its seed, which is what makes the double-execution hash-equality check meaningful.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel


class Stage(ABC):
    """A pure, versioned transform between named input and output blobs."""

    NAME: ClassVar[str]
    VERSION: ClassVar[str]
    # Maps each output name to the artifact kind recorded for it in the lineage ledger.
    OUTPUTS: ClassVar[dict[str, str]]

    @abstractmethod
    def parse_config(self, raw: dict[str, Any]) -> BaseModel:
        """Validate a raw configuration mapping into this stage's frozen config model."""

    @abstractmethod
    def run(self, inputs: dict[str, bytes], config: Any, seed: int) -> dict[str, bytes]:
        """Transform input blobs into output blobs deterministically."""


_REGISTRY: dict[str, Stage] = {}


def register(stage: Stage) -> None:
    """Register a stage instance under its name, rejecting duplicate registrations."""
    if stage.NAME in _REGISTRY:
        raise ValueError(f"stage {stage.NAME!r} is already registered")
    _REGISTRY[stage.NAME] = stage


def get_stage(name: str) -> Stage:
    """Return the registered stage with the given name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown stage {name!r}") from exc


def registered_stages() -> tuple[str, ...]:
    """Return the names of all registered stages, sorted."""
    return tuple(sorted(_REGISTRY))
