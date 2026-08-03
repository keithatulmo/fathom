"""Manifest models for the thin runner.

SD10 Section 6.2 fixes the pipeline form as a manifest-driven batch DAG executed by a thin
in-repo runner. A manifest names a run's seed and its nodes; each node names a stage, its input
bindings, and its configuration. An input binding is a string of the form ``<node>:<output>``
that wires one node's named output to another node's named input. The manifest is validated with
pydantic v2 and carries no external-orchestrator concepts.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

# The separator between a node identifier and one of its named outputs in an input binding.
REF_SEPARATOR = ":"


class NodeSpec(BaseModel):
    """One node of the pipeline DAG: a stage instance with bindings and configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    stage: str
    inputs: dict[str, str] = {}
    config: dict[str, Any] = {}


class Manifest(BaseModel):
    """A run manifest: a named, seeded set of DAG nodes and the terminal report node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    seed: int
    nodes: tuple[NodeSpec, ...]
    report_node: str | None = None
    description: str = ""

    @model_validator(mode="after")
    def _check(self) -> Manifest:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("node identifiers must be unique")
        known = set(ids)
        for node in self.nodes:
            for input_name, ref in node.inputs.items():
                if REF_SEPARATOR not in ref:
                    raise ValueError(
                        f"node {node.id!r} input {input_name!r} must be '<node>:<output>'"
                    )
                source = ref.split(REF_SEPARATOR, 1)[0]
                if source not in known:
                    raise ValueError(
                        f"node {node.id!r} input {input_name!r} refers to unknown node {source!r}"
                    )
        if self.report_node is not None and self.report_node not in known:
            raise ValueError(f"report_node {self.report_node!r} is not a node identifier")
        return self


def parse_ref(ref: str) -> tuple[str, str]:
    """Split an input binding ``<node>:<output>`` into its node and output names."""
    node, output = ref.split(REF_SEPARATOR, 1)
    return node, output
