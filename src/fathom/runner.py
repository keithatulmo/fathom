"""The thin manifest-driven DAG runner.

SD10 Section 6.2 fixes the pipeline form as a manifest-driven batch DAG executed by a thin in-repo
runner that writes the lineage ledger natively as it executes. The runner topologically orders the
nodes, computes a content-addressed computation key for each, consults the cache, executes the
stage on a miss, stores every output by its hash, and records the artifact, its parents, and the
stage execution. The intelligence lives in the ledger and the content addressing, not in the
runner, which stays deliberately small. There is no external orchestrator.

The reproduction path re-executes a recorded run's manifest from the ledger and verifies that every
recomputed output hash equals the recorded one, reporting any mismatch as a first-class failure.
"""

from __future__ import annotations

import secrets
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .artifacts import Report
from .determinism import (
    code_version,
    derive_seed,
    environment_fingerprint,
    pinned_threads,
)
from .hashing import canonical_json_bytes, hash_json, sha256_hex
from .ledger import Ledger
from .manifest import Manifest, parse_ref
from .serialize import decode_json
from .stages import get_stage

# Threads are pinned to one during scored runs so reductions are deterministic (SD10 6.6).
_THREAD_LIMIT = 1


@dataclass(frozen=True)
class RunResult:
    """The outcome of a run: its identity, terminal report, and reportability."""

    run_id: str
    report_hash: str | None
    reportable: bool
    node_outputs: dict[str, dict[str, str]]
    report_path: Path | None


@dataclass(frozen=True)
class Mismatch:
    """A single hash mismatch found during reproduction."""

    node_id: str
    output: str
    expected: str
    actual: str


@dataclass(frozen=True)
class ReproResult:
    """The outcome of a reproduction: how many artifacts matched and any mismatches."""

    run_id: str
    checked: int
    mismatches: tuple[Mismatch, ...]

    @property
    def ok(self) -> bool:
        """Return whether every recomputed artifact hash matched the recorded one."""
        return not self.mismatches


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _topological_order(manifest: Manifest) -> list[str]:
    """Return node identifiers in a dependency-respecting order, or raise on a cycle."""
    dependencies: dict[str, set[str]] = {node.id: set() for node in manifest.nodes}
    dependents: dict[str, set[str]] = {node.id: set() for node in manifest.nodes}
    for node in manifest.nodes:
        for ref in node.inputs.values():
            source = parse_ref(ref)[0]
            dependencies[node.id].add(source)
            dependents[source].add(node.id)

    ready = deque(sorted(nid for nid, deps in dependencies.items() if not deps))
    order: list[str] = []
    remaining = {nid: set(deps) for nid, deps in dependencies.items()}
    while ready:
        nid = ready.popleft()
        order.append(nid)
        for dependent in sorted(dependents[nid]):
            remaining[dependent].discard(nid)
            if not remaining[dependent]:
                ready.append(dependent)
    if len(order) != len(manifest.nodes):
        raise ValueError("manifest DAG contains a cycle")
    return order


class Runner:
    """Executes and reproduces manifest-driven runs over a store and lineage ledger."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        state = repo_root / ".fathom"
        from .store import ContentAddressedStore  # local import keeps module import order simple

        self._store = ContentAddressedStore(state / "store")
        self._ledger = Ledger(state / "ledger.db")
        self._runs_dir = state / "runs"
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._code_version = code_version(repo_root)
        self._env_fingerprint = environment_fingerprint(repo_root)

    @property
    def ledger(self) -> Ledger:
        """Return the lineage ledger this runner writes to."""
        return self._ledger

    def close(self) -> None:
        """Close the underlying ledger connection."""
        self._ledger.close()

    # -- execution --------------------------------------------------------------------------

    def _config_hash(self, stage_name: str, version: str, config: BaseModel) -> str:
        return hash_json(
            {"stage": stage_name, "version": version, "config": config.model_dump(mode="json")}
        )

    def _computation_key(
        self, config_hash: str, node_seed: int, input_hashes: dict[str, str]
    ) -> str:
        return hash_json(
            {
                "config_hash": config_hash,
                "code_version": self._code_version,
                "env_fingerprint": self._env_fingerprint,
                "seed": node_seed,
                "inputs": input_hashes,
            }
        )

    def _gather_inputs(
        self, node_inputs: dict[str, str], produced: dict[str, dict[str, str]]
    ) -> tuple[dict[str, str], dict[str, bytes]]:
        input_hashes: dict[str, str] = {}
        input_blobs: dict[str, bytes] = {}
        for input_name, ref in node_inputs.items():
            source, output = parse_ref(ref)
            digest = produced[source][output]
            input_hashes[input_name] = digest
            input_blobs[input_name] = self._store.get(digest)
        return input_hashes, input_blobs

    def run(self, manifest: Manifest) -> RunResult:
        """Execute a manifest end to end, writing lineage for every artifact as it goes."""
        run_id = self._make_run_id(manifest.name)
        manifest_blob = canonical_json_bytes(manifest.model_dump(mode="json"))
        manifest_hash = self._store.put(manifest_blob)

        with pinned_threads(_THREAD_LIMIT):
            self._ledger.insert_run(
                run_id=run_id,
                manifest_hash=manifest_hash,
                code_version=self._code_version,
                env_fingerprint=self._env_fingerprint,
                seed=manifest.seed,
                thread_limit=_THREAD_LIMIT,
            )
            self._ledger.insert_artifact(
                artifact_hash=manifest_hash,
                kind="manifest",
                producing_stage=None,
                stage_version=None,
                code_version=self._code_version,
                config_hash=hash_json({"manifest": manifest.name}),
                env_fingerprint=self._env_fingerprint,
                seed=manifest.seed,
            )
            try:
                produced = self._execute_all(run_id, manifest)
            except Exception as exc:  # record the failure, then re-raise for the caller
                self._ledger.append_run_status(run_id, "failed", detail=repr(exc))
                raise
            result = self._finalize(run_id, manifest, produced)
        return result

    def _execute_all(self, run_id: str, manifest: Manifest) -> dict[str, dict[str, str]]:
        nodes = {node.id: node for node in manifest.nodes}
        produced: dict[str, dict[str, str]] = {}
        for node_id in _topological_order(manifest):
            node = nodes[node_id]
            stage = get_stage(node.stage)
            config = stage.parse_config(node.config)
            config_hash = self._config_hash(stage.NAME, stage.VERSION, config)
            node_seed = derive_seed(manifest.seed, node_id)
            input_hashes, input_blobs = self._gather_inputs(node.inputs, produced)
            computation_key = self._computation_key(config_hash, node_seed, input_hashes)

            started = _now_iso()
            cached = self._ledger.cache_get(computation_key)
            if cached is not None and all(self._store.has(h) for h in cached.values()):
                outputs = cached
                reused = True
            else:
                output_blobs = stage.run(input_blobs, config, node_seed)
                outputs = {name: self._store.put(blob) for name, blob in output_blobs.items()}
                self._ledger.cache_put(computation_key, outputs)
                reused = False
            finished = _now_iso()

            parents = tuple(sorted(set(input_hashes.values())))
            for name, digest in outputs.items():
                self._ledger.insert_artifact(
                    artifact_hash=digest,
                    kind=stage.OUTPUTS.get(name, "unknown"),
                    producing_stage=stage.NAME,
                    stage_version=stage.VERSION,
                    code_version=self._code_version,
                    config_hash=config_hash,
                    env_fingerprint=self._env_fingerprint,
                    seed=node_seed,
                    parents=parents,
                )
            self._ledger.insert_stage_execution(
                exec_id=f"{run_id}:{node_id}",
                run_id=run_id,
                node_id=node_id,
                stage=stage.NAME,
                stage_version=stage.VERSION,
                computation_key=computation_key,
                config_hash=config_hash,
                reused=reused,
                started_at=started,
                finished_at=finished,
                inputs=input_hashes,
                outputs=outputs,
            )
            produced[node_id] = outputs
        return produced

    def _finalize(
        self, run_id: str, manifest: Manifest, produced: dict[str, dict[str, str]]
    ) -> RunResult:
        report_node = self._report_node(manifest)
        if report_node is None:
            self._ledger.append_run_status(run_id, "completed")
            return RunResult(run_id, None, True, produced, None)

        report_hash = produced[report_node]["report"]
        payload = Report.from_blob(self._store.get(report_hash)).payload
        reportable = bool(payload.get("reportable", False))
        self._record_metrics(run_id, payload, report_hash)

        report_path = self._runs_dir / f"{run_id}.json"
        report_document = {
            "run_id": run_id,
            "manifest": manifest.name,
            "seed": manifest.seed,
            "code_version": self._code_version,
            "env_fingerprint": self._env_fingerprint,
            "report_hash": report_hash,
            "report": payload,
        }
        report_path.write_bytes(canonical_json_bytes(report_document))

        self._ledger.append_run_status(run_id, "completed" if reportable else "invalidated")
        return RunResult(run_id, report_hash, reportable, produced, report_path)

    def _record_metrics(self, run_id: str, payload: dict[str, Any], report_hash: str) -> None:
        self._ledger.insert_metric(
            run_id=run_id,
            name="reportable",
            value=1.0 if payload.get("reportable") else 0.0,
            payload=None,
            config_hash=None,
            artifact_hash=report_hash,
        )
        if not payload.get("reportable"):
            return
        point = payload["reject_versus_miss"]["operating_point"]
        scalars = {
            "miss_rate": point["miss_rate"],
            "miss_rate_upper": point["miss_rate_upper"],
            "rejection_fraction": point["rejection_fraction"],
            "ece": payload["calibration"]["ece"],
            "coverage_fraction": payload["coverage"]["fraction"],
        }
        for name, value in scalars.items():
            self._ledger.insert_metric(
                run_id=run_id,
                name=name,
                value=float(value),
                payload=None,
                config_hash=None,
                artifact_hash=report_hash,
            )

    @staticmethod
    def _report_node(manifest: Manifest) -> str | None:
        if manifest.report_node is not None:
            return manifest.report_node
        for node in manifest.nodes:
            if node.stage == "scoring":
                return node.id
        return None

    @staticmethod
    def _make_run_id(name: str) -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{name}-{stamp}-{secrets.token_hex(3)}"

    # -- reproduction -----------------------------------------------------------------------

    def reproduce(self, run_id: str) -> ReproResult:
        """Re-execute a recorded run's manifest and verify every output hash matches.

        The manifest is retrieved from the store by the hash recorded against the run, then the
        DAG is re-executed with no cache consultation, and each recomputed output hash is compared
        to the hash the ledger recorded for that run. Any mismatch is a first-class failure.
        """
        run_row = self._ledger.get_run(run_id)
        if run_row is None:
            raise KeyError(f"unknown run {run_id!r}")
        manifest = Manifest.model_validate(decode_json(self._store.get(run_row.manifest_hash)))

        recorded = self._recorded_outputs(run_id)
        nodes = {node.id: node for node in manifest.nodes}
        produced: dict[str, dict[str, str]] = {}
        mismatches: list[Mismatch] = []
        checked = 0

        with pinned_threads(_THREAD_LIMIT):
            for node_id in _topological_order(manifest):
                node = nodes[node_id]
                stage = get_stage(node.stage)
                config = stage.parse_config(node.config)
                node_seed = derive_seed(manifest.seed, node_id)
                _, input_blobs = self._gather_inputs(node.inputs, produced)
                output_blobs = stage.run(input_blobs, config, node_seed)

                outputs: dict[str, str] = {}
                for name, blob in output_blobs.items():
                    actual = sha256_hex(blob)
                    outputs[name] = actual
                    expected = recorded.get(node_id, {}).get(name)
                    checked += 1
                    if expected is not None and expected != actual:
                        mismatches.append(Mismatch(node_id, name, expected, actual))
                produced[node_id] = outputs

        return ReproResult(run_id=run_id, checked=checked, mismatches=tuple(mismatches))

    def _recorded_outputs(self, run_id: str) -> dict[str, dict[str, str]]:
        recorded: dict[str, dict[str, str]] = {}
        for execution in self._ledger.get_stage_executions(run_id):
            outputs = {
                row.name: row.artifact_hash
                for row in self._ledger.get_stage_io(execution.exec_id, "output")
            }
            recorded[execution.node_id] = outputs
        return recorded
