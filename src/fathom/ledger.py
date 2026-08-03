"""The append-only lineage ledger.

The ledger is the SQLite sidecar of SD10 Section 6.5: a single auditable file recording corpus
sources, runs, artifacts and their parents, stage executions and their inputs and outputs, and
metrics. It is append-only by construction, enforced with triggers that abort any UPDATE or
DELETE, so the record of what produced what can only grow. The content-addressed cache lives
here too, mapping a computation key to the artifacts a prior identical computation produced.

The ledger stores metadata only. Timestamps and status transitions are recorded here and never
enter an artifact's content hash, so wall-clock time never perturbs reproducibility.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Every table is append-only; these triggers make that a property of the database, not a habit.
_APPEND_ONLY_TABLES = (
    "corpus_sources",
    "corpus_objects",
    "corpus_derivatives",
    "corpus_vessel_presence",
    "runs",
    "run_status",
    "artifacts",
    "artifact_parents",
    "stage_executions",
    "stage_io",
    "metrics",
    "computation_cache",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_sources (
    source_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    license          TEXT NOT NULL,
    truth_condition  TEXT NOT NULL,
    detection_range_r TEXT,   -- owner-certified figure r; NULL by discipline, never populated
    confirmer_pd      TEXT,   -- owner-certified confirmer P_d; NULL by discipline, never set
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    manifest_hash    TEXT NOT NULL,
    code_version     TEXT NOT NULL,
    env_fingerprint  TEXT NOT NULL,
    seed             INTEGER NOT NULL,
    thread_limit     INTEGER,
    started_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_status (
    seq              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    status           TEXT NOT NULL,
    detail           TEXT,
    at               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_hash    TEXT PRIMARY KEY,
    kind             TEXT NOT NULL,
    producing_stage  TEXT,
    stage_version    TEXT,
    code_version     TEXT NOT NULL,
    config_hash      TEXT NOT NULL,
    env_fingerprint  TEXT NOT NULL,
    seed             INTEGER,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_parents (
    artifact_hash    TEXT NOT NULL REFERENCES artifacts(artifact_hash),
    parent_hash      TEXT NOT NULL,
    PRIMARY KEY (artifact_hash, parent_hash)
);

CREATE TABLE IF NOT EXISTS stage_executions (
    exec_id          TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    node_id          TEXT NOT NULL,
    stage            TEXT NOT NULL,
    stage_version    TEXT NOT NULL,
    computation_key  TEXT NOT NULL,
    config_hash      TEXT NOT NULL,
    reused           INTEGER NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_io (
    exec_id          TEXT NOT NULL REFERENCES stage_executions(exec_id),
    direction        TEXT NOT NULL,   -- 'input' or 'output'
    name             TEXT NOT NULL,
    artifact_hash    TEXT NOT NULL,
    PRIMARY KEY (exec_id, direction, name)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    name             TEXT NOT NULL,
    value            REAL,
    payload          TEXT,
    config_hash      TEXT,
    artifact_hash    TEXT,
    PRIMARY KEY (run_id, name)
);

CREATE TABLE IF NOT EXISTS computation_cache (
    computation_key  TEXT PRIMARY KEY,
    outputs          TEXT NOT NULL    -- JSON mapping output name -> artifact hash
);

CREATE TABLE IF NOT EXISTS corpus_objects (
    sha256               TEXT PRIMARY KEY,
    source_id            TEXT NOT NULL,
    origin_url           TEXT NOT NULL,
    retrieved_at         TEXT NOT NULL,
    byte_count           INTEGER NOT NULL,
    raw_key              TEXT NOT NULL,
    license_class        TEXT NOT NULL,
    license_evidence_url TEXT NOT NULL,
    truth_condition      TEXT NOT NULL,
    training_eligible    INTEGER NOT NULL,  -- 0 quarantines the object from training partitions
    media_type           TEXT,
    site                 TEXT,
    instrument           TEXT,
    sample_rate_hz       REAL,
    band_low_hz          REAL,
    band_high_hz         REAL,
    band_partial         INTEGER NOT NULL DEFAULT 0,
    duration_s           REAL,
    batch_id             TEXT,
    created_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_derivatives (
    sha256           TEXT PRIMARY KEY,
    parent_sha256    TEXT NOT NULL REFERENCES corpus_objects(sha256),
    derived_key      TEXT NOT NULL,
    operation        TEXT NOT NULL,
    params_json      TEXT NOT NULL,
    byte_count       INTEGER NOT NULL,
    sample_rate_hz   REAL,
    band_low_hz      REAL,
    band_high_hz     REAL,
    created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corpus_vessel_presence (
    presence_id      TEXT PRIMARY KEY,   -- hash of (recording_sha, vessel_id)
    recording_sha    TEXT NOT NULL,
    site             TEXT,
    vessel_id        TEXT NOT NULL,
    mmsi             TEXT,
    imo              TEXT,
    name             TEXT,
    first_seen_s     REAL,
    last_seen_s      REAL,
    closest_range_m  REAL,
    registry_grade   INTEGER NOT NULL,   -- only registry-grade presence is tier-one truth
    truth_tier       INTEGER NOT NULL,
    ais_source_sha   TEXT,
    created_at       TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class RunRow:
    """A recorded run's immutable identity and environment fingerprint."""

    run_id: str
    manifest_hash: str
    code_version: str
    env_fingerprint: str
    seed: int
    thread_limit: int | None
    started_at: str


@dataclass(frozen=True)
class StageExecutionRow:
    """A recorded stage execution within a run."""

    exec_id: str
    run_id: str
    node_id: str
    stage: str
    stage_version: str
    computation_key: str
    config_hash: str
    reused: bool
    started_at: str
    finished_at: str


@dataclass(frozen=True)
class StageIORow:
    """A single input or output binding of a stage execution to an artifact."""

    exec_id: str
    direction: str
    name: str
    artifact_hash: str


class Ledger:
    """An append-only SQLite lineage ledger."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._install_append_only_guards()
        self._conn.commit()

    def _install_append_only_guards(self) -> None:
        for table in _APPEND_ONLY_TABLES:
            for verb in ("UPDATE", "DELETE"):
                trigger = f"{table}_no_{verb.lower()}"
                self._conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {trigger} "
                    f"BEFORE {verb} ON {table} "
                    f"BEGIN SELECT RAISE(ABORT, 'ledger is append-only'); END"
                )

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    # -- corpus -----------------------------------------------------------------------------

    def insert_corpus_source(
        self,
        *,
        source_id: str,
        name: str,
        license_: str,
        truth_condition: str,
    ) -> None:
        """Record a corpus source with its license and truth condition.

        The owner-certified figures ``detection_range_r`` and ``confirmer_pd`` are named columns
        that are never populated in this unclassified, synthetic repository; they are written as
        NULL by discipline so their absence is explicit in the schema rather than implied.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO corpus_sources "
            "(source_id, name, license, truth_condition, detection_range_r, confirmer_pd, "
            " created_at) VALUES (?, ?, ?, ?, NULL, NULL, ?)",
            (source_id, name, license_, truth_condition, _now_iso()),
        )
        self._conn.commit()

    def insert_corpus_object(self, *, batch_id: str, record: dict[str, object]) -> None:
        """Record an acquired corpus object; idempotent because identity is the content hash.

        The ``training_eligible`` flag is stored as an integer so the license rule of AC2 is a
        mechanical property of the ledger: a research-only or unknown-license object is filed with
        the flag cleared and can never be selected into a training-designated partition.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO corpus_objects "
            "(sha256, source_id, origin_url, retrieved_at, byte_count, raw_key, license_class, "
            " license_evidence_url, truth_condition, training_eligible, media_type, site, "
            " instrument, sample_rate_hz, band_low_hz, band_high_hz, band_partial, duration_s, "
            " batch_id, created_at) "
            "VALUES (:sha256, :source_id, :origin_url, :retrieved_at, :byte_count, :raw_key, "
            " :license_class, :license_evidence_url, :truth_condition, :training_eligible, "
            " :media_type, :site, :instrument, :sample_rate_hz, :band_low_hz, :band_high_hz, "
            " :band_partial, :duration_s, :batch_id, :created_at)",
            {**record, "batch_id": batch_id, "created_at": _now_iso()},
        )
        self._conn.commit()

    def corpus_object_exists(self, sha256: str) -> bool:
        """Return whether a corpus object with the given content hash is recorded."""
        row = self._conn.execute(
            "SELECT 1 FROM corpus_objects WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return row is not None

    def get_corpus_objects(self) -> list[dict[str, object]]:
        """Return every recorded corpus object as a dictionary, ordered by source and hash."""
        cursor = self._conn.execute("SELECT * FROM corpus_objects ORDER BY source_id, sha256")
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def insert_corpus_derivative(self, *, record: dict[str, object]) -> None:
        """Record a derived corpus object with its parent and the operation that produced it."""
        self._conn.execute(
            "INSERT OR IGNORE INTO corpus_derivatives "
            "(sha256, parent_sha256, derived_key, operation, params_json, byte_count, "
            " sample_rate_hz, band_low_hz, band_high_hz, created_at) "
            "VALUES (:sha256, :parent_sha256, :derived_key, :operation, :params_json, "
            " :byte_count, :sample_rate_hz, :band_low_hz, :band_high_hz, :created_at)",
            {**record, "created_at": _now_iso()},
        )
        self._conn.commit()

    def insert_vessel_presence(self, *, record: dict[str, object]) -> None:
        """Record a vessel's correlated presence in a recording; idempotent by presence id."""
        self._conn.execute(
            "INSERT OR IGNORE INTO corpus_vessel_presence "
            "(presence_id, recording_sha, site, vessel_id, mmsi, imo, name, first_seen_s, "
            " last_seen_s, closest_range_m, registry_grade, truth_tier, ais_source_sha, "
            " created_at) "
            "VALUES (:presence_id, :recording_sha, :site, :vessel_id, :mmsi, :imo, :name, "
            " :first_seen_s, :last_seen_s, :closest_range_m, :registry_grade, :truth_tier, "
            " :ais_source_sha, :created_at)",
            {**record, "created_at": _now_iso()},
        )
        self._conn.commit()

    def registry_vessel_counts_by_source(self) -> dict[str, int]:
        """Return the count of distinct registry-grade vessels correlated to each acoustic source.

        The count joins each presence to the recording it was correlated against and to that
        recording's source, so it feeds the audit's per-source registry-grade vessel column and the
        rule-of-three arithmetic that depends on target-side truth volume.
        """
        rows = self._conn.execute(
            "SELECT o.source_id, COUNT(DISTINCT p.vessel_id) "
            "FROM corpus_vessel_presence p "
            "JOIN corpus_objects o ON o.sha256 = p.recording_sha "
            "WHERE p.registry_grade = 1 "
            "GROUP BY o.source_id"
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def count_vessel_presence(self) -> int:
        """Return the total number of recorded vessel-presence rows."""
        row = self._conn.execute("SELECT COUNT(*) FROM corpus_vessel_presence").fetchone()
        return int(row[0])

    # -- runs -------------------------------------------------------------------------------

    def insert_run(
        self,
        *,
        run_id: str,
        manifest_hash: str,
        code_version: str,
        env_fingerprint: str,
        seed: int,
        thread_limit: int | None,
    ) -> None:
        """Insert a run's immutable identity row and its initial status event."""
        started = _now_iso()
        self._conn.execute(
            "INSERT INTO runs (run_id, manifest_hash, code_version, env_fingerprint, seed, "
            "thread_limit, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, manifest_hash, code_version, env_fingerprint, seed, thread_limit, started),
        )
        self.append_run_status(run_id, "started")
        self._conn.commit()

    def append_run_status(self, run_id: str, status: str, detail: str | None = None) -> None:
        """Append a status transition for a run; status is derived from the latest event."""
        self._conn.execute(
            "INSERT INTO run_status (run_id, status, detail, at) VALUES (?, ?, ?, ?)",
            (run_id, status, detail, _now_iso()),
        )
        self._conn.commit()

    def run_status(self, run_id: str) -> str | None:
        """Return the most recent status of a run, or None if the run is unknown."""
        row = self._conn.execute(
            "SELECT status FROM run_status WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        return None if row is None else str(row[0])

    def get_run(self, run_id: str) -> RunRow | None:
        """Return a run's identity row, or None if the run is unknown."""
        row = self._conn.execute(
            "SELECT run_id, manifest_hash, code_version, env_fingerprint, seed, thread_limit, "
            "started_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return RunRow(
            run_id=str(row[0]),
            manifest_hash=str(row[1]),
            code_version=str(row[2]),
            env_fingerprint=str(row[3]),
            seed=int(row[4]),
            thread_limit=None if row[5] is None else int(row[5]),
            started_at=str(row[6]),
        )

    # -- artifacts --------------------------------------------------------------------------

    def insert_artifact(
        self,
        *,
        artifact_hash: str,
        kind: str,
        producing_stage: str | None,
        stage_version: str | None,
        code_version: str,
        config_hash: str,
        env_fingerprint: str,
        seed: int | None,
        parents: tuple[str, ...] = (),
    ) -> None:
        """Record an artifact and its parents; idempotent because identity is content."""
        self._conn.execute(
            "INSERT OR IGNORE INTO artifacts (artifact_hash, kind, producing_stage, "
            "stage_version, code_version, config_hash, env_fingerprint, seed, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                artifact_hash,
                kind,
                producing_stage,
                stage_version,
                code_version,
                config_hash,
                env_fingerprint,
                seed,
                _now_iso(),
            ),
        )
        for parent in parents:
            self._conn.execute(
                "INSERT OR IGNORE INTO artifact_parents (artifact_hash, parent_hash) VALUES (?, ?)",
                (artifact_hash, parent),
            )
        self._conn.commit()

    def artifact_exists(self, artifact_hash: str) -> bool:
        """Return whether an artifact row exists for the given hash."""
        row = self._conn.execute(
            "SELECT 1 FROM artifacts WHERE artifact_hash = ?", (artifact_hash,)
        ).fetchone()
        return row is not None

    def list_run_artifacts(self, run_id: str) -> tuple[str, ...]:
        """Return the distinct artifact hashes bound to any execution of a run."""
        rows = self._conn.execute(
            "SELECT DISTINCT io.artifact_hash FROM stage_io io "
            "JOIN stage_executions ex ON ex.exec_id = io.exec_id "
            "WHERE ex.run_id = ? ORDER BY io.artifact_hash",
            (run_id,),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    # -- stage executions -------------------------------------------------------------------

    def insert_stage_execution(
        self,
        *,
        exec_id: str,
        run_id: str,
        node_id: str,
        stage: str,
        stage_version: str,
        computation_key: str,
        config_hash: str,
        reused: bool,
        started_at: str,
        finished_at: str,
        inputs: dict[str, str],
        outputs: dict[str, str],
    ) -> None:
        """Record a stage execution and its input and output artifact bindings."""
        self._conn.execute(
            "INSERT INTO stage_executions (exec_id, run_id, node_id, stage, stage_version, "
            "computation_key, config_hash, reused, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exec_id,
                run_id,
                node_id,
                stage,
                stage_version,
                computation_key,
                config_hash,
                int(reused),
                started_at,
                finished_at,
            ),
        )
        for name, artifact_hash in inputs.items():
            self._conn.execute(
                "INSERT INTO stage_io (exec_id, direction, name, artifact_hash) "
                "VALUES (?, 'input', ?, ?)",
                (exec_id, name, artifact_hash),
            )
        for name, artifact_hash in outputs.items():
            self._conn.execute(
                "INSERT INTO stage_io (exec_id, direction, name, artifact_hash) "
                "VALUES (?, 'output', ?, ?)",
                (exec_id, name, artifact_hash),
            )
        self._conn.commit()

    def get_stage_executions(self, run_id: str) -> tuple[StageExecutionRow, ...]:
        """Return a run's stage executions in insertion order."""
        rows = self._conn.execute(
            "SELECT exec_id, run_id, node_id, stage, stage_version, computation_key, "
            "config_hash, reused, started_at, finished_at FROM stage_executions "
            "WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ).fetchall()
        return tuple(
            StageExecutionRow(
                exec_id=str(row[0]),
                run_id=str(row[1]),
                node_id=str(row[2]),
                stage=str(row[3]),
                stage_version=str(row[4]),
                computation_key=str(row[5]),
                config_hash=str(row[6]),
                reused=bool(row[7]),
                started_at=str(row[8]),
                finished_at=str(row[9]),
            )
            for row in rows
        )

    def get_stage_io(self, exec_id: str, direction: str) -> tuple[StageIORow, ...]:
        """Return the input or output bindings of a stage execution, ordered by name."""
        rows = self._conn.execute(
            "SELECT exec_id, direction, name, artifact_hash FROM stage_io "
            "WHERE exec_id = ? AND direction = ? ORDER BY name",
            (exec_id, direction),
        ).fetchall()
        return tuple(
            StageIORow(
                exec_id=str(row[0]),
                direction=str(row[1]),
                name=str(row[2]),
                artifact_hash=str(row[3]),
            )
            for row in rows
        )

    # -- metrics ----------------------------------------------------------------------------

    def insert_metric(
        self,
        *,
        run_id: str,
        name: str,
        value: float | None,
        payload: dict[str, object] | None,
        config_hash: str | None,
        artifact_hash: str | None,
    ) -> None:
        """Record a scalar or structured metric produced by a run."""
        self._conn.execute(
            "INSERT INTO metrics (run_id, name, value, payload, config_hash, artifact_hash) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                name,
                value,
                None if payload is None else json.dumps(payload, sort_keys=True),
                config_hash,
                artifact_hash,
            ),
        )
        self._conn.commit()

    # -- content-addressed cache ------------------------------------------------------------

    def cache_get(self, computation_key: str) -> dict[str, str] | None:
        """Return the recorded outputs for a computation key, or None on a miss."""
        row = self._conn.execute(
            "SELECT outputs FROM computation_cache WHERE computation_key = ?",
            (computation_key,),
        ).fetchone()
        if row is None:
            return None
        loaded: dict[str, str] = json.loads(str(row[0]))
        return loaded

    def cache_put(self, computation_key: str, outputs: dict[str, str]) -> None:
        """Record the outputs a computation produced, keyed by its computation key."""
        self._conn.execute(
            "INSERT OR IGNORE INTO computation_cache (computation_key, outputs) VALUES (?, ?)",
            (computation_key, json.dumps(outputs, sort_keys=True)),
        )
        self._conn.commit()
