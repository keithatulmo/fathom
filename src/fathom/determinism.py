"""Determinism policy: seeds, thread pinning, and environment fingerprinting.

SD10 Section 6.6 states the reproducibility claim honestly: seeds are fixed and recorded,
numerical-library threads are pinned in scoring runs, and the environment is locked, so that
identical inputs under the identical locked environment yield identical numbers. Cross-platform
bitwise identity is not claimed. This module is where that policy is made mechanical.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

# A 63-bit mask, used to fold a SHA-256 digest down to a seed integer. Sixty-three bits keeps the
# value inside SQLite's signed 64-bit INTEGER so seeds record losslessly, and NumPy accepts it.
_SEED_MASK = (1 << 63) - 1

# Recorded when the lockfile is absent so the environment fingerprint is never silently empty.
UNLOCKED_FINGERPRINT = "unlocked"

# Recorded when no git commit is resolvable (for example a run made before the first commit).
UNVERSIONED_CODE = "unversioned"


def derive_seed(base_seed: int, *labels: str) -> int:
    """Derive a stable child seed from a base seed and a sequence of string labels.

    Each pipeline node draws its randomness from a seed derived deterministically from the
    run's base seed and the node's identity, so that node seeds are reproducible and distinct
    without the caller threading a counter through the graph.
    """
    hasher = hashlib.sha256()
    hasher.update(base_seed.to_bytes(8, "big", signed=False))
    for label in labels:
        label_bytes = label.encode("utf-8")
        hasher.update(len(label_bytes).to_bytes(8, "big"))
        hasher.update(label_bytes)
    return int.from_bytes(hasher.digest()[:8], "big") & _SEED_MASK


def rng(seed: int) -> np.random.Generator:
    """Return a NumPy generator seeded deterministically with the given integer."""
    return np.random.default_rng(seed)


@contextmanager
def pinned_threads(limit: int = 1) -> Iterator[None]:
    """Pin numerical-library thread pools for the duration of a scoring or scored run.

    Reduction order under multi-threaded BLAS is not guaranteed, which can perturb the low
    bits of a floating-point result and break hash equality. Pinning to a single thread during
    scored runs keeps the arithmetic deterministic within the locked environment.
    """
    with threadpool_limits(limits=limit):
        yield


def environment_fingerprint(repo_root: Path) -> str:
    """Return the environment fingerprint as the SHA-256 of the committed ``uv.lock``.

    The lockfile hash is the fingerprint of the whole resolved dependency closure, so recording
    it against every run and artifact ties the numbers to the exact environment that produced
    them. When no lockfile is present the fingerprint is an explicit sentinel rather than empty.
    """
    lock = repo_root / "uv.lock"
    if not lock.is_file():
        return UNLOCKED_FINGERPRINT
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def code_version(repo_root: Path) -> str:
    """Return the code version as the current git commit hash, or a sentinel if unavailable.

    The code version enters the computation key that gates the content-addressed cache, so a
    change of code correctly forces re-execution. It does not enter an artifact's content hash,
    which depends only on the produced bytes.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return UNVERSIONED_CODE
    return result.stdout.strip() or UNVERSIONED_CODE
