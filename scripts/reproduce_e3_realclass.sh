#!/usr/bin/env bash
# Regenerate the WO-9 real-class detection test that settles the SD5 close, from committed state.
#
# Prerequisites:
#   - The committed E3 lineage docs/e3_detection_lineage.json, which carries the three detectors'
#     frozen thresholds and the learned detector's committed weights; nothing is retrained or retuned.
#   - The corpus ledger (.fathom/ledger.db) for the held-out vessels' signal-to-noise and site and for
#     the E3 training background shas used in the disjointness attestation.
#   - The held-out quiet cohort: the cached E1 segments (.fathom/e1_segments.npz), or R2 credentials
#     (FATHOM_R2_*) sourced into the environment to re-gather it with --r2. No owner-certified value is
#     read or written.
#
# The run applies the three E3 detectors, frozen exactly as committed, to the real held-out quiet
# vessel windows as positives and real background as negatives, through the closed SD4 front end, and
# records each detector's detection rate on the real vessels and its false-alarm rate on real
# background at the frozen E3 threshold, per site, against the vessels' signal-to-noise, and at the
# four-hertz floor, then names the branch the numbers select. The committed lineage
# (docs/e3_realclass_lineage.json) makes the branch auditable from committed state without the ledger
# or R2, and the run is deterministic (no injection and no randomness).
#
# To audit the selected branch WITHOUT R2 or the environment, run scripts/verify_e3_realclass.py,
# which re-derives the branch and the disjointness attestation from the committed per-detector numbers
# using the standard library alone. This script is the deeper from-corpus regeneration.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m fathom.cli e3-realclass --r2 --seed 20260805 --out docs/e3_realclass_lineage.json
echo "done: docs/e3_realclass_lineage.json regenerated; exit code is 0 on an SD5-close branch, 4 otherwise"
