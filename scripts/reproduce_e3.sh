#!/usr/bin/env bash
# Regenerate the E3 detection bake-off that closes SD5, from the corpus and R2.
#
# Prerequisites:
#   - The corpus ledger (.fathom/ledger.db) with the acquired quiet, nominal, and busy site objects.
#   - R2 credentials sourced into the environment (FATHOM_R2_*); they live outside the repo and are
#     never committed. No owner-certified value is read or written.
#   - The committed closed SD2 hybrid surrogate at docs/e1_realdata_lineage.json, used as the
#     known-truth injection probe; E3 changes neither the closed SD4 front end nor the surrogate.
#
# The run gathers train and eval background windows from each site regime by a range read of the
# object head, splits them at the recording level so the learned detector never sees an eval
# recording (the leak rule), trains the learned detector on the train side, injects the closed hybrid
# across the signal-to-noise ladder onto the eval backgrounds through the closed front end, and scores
# each detector's sensitivity at a fixed false-alarm rate and its cross-site false-alarm stability
# under one threshold. The committed lineage (docs/e3_detection_lineage.json) carries every
# per-candidate, per-site, per-rung number, so the selection is auditable from committed state without
# the gitignored ledger or the R2 store; the seed is fixed for a byte-stable lineage.
#
# To audit the committed selection WITHOUT R2 or the environment, run scripts/verify_e3.py, which
# re-derives the winner and the flip clause from the committed per-candidate numbers using the
# standard library alone. This script is the deeper from-corpus regeneration.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m fathom.cli e3 --r2 --seed 20260805 --out docs/e3_detection_lineage.json
echo "done: docs/e3_detection_lineage.json regenerated; exit code is 0 on an SD5-close, 4 otherwise"
