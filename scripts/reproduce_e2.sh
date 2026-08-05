#!/usr/bin/env bash
# Regenerate the E2 front-end sweep result that closes SD4, from the corpus and R2.
#
# Prerequisites:
#   - The corpus ledger (.fathom/ledger.db) with the acquired quiet, nominal, and busy site objects.
#   - R2 credentials sourced into the environment (FATHOM_R2_*); they live outside the repo and are
#     never committed. No owner-certified value is read or written.
#   - The committed closed SD2 hybrid surrogate at docs/e1_realdata_lineage.json, used as the
#     known-truth injection probe; E2 re-fits nothing and reads no train or held-out split.
#
# The run gathers a real background window from each site regime by a range read of the object head,
# injects the closed hybrid surrogate at the operating-point signal-to-noise as a known-truth comb,
# and sweeps the linear split-window front end, scoring tonal separability as recovered comb
# prominence and background flatness as the cross-site false-alarm invariance of one threshold. The
# committed lineage (docs/e2_front_end_lineage.json) carries every per-configuration score, so the
# selection is auditable from committed state without the gitignored ledger or the R2 store; the seed
# is fixed for a byte-stable lineage.
#
# To audit the committed selection WITHOUT R2 or the environment, run scripts/verify_e2.py, which
# re-derives the selection and the flip clause from the committed per-configuration numbers using the
# standard library alone. This script is the deeper from-corpus regeneration.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m fathom.cli e2 --r2 --seed 20260804 --out docs/e2_front_end_lineage.json
echo "done: docs/e2_front_end_lineage.json regenerated; exit code is 0 on an SD4-close, 4 on a flip"
