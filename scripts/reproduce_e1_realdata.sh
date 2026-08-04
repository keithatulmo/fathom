#!/usr/bin/env bash
# Regenerate the E1 real-data result that closes SD2, from the corpus and R2.
#
# Prerequisites:
#   - The corpus ledger (.fathom/ledger.db) with the audio-backed quiet-tail cohort and the
#     vessel-level splits. If it is not present, run scripts/reproduce_audio_backing.sh first, which
#     rebuilds the audio-backed cohort from the corpus and R2, then re-run assign-splits.
#   - R2 credentials sourced into the environment (FATHOM_R2_*); they live outside the repo and are
#     never committed. No owner-certified value is read or written.
#
# The run fits the surrogate line statistics to train-side quiet-tail audio only, opens the held-out
# class solely at the comparison, sweeps the signal-to-noise ladder through the identical reference
# front end, detector, and reference rejector, and decides realism by whether the surrogate-to-class
# Wasserstein distance falls inside the null distribution derived from the held-out class's own
# halves. The committed lineage (docs/e1_realdata_lineage.json) and its object hashes let a third
# party rebuild and check the verdict without the gitignored ledger. The seed is fixed for a
# byte-stable lineage; the object hashes resolve against the R2 store.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m fathom.cli e1-realdata --r2 --seed 20260804 --out docs/e1_realdata_lineage.json
echo "done: docs/e1_realdata_lineage.json regenerated; exit code is 0 on a pass, 4 on a fail"
