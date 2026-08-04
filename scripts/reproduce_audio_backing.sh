#!/usr/bin/env bash
# Reproduce the audio-backed quiet-tail cohort from persisted state.
#
# Prerequisites:
#   - The base corpus ledger (.fathom/ledger.db) with corpus objects, registered SanctSound windows,
#     and the kinematic quiet-tail correlation already present (the state the growth work produced).
#   - R2 credentials sourced into the environment (FATHOM_R2_* and, if re-acquiring, FATHOM_ONC_TOKEN);
#     these live outside the repo and are never committed.
#
# This script performs only the audio-backing step the second corpus review directed: it fetches the
# audio for the kinematically flagged quiet-tail passages, measures each one's in-band SNR at closest
# approach, and persists the audio objects (with sample rate and duration), the SNR values, and the
# resulting flags into the ledger. Every count downstream then regenerates from that persisted state.
set -euo pipefail
cd "$(dirname "$0")/.."

SITES=(
  "florida_keys_fk01 sanctsound_fk01"
  "stellwagen_sb01 sanctsound_sb01"
  "monterey_mb01 sanctsound_mb01"
  "stellwagen_sb03 sanctsound_sb03"
  "grays_reef_gr01 sanctsound_gr01"
  "channel_islands_ci01 sanctsound_ci01"
)
RUN=corr-grow-20260803T224340Z   # append audio-backed presences into the existing correlation run

# 1. Fetch the audio: first maximise distinct-vessel coverage, then top up by closest passage, since
#    audibility falls off with range. --exclude-audible so a top-up pursues vessels not yet audible.
uv run python -m fathom.cli fetch-quiet-tail --target-vessels 60 --max-files 80 --r2
uv run python -m fathom.cli fetch-quiet-tail --prioritize closest --exclude-audible \
  --target-vessels 60 --max-files 80 --r2

# 2. Measure in-band SNR for the fetched passages and persist audio-backed presences.
for spec in "${SITES[@]}"; do
  set -- $spec
  uv run python -m fathom.cli corpus-truth --site "$1" --source "$2" \
    --ais-source marinecadastre_ais --corr-run "$RUN" --audio-only --r2
done

# 3. Capture sample rate/duration from the ADEON and ONC audio headers so CA6 is measured, not
#    asserted, on the audio that is actually present.
uv run python -m fathom.cli capture-audio --r2

# 4. Assign vessel-level splits seeded from the audio-backed cohort, then score and export lineage.
uv run python -m fathom.cli assign-splits
uv run python -m fathom.cli corpus-adequacy
uv run python -m fathom.cli audio-lineage
echo "done: audio-backed cohort, adequacy scorecard, and docs/audio_backing_lineage.json regenerated"
