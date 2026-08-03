# Fathom

Fathom is the evaluation harness and provenance-native pipeline scaffold for the Tuor build.
This repository is the first code of the Fathom restart, built on the ratified SD10 stack, and
it exists so that every later design decision can close on numbers that this harness produces.
Provenance and self-measurement are native from the first commit: every artifact is
content-addressed and recorded in an append-only lineage ledger, and every run can be
re-executed and checked hash-for-hash.

The harness accepts when it scores a deliberately trivial detector end to end and reproducibly.
There is no committed binary data; the acceptance corpus is generated at run time from a seed,
with known truth, so the whole path is self-contained and needs no network access to execute.

## Requirements

The build targets Python 3.12 and is managed with [uv](https://docs.astral.sh/uv/) against a
committed lockfile. A clean run needs only git, uv, and one-time network access to the package
registry to install the locked dependencies.

## Running the acceptance path

Install the locked environment and run the acceptance manifest end to end:

```bash
uv sync
uv run fathom run manifests/acceptance.json
```

The run prints a run identifier and writes a scored report with a lineage row for every
artifact. To reproduce a recorded run and verify that every output hash is identical, pass the
run identifier to the reproduction command:

```bash
uv run fathom repro <run-id>
```

A reproduction that reports any hash mismatch is a first-class failure, not a warning. The
reproducibility claim is bounded to the locked environment, as SD10 Section 6.6 states.

## Checks

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

Continuous integration runs lint, strict types, the unit tests, and the acceptance run with its
double-execution hash-equality check on every push.

## Corpus acquisition (WO-2)

The corpus subsystem acquires the unclassified background and clutter corpus into an S3-compatible
object store under a content-addressed `raw/`, `derived/`, and `manifests/` layout, records every
object with its license and truth condition, and generates the audit report that closes SD1.
Destination credentials are read from the environment only and never entered into the repository.

To point the tooling at the R2 bucket, provide the wiring as environment variables — for example in
a file outside the repository that you source before running:

```bash
export FATHOM_R2_ENDPOINT="https://<account>.r2.cloudflarestorage.com"
export FATHOM_R2_ACCESS_KEY_ID="<access-key-id>"
export FATHOM_R2_SECRET_ACCESS_KEY="<secret-access-key>"
export FATHOM_R2_BUCKET="fathom-corpus"
```

Each source family is reached its own way, and one `acquire` command covers all of them. Probe a
listable source first, then acquire, then generate the audit report:

```bash
# Family 2 — MBARI (anonymous S3): list a prefix, then acquire a bounded batch.
uv run fathom corpus-probe mbari_pacific_sound_2khz --limit 8
uv run fathom acquire mbari_pacific_sound_2khz --prefix "2015/07/" --limit 20 --r2

# Families 1, 2, 4 — NCEI (public Google bucket): list under a prefix, then acquire.
uv run fathom corpus-probe adeon_ncei --prefix "ADEON/" --limit 20
uv run fathom acquire adeon_ncei --prefix "ADEON/" --limit 20 --r2

# Family 1 — OOI (Apache archive): enumerate one named month directory, then acquire.
uv run fathom acquire ooi_slope_base_broadband --index-url "https://rawdata.oceanobservatories.org/files/<site>/<node>/<instrument>/2015/07/28/" --limit 24 --r2

# Family 3 — MarineCadastre AIS (bulk zips): acquire the days overlapping your recordings.
uv run fathom acquire marinecadastre_ais --date-start 2015-07-28 --date-end 2015-07-31 --r2

# Family 2 — ONC (credentialed): needs FATHOM_ONC_TOKEN; give a location, device category, and dates.
uv run fathom acquire onc_strait_of_georgia --onc-location SOG --onc-device-category HYDROPHONE --date-start 2019-06-01 --date-end 2019-06-02 --r2

# Any source: acquire explicit object URLs (for example a DCLDE set).
uv run fathom acquire dclde_baleen --url "https://.../set/file01.wav" --url "https://.../set/file02.wav" --r2

uv run fathom corpus-audit
```

Acquisition is idempotent because objects are addressed by content hash, per-family volume caps are
respected, and a research-only or unknown-license source is quarantined from training-designated
partitions mechanically. Deviations and access-friction findings are recorded in
[`docs/WO2_DEVIATIONS.md`](docs/WO2_DEVIATIONS.md).

## Governing documents

This repository implements two ratified decision memos, which live in the Engineering & Product
folder on the owner's machine and govern where this code and its comments differ from them:

- `Fathom_SD_memos/Fathom_SD10_stack_memo_v0.1.md` fixes the stack and its interior selections.
- `Fathom_SD_memos/Fathom_SD3_evaluation_protocol_memo_v0.1.md` fixes the evaluation protocol,
  the metric estimators, and the acceptance test.

The work order this repository delivers is `Fathom_work_orders/Fathom_WO1_scaffold_and_harness.md`,
and the surrounding sequence is in `Fathom_Tuor_build_definition_plan_v1.0.md`. Deviations from the
memos are recorded in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## Scope

This work order delivers the scaffold and harness only. There are no corpus downloads, no real
detectors beyond the trivial acceptance detector, no learned models, no tracker, and no
supervision surface; those arrive in later work orders. Only unclassified, synthetic data is
touched, and no owner-certified value appears anywhere in the code, its fixtures, or its comments.
