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
