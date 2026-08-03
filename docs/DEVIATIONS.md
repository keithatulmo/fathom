# Deviations and narrowings for WO-1

This document records every point where the WO-1 implementation departs from, narrows, or makes a
choice not fully specified by the SD10 stack memo or the SD3 evaluation-protocol memo, however
small, per the work order's reporting rule. Where a memo governs and this build narrows it, the
narrowing is scoped to the scaffold and names what closes it. Nothing here changes a ratified
decision; each item is an implementation choice or a scaffold-scope limitation.

## Implementation choices not fixed by the memos

1. **Array artifact encoding.** SD10 Section 6.3 fixes Zarr version 3 as the store for derived
   arrays and requires content addressing at artifact granularity. This build stores each array as
   a Zarr version 3 array with compression disabled (the identity `bytes` codec) and reduces the
   Zarr store's key-to-chunk mapping to one deterministic blob, which is then hashed with SHA-256.
   Disabling compression removes codec state as a source of byte variation, so the double-execution
   hash-equality check is exact. The Zarr selection itself is unchanged, and a compressor can be
   introduced later behind the same serialization seam without altering the artifact model.

2. **Manifest format.** SD10 Section 6.2 fixes a manifest-driven batch DAG but does not fix the
   manifest's serialization. Manifests are JSON, parsed with the standard library and validated
   with pydantic v2, which keeps the closure minimal and the manifest hash canonical.

3. **Repository layout.** The package is under a `src/` layout (`src/fathom`), which is the
   conventional uv layout and keeps tests running against the installed package rather than the
   working tree. The work order's phrase "package layout under `fathom/`" is read as naming the
   package, which is `fathom`.

4. **Seed width.** Derived node seeds are folded to 63 bits so they record losslessly in SQLite's
   signed 64-bit `INTEGER` column, which NumPy's generator accepts without loss.

5. **Thread pinning mechanism.** SD10 Section 6.6 requires numerical-library threads to be pinned
   in scored runs. This is implemented at run time with `threadpoolctl` around the run and
   reproduction paths, and the pinned limit is recorded in the run row, rather than being set
   through environment variables before interpreter start.

## Scaffold-scope narrowings that close with later work

6. **Temporal-guard verification.** SD3 Section 5.6 requires the audit to verify temporal guards.
   The synthetic acceptance corpus carries no acquisition timestamps, so the audit verifies the
   guard the registry declares against the required guard and records both, rather than checking
   real inter-segment intervals. This narrowing closes when the real corpus with timestamps lands
   in a later work order, at which point the audit checks actual boundaries.

7. **Parquet metric export.** SD10 Section 6.3 notes that metrics additionally export to Parquet
   for analysis. WO-1 records metrics in the SQLite ledger and in the JSON report, which satisfies
   this work order's deliverables; the Parquet export is deferred as an analysis convenience and is
   not required by any WO-1 acceptance criterion.

8. **Per-site reject-versus-miss reporting.** SD3 Section 5.5 reports the reject-versus-miss curve
   per site, and the estimator carries a site field. The acceptance report scores the test split as
   a single evaluation population, because the site-holdout axis for shift measurement belongs to
   experiment E6 and is out of WO-1 scope. Per-site partitioning composes on top of the same
   estimator when the corpus and the site-holdout axis arrive.

9. **Custody metrics are exercised by unit tests, not by the acceptance run.** SD3 Section 5.7
   calls for custody placeholders in the acceptance report, because the trivial detector produces
   no tracks. The custody estimators (label consistency, swaps, losses, and time-to-reacquire) are
   implemented fresh and covered by hand-computed unit tests; the acceptance report carries explicit
   placeholders. The estimators are exercised end to end when the tracker arrives.

## Statistical-power note carried honestly

10. **Small exchangeable-unit counts in the acceptance corpus.** The seeded synthetic corpus has a
    handful of vessels per split, so the near-zero-miss bound is illustrative rather than tight and
    the cluster-bootstrap intervals are wide. This is the honest consequence SD3 Sections 5.5 and 7
    anticipate: the harness reports the exact one-sided bound and the vessel-level interval, and the
    proof's injected-target count is sized against that arithmetic by the owner at Step 15. Nothing
    in the harness hides the looseness; it surfaces it.

## Open questions returned rather than resolved silently

- SD3 Section 5.2 sizes the calibration split toward the order of five hundred exchangeable scoring
  events per calibrated output. The acceptance corpus does not reach that scale, and the figure is
  carried as a provisional default in configuration and in the registry rather than being exercised.
  This is expected for a scaffold, and it is flagged so the sizing is validated when a real corpus
  and the SD8 calibration method arrive, not assumed satisfied here.
