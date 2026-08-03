# Deviations and narrowings for WO-2

This document records every point where the WO-2 corpus-acquisition implementation departs from,
narrows, or makes a choice not fully fixed by the work order, per the reporting rule. Acquisition
friction is itself an SD1 criterion being measured, so sources that are deferred, quarantined, or
unreachable are recorded here rather than silently dropped.

## Execution posture

1. **The agent builds the tooling and proves the path; the owner runs the bulk.** Per the owner's
   ruling, this work delivers the acquisition subsystem and a bounded proof pull rather than the
   full ~400 GB first wave. The subsystem is complete and offline-tested; the bulk transfer is run
   by the owner in their own credentialed session with the same commands.

2. **Credentials are read from the environment only.** The R2 wiring is taken from
   `FATHOM_R2_ENDPOINT`, `FATHOM_R2_ACCESS_KEY_ID`, `FATHOM_R2_SECRET_ACCESS_KEY`, and
   `FATHOM_R2_BUCKET`. No credential is written to the repository or its history, satisfying AC6.

## Implementation choices not fixed by the work order

3. **Derivative lineage.** WO-2 Section 4 asks that working derivatives be produced through the
   WO-1 pipeline so each carries ledger ancestry. For bulk acquisition this build records
   derivatives in a dedicated append-only `corpus_derivatives` table that names each derivative's
   parent raw object, the operation, and the recorded anti-alias parameters, rather than routing
   every object through the manifest DAG runner. This preserves full parent-to-child lineage while
   scaling to bulk; the DAG runner remains available for evaluation pipelines.

4. **Verification depth.** Every stored object is verified by comparing its stored size to the
   size fetched. Objects at or below 64 MiB are additionally re-read and re-hashed for end-to-end
   checksum verification; larger objects are not re-downloaded, because a full re-hash would double
   the bandwidth, and their identity is already fixed by the content hash computed while streaming.

5. **Manifest format and idempotency.** Manifests are JSON, one per batch, and are authoritative in
   manifest-first mode; ledger rows are written inline when a ledger is supplied and are idempotent
   by content hash. Re-running an acquisition stores nothing new, satisfying AC4.

## Scaffold-scope narrowings that close with real acquisition

6. **AIS correlation is implemented and tested but not yet bound to specific recordings.** The
   correlation into registry-grade, tier-one vessel-presence truth is written fresh and unit-tested,
   including the MMSI-to-IMO/name reconciliation SD3 Section 5.1 requires. Populating per-recording
   truth rows requires the overlap windows between acoustic recordings and AIS, which are defined
   during real acquisition; until then the audit reports the vessel-truth column as not yet
   quantified rather than asserting a count.

7. **Decimation reads WAV and FLAC.** The decimation derivative decodes WAV and FLAC via libsndfile.
   Sources delivered in other container formats are recorded honestly and decimated once a decoder
   is added; no such source is in the first-wave proof path.

## Access-friction findings (an SD1 criterion)

8. **Ocean Networks Canada is now unblocked.** ONC was initially deferred because it requires a free
   account and an API token. The owner has since provisioned a token, so ONC is wired through its
   Oceans 3.0 archive-file API and the Strait of Georgia busy site is available. The token is read
   from `FATHOM_ONC_TOKEN` in the environment and appears only in the transient download URL; the
   provenance URL recorded in the manifest and ledger is token-free. The token was validated live
   against the ONC API (read-only) before this was recorded.

9. **DeepShip and ShipsEar are quarantined and not acquired.** DeepShip carries no license and is
   email-to-author only; ShipsEar is research-only. Both are quarantined from training-designated
   partitions mechanically, and neither is required for the wave because evaluation target truth
   comes from AIS-correlated real vessels. Whether either is usable as evaluation reference is an
   owner ruling once written terms exist.

## Open questions returned rather than resolved silently

- The proof pull uses one MBARI daily 2 kHz file (~518 MB), the smallest content-addressable unit
  that bucket offers; there is no smaller bounded object there. If a lighter proof is preferred, a
  single SanctSound or DCLDE annotation object would be smaller, and the owner can say which.
- The R2 endpoint's exact form (account-scoped Cloudflare endpoint versus a custom domain) is taken
  from `FATHOM_R2_ENDPOINT` as provided; the tooling does not assume a particular endpoint shape.
