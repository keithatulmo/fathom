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

6. **AIS correlation is wired to recordings.** The correlation into registry-grade, tier-one
   vessel-presence truth is written fresh and unit-tested, including the MMSI-to-IMO/name
   reconciliation SD3 Section 5.1 requires. It is now bound to recordings by the `fathom
   corpus-truth` command, which reads the acquired AIS zips, filters to a site's bounding box and
   each recording's time window, correlates, and records vessel presence in an append-only ledger
   table. The distinct registry-grade vessel counts feed the audit's per-source column. AIS
   timestamps are pinned to UTC explicitly, since MarineCadastre writes them without an offset.
   Site coordinates are carried in a small in-code registry keyed by site.

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

## Addendum: audio-backed quiet-tail cohort (response to the corpus-growth review, 2026-08-04)

The corpus-growth review (Fathom_reviews/Fathom_corpus_growth_review_v0.1) found the review's
intended failure mode: the quiet-tail cohort had grown in AIS correlations against un-downloaded
windows, so a vessel counted as quiet-tail from its kinematics alone with no audio behind it. Of the
87 kinematically flagged quiet-tail vessels, exactly one had decodable audio. The factual question
the ledger could not answer was settled directly against the R2 bucket: it held exactly the 3,006
ledgered non-SanctSound objects (MBARI, ADEON, ONC, AIS) and no SanctSound audio, so the near-shore
audio had been enumerated but never fetched. The window pre-correlation was therefore a targeting
list, and only the bounded fetch remained.

10. **The quiet-tail cohort is now audio-backed and SNR-verified.** A `fetch-quiet-tail` driver
    downloads the audio for the flagged passages, greedily by distinct-vessel coverage or, with
    `--prioritize closest`, by closest approach first, since audibility falls off with range. Each
    fetched object records its sample rate and decoded duration, which is what CA6 needs. A new
    `snr.py` measures in-band (4-150 Hz) signal-to-noise at each passage's closest approach: it
    decodes a short window at closest approach and a spread of reference windows for the ambient
    floor (a low percentile, the quiet baseline), never the whole multi-hour file, and reports the
    ratio in decibels. `corpus-truth --audio-only` correlates the fetched objects and writes
    audio-backed vessel presence carrying the measured SNR and an `audio_quiet_tail` flag, added to
    the presence table by migration. The adequacy scorer now counts a quiet-tail vessel toward CA2
    and CA3 only when it is audio-backed and audible, gates CA6 on a captured (not asserted) sample
    rate, and seeds the vessel-level split from the audio-backed cohort.

11. **The measurement is physically consistent, and CA3 is held open by one vessel at a strict
    6 dB.** Across the fetched passages the mean in-band SNR is about 8.5 dB within 1.5 km and about
    3.7 dB beyond 3 km, so the measure tracks range as a real signal should. At a 6 dB audibility
    threshold the audio-backed quiet-tail cohort is 29 distinct vessels, up from one; CA2's held-out
    and train-side minimums are met with a disjoint split (held-out 13, train-side 14). CA3's count
    of 30 is not met: the cohort is 29, and the thirtieth vessel sits at 5.82 dB, 0.18 dB below the
    cut. Three closest-passage fetch passes (about 70 files, 40 GB of audio) moved the cohort from
    25 to 28 to 29 and then plateaued, because the near-shore sites with abundant quiet-tail traffic
    are busy: Florida Keys close passages within a kilometre added no audible vessels, their in-band
    level sitting at the Straits' ambient floor, while the quiet sites carry few quiet-tail vessels.
    So reaching 30 at a strict 6 dB is bounded by site ambient, not by fetching. The full
    distribution is recorded (41 vessels at 3 dB, 29 at 6 dB, 22 at 8 dB). The owner elected to hold
    the round 6 dB bar and record CA3 as narrowly unmet rather than ratify a lower threshold in the
    5.7 to 6.2 dB cluster the cohort occupies; the SNR-threshold decision is carried into the
    revised adequacy note. No owner-certified value is introduced anywhere in this work.

## Addendum: auditable-by-construction audio backing (response to the second review, 2026-08-04)

The second corpus review recomputed the audit from a ledger that predated the audio-backing runs
and, seeing no SanctSound audio and no `audio_quiet_tail` column, read the cohort as one. The
audio-backing was in fact fully persisted in the working ledger, verified with a raw query that
bypasses the schema migration: the columns exist, seventy SanctSound audio objects and one hundred
ten in-band SNR values are stored, and twenty-nine distinct vessels are audio-backed and join to a
stored object. But that ledger is gitignored local state and the audio lives in R2, neither visible
to the review, so from the outside the count was unauditable. The review's structural lesson holds
regardless of which ledger it read, and it is adopted.

12. **The scorer counts only persisted, object-joined state.** It no longer trusts a stored flag: a
    quiet-tail vessel counts toward CA2 and CA3 only when its recording joins a stored audio object
    that carries a recorded sample rate and duration and its persisted in-band SNR meets the bar,
    which is now a scoring parameter (`corpus-adequacy --snr-db`). The split assignment uses the
    same derivation (`audio_backed_vessels`), so the split and the audit agree and neither can
    credit a vessel whose audio was never fetched. Made to read only persisted state, the count
    regenerates from the ledger by construction.

13. **CA6 rests on measured audio, and the cohort is exported as committed lineage.** A
    `capture-audio` command reads sample rate, duration, and channels from each stored audio header
    that lacked a rate at acquisition (ADEON at 16 kHz, ONC at 64 kHz) and persists them in an
    append-only side table, so CA6 confirms the working band from the audio actually present rather
    than from window metadata for audio that is absent; every audio source now has a captured rate.
    An `audio-lineage` command writes the cohort to `docs/audio_backing_lineage.json`, tying each
    vessel to the object sha that backs it, its rate and duration, its closest-approach range, and
    its measured SNR, so the count is auditable from the committed repository without the live
    ledger or R2. `scripts/reproduce_audio_backing.sh` documents the fetch-measure-persist sequence
    so the audio-backed ledger regenerates from the base corpus and R2. The audited count is
    twenty-nine at the 6 dB bar, unchanged; CA2 passes on evidence, CA3 remains one vessel short,
    and the count-versus-bar question is the owner's, as the review notes.
