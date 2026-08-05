# Deviations and narrowings for WO-9

This document records every point where the WO-9 real-class detection test departs from, narrows, or
makes a choice not fixed by the SD5 disposition or the work order. No owner-certified value and no
absolute level appears anywhere: every quantity is a detection rate or a false-alarm rate on data
against signal-to-noise relative to the measured background. Nothing is retrained and nothing is
retuned: the learned detector is applied with its E3-committed weights and standardisation restored
verbatim, and every detector keeps the single threshold E3 fixed at the five-percent false-alarm rate.
The closed SD4 front end and the closed SD2 surrogate are unchanged.

1. **The positives are the E1 held-out cohort, processed as the E3 trials were.** The real
   quiet-vessel positives are the thirteen held-out closest-approach windows E1 used, and the
   negatives are the same real background windows E1 used, both from the committed E1 segment cache.
   Each window is sliced into the E3 dwell subwindows of eight seconds and scored through the closed
   front end, so a detector sees the real class on exactly the trial structure and threshold it saw the
   surrogate on; the detection rate on a vessel is the fraction of its subwindows that clear the frozen
   threshold, and the false-alarm rate is the fraction of pooled background subwindows that clear it.
   Reusing the E1 cohort is the point of the test, since it is the same real class the surrogate's
   realism was measured against at E1.

2. **The vessels' signal-to-noise and site come from the ledger; the operating-point subset is the
   near-six-decibel vessels.** Each held-out vessel's measured in-band signal-to-noise and its site are
   read from the audio-backed presences and corpus objects in the ledger, and the operating-point
   detection rate is the mean over the vessels whose signal-to-noise sits within two decibels of the
   six-decibel operating point, so the real-class result can be read at the operating point and against
   each vessel's own level. The full detection-versus-signal-to-noise points are recorded per detector
   so the curve is legible, not only the operating-point summary.

3. **The success floor and the branch order.** A detector "detects" the real class when its detection
   rate reaches a success floor of one half, a majority of the vessel's windows, and exceeds twice its
   own real-background false-alarm rate, so a detector that merely fires at its false-alarm rate is not
   counted as detecting. The branches are ordered so a surrogate defect surfaces before a close: a line
   success is checked first, then a learned confirmation, then a learned artifact. The floor is a
   proof-scope threshold recorded in the lineage, and the raw per-detector rates are reported so the
   Engineering and Product surface can judge the call independently of the floor.

4. **The leak rule is attested by disjoint recording shas, re-checkable from committed state.** The
   held-out vessel recording shas and the E3 training background shas are both written into the
   lineage, the latter reconstructed by the same deterministic smallest-objects-per-regime order the E3
   gather used, and their disjointness is attested and re-derived by the standard-library verifier, so
   the claim that no detector was trained on a test recording is auditable without the ledger.

5. **The run is deterministic and injects nothing.** The test scores real recordings through frozen
   detectors, so there is no injection and no randomness; the seed is a run tag only, and the lineage
   is byte-stable on re-run from the cached cohort.

## The verdict and what reopens

The numbers select the **learned-artifact branch**, and SD5 does not close. The learned detector, which
won E3 at ninety-five percent on the surrogate, detects the real held-out quiet vessels at one percent,
essentially at its own six-percent false-alarm floor, so its E3 sensitivity was an artifact of the
surrogate's broadband lift, the fitted boundary-member axis no experiment had validated against the
real class. The disposition's second branch is the one the numbers took: the learned win was a
surrogate artifact, so SD2's broadband balance and E3 reopen.

The line detectors carry a corroborating secondary finding, recorded though not the selected branch.
They detect the real vessels at twenty-four and nineteen percent, below the success floor so not a line
success, but well above the zero percent they scored on the surrogate. That the classical line
detectors do better on real quiet vessels than on the closed hybrid is direct evidence that the
surrogate's lines are too weak relative to its broadband, which is the same SD2 broadband-balance defect
the primary branch points to, approached from the other side. Both the learned collapse and the line
improvement say the surrogate's broadband is too strong and its lines too weak on the detection axis.

The two owner-level findings the SD5 disposition escalated are now on this evidence rather than
pending. The measured detection signature of the quiet target on the surrogate was broadband, but that
signature does not carry to the real class, so the architectural re-weighting question is not a simple
move from lines to broadband; it is that neither the surrogate's broadband nor its lines, as closed,
detects the real quiet vessel well through the closed front end at the operating point, and the highest
real-class rate any detector reaches is the peak-picker's twenty-four percent. The SD4-SD5 coherence
question stands: the split-window normalizer removes the broadband the E3 winner keyed on, and the
four-hertz-floor detection is near zero on the real class as it was on the surrogate. Both go to the
owner with the real-class numbers behind them. The standing obligations carry unchanged: the front-end
and rejector re-confirmation owed at SD6 is the same real-class re-scoring, and this test is the
detection part of it delivered early.
