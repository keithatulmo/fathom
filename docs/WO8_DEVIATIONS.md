# Deviations and narrowings for WO-8

This document records every point where the WO-8 detection build and the E3 bake-off depart from,
narrow, or make a choice not fixed by the SD5 memo or the work order. No owner-certified value and no
absolute level appears anywhere: sensitivity is a detection rate on injected surrogate targets at a
fixed false-alarm rate, the false-alarm rate is a harness count on real background, and every level is
a signal-to-noise ratio relative to the measured background. The closed SD4 front end and the closed
SD2 hybrid surrogate are unchanged; E3 measures only the detector. The leak rule of SD3 binds: the
learned detector trains on train-side backgrounds only, disjoint from the eval backgrounds it is
scored on.

1. **The detectors are blind; only the scoring knows the injected truth.** A bake-off measures
   detectors, so none of the three is handed the injected line frequencies; the injected truth is used
   only to label a trial injected-or-background when computing detection and false-alarm rates. The
   constant-false-alarm-rate candidate is the strongest normalized cell in the band, the peak-pick on
   the already-flattened gram. The learned candidate reads a fixed comb-agnostic feature vector. This
   is the honest reading of a bake-off; a detector given the exact per-target comb would be an oracle,
   not a detector.

2. **Line and harmonic integration is temporal integration over a scanned shaft comb.** The SD5 memo
   describes the integration candidate as integrating each line along its track over a dwell and
   summing across the harmonic comb at the shaft-harmonic positions. It is implemented faithfully to
   that: every in-band bin is integrated over the dwell (its mean over the dwell frames, the line
   integration that lifts a persistent line while noise averages down), and the detector then scans a
   grid of shaft-rate hypotheses across the machinery range, sums the dwell-integrated value at the
   first several harmonics of each, and takes the best-fitting comb. It is not handed the target's
   shaft rate, so the scan keeps it honest. A blind top-line-sum variant was tried during development
   and rejected: it scored higher on clean synthetic noise but could not distinguish a genuine
   harmonic comb from unstructured real-ambient tonals, which is exactly the discrimination the memo's
   harmonic summation is for, so the harmonic-comb scan is the committed detector.

3. **The learned detector is a bounded, deterministic logistic regression, train-side only.** It is a
   logistic regression on nine fixed comb-agnostic features (peak strength, the tail of the normalized
   distribution, and blind temporal-and-harmonic integration summaries), trained by deterministic
   gradient descent from a zero initialisation with no randomness, so the fit is bit-level
   reproducible and carries no PyTorch dependency, keeping it inside the acceptance closure. It trains
   on train-side backgrounds with the closed hybrid injected as labeled targets, disjoint at the
   recording level from the eval backgrounds, which is the leak rule made concrete. Its weights and
   standardisation are written into the committed lineage so the trained detector regenerates from
   committed state.

4. **The trial structure, the single threshold, and the fixed false-alarm rate.** Each background is
   sliced into non-overlapping subwindows of the dwell length; each eval subwindow is one background
   trial, and the same subwindow with a hybrid injected across the ladder is the detection trial, so
   the sensitivity curve is one target seen at each signal-to-noise. For each detector one threshold is
   set across the pooled eval background at the target false-alarm quantile, five percent, and held
   across all sites; sensitivity is the detection rate above that threshold and false-alarm stability
   is whether each site's realized false-alarm rate sits within a factor of two and a half of the
   target. The operating point is six decibels and the four-hertz floor is the four-to-twelve-hertz
   band SD4 flagged; both are recorded in configuration. Six eval backgrounds per regime are gathered
   rather than three, so a site's false-alarm rate is measured over enough trials that it is not read
   as spuriously zero from a small sample; the learned detector's cross-site false-alarm stability
   turns on this measurement, so it is made robustly.

5. **The implementation-risk trade is an explicit margin.** The SD5 memo directs that the learned
   detector, carrying the highest implementation risk and the false-alarm-stability exposure, is
   selected over hand-built integration only when it beats it by a margin that outweighs those costs.
   That margin is encoded as a proof-scope threshold, a tenth in operating-point detection rate, and
   recorded in configuration and in the lineage: if the learned detector wins operating-point
   sensitivity while stable but does not beat the best hand-built stable detector by that margin,
   integration is selected and the learned detector's ceiling is the runner-up. The selection rule and
   this trade are re-derived by the standard-library verifier from the committed numbers, so the close
   is auditable exactly as the E1 and E2 verifiers are.

6. **Latency and implementation risk are recorded judgments.** Latency is the detection delay each
   candidate needs: the peak-pick needs a single frame, so its latency is the front end's window
   length; integration and the learned detector need the dwell, so theirs is the subwindow length.
   Implementation risk is a recorded judgment of maturity and effort, low for the peak-pick, moderate
   for integration, and high for the learned detector, not a measured quantity, as the memo specifies.
   Downstream compatibility is recorded as whether the detector emits a graded soft score, which all
   three do, with the score-versus-signal-to-noise correlation carried as a calibration indicator.

## The verdict and standing obligations

E3 closes SD5 on the **learned detector**, and the flip clause does not fire. The learned detector
maximizes operating-point sensitivity at ninety-five percent while holding cross-site false-alarm
stability, so it is selected with its implementation-risk cost named, which is the exact case the SD5
memo said would overturn its integration prior: the learned detector materially ahead on sensitivity
while holding false-alarm stability across sites. Its four-hertz-floor sensitivity is fifty-four
percent, moderate, as expected at the band's weakest point.

The measured result is that the two hand-built detectors, constant-false-alarm-rate peak-picking and
harmonic-comb integration, have essentially zero operating-point sensitivity on this problem, and this
is a genuine finding rather than a defect. The closed SD2 hybrid is a broadband-dominated quiet target
whose line-to-broadband ratio the WO-5 fit put well below unity, so its tonal lines are weak, and real
ambient carries its own narrowband structure that a peak or comb statistic picks up; the target is
therefore detectable by its broadband and distributional signature, which the learned detector reads,
rather than by its weak lines, which the peak and comb detectors depend on. Harmonic integration also
failed cross-site false-alarm stability at proof scope. This ties the detection result back to SD2: the
quiet class is broadband-dominated, so the detector that wins is the one that uses the broadband, not
the comb.

Two honest qualifications and two standing obligations bound the close. The learned detector's edge is
measured under the placeholder line-prominence rejector, so the selection is provisional until the SD6
rejector consumes the detector's soft scores and E3 re-confirms; the detector's graded soft scores are
a downstream obligation to the SD6 rejector and the SD9 tracker, not only a local product. The false-
alarm stability was made robust by measuring over six eval backgrounds per regime rather than three,
which removed a small-sample zero that had spuriously failed the learned detector's stability in a
first pass; the corrected measurement is what the committed lineage carries. The front end of SD4 and
the surrogate of SD2 are unchanged, so E3 measures only the detector, and the broadband and transient
channels stay scoped to the narrowband proof and are not built out here.
