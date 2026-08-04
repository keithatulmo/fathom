# Deviations and narrowings for WO-4

This document records every point where the WO-4 E1 real-data run departs from, narrows, or makes a
choice not fully fixed by the SD2 memo or the work order, per the reporting rule. None invokes an
owner-certified value or an absolute source level, none consults prior code, and the injection-leak
rule of SD3 Section 5.4 is enforced by construction: the surrogate is fit from train-side audio only,
and the held-out class is opened solely at the comparison.

1. **The structural frequencies stay first-principles; the fit refines amplitudes, roll-off, width,
   and wander.** SD2 Section 4 places the line frequencies from machinery kinematics and draws the
   amplitude, width, and wander from distributions fit to train-side vessels, and that is what is
   done: the real-data fit measures the amplitude spread, the amplitude-versus-frequency roll-off,
   the peak width, and the frequency wander from the train-side closest-approach spectra and sets the
   surrogate distributions from them, while the shaft, blade, electrical, and auxiliary frequencies
   remain kinematic. Blind shaft-rate recovery from a single quiet vessel is not reliable, so the
   structural placement is not inferred from audio; this matches SD2 Section 4 rather than departing
   from it, and is stated so the boundary of what the audio fit sets is explicit.

2. **The held-out class is processed as-is; the surrogate is swept.** SD2 Section 7 says to process
   the held-out class through the identical front end, detector, and rejector. The held-out are real
   recordings, so they are processed as-is: their response is a fixed distribution, measured once,
   independent of any injected level. The surrogate, being synthetic, is injected onto real
   background across the ladder. A first attempt instead re-levelled each held-out segment as a
   signal onto fresh background at each rung; that was recorded here as a deviation before the run,
   and the run confirmed the concern, because re-injecting a real recording over-amplifies its own
   broadband ambient and drove the measured held-out detectability to a floor that made every high
   signal-to-noise rung fail. Processing the held-out as-is is the faithful reading of SD2 Section 7
   and the method the reported verdict uses; the change corrects the comparison, and it is not a
   change to the tolerance, which remains the held-out null. The operating band is then the set of
   rungs where the surrogate is as detectable as the real class, its detectability interval
   overlapping the held-out's, and the realism verdict is taken across that band.

3. **The tail-sensitive distance is the Wasserstein distance.** WO-4 Section 4 allows Anderson-Darling
   or Wasserstein or an operating-quantile evaluation; the Wasserstein distance is chosen because it
   integrates the difference of the empirical distribution functions across the whole support, so it
   registers the tail-mass differences the operating point depends on, and because it is stable at the
   small held-out sample size where the Anderson-Darling statistic is noisy. The Kolmogorov-Smirnov
   value is reported alongside for continuity, but the verdict rests on the Wasserstein test.

4. **The null and the test are computed at matched sample sizes.** The work order specifies the null
   as the distance between random halves of the held-out class. To keep the surrogate-to-class test
   comparable to that null rather than biased by sample size, both are computed between a
   ceil-half-size sample and a floor-half-size sample, differing only in whether the first half is
   drawn from the held-out class (null) or the surrogate pool (test). Without this the larger
   surrogate pool would shrink its distance artificially and bias the verdict toward a pass; this
   refinement removes that bias and is a strengthening of the work order's method, not a relaxation.

5. **Segment and background selection.** Each vessel is represented by the closest-approach segment
   of its best-signal-to-noise audio-backed passage, a window centred on the recorded closest-approach
   time. The background for injection is a window in the same recording taken far from that time, where
   the vessel is distant, which is the same reference-window logic the audibility measurement used.
   These are proof-scope choices recorded in configuration.

6. **The rejector is the placeholder line-prominence statistic.** As in WO-3 the reference rejector is
   the in-band line prominence, standing in until SD6 selects the real rejector. The realism verdict is
   therefore provisional under the placeholder and is re-confirmed under the real SD6 rejector when
   that closes, which is a standing note carried forward rather than a blocker now.

7. **The line-statistic extraction is proof-scope.** The amplitude, roll-off, width, and wander
   statistics are measured by peak-picking the Welch spectrum of each train-side closest-approach
   segment. On real quiet vessels this is noisy: some segments yield a clean set of narrow lines,
   others a single broad feature or none, so the fitted width and roll-off ranges are wide. This is
   recorded honestly as a limit of the audio fit; it does not touch the held-out class's rejector
   response, which is measured directly on the recordings and is what the realism divergence rests
   on, and the fitted lines are if anything broad, which lowers rather than inflates the surrogate's
   prominence.

## The verdict and what it means

E1 returns **fail** on this measurement, so SD2 does not close, and the SD2 Section 8 flip condition
of Section 5 governs the next step. The tolerance was not moved and the operating band was not
re-drawn to force a pass.

The finding is specific and is a real property of the narrowband Option A recipe rather than a
methodological artifact, and it survives the two corrections made during the run (processing the
held-out as-is, and fitting from train-side audio). The held-out class's rejector response, measured
directly on the real recordings, is a line prominence of about three and a half to twenty-five with a
median near seven, and the real class detects at only about fifteen percent under the reference
detector, because real quiet vessels carry broadband energy that fills in between their lines. The
pure-narrowband surrogate, at the signal-to-noise ratio where it is as detectable as the real class,
around seven to nine decibels, is more line-prominent than the real class: it matches inside the
held-out null at zero to six decibels and diverges above nine, with the operating point at the
boundary. In plain terms the surrogate is spectrally too clean, because it has lines and no broadband
where a real quiet vessel has both.

Two honest qualifications bound the strength of the result. It is provisional under the placeholder
line-prominence rejector and the reference detector, and must be re-confirmed under the real SD6
rejector and SD5 detector when those close, which the payload records. And the gap is a
broadband-versus-narrowband difference, so the Section 8 flip's recalibration of line-amplitude and
roll-off statistics against other train-side vessels may not close it on its own; adding broadband
structure moves the recipe toward the hybrid, which is an Engineering and Product decision on the
returned result rather than a Code-surface choice.
