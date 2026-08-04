# Deviations and narrowings for WO-5

This document records every point where the WO-5 hybrid surrogate and the E1 re-run depart from,
narrow, or make a choice not fully fixed by the SD2 memo or the work order, per the reporting rule.
None invokes an owner-certified value or an absolute source level, none consults prior code, and the
injection-leak rule of SD3 Section 5.4 is enforced by construction and now extends to the broadband
fit: every statistic the surrogate is fit to, line and broadband alike, is read from train-side
vessels only, and the held-out class is opened solely at the comparison. The WO-4 deviations carry
forward unchanged except where noted here; this file records only what WO-5 adds.

1. **The broadband continuum recipe, the one open parameterization WO-5 Section 3 flags.** The
   between-line continuum is specified by two fitted quantities and one generation rule, chosen
   deliberately and recorded here so the recipe is auditable, with its correctness left to the re-run
   E1 against the held-out null rather than asserted in advance. The continuum spectral shape is a
   power law: for each train vessel the detected lines are masked out and the slope of the log power
   spectral density on log frequency is fit to the remaining between-line floor, clamped to the
   physically red band of minus three to zero, and the surrogate draws an exponent from the
   train-cohort spread. The continuum level is set by the line-to-broadband power ratio: for each
   train vessel the summed line excess over the local continuum divided by the summed continuum floor
   across the band, in decibels, and the surrogate draws a ratio from the train-cohort spread. The
   continuum is generated as band-limited coloured noise shaped to the drawn exponent, scaled so the
   line-to-continuum in-band power equals the drawn ratio, and added to the line signal before the
   injection levels the whole hybrid to the requested signal-to-noise ratio. Because the injection
   scales lines and continuum together, the ratio is preserved through injection, so the surrogate's
   line prominence is bounded by its own between-line energy rather than climbing without bound as
   the pure-narrowband surrogate's did. This is a recipe choice among several defensible ones; it is
   the between-line power-law-plus-ratio reading of SD2 Section 4's narrowband-spine-plus-broadband
   description, and the re-run validates it.

2. **The line-to-broadband ratio is drawn independent of the quiet-target flag.** The ratio is drawn
   from the single train-cohort distribution for every target, quiet or not, rather than from a
   quiet-conditioned sub-distribution. The train cohort already spans the quiet, broadband-dominated
   vessels, so the low-ratio tail the quiet case needs is represented, and a joint fit of the ratio
   against the quiet flag is a refinement the proof-scope cohort does not warrant. The consequence is
   that a quiet-mode draw takes a ratio spanning the whole range rather than a systematically lower
   one; this is recorded as a deliberate simplification, not a hidden default.

3. **The continuum is generated as coloured noise, not resampled from real audio.** The between-line
   energy is synthesised as shaped random noise at the fitted exponent, not carved from a real
   recording, so no held-out or train audio is replayed into the surrogate and no absolute level
   enters. This keeps the surrogate fully synthetic and regenerable from its seed, and it keeps the
   leak rule mechanical: only the two dimensionless statistics cross from the train side, never audio.

4. **The hybrid is applied uniformly, in the injection stage and the E1 real-data path.** The
   broadband continuum is added in the shared injection orchestrator and in the E1 real-data synth
   loop, so the surrogate is hybrid everywhere it is built, not only in the run that closes SD2. The
   acceptance path does not exercise the injector, so its double-execution hash equality is
   unaffected; the injection record now carries a broadband parameter block alongside the machinery
   block, and it carries no absolute-level field.

5. **The operating-band definition and the tolerance are carried forward from WO-4 unchanged.** The
   operating band is still the set of rungs whose surrogate detectability interval overlaps the
   held-out's, and the tolerance is still the ninety-fifth percentile of the held-out null derived
   from the class's own random halves at matched sample sizes. Neither was redrawn or moved for the
   re-run; the verdict below is rendered under the same criterion the failing run used, so the two
   runs are comparable rung for rung, which is what Section 4 asks.

6. **The owner-certified guard extends to the broadband fit.** The broadband parameters are a
   dimensionless spectral exponent and a decibel power ratio only; no absolute spectral density and
   no owner-certified figure appears at any point in the continuum fit, synthesis, or record, and the
   owner-certified fields remain unset exactly as before.

## The verdict and what it means

The re-run returns **fail** under the unchanged Section 4 criterion, but the recipe fix worked, and
the character of the result is the substance to carry to the Engineering and Product surface, not the
bare verdict bit. The tolerance was not moved and the operating band was not redrawn.

The pure-narrowband surrogate's failure was a runaway: its Wasserstein distance to the held-out class
climbed from about eight at zero decibels to about ninety-seven at twenty-four, and it fell inside
the held-out null only at the three lowest rungs. The hybrid surrogate removes that runaway
completely. Its Wasserstein distance is flat at about seven and a half across the whole ladder, and
it falls inside the held-out null at eight of the nine rungs, including the true operating point near
six to twelve decibels where the surrogate is as detectable as the real class. The surrogate now sits
at the outer edge of the held-out class's own spread, its distance above the null's median of about
three point eight but under the ninety-fifth-percentile tolerance of about eight, rather than far
outside it. In plain terms the surrogate now carries the between-line broadband a real quiet vessel
carries, and its line prominence is bounded the way a real vessel's is.

The single rung that fails is twenty-one decibels, where the surrogate-to-class distance is eight
point one five against a locally low tolerance of eight point zero two, a miss of under two percent.
That rung is also one where the surrogate detects at about forty-eight percent against the real
class's fifteen, three times as often, and it counts as operating only because the thirteen-vessel
held-out detectability interval is wide enough to overlap. Under the work order's own description of
the operating band as the rungs where the surrogate is as detectable as the real class, twenty-one
decibels is not an operating rung and the run passes; under the implemented overlap test carried
forward from WO-4, it is, and the run fails by that one rung. Both readings are reported here rather
than the operating band being redrawn to force either answer.

This places the result in the second fork of Section 5. The surrogate is now physically faithful in
its between-line content, the divergence the fail exposed is gone, and the residual is a sub-two-
percent miss at a single high-signal-to-noise rung that is marginal by definition and provisional
under the placeholder line-prominence rejector and the reference detector. Whether that closes SD2 on
measurement, or waits for the real SD5 detector and SD6 rejector to re-confirm under a sharper lens,
is an Engineering and Product decision on the returned result. The recalibration fork remains open as
well: if E&P reads the residual as a broadband-shape mismatch rather than a lens limit, the continuum
exponent and ratio recalibrate against other train-side quiet vessels, never against the held-out
class and never by moving the tolerance.
