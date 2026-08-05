# Deviations and narrowings for WO-6

This document records every point where the WO-6 front-end build and the E2 sweep depart from,
narrow, or make a choice not fixed by the SD4 memo or the work order, per the reporting rule. No
owner-certified value and no absolute source level appears anywhere: the front end operates on
normalized grams, the injected level is a signal-to-noise ratio relative to the measured background,
and every level in the normalizer is a ratio. The flatness discipline is enforced by construction:
the operating-point signal-to-noise and the single detector threshold are set once across the whole
site set and never curated per site.

1. **The sweep grid is a proof-scope bracket, not exhaustive.** The grid crosses three window lengths
   (two, four, and eight seconds, so frequency resolutions of a half, a quarter, and an eighth of a
   hertz), two overlaps (a half and three-quarters), two integration counts (one and four), the two
   tapers (a single Hann and a five-taper Thomson multitaper at time-bandwidth three), and the two
   normalizers (split-window against temporal median), for forty-eight configurations. The
   split-window half-window, guard, and truncation (six hertz, one hertz, three) and the
   temporal-median window (eight frames) are fixed at defensible values rather than swept, and the
   multitaper is a single configuration rather than a set. These are recorded in configuration and
   are the parameters E&P can widen if the close warrants it; the grid brackets the resolution the
   memo names and is enough to fire or not fire the flip clause.

2. **Separability is a detection index against the achieved between-line floor.** The work order asks
   for the recovered line prominence or detection index of the injected comb. It is scored as the
   median, across the comb's known in-band lines, of the recovered line level divided by the
   between-line floor the configuration actually achieves, in decibels, with the recovery tolerated
   inside a fixed half-hertz window that matches the surrogate's wander and is stated in hertz rather
   than bins. Measuring the line against the achieved floor rather than against unity makes the score
   fair across normalizers, so a normalizer that lifts the whole in-band region uniformly earns no
   free prominence, and stating the tolerance in hertz stops a coarse configuration from inflating its
   score by letting a wide bin reach across a comb interval. This is a specific reading of the
   work order's "recovered line prominence or detection index," recorded so the metric is legible.

3. **Flatness is the cross-site false-alarm invariance of one threshold.** One threshold is set across
   the pooled normalized background at the target false-alarm quantile, five percent, and the realized
   false-alarm rate is measured per site; flatness holds when every site's rate sits within a factor
   of two and a half of the target. This is the single-threshold-across-sites property the criterion
   names, made concrete, with the cross-bin coefficient of variation recorded alongside.

4. **The operating point, the separability floor, and the compute budget are proof-scope constants.**
   The operating-point signal-to-noise is six decibels, one value across all sites, taken from the E1
   operating band where the surrogate is as detectable as the real quiet class. The separability floor
   that the flip clause tests adequacy against is six decibels of recovered prominence, and the compute
   budget is a proxy ceiling a few times the reference front end's cost. Each is recorded in
   configuration and in the lineage; none is tuned to force a selection.

5. **The corpus slice is three background windows per regime, read from the object head.** One
   background window of thirty seconds is drawn from each of three recordings in each regime, the
   quiet ADEON Atlantic site, the nominal Monterey mooring, and the busy Strait of Georgia site, by a
   range read of the object head rather than a full download, because a streamable FLAC decodes its
   leading frames from the head and a multi-hour object need not be fetched in full to sample a
   window. The smallest objects are preferred so the head read is light. These are proof-scope
   sampling choices recorded in the lineage's corpus slice; the injection probe is the already-closed
   SD2 hybrid surrogate loaded from the committed E1 lineage, so E2 re-fits nothing and reads no train
   or held-out split.

6. **The selected normalizer is the temporal median, narrowly, over the memo's split-window primary.**
   This is the one material place the measurement departs from the memo's structural expectation and
   is reported rather than papered over. The SD4 memo selects the split-window normalizer as the
   structural primary and carries the temporal median as the alternative arm. On this corpus slice the
   sweep's highest-separability configuration that holds cross-site flatness is a Hann, quarter-hertz,
   temporal-median configuration at ten point four five decibels, and the best split-window
   configuration is a Hann, half-hertz configuration at ten point two one decibels, a gap of under a
   quarter of a decibel with both holding flatness cleanly. The selection rule of Section 6 is
   maximize separability subject to flatness, so the temporal median is the literal argmax and is
   recorded as the selection, but the two are a statistical tie at proof scope, the memo's structural
   argument for the split-window primary stands on grounds the sweep does not measure here, its
   per-frame adaptivity with no stationarity window and its behaviour at scale and over longer dwells,
   and the thirty-second windows give the temporal median only a handful of frames to estimate from.
   Which normalizer family the closing memo fixes is therefore an Engineering and Product decision on
   the returned result, not one the Code surface makes by committing the bare argmax; both are exported
   with their full per-configuration numbers so the choice is reviewable from committed state.

## The verdict and what closes

E2 renders a selection and the flip clause does not fire, so SD4 closes on measurement under the
proof-scope criterion, subject to the normalizer-family decision of deviation six and to the standing
provisionality below.

The selected configuration is a linear Hann gram at a quarter-hertz resolution, a half overlap, no
extra integration, with the temporal-median normalizer, at ten point four five decibels of recovered
comb separability with cross-site flatness holding at a five-percent false-alarm rate that is
site-invariant to within a tenth of a percent across the quiet, nominal, and busy sites. The flip
clause does not fire because a single-taper Hann configuration reaches the separability floor with
flatness; the multitaper arm scores far lower, near five decibels, because its wider main lobe spreads
the line energy, so it is not adopted and pays no compute. The reference front end that E1 used scores
a higher raw separability, twelve decibels, but fails cross-site flatness outright, holding a
fifteen-percent false-alarm rate at the quiet site against near zero at the busy site under one
threshold, which is exactly the site-transferability the median-over-frequency reference cannot
deliver and the reason SD4 selected an adaptive normalizer over it. The working-band endpoints hold:
the recovered comb prominence at the four-hertz floor and the one-hundred-fifty-hertz edge are both
within three decibels of the mid band, so no band adjustment is indicated, which is reported as
feedback to the design-assumptions band entry for the owner to rule on and is not changed here.

Two standing qualifications bound the close. The selection is provisional under the placeholder
line-prominence detector until SD5 delivers the real detector, exactly as the E1 realism verdict is
provisional under the placeholder rejector until SD6. And the closed front end differs materially from
the reference front end E1 ran on, a different resolution, taper handling, and normalizer, so the E1
realism comparison is owed a re-run on the closed front end; that folds into the SD6 re-confirmation
already owed from the WO-5 disposition, since both are the same act of re-scoring once the real front
end and rejector exist.
