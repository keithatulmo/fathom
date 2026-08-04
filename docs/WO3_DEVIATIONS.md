# Deviations and narrowings for WO-3

This document records every point where the WO-3 injector and E1 harness implementation departs
from, narrows, or makes a choice not fully fixed by the SD2 memo or the work order, per the
reporting rule. None invokes an owner-certified value or an absolute source level, and none consults
prior code.

1. **The reference rejector is an in-band line-prominence statistic.** SD2 Section 7 calls for a
   reference rejector standing in until SD6 closes but does not fix its form. This build uses the
   ratio of the strongest in-band spectral bin to the median in-band bin, which is large for a
   narrowband target and near one for broadband clutter, as the reference rejector response. It is a
   documented placeholder; the real rejector is selected at SD6, and the E1 comparison rebinds to it
   without change to the harness.

2. **Propagation is applied as a spectral shaping plus a range envelope, not a full range-and-
   frequency-dependent field.** At proof scope, absorption and multipath reshape the line spectrum
   at the reference range and a separate spreading envelope varies the level over the pass. This is
   the simple multipath-or-modal surrogate the work order asks for, explicitly short of the
   track-two world model; a fuller model is a track-two decision.

3. **Bearing is recorded but not consumed by the single-channel reference detector.** The kinematic
   track produces a bearing over time for the custody scenario and records it, but the reference
   front end and detector operate on a single channel, so only the Doppler and the range envelope
   affect the waveform. Bearing enters processing when an array front end is selected, which is out
   of scope here; recording it now keeps the custody scenario regenerable.

4. **The amplitude, width, and wander statistics are first-principles priors at proof scope.** The
   fitting reads the train roster and the train-side quiet-target fraction and refuses every other
   split, but the line-amplitude and harmonic-roll-off distributions are first-principles machinery
   priors rather than statistics fit to real train-side audio. Fitting those to real audio is the
   refinement the SD2 Section 8 flip condition names and the E1 real-data run performs; the priors
   are recorded as priors in the fitted distributions.

5. **The E1 held-out arm on fixtures is a placeholder synthetic class.** WO-3 AC4 asks the harness to
   run end to end on fixtures with placeholder held-out data. The placeholder is an independent draw
   from the same fitted distributions, so on fixtures the two arms are statistically similar and the
   harness reports the indistinguishable verdict, proving the comparison path. The real held-out
   quietest vessel class is read only at AC7, is processed through the identical path, and is never
   consulted in fitting.

6. **The two-sample distance is the Kolmogorov-Smirnov statistic.** SD2 Section 7 asks for a
   two-sample distance below a stated tolerance on the rejector-response distribution but does not
   fix its form. This build uses the two-sample KS distance, bounded in the unit interval, with the
   tolerance recorded in configuration, provisional and revisable when the held-out class is
   measured.

7. **Injection places one surrogate per background channel.** The injection stage injects a surrogate
   onto each background channel rather than a single scenario, so the E1 ladder carries several
   independent channels per rung for the cluster bootstrap. Each channel's parameters and seed are
   recorded, so every channel is regenerable.

## Acceptance criteria status

AC1 through AC6 are met now: the injector places lines at the specified kinematic frequencies with an
achieved signal-to-noise within tolerance, injection is deterministic and byte-identical on
regeneration, the leak guard refuses held-out and evaluation vessels by raising, the E1 harness runs
end to end on fixtures and renders the realism report, no owner-certified value or absolute source
level appears anywhere, and ruff, strict mypy, and the full suite pass with no network dependency on
the offline path. AC7 is deferred by the work order: the E1 comparison against the real held-out
quietest vessel class, and the pass-or-fail that closes SD2, wait on that class, which is read
through the same harness the moment the corpus delivers it.
