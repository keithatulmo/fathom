# Deviations and narrowings for WO-7

WO-7 makes committed state match the SD4 closing memo (v1.0). It re-designates the E2 selection from
the site-averaged argmax the agent committed to the quiet-site operating-point configuration the close
fixes, and it does so over the already-committed per-configuration numbers with no re-measurement. No
owner-certified value and no absolute level appears; every level in the front end is a ratio, and the
operating point and threshold stay set once across sites.

1. **The selection objective changed from site-averaged to quiet-site operating-point separability.**
   The E2 selector previously maximized separability averaged across the quiet, nominal, and busy
   sites; it now maximizes separability at the operating-point site, which is the quiet site, subject
   to holding cross-site flatness. The justification, recorded in the selector and here, is that the
   front end serves the quiet-site operating point the proof is staged at, so the site that sets the
   selection is the quiet site, not the average of three sites of which two are not the operating
   point. This is encoded as the ``operating_point_regime`` field of the E2 configuration and read by
   both the sweep selector and the standard-library verifier, so the change is a rule the verifier
   re-derives, not a hand-placed selection. The corrected objective and the operating-point regime are
   written into the committed lineage.

2. **No re-measurement: the per-configuration scores stand.** The sweep was not re-run in the sense
   that matters; the per-configuration separability, flatness, and constraint numbers are byte-for-byte
   identical to those committed at WO-6, which was confirmed by diffing the per-configuration table
   before and after. Only the selection, the runner-up ordering, the verdict, the band-endpoint
   finding, and the recorded objective changed, all derived from the unchanged numbers. The lineage was
   regenerated from the committed background cache under the corrected objective so the whole artifact
   is internally consistent, and it is byte-stable on a cache re-run.

3. **The re-derived selection and the retained runner-up.** Under the quiet-site objective the argmax
   over the flatness-holding candidates is the split-window configuration at a two-second window and
   half-hertz resolution, three-quarters overlap, a single Hann taper, unit integration, and a
   two-pass split-window order-truncated-average normalizer with a six-hertz half-window, a one-hertz
   guard, and a truncation order of three, at ten point two decibels of quiet-site separability and
   site-invariant to a tenth of a decibel across the three sites at ten point two, ten point two, and
   ten point three. The temporal-median configuration the agent committed as the site-averaged argmax
   is retained as a runner-up with its full per-site numbers, where it reads nine point nine decibels
   at the quiet operating point and ranks third under the corrected objective, its site-averaged lead
   manufactured at the busy site where it reaches eleven point nine. The reference front end is
   unchanged in its committed numbers: it still scores the highest raw separability and still fails
   cross-site flatness, the trap the flatness criterion exists to catch.

4. **The band-endpoint finding is re-reported on the closed configuration.** The endpoints were
   previously measured on the temporal-median configuration; they are re-measured on the closed
   split-window configuration from the committed background cache. Both hold: recovered comb prominence
   is eight point eight decibels at the four-hertz floor and ten point eight decibels at the
   one-hundred-fifty-hertz edge, both within three decibels of the ten point two decibel mid band, so
   no band change is indicated. This is feedback to the design-assumptions band entry the owner rules
   on, not a change made here.

5. **The production front-end default is set to the closed configuration.** A named
   ``closed_front_end`` accessor returns the closed split-window parameters as the SD4-closed default
   the detection and tracking layers build on, and the WO-3 reference front end stays in place
   unchanged for comparison.

## Standing obligations, carried unchanged

The front-end choice is provisional under the placeholder SD5 detector, exactly as the E1 realism
verdict is provisional under the SD6 rejector, because the separability metric is recovered-line
prominence through a placeholder detection path rather than the real detector. E3, the SD5 detection
bake-off, runs on this closed front end at the operating point and is both the measured confirmation of
the front-end choice and the one basis on which the normalizer family would be revisited: only a
measured result there that temporal-median delivers materially better operating-point detection with a
transferable threshold reopens the family. And the closed front end differs materially from the
reference front end E1 ran on, now a half-hertz split-window gram, so the E1 realism re-run is owed on
it and folds into the SD6 re-confirmation already owed from the WO-5 disposition, since both are the
same act of re-scoring once the real front end and rejector exist.
