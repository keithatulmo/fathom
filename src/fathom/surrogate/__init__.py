"""The surrogate-target injector (SD2 Option A).

The injector builds a manned-vessel acoustic surrogate from machinery kinematics as narrowband
line families, conditions it by propagation and a kinematic bearing track, and injects it onto real
or fixture background at a controlled signal-to-noise ratio. Everything is dimensionless or
signal-to-noise relative: no absolute radiated source level, detection range, or confirmer
probability appears anywhere, per SD2 Section 2. Surrogate parameters derive from train-side vessels
and first-principles structure only, and the held-out quietest class is opened solely at the E1
comparison, per the SD3 Section 5.4 injection-leak rule. The governing memo is
Fathom_SD_memos/Fathom_SD2_surrogate_methodology_memo_v0.1.
"""
