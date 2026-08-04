# The Surrogate Injector and the E1 Realism Harness

This note states, in complete sentences, what the surrogate injector is, how its level is defined
against the background, and where the methodology it follows is fixed. The governing memo is
`Fathom_SD_memos/Fathom_SD2_surrogate_methodology_memo_v0.1`, which selects the physics-parametric
recipe (its Option A) and designs the E1 validation experiment; the injection-leak rule that binds
the fitting is Section 5.4 of `Fathom_SD_memos/Fathom_SD3_evaluation_protocol_memo_v0.1`. This code
consults no prior codebase and touches only unclassified, synthetic data at proof scope.

## What the injector is

The injector builds a manned-vessel acoustic surrogate from machinery kinematics as narrowband line
families and injects it onto real or fixture background. The line set has a shaft-rate fundamental
and its harmonics, a blade-rate line at the shaft rate times the blade count with harmonics, an
electrical line at the supply frequency with harmonics, and a small set of auxiliary-machinery
lines; higher harmonics roll off, which is the first-principles structure a physics model supplies.
The quiet-target case is the hard one the workflow matrix names: the propulsion lines may be absent
and the auxiliary lines carry the signature at low level. Each line is realised in the time domain
as a coherent tone whose instantaneous frequency is scaled by a kinematic Doppler factor and
perturbed by a finite width and a slow wander, and whose amplitude follows the propagation spectral
gain and a range envelope. The construction lives under `src/fathom/surrogate/`.

## How the level is defined

The level is a signal-to-noise ratio relative to the measured in-band background of the target
channel, and nothing else. The injector measures the in-band power of the surrogate and of the
background, scales the surrogate so their ratio equals the requested ratio, adds it to the
background, and re-measures the achieved ratio on the combined channel to a stated tolerance. There
is no absolute radiated source level anywhere in the interface, no detection range, and no confirmer
probability; the quietness of a target is represented as a low signal-to-noise ratio and sparse line
content, per SD2 Section 2. Every amplitude in the line set is a dimensionless family weight, and the
propagation and kinematic parameters are dimensionless proxies, so no owner-certified value is
expressible.

## Propagation, kinematics, and the train-only fit

Propagation is conditioned at proof scope by a frequency shaping, in which absorption attenuates
higher lines and a multipath comb adds an interference ripple, and by a range envelope from
spreading that makes the surrogate loudest at the closest point of approach; it is explicitly short
of the track-two world model. The kinematic drive places the surrogate on a straight-line pass in
dimensionless proxy units, so it approaches and recedes, its bearing sweeps, and its Doppler shifts
up while closing and down while opening. The surrogate distributions are fit from train-side vessels
only: the fitting refuses any calibration, test, or held-out vessel by raising, so no code path can
let the held-out quietest class reach surrogate construction, which is the SD3 Section 5.4 leak rule
made mechanical.

## Injection and E1

Injection is a pipeline stage that consumes background and emits an injected channel with its full
parameter set and seed recorded in the lineage ledger, so any injected scenario regenerates byte for
byte through the reproduction command. The E1 harness injects surrogates across a swept
signal-to-noise ladder, processes them and the held-out class through an identical documented
reference front end and detector, and compares the detectability-versus-signal-to-noise response and
the rejector-response distribution; the realism criterion is met when the two are indistinguishable
across the operating band, judged by overlapping confidence intervals on the detectability curve and
a two-sample distance below tolerance on the rejector response. A reference rejector stands in until
SD6 closes. On synthetic fixtures the held-out arm is a placeholder class, which proves the
comparison machinery; the real held-out quietest vessel class is read only when the corpus provides
it, and is never consulted in surrogate fitting.
