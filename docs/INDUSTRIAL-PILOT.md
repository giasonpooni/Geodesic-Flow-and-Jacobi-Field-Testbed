# Curved-surface manufacturing and inspection pilot

A CNC, small robot, or custom positioning rig traces neighbouring courses on
interchangeable coupons with deliberately varied starting position and heading.

## Start with plate against cylinder

The first experiment should be a **matched plate and rolled cylinder**, not a
curved coupon. It is the cheapest decisive control available, because it tests
the one prediction here that is not obvious to the eye: a cylinder is visibly
curved and its geodesics are helices, yet it is intrinsically flat, and the
runtime says its path sensitivity is *identical* to the flat plate's -- to
1e-13, from a completely different parameterisation. A bench that reproduces
that has shown the instrument keys on intrinsic curvature; a bench that does
not has found something real.

Only then add a spherical cap, for focusing, and a measured saddle or torus
patch for varying curvature. A pseudosphere -- the only embedded surface of
constant negative curvature -- is worth making only if the singular edge can be
handled, and is not needed before the variable-curvature runtime has been used
on a measured patch.

**A hyperbolic-paraboloid saddle does not have constant curvature.** Its `K`
runs from about -0.78 at the centre to -0.03 two units out. Comparing it with a
constant-`K` prediction and reading the mismatch as experimental error is the
easiest available way to get a wrong answer; use the surface's own curvature
profile, which the runtime takes directly from `r(u, v)`.

## Measurements

- the **observation mode** of every recorded comparison -- an in-surface
  distance and an ambient chord differ at second order in the perturbation,
  which is the same order as the model's own failure, so an untagged number is
  not evidence;
- the as-built coupon geometry, scanned; nominal CAD is not enough, since the
  prediction is a functional of the actual surface;
- calibrated coupon geometry and datum frame;
- commanded and measured path coordinates;
- starting-position and heading perturbations;
- tool or sensor width;
- timestamped encoder and external metrology observations;
- calibration, fixture, software, and dataset identifiers.

## Validation sequence

1. Calibrate the machine and external measurement system independently.
2. Size the experiment by signal-to-noise **before** running it. The nonlinear
   residual the trial is meant to resolve must be several times the combined
   measurement uncertainty. On a 300 mm coupon over a 600 mm path, a 2 degree
   aiming error leaves a first-order residual of about 27 um on the saddle and
   1 um on the plate; against a 25 um measurement uncertainty the first is not
   a convincing validation point and the second is no point at all. Longer
   paths, larger perturbations or better metrology, chosen deliberately.
3. Declare position and heading perturbations before running a trial, and
   perturb **both** -- the transfer map has two columns and the lateral one has
   never been exercised physically.
4. Generate the prediction and the envelope, in the observation mode the
   instrument will actually report.
5. Execute repeated paths on each coupon, with randomised run order.
6. Measure transverse path separation as a function of arclength.
7. Compare prediction and measurement on held-out trials. Calibration runs and
   validation runs must be separate data; if the same measurements tune and
   demonstrate the model, the evidence is worth much less.
8. Record where the first-order approximation ceases to meet its declared
   error tolerance, and compare that boundary with the predicted one.

## Filtering

Raw metrology is noisy and will be filtered. Filtering is legitimate and it is
also the easiest way in this whole programme to manufacture agreement with the
prediction, so it is confined to the measurement and surface-reconstruction
layers. **No filter runs inside the Jacobi solver.** `engine/flows.py` and
`engine/transfer.py` integrate a declared equation at a declared step and have
no smoothing in them at all; a filter that touched them would be tuning the
model to the data through the model's own machinery.

A filter is instead part of the observation instrument, and therefore changes
the observation model. If a linear temporal filter is the matrix `F`, then

```text
y_f = F H Phi dz0 + F eta        R_f = F R F^T
```

Comparing filtered measurements against an unfiltered prediction while still
quoting the original `R` overstates the agreement by exactly the factor the
filter removed. `engine/observation_model.py` carries `TemporalFilter` and
`filtered_noise_covariance(F, R)`, `ObservationModel` holds the filter it was
built with, and `measurement.compare` raises on a trial whose
`filter_identifier` is not `"none"` unless the caller passes
`filtered_prediction=True` — that is, unless the prediction has been put
through the same `F`. The default refuses the mismatch; the flag is an
assertion that the work was done, and it is recorded in the result.

### Prohibitions

- Do not tune filter parameters on validation trials.
- Do not choose smoothing strength to improve agreement with the Jacobi
  prediction. Strength is fixed from calibration data, before the trial.
- Do not use a zero-phase or future-looking smoother in an experiment claimed
  to be real time. `filter_causal` records which it was, and `offline` is an
  honest answer; a mislabelled one is not.
- Do not suppress samples near a focus because they look unstable. That is
  precisely where `rho` collapses and where the prediction is most falsifiable.
- Do not report filtered residuals with raw-sensor uncertainty.
- Do not discard outliers without recording the rule and the rejected samples.

### What each MeasurementRecord carries

`engine/measurement.py` requires, alongside the geometry and calibration
provenance: `filter_identifier`, `filter_version`, `filter_parameters`,
`filter_causal`, `filter_group_delay`, `filter_tuned_on`,
`input_sampling_rate`, `output_sampling_rate`, `measurement_covariance`,
`rejected_sample_mask`, `outlier_rule` and `raw_data_digest`. A record without
them does not describe an instrument, and `MEASUREMENT_SCHEMA` is
`path-sensitivity-observation-v1`.

### Three datasets, preserved separately

For the plate--cylinder trial and every one after it:

1. **raw observations**, as the sensor emitted them;
2. **calibrated but unfiltered** observations;
3. the **final filtered and reconstructed** trajectory, with `F`, `R_f` and the
   rejected-sample mask that produced it.

Keeping all three is what makes it possible, later, to ask whether an agreement
was in the surface or in the smoothing.

## Acceptance evidence

- numerical error against constant-curvature reference solutions;
- measured versus predicted transverse separation;
- repeatability and measurement uncertainty;
- maximum observed and bounded gap or overlap;
- inspection coverage margin;
- model-validity boundary versus perturbation size;
- explicit accounting for backlash, compliance, fixture error, and surface
  geometry error.

## Next adapter

Parametric surfaces are already handled here: `engine/surfaces.py` takes
`r(u, v)`, and `engine/envelope.py` traces the nominal path and both transfer
columns along it.

Triangulated surfaces are **not**, and will not be. Meshes, discrete curvature
estimators and mesh path convergence belong to the Intrinsic Surface Geodesics
Testbed. The adapter this repository needs is therefore a *consumer*, not a
tracer: it accepts a versioned path artefact produced there -- sampled
position, tangent, Gaussian curvature, frame, units, provenance and
uncertainty -- and feeds it to the existing transfer-map integrator. Tracing a
second mesh solver here would duplicate the one that project exists to build.

The application contracts do not change when that adapter arrives, because both
sides of it speak the same transfer record (`engine/record.py`).
