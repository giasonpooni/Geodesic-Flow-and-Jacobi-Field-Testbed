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

After the coupon experiment, add a surface adapter that accepts a triangulated
or parametric surface, traces a nominal geodesic, and supplies sampled Gaussian
curvature to the existing Jacobi integrator. The application contracts should
not change when that adapter arrives.
