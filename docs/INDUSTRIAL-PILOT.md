# Curved-surface manufacturing and inspection pilot

The first physical pilot should use three interchangeable coupons: a plane, a
spherical cap, and a negative-curvature saddle. A CNC, small robot, or custom
positioning rig traces neighbouring courses with deliberately varied starting
position and heading.

## Measurements

- calibrated coupon geometry and datum frame;
- commanded and measured path coordinates;
- starting-position and heading perturbations;
- tool or sensor width;
- timestamped encoder and external metrology observations;
- calibration, fixture, software, and dataset identifiers.

## Validation sequence

1. Calibrate the machine and external measurement system independently.
2. Declare position and heading perturbations before running a trial.
3. Generate the Jacobi prediction and deterministic envelope.
4. Execute repeated paths on each coupon.
5. Measure transverse path separation as a function of arclength.
6. Compare prediction and measurement on held-out trials.
7. Record where the first-order approximation ceases to meet its declared
   error tolerance.

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
