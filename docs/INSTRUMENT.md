# From a verified solver to an instrument

**Positioning.** Curvature-aware path sensitivity: given a surface, a nominal
path along it, and a tolerance on how the path is started, how large is the
resulting deviation further along — and where does the cheap first-order answer
stop being an answer?

A Jacobi field is exactly the first-order separation between a nominal geodesic
and a neighbouring one. Read as an instrument:

| behaviour of `j(s)` | reading |
|---|---|
| `j` grows slowly | the path tolerates setup error |
| `j` grows fast | small aiming errors are amplified downstream |
| `j(s) = 0` away from the start | neighbouring paths refocus; the path-to-endpoint map is ill-conditioned there |
| `εj(s)` departs from the measured separation | the first-order model has left its validity range |

This repository implements and verifies the **mathematical calibration stage**
of that instrument, on the three surfaces where a closed-form answer exists to
check against. It is not the application.

---

## What the current repository does and does not establish

Three levels, in increasing order of what they would be worth:

| level | meaning | status here |
|---|---|---|
| **Proof of work** | derivation, solver, numerical tests, reproducible reports | **done** — 109 declared checks, orders 1/2/4 recovered, the `ε²` coefficient matched to ~1e-4 relative, conjugate point located to 2e-15, all in [`validation/report-v1.json`](../validation/report-v1.json) |
| **Proof of function** | measured physical path separation agrees with the Jacobi prediction inside a quantified error budget | **not attempted** — no physical measurement exists in this repository |
| **Proof of industrial relevance** | using the sensitivity model produces a better decision: fewer gaps, better coverage, lower endpoint error | **not attempted** |

Nothing in this repository should be read as a claim at the second or third
level. The distinction is the point of stating it.

---

## The readout, today

`results.path_sensitivity` in the report already answers the instrument's three
questions for the three model surfaces. Lengths are in units of the radius of
curvature `R`; a spherical coupon of radius 300 mm makes `s = 1` a 300 mm path.

**Amplification** `j(s)/s`, relative to the flat case:

| `s` | `K=0` | `K=+1` | `K=-1` |
|---|---|---|---|
| 0.25 | 1.000 | 0.990 | 1.010 |
| 0.50 | 1.000 | 0.959 | 1.042 |
| 1.00 | 1.000 | 0.842 | 1.175 |
| 1.50 | 1.000 | 0.665 | 1.420 |
| 2.00 | 1.000 | 0.455 | 1.813 |

**Tolerance budget** — the largest initial aiming error that keeps the
transverse deviation within `1e-3 R` (0.3 mm on a 300 mm coupon):

| `s` | `K=0` | `K=+1` | `K=-1` |
|---|---|---|---|
| 0.50 | 0.115° | 0.120° | 0.110° |
| 1.00 | 0.057° | 0.068° | 0.049° |
| 2.00 | 0.029° | 0.063° | 0.016° |

At `s = 2` the saddle demands aiming four times tighter than the sphere for the
same deviation. That ratio, not the absolute numbers, is what a curvature-aware
planner would act on.

---

## Predictions a bench could falsify

These are computed in advance, from the closed form, and cross-checked against
an independent RK4 flow to better than `1e-10` relative. Path length `s = 2`
(600 mm on a coupon of radius 300 mm); separation in mm at `R = 300 mm`.

| initial angle | `K=0` | `K=+1` | `K=-1` | first-order error, `K=-1` |
|---|---|---|---|---|
| 0.25° | 2.618 mm | 1.190 mm | 4.747 mm | 1.1e-05 (0.05 µm) |
| 0.50° | 5.236 mm | 2.381 mm | 9.495 mm | 4.5e-05 (0.43 µm) |
| 1.00° | 10.472 mm | 4.761 mm | 18.990 mm | 1.8e-04 (3.4 µm) |
| 2.00° | 20.944 mm | 9.522 mm | 37.980 mm | 7.2e-04 (27 µm) |

The right-hand column is the honest part: even at 2°, the first-order
prediction on the saddle is wrong by only 27 µm out of 38 mm. A bench whose
measurement uncertainty is worse than about 25 µm cannot detect the breakdown
of the first-order model at these angles at all — it would have to run larger
angles, longer paths, or better metrology. Knowing that *before* building the
bench is worth more than the bench.

---

## The bench that would establish proof of function

Three interchangeable coupons, one per curvature sign, on a positioning stage
with a contact marker or probe, measured by photogrammetry or a line-laser
scanner, with a repeatable datum and adjustable initial position and angle.
Run nominal paths, then perturb the initial direction by 0.25°, 0.5°, 1°, 2°,
and compare the measured transverse separation with `ε j(s)`.

Expected qualitative result, which the reader can check against the figure:
linear spreading on the flat plate, refocusing on the spherical cap, rapid
growth on the saddle, and a first-order prediction that departs from the
measurement as the angle grows.

**One caveat that must not be glossed over.** A flat plate has `K = 0` exactly
and a spherical cap has constant `K > 0` exactly, but a hyperbolic-paraboloid
saddle does **not** have constant curvature — `K` varies across it. For the
`K = -1` coupon either machine a surface of genuinely constant negative
curvature (a pseudosphere/tractricoid patch, which has a singular edge to
design around) or accept that the constant-curvature prediction is an
approximation and do step 3 of the roadmap first. Comparing a variable-`K`
coupon against a constant-`K` prediction and calling the mismatch an
experimental error would be the easiest way to get a wrong answer here.

What a run should report: prediction-versus-measurement RMSE, repeatability
across runs, maximum gap or overlap, sensitivity to initial position as well as
direction, the measured first-order validity range, an uncertainty budget,
the contribution of backlash and fixture error, and full calibration and data
provenance. Calibration runs and validation runs must be separate data: if the
same measurements tune and demonstrate the model, the evidence is worth much
less.

---

## Roadmap

1. **Complete the constant-curvature numerical experiment.** — done; this repository.
2. **Verify Jacobi solutions against finite differences between nearby geodesics.** — done;
   `results.first_order_validity`, including the `ε²` coefficient and both failure modes.
3. **Arbitrary triangulated or parametric surfaces**, where `K` varies along the
   path, the Jacobi equation has no closed-form solution, and the reference must
   come from a converged fine-step solution instead. This is the next piece of
   work, and the precondition for a meaningful saddle coupon.
4. **A sensitivity envelope around every nominal path**, not just a scalar at
   selected arc lengths.
5. **The three-coupon bench.**
6. **Controlled fixture, angle and backlash errors.**
7. **A robust path selector** that prefers a starting path with lower sensitivity.
8. **Configuration space**, where the manifold is the robot's, not the workpiece's.

Steps 1 and 2 are the content of this repository. Steps 3 onward are not
started, and the README does not claim otherwise.

## Where it would apply

Automated fibre and tape placement, filament winding, coating and welding on
curved surfaces; robotic inspection path planning under starting-pose and
calibration error; and, further out, trajectory robustness and pose-estimation
uncertainty on `SO(3)` and `SE(3)`, where the manifold is the robot's
configuration space rather than the part. Those are destinations, not claims:
the only thing verified here is the mathematics of the constant-curvature case.
