# Interpretation and validation limits

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

This repository implements and verifies the **mathematical stages** of that
instrument: first on the three surfaces where a closed-form answer exists to
check against, then on arbitrary parametric surfaces anchored to those. It is
not yet the application, because nothing here has been measured physically.

---

## What the current repository does and does not establish

Three levels, in increasing order of what they would be worth:

| level | meaning | status here |
|---|---|---|
| **Proof of work** | derivation, solver, numerical tests, reproducible reports | **done** — 235 declared checks across two stages: orders 1/2/4 recovered, the `ε²` coefficient matched to 4e-4 relative or better, the conjugate point located to 2e-15, the full 2x2 transfer map with `det Phi = 1` held to 2e-14, and the general parametric solver recovering both columns to 4e-13 before it is trusted on a saddle. [`report-v1.json`](../validation/report-v1.json), [`report-v2-surfaces.json`](../validation/report-v2-surfaces.json) |
| **Proof of function** | measured physical path separation agrees with the Jacobi prediction inside a quantified error budget | **not attempted** — no physical measurement exists in this repository |
| **Proof of industrial relevance** | using the sensitivity model produces a better decision: fewer gaps, better coverage, lower endpoint error | **not attempted** |

Nothing in this repository should be read as a claim at the second or third
level. The distinction is the point of stating it.

---

## Observation modes

Since a prediction and a measurement are only comparable if they are the same
quantity, every recorded comparison names its mode
(`geodesic_testbed.engine.observation`):

| mode | version | implemented |
|---|---|---|
| `intrinsic-surface-distance` | 1 | yes |
| `ambient-euclidean-chord` | 1 | yes |
| `scanner-reconstructed-chord` | 0 | no — needs a characterised instrument |
| `camera-image-residual` | 0 | no |

A camera does not measure a chord directly; it measures image coordinates, and
a chord appears only after calibration, reconstruction and registration. The
last two modes are declared and left unimplemented rather than quietly
conflated with the second, and every experimental record should carry its mode,
calibration, frames, units and uncertainty alongside the number.

## Observation-model distinction

Stage two measures the Jacobi field two independent ways: by integrating
`j'' + K j = 0`, and by central-differencing the geodesic flow itself. They
agree to second order in the differencing step, with coefficient

```text
( cn_K(s)² + κ_n² sn_K(s)² ) / 6
```

where `κ_n` is the normal curvature transverse to the path. The second term
exists because the differencing measures a **straight-line chord in space**,
while the Jacobi field is a **distance in the surface**. On a plate the term
vanishes; on the pseudosphere it is 2.5 times the intrinsic term.

A photogrammetry rig or a line-laser scanner reports image or range
observations; a chord appears only once those are calibrated, reconstructed and
registered into 3-D points. A bench that compares the resulting chord directly
against `ε j(s)` will see a discrepancy that is second order in the perturbation — the same order as the
first-order model's own failure — and will mistake one for the other. Either
convert the measurement to an in-surface distance, or predict the chord. The
coefficient above says how much it matters for a given coupon, before any
hardware is bought.

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

## Closed-form numerical reference

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
angles, longer paths, or lower measurement uncertainty. These are numerical
reference values, not results from a physical trial.

The supported measurement and filtering records are documented in
[MEASUREMENT.md](MEASUREMENT.md). Physical validation remains `not_started`.
