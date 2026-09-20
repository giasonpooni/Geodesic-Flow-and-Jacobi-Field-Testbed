# Stage two: surfaces where the curvature varies

**One question:** the constant-curvature stage could check every number against
a closed form. A real workpiece has none. What replaces it?

Produced by `python examples/run_experiment.py --stage surfaces` and recorded
in [`validation/report-v2-surfaces.json`](../validation/report-v2-surfaces.json)
under schema `geodesic-jacobi-surfaces-v3`: **181 declared checks, 0 failed.**

![Curvature-aware path sensitivity on surfaces where the curvature varies](../figures/surfaces-testbed-v1.png)

---

## What changes

A parametric surface `r(u, v)` in `R^3` carries its own metric. From one jet —
`r` and its first two partial derivatives — come the first fundamental form,
the Christoffel symbols and the Gaussian curvature, and the geodesic and its
Jacobi field are advanced together in the surface's own coordinates:

```text
u'' = -(G1_11 u'^2 + 2 G1_12 u'v' + G1_22 v'^2)
v'' = -(G2_11 u'^2 + 2 G2_12 u'v' + G2_22 v'^2)
a'' = -K(u, v) a        (lateral offset column)
b'' = -K(u, v) b        (heading error column)
```

The parts cannot be separated any more: the Jacobi equation needs `K` at the
moving point, so the eight-component state `(u, v, u', v', a, a', b, b')` is
integrated as one system. Both columns of the transfer map ride along for the
price of one curvature evaluation.

Everything is intrinsic, so the path cannot drift off the surface — it is
*defined* in the surface's coordinates. What can drift is unit speed, and it is
never re-imposed, which leaves it free as a diagnostic. Across every case it
stays below `2e-12`.

## The catalogue

| case | surface | `K` | why it is here |
|---|---|---|---|
| plate | plane | 0 | the flat anchor |
| rolled-sheet | unit cylinder | 0 | extrinsically curved, intrinsically flat |
| spherical-cap | unit sphere | +1 | the positive anchor, and a conjugate point |
| pseudosphere | tractricoid | -1 | the negative anchor, and what a real `K < 0` coupon must be |
| saddle | `z = (u² - v²)/2` | -0.78 … -0.03 | varying `K`, one sign |
| torus | `R = 2, r = 1` | -0.07 … +0.32 | varying `K`, both signs |

## Four sources of verification, since closed forms have run out

### 1. Anchoring

The general machinery is run on the surfaces whose answer is already known and
must reproduce it. If it cannot recover the calibrated case it has no business
on a saddle.

| case | `b` must be | `max ǀb - refǀ` | `a` must be | `max ǀa - refǀ` | `det Phi` drift | speed drift |
|---|---|---|---|---|---|---|
| plate | `s` | 1.1e-13 | `1` | 0 | 0 | 0 |
| rolled-sheet | `s` | 1.1e-13 | `1` | 0 | 0 | 0 |
| spherical-cap | `sin s` | 4.4e-13 | `cos s` | 4.0e-13 | 3.1e-15 | 1.6e-12 |
| pseudosphere | `sinh s` | 1.2e-14 | `cosh s` | 1.4e-14 | 2.0e-14 | 9.3e-15 |

The rolled sheet is the one worth pausing on. A cylinder is visibly curved and
its geodesics are helices, but it is intrinsically flat, and the solver returns
exactly the plate's answer — to `1e-13`, from a completely different
parameterisation. Rolling a flat sheet onto a drum does not change how far an
aiming error is carried.

### 2. Two independent routes to each column

Each column is obtained twice. `b` by integrating `j'' + K j = 0` along the
path, and by central-differencing the flow itself in the **initial heading**.
`a` by the same equation, and by moving the **start point** sideways along the
perpendicular geodesic and parallel-transporting the initial direction to it —
a structurally different construction that also never touches the Jacobi
equation. Its measured exponents:

| case | `a` from equation | regime | fitted exponent |
|---|---|---|---|
| plate | 1.000000 | exact to roundoff | — |
| rolled-sheet | 1.000000 | second order | 1.9999 |
| spherical-cap | 0.540302 | second order | 1.9999 |
| pseudosphere | 1.543081 | second order | 1.9993 |
| saddle | 1.252333 | second order | 1.9998 |
| torus | 0.857576 | second order | 1.9999 |

The plate is exactly right at every offset, as it must be: two parallel lines
stay exactly as far apart as they started.

For the heading column the same comparison runs as follows. Their relative difference is second
order in the differencing step `eps`, with coefficient

```text
( cn_K(s)² + κ_n² sn_K(s)² ) / 6
```

where `κ_n` is the normal curvature transverse to the path. The first term is
intrinsic — a finite variation is not a derivative. The second is the price of
measuring a straight-line chord in space rather than a distance in the surface.

| case | fitted exponent | fitted coefficient | excess over `1/6` | chord closed form |
|---|---|---|---|---|
| plate | 2.0001 | 0.1667 | +0.0000 | 8.3e-11 |
| spherical-cap | 2.0000 | 0.1667 | -0.0000 | 1.8e-11 |
| rolled-sheet | 2.0000 | 0.1836 | +0.0170 | — |
| torus | 1.9999 | 0.1780 | +0.0113 | — |
| saddle | 2.0000 | 0.2188 | +0.0522 | — |
| pseudosphere | 1.9997 | 0.4090 | +0.2423 | — |

On a plate (`κ_n = 0`) and on a unit sphere (`cn² + sn² = 1`) the two terms
collapse to exactly `1/6`, the whole finite-`eps` answer is known in closed
form — `ǀJǀ = sn_K(s)·sin(eps)/eps` — and it is checked sample by sample rather
than asymptotically.

Elsewhere the excess is real. The cylinder's is the cleanest illustration:
intrinsically flat, so `cn = 1` and `sn(1) = 1`, and at heading `θ` the
transverse normal curvature is `sin²θ`, predicting `(1 + sin⁴θ)/6 = 0.18360`
at `θ = 0.6`. Measured: 0.18363. That prediction is a unit test.

**This matters for hardware.** A reconstructed pair of 3-D points gives a chord
in space; the Jacobi field is a distance in the surface. On the pseudosphere at
`s = 1` the measured second-order coefficient is 0.4090, of which the intrinsic
term is 0.1667 and the chord excess 0.2423 — the excess is 1.45 times the
intrinsic term, and a chord measurement therefore sees 2.5 times the
coefficient an in-surface distance would. And a camera does not report even
the chord directly — it reports image coordinates, and a chord appears only
after calibration, reconstruction and registration, each with its own error.
Any bench that compares a measured separation with `ε j(s)` has to account for
the whole chain or it will report a model failure that is really an instrument
definition.

### 3. Self-convergence

Where there is no reference solution, the step size is halved against a finer
run of the same solver.

| case | order, `j` | order, path |
|---|---|---|
| spherical-cap | 3.960 | 4.012 |
| pseudosphere | 3.984 | 4.010 |
| saddle | 4.003 | 3.922 |
| torus | 4.004 | 3.995 |
| plate, rolled-sheet | exact to roundoff | exact to roundoff |

In the plate's and the cylinder's charts every Christoffel symbol vanishes and
`K = 0`, so the whole system is linear and RK4 is exact at any step size. The
report says `exact-to-roundoff` rather than fitting an order to noise.

### 4. The cost of not supplying derivatives

A surface can be given by `r(u, v)` alone, in which case the jet is taken by
central differences — one function, and it works. That convenience is measured,
not assumed: each surface is run both ways.

| case | relative difference in `j` | difference in `K` |
|---|---|---|
| saddle | 6.0e-12 | 3.2e-09 |
| torus | 3.0e-10 | 3.8e-08 |
| pseudosphere | 1.0e-09 | 7.5e-08 |
| spherical-cap | 8.1e-09 | 5.2e-08 |

So a surface brought in as a black box — a CAD patch, a fitted scan — costs
roughly eight significant figures. That is far below the accuracy any physical
measurement will have, and far above what a convergence study can ignore, which
is why analytic derivatives are used wherever the surface knows them.

---

## The readout

For each nominal path, `results.envelopes` reports the curvature along it, the
Jacobi field, the amplification `j(s)/s`, the largest initial heading error
that keeps the deviation inside a tolerance, and whether the path passes a
focus.

| case | `K` along the path | `j` at the end | amplification | focus | heading budget for `1e-3 R` |
|---|---|---|---|---|---|
| plate | 0 | +2.000 | 1.000 | none | 0.0286° at `s = 2` |
| rolled-sheet | 0 | +2.000 | 1.000 | none | 0.0286° at `s = 2` |
| spherical-cap | +1 | -0.757 | -0.189 | `s = 3.1415926535` | 0.0573° at `s = 4` |
| pseudosphere | -1 | +2.129 | 1.420 | none | 0.0269° at `s = 1.5` |
| saddle | -0.78 … -0.03 | +2.254 | 1.127 | none | 0.0254° at `s = 2` |
| torus | -0.07 … +0.32 | +1.776 | 0.888 | none | 0.0323° at `s = 2` |

The budget is cumulative — governed by the worst deviation reached anywhere so
far, not by the deviation at the endpoint — because a path that must stay
inside a tolerance has to stay inside it the whole way.

## The decision it supports, and the trap in it

Twenty-four candidate starting headings from one point on the torus, each
flowed for the same path length of 6, ranked by forward angular-error
amplification `max ǀb(s)ǀ`:

| | heading | `max ǀbǀ` | focus margin | passes a focus | mean `K` |
|---|---|---|---|---|---|
| lowest amplification | 97° | 1.74 | 0.000106 | **yes**, at `s = 5.50` | +0.325 |
| highest amplification | 15° | 36.90 | 1.136 | no | -0.328 |

A factor of **21** between the extremes, from the same start and the same path
length, and the reason is in the last column: the low-amplification headings
stay on the positively curved outer region, where neighbouring paths refocus
and an aiming error is bounded; the intolerant one wanders onto the negatively
curved inner region, where it grows.

But that is the whole story only if forward separation is the whole objective,
and it is not. Seven of the twenty-four headings pass through a focus — and the
**six best-scoring headings are all among them**. A zero of `b` means the map
from starting heading to endpoint is ill conditioned, the path is at a
conjugate point, local minimality can be lost, and a family of such paths
crowds together instead of covering. Choosing purely by minimum amplification
walks straight into one.

### Why the criterion is not a threshold on `ǀbǀ`

The obvious patch is to require a focus margin above some floor. It is the
wrong kind of quantity. `ǀbǀ` has units of length per radian, so a floor of
0.25 means one thing on a 300 mm coupon in radians and something else on a
3 m part in degrees; it is a number chosen here, not a property of the job.

The criterion the scan actually uses is the one the instrument imposes.
Introduce the observation model — `δz(s) = Φ(s) δz₀`, `y(s) = H(s) δz(s) +
η(s)`, `Cov(y) = H Φ C₀ Φᵀ Hᵀ + R` — and the heading component of a starting
pose error is visible to the sensor at

```text
rho(s) = ǀb(s)ǀ · sigma_alpha / sigma_measurement
```

which is dimensionless. `sigma_alpha = 0.1°` of aiming uncertainty against a
25 µm metrology system on a 300 mm coupon is the declared example here, and
the report checks that `rho` does not move when the same physical situation is
drawn at twice the size (`max_relative_difference` 0.0).

`rho` is also not used as a minimum, because `rho(0) = 0` on every route — a
whole-path minimum is unsatisfiable, and a route first crossing the threshold
at its last sample would pass vacuously. It is used as a **schedule**
(`engine/tracking.py`): acquire above `rho = 5` within the first 1.0 of path
length, hold above `rho = 3` thereafter, tolerate at most 0.1 of continuous
loss, and require at least 3.0 of tracked distance. Outcomes are explicit:
`TRACKED`, `NEVER_ACQUIRED`, `LATE_ACQUISITION`, `TRACK_LOST`,
`INSUFFICIENT_TRACKED_DISTANCE`.

### What the instrument decides

| | declared process limits only | plus the acquisition schedule |
|---|---|---|
| feasible routes, of 24 | 15 | 8 |
| recommended | 97°, `max ǀbǀ` = 1.74 | 120°, `max ǀbǀ` = 1.82 |
| fate of 97° | feasible | `TRACK_LOST` — `rho` below 3 from `s = 5.354` |

Five per cent more amplification, for a route the scanner can actually follow.
And the routes it cannot follow are **exactly** the seven that pass through a
focus — two independent computations agreeing, the zeros of `b` and the
schedule of `rho`, recorded as a declared check rather than an observation.

That check is named
`surface-track-loss-coincides-with-focus-in-this-configuration`, and the
qualifier is the point. The agreement holds for *this* torus, this starting
uncertainty, this `H`, this scanner noise, this schedule and this candidate
set. It is corroboration, not an identity, and the two quantities come apart
in both directions:

| | focus present? | resolvable? |
|---|---|---|
| plate, heading tolerance shrunk 10⁴× | no — `b(s) = s` never vanishes | no: `rho` peaks at 0.0042 |
| spherical cap, 0.01 before `s = pi` | yes | yes, given 1.0 µm metrology on a 300 mm coupon |

A noisier scanner loses a path with no focus in it; a better one holds a path
that has one, because `ǀbǀ` beside a conjugate point is small but not zero.
Both are declared checks (`surface-unresolvable-without-a-focus`,
`surface-resolvable-beside-a-focus`), and both outputs are reported:
`focus_points` is the geometric claim, the tracking outcome is the
instrument's, and only the second decides a route.

Note what this costs the old floor: 120° has a focus margin of 0.200, *below*
the 0.25 that a hand-chosen threshold would have demanded, and the scanner
holds it for the whole path. The floor was rejecting a route on a number that
did not describe the sensor. `focus_margin_report_level` survives in the report
as a reported quantity and is explicitly not a criterion.

The ranking scalar is therefore recorded as
`minimum-forward-angular-error-amplification` — not "most robust" — and it
decides nothing. The recommendation comes from `engine/routing.py`:
cross-track error, heading error, coverage margin, and the tracking outcome.
Boundary clearance, curvature exposure and path length are declared in
`RouteConstraints` but left unbounded in this example.

---

## What is still not here

* **No physical measurement.** Unchanged from stage one: see
  [INSTRUMENT.md](INSTRUMENT.md).
* **No triangulated meshes.** Surfaces are parametric. A mesh needs a discrete
  curvature estimator, which is a different problem with its own error analysis.
* **No obstacles, no cut locus search.** Leaving the declared parameter domain
  *is* now detected — `ParametricSurface.chart_validity` reports domain
  containment, orthogonality and scale ratio, `require_valid_chart` refuses to
  continue without it, and every envelope carries a `surface-chart-validity`
  check. What is absent is anything beyond the chart: obstacles, fixtures, and
  the cut locus. `RouteConstraints.min_boundary_clearance` exists and is left
  unbounded in the example.
* **Full envelope only to first order.** `Phi(s)` maps a starting pose error to
  a downstream one linearly. Stage one measured exactly where that stops being
  the truth; the same limit applies here and has **not** been re-measured on
  the varying-curvature cases. A path-specific nonlinear validity region is the
  next thing this stage needs.
* **Observability is a threshold schedule, not an integral.** Tracking asks
  whether `rho(s)` clears declared levels sample by sample. The accumulated
  observability Gramian `W(s0, s1) = ∫ Φᵀ Hᵀ R⁻¹ H Φ ds`, which would say how
  much information a whole span carries rather than how much any one sample
  does, is not computed.
* **`R` is diagonal and stationary.** The observation noise covariance is
  declared constant and uncorrelated along the path. Real scanner noise is
  neither. Correlated and block-structured `R` are not modelled.
* **Fixed step, no adaptive error estimate.** Step size is declared, not
  chosen. There is no step-doubling estimate and therefore no per-run numerical
  error budget; what exists instead is self-convergence measured after the
  fact, and the invariants.
