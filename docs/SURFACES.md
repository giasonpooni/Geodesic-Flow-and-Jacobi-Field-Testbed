# Stage two: surfaces where the curvature varies

**One question:** the constant-curvature stage could check every number against
a closed form. A real workpiece has none. What replaces it?

Produced by `python examples/run_experiment.py --stage surfaces` and recorded
in [`validation/report-v2-surfaces.json`](../validation/report-v2-surfaces.json):
**47 declared checks, 0 failed.**

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
j'' = -K(u, v) j
```

The two halves cannot be separated any more: the Jacobi equation needs `K` at
the moving point, so the six-component state `(u, v, u', v', j, j')` is
integrated as one system.

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

| case | must reproduce | `max ǀj - referenceǀ` | speed drift | `K` error |
|---|---|---|---|---|
| plate | `s` | 1.1e-13 | 0 | 0 |
| rolled-sheet | `s` | 1.1e-13 | 0 | 0 |
| spherical-cap | `sin s` | 4.4e-13 | 1.6e-12 | 8.9e-16 |
| pseudosphere | `sinh s` | 1.2e-14 | 9.3e-15 | 8.9e-16 |

The rolled sheet is the one worth pausing on. A cylinder is visibly curved and
its geodesics are helices, but it is intrinsically flat, and the solver returns
exactly the plate's answer — to `1e-13`, from a completely different
parameterisation. Rolling a flat sheet onto a drum does not change how far an
aiming error is carried.

### 2. Two independent routes to the same field

The Jacobi field is obtained twice: by integrating `j'' + K j = 0` along the
path, and by central-differencing the flow itself in the initial heading. The
second never touches the Jacobi equation. Their relative difference is second
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

**This matters for hardware.** A camera or a laser scanner measures chords in
space; the Jacobi field is a distance in the surface. On the pseudosphere the
difference is 2.5 times the intrinsic effect. Any bench that compares a
measured separation with `ε j(s)` has to account for it or it will report a
model failure that is really an instrument definition.

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

## The decision it supports

Twenty-four candidate starting headings from one point on the torus, each
flowed for the same path length of 6, ranked by the worst `ǀjǀ` anywhere along
the path:

| | heading | worst `ǀjǀ` | mean `K` along the path | focus |
|---|---|---|---|---|
| most tolerant | 97° | 1.74 | +0.325 | `s = 5.50` |
| least tolerant | 15° | 36.90 | -0.328 | none |

A factor of **21** between the best and worst choice, from the same start and
the same path length. The reason is visible in the last two columns: the
tolerant heading stays on the positively curved outer region, where
neighbouring paths refocus and an aiming error is bounded; the intolerant one
wanders onto the negatively curved inner region, where it grows.

That is the instrument's point. Curvature along a path is not a diagnostic to
look at afterwards — it is something to choose.

---

## What is still not here

* **No physical measurement.** Unchanged from stage one: see
  [INSTRUMENT.md](INSTRUMENT.md).
* **No triangulated meshes.** Surfaces are parametric. A mesh needs a discrete
  curvature estimator, which is a different problem with its own error analysis.
* **No boundaries, no obstacles, no cut locus search.** A geodesic that leaves
  the chart is not detected; the domains here are chosen so that none does.
* **Full envelope only to first order.** `± ε j(s)` is the first-order tube.
  Stage one measured exactly where that stops being the truth; the same limit
  applies here and has not been re-measured on the varying-curvature cases.
