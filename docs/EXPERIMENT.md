# The experiment

**One question:** on a surface of constant curvature, how far does a nearby
geodesic drift from a nominal one — and over what range is the first-order
(Jacobi) answer to that question actually correct?

Everything below is produced by `examples/run_experiment.py` and recorded in
[`validation/report-v1.json`](../validation/report-v1.json) with a pass/fail
threshold attached to each claim.

---

## 1. One chart for three surfaces

The three simply connected constant-curvature surfaces are carried in a single
ambient representation: `R^3` with the symmetric bilinear form

```text
<x, y>_K = x0 y0 + x1 y1 + K x2 y2
```

| K | form | surface | base point | geodesic |
|---|---|---|---|---|
| `+1` | Euclidean | `<x,x> = +1`, the unit sphere | `(0,0,1)` | great circle |
| `0` | degenerate in `x2` | the affine plane `x2 = 1` | `(0,0,1)` | straight line |
| `-1` | Minkowski | `<x,x> = -1`, upper sheet of the hyperboloid | `(0,0,1)` | hyperbola |

Restricting to `K ∈ {0, +1, -1}` costs nothing: rescaling the metric by `λ²`
rescales curvature by `λ⁻²`, so every constant-curvature surface is one of
these three in a suitable unit of length. All lengths below are in units of
the radius of curvature `R`.

Define the generalised sine and cosine as the two solutions of the very
equation the experiment is about:

```text
sn_K'' + K sn_K = 0,   sn_K(0) = 0, sn_K'(0) = 1     ->   s,  sin s,  sinh s
cn_K'' + K cn_K = 0,   cn_K(0) = 1, cn_K'(0) = 0     ->   1,  cos s,  cosh s
```

Then the whole of the geometry is one formula per line, valid for all three
curvatures at once:

| object | formula |
|---|---|
| geodesic equation | `x'' = -K <x', x'>_K x` |
| exponential map | `exp_p(sv) = cn_K(s) p + sn_K(s) v` |
| Pythagoras | `cn_K² + K sn_K² = 1` |
| ambient chord | `<q - p, q - p>_K = 4 sn_K(d/2)²` |
| distance | `d = 2 sn_K⁻¹(chord / 2)` |

The last two lines matter numerically. Writing distance through the chord keeps
full precision at small separations, where the textbook `arccos` and `arccosh`
forms lose most of their digits — and small separations are the entire subject
here. The sphere needs one extra ingredient at the far end of its range, where
`arcsin` saturates near the antipode: the companion identity
`K <q + p, q + p>_K = 4 cn_K(d/2)²` gives `d = 2·atan2(chord, sum)`, which is
well conditioned across the whole of `[0, π]`.

Notice that `sn_K(s)` appears in the exponential map as the coefficient of the
initial direction. That is not a coincidence — it is the reason a Jacobi field
and the spreading of geodesics are the same object.

---

## 2. Convergence of the two solvers

Three explicit fixed-step integrators of formal order 1, 2 and 4 (Euler,
midpoint, RK4) are run over a geometric ladder of nine step sizes on `s ∈ [0,2]`,
and the maximum error over the grid is fitted against `h` in log-log space.

Measured order (committed report):

| method | formal | Jacobi ODE, `K=+1` | Jacobi ODE, `K=-1` | geodesic flow, `K=+1` | geodesic flow, `K=-1` |
|---|---|---|---|---|---|
| euler | 1 | 1.021 | 0.968 | 0.996 | 0.973 |
| midpoint | 2 | 1.960 | 1.979 | 2.007 | 1.968 |
| rk4 | 4 | 3.934 | 3.967 | 3.967 | 4.023 |

Three integrators rather than one is the point: a single method can only show
that *something* converges, while methods of order 1, 2 and 4 separate exactly
as the theory says they must.

Two details keep the fit honest.

* **The roundoff floor.** Once an error reaches double-precision noise it stops
  shrinking, and including those levels flattens the slope. Levels below `1e-12`
  are excluded and the report records how many were dropped and over which
  window the fit was taken.
* **`K = 0` is exact.** On the flat model both the Jacobi solution and the
  geodesic are linear in `s`, so every method — Euler included — reproduces them
  to roundoff (`< 1e-13`). There is no order to measure, and the report says so
  (`regime: exact-to-roundoff`) rather than fitting noise.

The geodesic sweep also records how far the numerical state drifts off the unit
tangent bundle. At the finest step RK4 holds the constraints to `4e-14` on the
sphere and `2e-13` on the hyperbolic plane; no projection or re-normalisation is
applied anywhere, so this is the raw integrator.

---

## 3. Where the first-order variation stops working

Two unit-speed geodesics leave the same point, their initial directions
separated by an angle `ε`. Because the two exponential images differ only in
the direction factor, the ambient chord computation collapses to a single law
of cosines covering all three curvatures:

```text
sn_K(d/2) = sn_K(s) · sin(ε/2)
```

Expanding it gives both the first-order prediction and the first thing it gets
wrong:

```text
d(s; ε) = ε · sn_K(s) · [ 1 - cn_K(s)² ε² / 24 + O(ε⁴) ]
```

So the Jacobi field `ε·sn_K(s)` is in relative error by `cn_K(s)²ε²/24` — a
prediction with a *coefficient*, not just a slope. Sweeping `ε` over eight
decades and fitting recovers both:

at `s = 1`:

| K | fitted exponent | fitted coefficient | `cn_K(1)²/24` | valid to 1e-3 | valid to 1e-6 |
|---|---|---|---|---|---|
| `0` | 2.0000 | 4.166534e-02 | 4.166667e-02 | 8.88° | 0.281° |
| `+1` | 2.0000 | 1.216475e-02 | 1.216361e-02 | 16.43° | 0.519° |
| `-1` | 2.0000 | 9.920410e-02 | 9.921241e-02 | 5.75° | 0.182° |

The coefficients agree with theory to between 3e-5 and 1e-4 relative. The last
two columns are the practical answer to "how big may `ε` be?" and they differ
by a factor of three between the sphere and the saddle at the same path length.

### The other failure mode

The same sweep also measures the separation by flowing both geodesics
numerically. Something initially surprising happens: with the whole fan of
perturbed geodesics advanced by **one shared step sequence**, the separation
agrees with the closed form to `3e-13` relative *at every `ε` down to 1e-8* —
far better than either trajectory's own error. The truncation error is
common-mode and cancels in the difference.

Flow the two geodesics at **different resolutions** and it does not cancel. The
sweep does that too, and the resulting error grows like `1/ε`: the same coarse
integrator that resolves a 1-radian perturbation to 5e-7 knows nothing at all
about a 1e-8 one. The measured crossover at `s = 1`, for a 0.1%-accurate
answer, is `ε ≈ 1.8e-5` on the sphere and `1.0e-5` on the hyperbolic plane; on
the flat model the flow is exact at any step size and there is no floor.

One honest breakdown is recorded rather than hidden. On the hyperbolic plane
the ambient form is indefinite, so once two flowed points have drifted off the
hyperboloid by more than they are apart, their difference stops being spacelike
and there is no distance left to extract. `SpaceForm.distance` clips to the
closest legal value; `SpaceForm.chord_squared` exposes the raw quantity, and
the report counts how many samples went timelike (5 to 7 of 33, all inside the
noise floor).

---

## 4. A zero of the Jacobi field

On the sphere `sn_K(s) = sin s` vanishes at `s = π`. The RK4-integrated Jacobi
field finds that zero at `s = 3.141592653589791`, in error by `2.2e-15`.

That zero is a conjugate point, and past it the geodesic stops being the
shortest path. The experiment checks this on the flowed great circle rather
than asserting it:

| `s` | distance back to the start | still minimising? | excess length |
|---|---|---|---|
| `π/2` | 1.5707963267948983 | yes | -1.8e-15 |
| `π` | 3.141592653589789 | yes | 4.0e-15 |
| `3π/2` | 1.5707963267948977 | **no** | π |
| `1.9π` | 0.314159265358986 | **no** | 5.65 |

The flowed distance reproduces `min(s, 2π - s)` to `7.5e-15` across the full
turn, the excess length past the conjugate point matches `2(s - π)` to
`7.1e-15`, and the circle closes on its starting point at `s = 2π` to `7.2e-15`.

This is the second boundary the project insists on: finding a geodesic does not
establish that an arbitrarily long segment is globally shortest, and here is
exactly where and by how much that fails.

---

## 5. What is deliberately not here

* **No claim beyond the closed forms.** The report's `claim_scope` is
  `numerical-verification-against-closed-form-solutions`. Every number is
  checked against an analytic solution on a surface where one exists.
* **No general surfaces.** Triangulated or parametric surfaces, where the
  curvature varies along the path and there is no closed form to check against,
  are the obvious next step and are not attempted here.
* **No physical measurement.** See [INSTRUMENT.md](INSTRUMENT.md) for what
  would be required to turn this into an instrument, and for the line between
  what is verified here and what is not.
* **No adaptive stepping, no symplectic integrator, no compiled backend.** The
  fixed-step methods are what make the order measurement legible.
