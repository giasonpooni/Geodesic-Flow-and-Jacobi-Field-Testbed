# Methods

## Transverse first variation

For a unit-speed surface geodesic, a transverse scalar Jacobi component
satisfies

```text
j''(s) + K(s) j(s) = 0.
```

The runtime propagates two fundamental solutions:

```text
a(0) = 1, a'(0) = 0     initial lateral-position sensitivity
b(0) = 0, b'(0) = 1     initial-heading sensitivity
```

An initial transverse offset `delta_p` and small heading change
`delta_alpha` therefore produce the first-order field

```text
j(s) = a(s) delta_p + b(s) delta_alpha.
```

Written as a map rather than a sum, the same statement is

```text
         | a(s)   b(s) |            | delta_p(s)     |          | delta_p(0)     |
Phi(s) = |              |     and   |                | = Phi(s) |                |
         | a'(s)  b'(s)|            | delta_alpha(s) |          | delta_alpha(0) |
```

which is what the runtime actually propagates. The matrix form is needed for
anything that must carry a *distribution* rather than a bound: a starting-pose
covariance `C` becomes `Phi C Phi^T`.

For constant Gaussian curvature, the package evaluates these bases
analytically. For a varying declared curvature profile it integrates both
bases with fourth-order Runge--Kutta on the caller's arclength grid. For a
real parametric surface it integrates them together with the geodesic itself,
since `K` is then a function of the moving point.

## The Wronskian

The equation has no first-derivative term, so its Wronskian is conserved:

```text
det Phi(s) = a(s) b'(s) - a'(s) b(s) = 1     for every s.
```

This is an identity, not an approximation, and the solver never enforces it, so
its drift is a free diagnostic of the integration — available on any surface,
at any arc length, with no reference solution.

For a constant `K` and a fixed step the drift is known exactly. The one-step
map is a polynomial in `hA` with `A^2 = -K I`, and the determinant of a product
is the product of determinants, so `det Phi = d^n` where

| method | one-step `d` | drift order in `h` |
| --- | --- | --- |
| Euler | `1 + K h^2` | 1 |
| midpoint | `1 + K^2 h^4 / 4` | 3 |
| RK4 | `(1 - Kh^2/2 + K^2h^4/24)^2 + K(h - Kh^3/6)^2` | 5 |

## Focus points

A zero of `b` away from the start is a conjugate point of the heading
variation: geodesics that left at different angles meet again. A zero of `a` is
the corresponding focus of the lateral variation, and it is generally somewhere
else — on a unit sphere `b` vanishes at `s = pi` and `a` at `s = pi/2`.

At a focus the first-order separation is small, which reads as low sensitivity,
but the map from starting pose to endpoint is ill conditioned there, local
minimality can be lost, and a family of paths crowds rather than covers. Low
amplification and robustness are not the same property, and a route score that
uses only the first will select for the second's failure.

## Deterministic tolerance envelope

Given nonnegative starting bounds `epsilon_p` and `epsilon_alpha`, the default
worst-case envelope for one path is

```text
e(s) = |a(s)| epsilon_p + |b(s)| epsilon_alpha.
```

This is not a probability distribution. An RSS option is available only when
the caller deliberately chooses that combination rule.

The default relative factor is two because two neighbouring paths may deviate
in opposite directions by the full one-path envelope. A different factor must
be an explicit process-model choice.

## Manufacturing interpretation

The nominal separation between neighbouring courses is propagated as another
Jacobi variation. Starting-pose tolerance expands that nominal value into a
minimum and maximum possible separation. Course width then yields deterministic
gap and overlap bounds.

This first slice does not model compaction, tow steering limits, material
shear, adhesion, thermal distortion, or machine tracking dynamics.

## The observation model

The transfer map propagates a starting-pose error. No instrument reports that
error. It reports some linear functional of it, corrupted by its own noise:

```text
delta_z(s) = Phi(s) delta_z0
y(s)       = H(s) delta_z(s) + eta(s)
Cov(y)     = H Phi C0 Phi^T H^T + R
```

`H` is the observation mode made concrete — a scanner that reports transverse
deviation only is `[1, 0]`; one that also tracks orientation is the identity —
and `R` is the metrology covariance. `C0` is the starting-pose covariance, or
`diag(dp^2, da^2)` for a tolerance box treated as independent.

### Resolvability, which is not a focus

```text
rho(s) = sqrt(diag(H Phi C0 Phi^T H^T)) / sqrt(diag R)
```

With `H = [1, 0]` and uncertainty in heading alone this is exactly
`|b(s)| sigma_alpha / sigma_measurement`. Below one, the instrument cannot tell
two admissible starting headings apart.

This is the quantity a route constraint must use. `|b|` has units of length per
radian, so a threshold on it — the repository briefly used 0.25 — is specific
to one part size and one angle unit: the same physical situation drawn at twice
the scale, or stated in degrees, changes the verdict. `rho` is dimensionless
and does not, and that invariance is a declared check.

**It is still a different quantity from a focus**, and the two must not be
collapsed into one name. A geometric focus is a zero of a transfer column: a
property of the surface and the path, present whatever instrument is pointed
at it. Low resolvability is a property of the whole chain. They come apart in
both directions, and both directions are measured:

| | focus present? | `rho` above the acquire threshold? |
|---|---|---|
| plate, heading tolerance shrunk 10⁴× | no — `b(s) = s` has no zero | no: peaks at 0.0042 |
| spherical cap, 0.01 before the conjugate point | yes, at `s = pi` | yes, for a metrology sigma of 1.0 µm on a 300 mm coupon |

So a tight tolerance produces low `rho` with no focus anywhere, and a sharp
enough scanner stays resolvable beside a real conjugate point, because `|b|`
near a focus is small but not zero. `results.focus_versus_resolvability`
carries both, with `surface-unresolvable-without-a-focus` and
`surface-resolvable-beside-a-focus` as declared checks.

Both outputs therefore survive: `focus_points` on the transfer map is the
geometric claim, which no scanner can move; the tracking outcome is the
instrument's, and it is the one the route constraint uses. Where the two agree
— as they do for every candidate in the torus scan — that is corroboration
under one configuration, which is what the check is named for, and not an
identity.

## Chart validity

A parameterisation is not the surface. Each surface declares a `Chart`: the
parameter region where its chart is a chart, plus a conditioning floor. A chart
fails two independent ways, and they need two numbers rather than one.

**The coordinate curves become parallel.** Measured by `chart_orthogonality`,
the ratio of singular values of the *direction-normalised* Jacobian, which with
`cos t = |F| / sqrt(EG)` is

```text
sqrt((1 - cos t) / (1 + cos t))
```

— 1 where the coordinate directions meet at a right angle, 0 where they
coincide. Normalising each column first is the whole point: this then depends
only on the angle between the directions and is unchanged when `u` and `v` are
rescaled independently.

The obvious alternative, `sqrt(EG - F^2) / max(E, G)`, is **not**, and that is
not a fine distinction. It confuses an anisotropic chart with a degenerate one:
reparameterise the torus by `v -> v/3` and it falls from 0.3339 to 0.1113 — by
exactly the factor of three, since `r_v -> 3 r_v` puts a 3 in the numerator and
a 9 in the denominator — for a chart that has not become any less regular.
`chart_orthogonality` moves by 1.3e-12 across the same reparameterisation.
The repository used the wrong one; both halves of that statement are declared
checks (`surface-chart-orthogonality-rescaling-invariant` and
`surface-retired-chart-criterion-is-not`).

**A coordinate direction collapses**, as longitude does at a sphere's pole.
Measured by `chart_scale_ratio`: the shorter of `|r_u| u_scale` and
`|r_v| v_scale` against the chart's declared `reference_length`. This one
genuinely needs declared scales, because "short" means nothing on its own.

`chart_conditioning` is the worse of the two, and it is the quantity the floor
applies to.

A start the chart cannot represent is refused. A path that leaves the valid
region is reported, with where and why, rather than being allowed to become
NaNs or — worse — a plausible-looking envelope computed from a degenerate
metric.

## Derivatives

A surface may supply analytic first and second derivatives; a surface given as
`r(u, v)` alone gets them by central differences. The differencing step is
**relative** to the parameter scale, because an absolute one is a scale defect
that second derivatives amplify, and `derivative_convergence` reports how much
the curvature moves when the step is doubled — so a surface supplied as a black
box states how far it can be trusted instead of having it assumed. For scanned
geometry, fit a smooth surface and difference the fit; do not difference raw
points.

## Observation modes

A prediction is comparable with a measurement only if both are the same
quantity. Three appear here and they differ at second order in the
perturbation -- the same order as the first-order model's own failure:

| mode | quantity | implemented |
| --- | --- | --- |
| `intrinsic-surface-distance` | distance measured *in the surface*; what a Jacobi field predicts | yes |
| `ambient-euclidean-chord` | straight-line distance through space between the same two points | yes |
| `scanner-reconstructed-chord` | the ambient chord after calibration, registration and fitting | no |
| `camera-image-residual` | residual in image coordinates, before reconstruction | no |

The gap between the first two is `kappa_n^2 sn_K(s)^2 / 6` in the relative
`eps^2` coefficient, `kappa_n` being the normal curvature transverse to the
path. Every recorded comparison names its mode; an untagged one is not
evidence.

## Inspection interpretation

Inspection uses the same path-separation bound. The maximum separation is
compared with the declared sensor swath, while one-path uncertainty is compared
with a declared cross-track-error limit.

The result is a path-geometry assessment. It does not claim a probability of
detection or model the sensor's signal response.

## Reference validation

For constant curvature, the exact bases are

```text
K > 0: a = cos(sqrt(K)s),  b = sin(sqrt(K)s)/sqrt(K)
K = 0: a = 1,              b = s
K < 0: a = cosh(sqrt(-K)s), b = sinh(sqrt(-K)s)/sqrt(-K)
```

The test suite compares the numerical propagation against these expressions.
It also compares `|b(s) delta_alpha|` with the exact finite distance between
two constant-curvature geodesic rays and confirms convergence as the angular
perturbation tends to zero.
