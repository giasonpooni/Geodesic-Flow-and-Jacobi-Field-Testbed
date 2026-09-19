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
