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

For constant Gaussian curvature, the package evaluates these bases
analytically. For a varying declared curvature profile it integrates both
bases with fourth-order Runge--Kutta on the caller's arclength grid.

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
