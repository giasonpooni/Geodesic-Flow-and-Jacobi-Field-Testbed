# Geodesic Flow and Jacobi-Field Testbed

Plane, sphere, and hyperbolic-plane experiments comparing numerical
trajectories with analytical first-order variation.

This is the **second** project in the portfolio

> Computational Geometry, Geodesic Dynamics, and Invariant Representations

The first build is the [Flat-Torus Moduli and Geodesic Explorer](https://github.com/giasonpooni/Flat-Torus-Moduli-and-Geodesic-Explorer).
That repository is where lattice shape, closed-loop lengths, and modular
equivalence live. This repository is the numerical companion: how a small
change in a geodesic's initial conditions affects its trajectory.

## Planned first experiment

For unit-speed geodesics on constant-curvature surfaces, a transverse
Jacobi component satisfies

```text
j''(s) + K j(s) = 0,    j(0) = 0,    j'(0) = 1
```

with reference solutions `j(s) = s` (K = 0), `sin s` (K = 1), and
`sinh s` (K = -1). The testbed should compare numerically traced nearby
geodesics against those predictions while varying step size and
perturbation magnitude.

Two boundaries stay explicit:

- A Jacobi field is a first-order variation, not the exact finite
  distance between trajectories.
- Finding a geodesic does not establish that an arbitrarily long segment
  is globally shortest.

## Status

Scaffold only. Implementation starts after the torus first release is
in use. Do not read this README as a claim of implemented capability.
