# Runtime boundary

This repository implements a NumPy runtime for geodesic flow, Jacobi transfer
maps and application contracts. Matplotlib supplies figures. There is no CUDA,
Rust or compiled numerical backend.

A Jacobi field is a first-order variation, not a finite distance. The
constant-curvature experiment checks this distinction against exact finite
separations and reports its numerical limits; see
[EXPERIMENT.md](EXPERIMENT.md).

There is one integrator and one transfer map. `geodesic_testbed.jacobi` delegates
to `engine.integrators` and `engine.transfer`; preserve that shared implementation
instead of introducing divergent copies.
