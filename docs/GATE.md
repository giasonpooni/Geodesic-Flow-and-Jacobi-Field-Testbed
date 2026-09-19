# Compiled stack

This repository owns a tested NumPy runtime: geodesic flow, Jacobi transfer
maps, and the application contracts built on them. Matplotlib is used for the
figures and nothing else. There is no CUDA or Rust runtime here and there is
no case for one -- both experiment stages run in under a minute.

A Jacobi field is a first-order variation, not a finite distance. The
constant-curvature stage checks that distinction against the exact finite
separation of geodesic rays and reports where it fails; see
`docs/EXPERIMENT.md` section 3.

Two standing prohibitions:

- **No second solver.** One integrator, one transfer map. `geodesic_testbed.jacobi`
  delegates to `engine.integrators`; a third finite-difference helper is a
  defect.
- **No local GPU stack.** If a future stage consumes JSPT local maps or the
  later Rust gate, it enters through a new module with its own report
  contract.
