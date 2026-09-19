# Compiled stack

This repository now owns a tested NumPy Jacobi-propagation runtime. It does
not own a CUDA or Rust runtime.

A Jacobi field is first-order variation, not finite distance. The implemented
constant-curvature experiment checks that distinction against exact finite
geodesic-ray separation. Do not add a third finite-difference helper or a
local GPU stack.
