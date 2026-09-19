# Compiled stack

This repository is pure Python and NumPy, with Matplotlib for the figure only.
It does not own a CUDA or Rust runtime, and does not need one: the whole
experiment runs in a few seconds.

A Jacobi field is a first-order variation, not a finite distance. Where that
distinction is measured rather than asserted, see `docs/EXPERIMENT.md` §3.

If a future stage consumes JSPT local maps or the later Rust gate, it enters
through a new module with its own report contract. Do not add a third
finite-difference helper or a local GPU stack.
