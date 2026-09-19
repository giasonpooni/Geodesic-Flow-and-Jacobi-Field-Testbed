# Development workflow

- The constant-curvature experiment is implemented and verified. Keep the
  README limited to what the committed report actually shows.
- Every quantitative claim in the README or the docs must correspond to a
  declared check in `src/geojac/experiment.py`. If a number is worth stating,
  it is worth a threshold.
- After changing anything numerical, regenerate the tracked artefacts:
  `python examples/run_experiment.py --update-committed`. `pytest` fails if the
  committed report no longer describes the current code.
- `ruff check .` and `pytest -q` both pass before a commit. CI runs them on
  Python 3.10-3.13 and re-runs the experiment from a clean checkout.
- Do not add a third finite-difference helper, a local GPU stack, or a compiled
  backend. Fixed-step methods of known order are what make the convergence
  measurement legible.
- Stage two (parametric surfaces of varying curvature) is implemented. Anything
  added there must stay anchored: the general machinery has to keep reproducing
  the constant-curvature closed forms, and that anchor is a declared check, not
  a comment.
- A new surface needs only `r(u, v)`; supply analytic derivatives too if the
  surface has them, since the finite-difference fallback costs about eight
  significant figures and `experiment_surfaces.py` measures that cost.
- Next pieces, in order: triangulated meshes (needs a discrete curvature
  estimator with its own error analysis), then the physical bench. See
  `docs/INSTRUMENT.md` for the full roadmap and for what is still unproven.
