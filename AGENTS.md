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
- Next foundation piece is surfaces of varying curvature, where there is no
  closed form to check against and the reference must come from a converged
  fine-step solution. See `docs/INSTRUMENT.md` for the roadmap.
