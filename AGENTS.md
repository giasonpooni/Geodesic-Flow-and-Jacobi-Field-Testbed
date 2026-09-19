# Development workflow

- Maintain Curved-Surface Geodesic Sensitivity as one project. `main` carries
  the application contracts; implementation work lands on a branch and is
  reconciled with `main` before it is tagged.
- `geodesic_testbed` is the public API: tolerances, manufacturing and
  inspection assessments, reports. `geodesic_testbed.engine` is the verified
  numerical core underneath it. Application code imports the former; only the
  engine and its own tests import the latter.
- There is one integrator and one transfer map. `geodesic_testbed.jacobi`
  delegates to `engine.integrators` and `engine.transfer` rather than carrying
  a second copy, and a new solver anywhere is a defect, not a feature.
- Every quantitative claim in the README or the docs must correspond to a
  declared check in `engine/experiment.py` or `engine/experiment_surfaces.py`.
  If a number is worth stating, it is worth a threshold.
- After changing anything numerical, regenerate the tracked artefacts:
  `python examples/run_experiment.py --update-committed` and
  `python examples/write_reference_report.py`. `pytest` fails if a committed
  report no longer describes the current code.
- `ruff check .` and `pytest -q` both pass before a commit. CI runs them and
  re-runs both experiment stages from a clean checkout.
- Anything added to stage two must stay anchored: the general machinery has to
  keep reproducing the constant-curvature closed forms, and that anchor is a
  declared check, not a comment.
- A new surface needs only `r(u, v)`; supply analytic derivatives too if the
  surface has them, since the finite-difference fallback costs about eight
  significant figures and `experiment_surfaces.py` measures that cost.
- Every comparison between a prediction and a measurement names its
  observation mode (`engine/observation.py`). An in-surface distance and an
  ambient chord differ at the same order as the effect being measured, so an
  untagged comparison is not evidence.
- Repository boundaries. Mesh work -- triangulated surfaces, discrete
  curvature estimators, mesh path convergence -- belongs in the Intrinsic
  Surface Geodesics Testbed, not here. This runtime should eventually consume
  a versioned path artefact from it (sampled position, tangent, curvature,
  frame, units, provenance, uncertainty) rather than growing a mesh solver.
  Covariance representations and SPD geometry belong in the Covariance
  Geometry Testbed; downstream decisions belong in the Construction State
  Estimator.
- No CUDA stack, no Rust gate, no compiled backend. Fixed-step methods of
  known order are what make the convergence and invariant measurements
  legible.
