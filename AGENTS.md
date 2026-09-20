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
  untagged comparison is not evidence. Support is recorded **per domain**
  (`constant-curvature`, `parametric-surface`, `physical-instrument`): a mode
  that is exact on a model space can be unavailable on a real surface, and a
  single "implemented" flag would have to lie about one of them.
- `TransferRecord` (`engine/record.py`) is the interchange type. Anything that
  produces a transfer map presents one, and anything that consumes a path takes
  one. New consumers accept `to_transfer_record(source)`, never a concrete
  producer type.
- **No dimensionful thresholds in a criterion.** `|b|` is a length per radian:
  a cutoff on it is specific to one part size and one angle unit. Route
  decisions use quantities that survive rescaling -- the dimensionless
  transfer `S^-1 Phi S`, and resolvability
  `rho = sqrt(diag(H Phi C0 Phi^T H^T)) / sqrt(diag R)`, which asks whether the
  declared instrument can distinguish the starting poses the tolerance admits.
  Dimensionful quantities may be reported; they may not decide.
- **A declared constraint is never skipped.** If its evidence is missing,
  `assess_route` raises. An acquisition schedule with no observation model or
  no starting covariance, or a boundary limit with no boundary data, is an
  error, not a pass: a route must never look feasible *because* the data for a
  limit it has to meet was absent.
- **An event and the knowledge of it are different times.** A sustained window
  is recognised only when it closes, so every tracking outcome carries both
  `acquisition_window_started_at` / `acquisition_declared_at` and
  `loss_started_at` / `track_loss_declared_at`. `processing="causal"` acts on
  the declaration; `offline` may use the retrospective start. Reporting an
  offline schedule as a real-time result is a defect.
- **A geometric focus is not low resolvability.** A focus is a zero of a
  transfer column and belongs to the surface and the path; resolvability
  belongs to the surface, the tolerance and the instrument together. A tight
  tolerance gives low `rho` with no focus present, and a sharp instrument
  stays resolvable beside a real conjugate point. Report both, never rename
  one as the other, and let only the instrument's answer decide a route.
- **Filter agreement is proved, not asserted.** A comparison against a filtered
  trial takes a `FilteredPrediction` whose identifier, version, operator digest
  and causality all match the record. A boolean cannot tell the declared
  operator from a different one with the same name, which is the case that
  manufactures agreement.
- **Thresholds that decide come from the instrument protocol.** The numerical
  layer reports signal-to-noise, resolvability and margins; it does not decide
  what counts as resolved. A constant like `SNR >= 3` compiled into the
  comparison is a declared limit smuggled into arithmetic.
- A path that leaves its chart must say so. Every surface declares a `Chart`
  and a conditioning floor; `require_valid_chart` refuses a start the
  parameterisation cannot represent, and `PathEnvelope.chart` records where a
  path left the valid region. Silent NaNs, or an envelope computed from a
  degenerate metric, are defects.
- The finite-difference fallback jet uses a step **relative** to the parameter
  scale, and reports `derivative_convergence`. An absolute step is a scale
  defect, and second derivatives amplify it.
- **No filtering inside the solver.** Smoothing belongs to the measurement and
  surface-reconstruction layers, never to `engine/flows.py` or
  `engine/transfer.py`; a filter there would tune the model to the data through
  the model's own machinery. A filter is part of the observation instrument, so
  it changes the observation model: compare `y_f = F H Phi dz0 + F eta` against
  `R_f = F R F^T`, never a filtered measurement against an unfiltered
  prediction with the original `R`. Every `MeasurementRecord` carries the
  filter's identity, version, parameters, causality, group delay, sampling
  rates, the dataset it was tuned on, the rejected-sample mask and the
  outlier rule; see `docs/INDUSTRIAL-PILOT.md` for the prohibitions.
- Repository boundaries. Mesh work -- triangulated surfaces, discrete
  curvature estimators, mesh path convergence -- belongs in the Intrinsic
  Surface Geodesics Testbed, not here. This runtime should eventually consume
  a versioned path artefact from it (sampled position, tangent, curvature,
  frame, units, provenance, uncertainty) rather than growing a mesh solver.
  Covariance representations and SPD geometry belong in the Covariance
  Geometry Testbed; downstream decisions belong in the Construction State
  Estimator.
- Physical validation is `not_started` and stays so until held-out measured
  data agrees. The Wronskian validates internal propagation consistency -- not
  the surface model, not the observation model, not a physical prediction.
- No CUDA stack, no Rust gate, no compiled backend. Fixed-step methods of
  known order are what make the convergence and invariant measurements
  legible.
