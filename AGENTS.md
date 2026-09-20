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
- **A number the record carries is a number something checks.** The path
  geometry is not decoration: the Darboux triad is validated on construction
  (unit, orthogonal, `transverse = normal x tangent`) and the two normal
  curvatures are held to Euler's theorem, `kappa_n(along) + kappa_n(across) =
  2H`, against a mean curvature computed from the second fundamental form -- on
  every surface, with no closed form required. Adding a field to the contract
  without a declared check on it is how a contract starts lying.
- **An error estimate that understates the error is worse than none**, because
  it is acted on. `estimate_convergence` halves the step and Richardson
  -extrapolates per quantity -- position, transfer, curvature, focus,
  covariance, which do not converge together -- and the surfaces experiment
  checks it against the closed forms where one exists rather than trusting it.
  It costs a second integration, so it is never the default: a record whose
  convergence says `not-established` has not paid for one, which is a true
  statement and not a missing feature.
- **The record is the whole of the outward interface.** This runtime is a
  parallel computational substrate that an instrument may consume, not a module
  inside an instrument workbench. `geodesic_testbed.boundary` presents the
  contract on its own, `docs/BOUNDARY.md` states it, and the record carries the
  seven things a consumer cannot reconstruct: units, frame, arclength grid,
  covariance, provenance, calibration IDs and observation mode -- and, for a
  record built from a surface, the path itself: positions, the frame as
  vectors, both normal curvatures, the chart validity, the path type and the
  convergence estimate. Anything a downstream system needs goes in the record;
  nothing else is public.
- **The one-way rule is checked, not intended.** `boundary.SUBSTRATE` may not
  import `boundary.INSTRUMENT_FACING`, and `boundary.CONTRACT` may import
  neither -- `record` reaches `transfer` for the map it wraps and nothing else,
  and `engine/contract.py` imports only the standard library and NumPy.
  `tests/test_boundary.py` reads this out of the syntax tree. A new module goes
  in one of the three lists.
- **A field that may say no is never silently filled in.** Covariance,
  calibration and validity each have an undeclared state, and undeclared is the
  correct answer for a record computed from an analytic surface.
  `propagate_declared_covariance()` raises rather than substitute a `C0`; two
  unbound calibration bindings do **not** agree, because absence of a
  calibration is not evidence of a shared one. Converting a tolerance box to a
  covariance records its `coverage_factor`: a box is a bound and a covariance
  is a distribution, and the convention between them is not a silent choice.
- The version is declared once, in `engine/contract.py`. `pyproject.toml`,
  `geodesic_testbed.__version__` and every record's `producer_version` read it,
  and a test holds the three together. A record carries no timestamp and no
  hostname: both would make two runs of the same computation differ, and a
  report that cannot be regenerated and compared is not worth shipping.
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
- **A path that leaves its chart is stopped, not cleaned up afterwards.** Every
  surface declares a `Chart` and a conditioning floor; `require_valid_chart`
  refuses a start the parameterisation cannot represent, and `integrate_paths`
  defaults to `on_chart_exit="truncate"`, which runs `integrate_guarded` so the
  right-hand side is never evaluated past the exit. Truncating the output
  afterwards removes the invalid samples and not the arithmetic that produced
  them, and on a degenerate chart that arithmetic overflows. `"report"` is for
  the experiments that measure the exit itself, and is the only policy allowed
  to integrate past it. `PathEnvelope.chart` records what happened either way,
  and `as_transfer_record()` refuses an envelope with invalid samples: a record
  carries no chart and cannot warn a consumer.
- **Every float in a committed artefact is canonicalised**
  (`engine/canonical.py`, twelve significant digits, no signed zero) before it
  is serialised or hashed. A content hash that moves because a BLAS reduction
  summed in a different order is not an identity. Rounding is relative, so a
  1e-16 residual is still recorded as 1e-16, and no declared threshold is
  anywhere near the floor. A consistency check on *reported* values may
  therefore not demand better than `10^-(CANONICAL_DIGITS - 1)`. Anything a
  report names -- a route label included -- must be built so that it cannot
  move on the last bit: `heading_label` uses a decimal place because the scan's
  headings land exactly on the `:.0f` rounding boundary.
- **An invariant is checked directly, never through a decomposition.**
  `det Phi = a b' - a' b` is two products and a subtraction; reading it off
  `sigma_1 sigma_2` instead measures the SVD's conditioning on the tolerance
  box, which on a 10^4 aspect ratio is three orders of magnitude worse than the
  quantity being tested. Report the decomposition if it is interesting; do not
  let it carry the check.
- **Declared evidence is validated, not merely present.** A limit that admits
  nothing (zero, negative, non-finite) is refused at declaration; a boundary
  array must be finite and on the record's own arclength grid, or it pairs
  clearances with the wrong arc lengths; a covariance goes through the single
  `contract.validated_covariance`, because two implementations of "is this a
  covariance" eventually disagree and the laxer one wins. An observation mode
  and a domain are checked **together**: each is real on its own, and the pair
  is what is false.
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
