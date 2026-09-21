# Contributing

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
- **The starting pose is one term in a budget.** `engine/uncertainty.py` holds
  the rest -- surface reconstruction, fixture and datum, calibration transform,
  path registration, sensor noise -- and each carries its *shape*. Most are
  systematic: one unknown felt the same way at every sample, which is a
  rank-one covariance and not a per-sample variance, and confusing the two is
  how a budget comes out wrong by an order of magnitude rather than by a few
  per cent. The total is a sum of matrices; a sum of variances would destroy
  the off-diagonal structure that distinguishes a bias from noise. A budget of
  purely systematic terms is singular and that is correct, not a bug.
- **A physical programme is declared before the data, not after.**
  `engine/campaign.py` holds the coupon stages, their dependencies, the
  perturbation plan and the scales, with status `not-started`. A campaign
  design chosen after seeing the residuals is not a test. Every stage perturbs
  **both** axes -- one that perturbs only the heading measures `b` and says
  nothing about `a` while producing a full set of plots; calibration and
  validation are split by **coupon**, not by run, because two runs share a
  coupon's as-built geometry, its fixturing and its calibration; and the
  comparison uses the **achieved** perturbation, since the gap between
  commanded and achieved is a starting-pose error of exactly the kind under
  test. Conformance is bookkeeping and never reports agreement.
- **No dimensionful thresholds in a criterion.** `|b|` is a length per radian:
  a cutoff on it is specific to one part size and one angle unit. Route
  decisions use quantities that survive rescaling -- the dimensionless
  transfer `S^-1 Phi S`, and resolvability
  `rho = sqrt(diag(H Phi C0 Phi^T H^T)) / sqrt(diag R)`, which asks whether the
  declared instrument can distinguish the starting poses the tolerance admits.
  Dimensionful quantities may be reported; they may not decide.
- **Route selection reports a front, not a score.** Amplification, cross-track
  error, heading error, accumulated observability, boundary clearance and path
  length stay separate, each divided by its own declared limit so they are
  comparable. `pareto_front` returns what nothing else beats on every one at
  once; `weighted_cost` collapses that only with weights the caller declared,
  which must sum to one and must name every objective -- an omitted weight is a
  zero chosen by accident -- and it refuses an objective with no limit, because
  the limit is what keeps the weights from carrying units.
- **Accumulated observability is not sampled resolvability.** `rho(s)` asks
  about one arc length; `W = int Phi^T H^T R^-1 H Phi ds` asks what the whole
  path says, and a route can pass the first everywhere while saying almost
  nothing about one direction of the starting pose. The Gramian is reported
  only in a declared dimensionless scaling -- a tolerance box or a prior
  covariance -- because its entries do not share units. Its *density* is scale
  invariant and the accumulated figure is not, and both are declared checks.
- **Boundaries are computed from the declared part**, not supplied as an array
  that might be on the wrong grid. An obstacle clearance uses the ambient
  chord, which is never longer than the in-surface distance, so it errs towards
  rejecting a route rather than towards clearing one.
- **Generate routes over both degrees of freedom.** A heading fan exercises the
  `b` column and a set of offset courses exercises `a`, and the two columns
  focus in different places. An offset course starts where a geodesic
  perpendicular to the seed reaches the declared spacing, with its direction
  parallel-transported along it -- stepping linearly in `u` and `v` instead is
  right to first order and wrong by `O(K spacing^2)`, which is the size of the
  effects measured here.
- **Every event is located between samples.** Threshold crossings in
  `evaluate_tracking` are interpolated and a track loss is declared exactly one
  tolerated length after the excursion began, for the same reason focus
  locations are Hermite-refined: an event snapped to the next sample is known
  only to the sample spacing, and the rounding is one-sided.
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
- **A prediction carries its transformations, not a label for them.** Between
  `Phi dz0` and an ambient chord sit two second-order corrections, and between
  that and an instrument output sit `H` and the filter. `engine/prediction.py`
  makes each one a named step on a `Prediction` that records its whole chain,
  and the chain runs forward only -- a chord that could be stepped back into an
  intrinsic distance is the relabelling the module exists to prevent. A
  transformation that is not computable is *declared* missing on the
  prediction, never omitted silently: `chord_from_tangent` applies the chord
  correction on a varying-curvature surface and says in `chain`, `note` and
  `extra["intrinsic_correction"]` that the finite-separation one was not
  applied.
- **A comparison statistic is not a scalar.** A maximum absolute residual
  throws away the covariance, cannot be compared between instruments, and
  cannot be held to any threshold that is not already in the measurement's
  units. `residual_statistics` returns the full residual covariance, the
  whitened residual and the chi-square. A filter correlates arc lengths, so a
  filtered comparison gets one covariance over every scalar residual rather
  than a per-sample stack; keeping the diagonal blocks alone would treat as
  independent exactly the samples the filter made dependent.
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
  summed in a different order is not an identity.
- **`tools/e2e.py` is the end-to-end gate, and it varies something each cycle.**
  Hash seed, BLAS thread count and working directory, because a single run
  agrees with itself by construction and cannot see a dependence on any of
  them. It drives the shipped entry points as subprocesses, not imports -- an
  imported command shares this process's interpreter and directory, which are
  two of the three things being varied. `--baseline committed` compares with
  the tracked artefacts and is meaningful only where they were generated;
  `--baseline self` compares every cycle with the first and is what CI runs.
- **A content hash is an identity within one environment and nowhere else.**
  It answers "has anything changed since this was last run here", which is what
  `tools/e2e.py` asserts across a hundred cycles of varying hash seed, BLAS
  thread count and working directory. It cannot answer "do two numpy builds
  agree", because they do not in the last digits and no rounding rule makes
  them: four builds here give four hashes.
- **Across environments the claim is the verdict, not the value.** Comparing
  values to a tolerance was tried and measured against a GitHub runner: a
  fitted order moved by 3e-6 relative, a `coefficient_relative_error` by 9%,
  and a finite-difference jet value at a step of 1e-6 by a factor of 2.8. The
  last two are a residual and a cancellation-limited probe -- this artefact is
  a report *about* numerical error, so most of its numbers are numerical
  error, and that is the thing two builds of libm disagree about. A tolerance
  loose enough to admit them admits a regression.
  `_assert_same_verdicts` compares every check's pass/fail against its own
  threshold instead, and that is sufficient rather than merely possible:
  every quantitative claim here must have a declared check, so a regression
  large enough to matter flips one. Do not assert hash or value equality in
  the test suite; assert bytes in the end-to-end harness, where the
  environment is fixed. Rounding is relative, so a
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
  outlier rule; see [the measurement contract](docs/MEASUREMENT.md) for the prohibitions.
- Repository boundaries. Mesh work -- triangulated surfaces, discrete
  curvature estimators, mesh path convergence -- belongs in the Intrinsic
  Surface Geodesics Testbed, not here. A geometric input must retain sampled
  position, tangent, curvature, frame, units, provenance and uncertainty.
  Covariance-manifold geometry and downstream decision authority are outside
  this runtime.
- Physical validation is `not_started` and stays so until held-out measured
  data agrees. The Wronskian validates internal propagation consistency -- not
  the surface model, not the observation model, not a physical prediction.
- No CUDA stack, no Rust gate, no compiled backend. Fixed-step methods of
  known order are what make the convergence and invariant measurements
  legible.

Preserve concurrent work and existing validated behavior. Do not force-push
shared history.

## The contracts added in 0.3.0

- **A covariance is admitted or refused, never repaired.** There is one
  validator, `contract.validated_covariance`, and `transfer`,
  `observation_model`, `measurement` and `output_covariance` all delegate to
  it. It tests in correlation coordinates, checks **both** stored triangles,
  and returns the caller's values unaltered. Symmetrising an input before
  testing it is the specific thing it must not do: `[[1, .2], [.1, 1]]` has two
  triangles that disagree -- a caller bug, and the only evidence of it -- and
  averaging them produces a plausible matrix that passes. A congruence is
  validated in its *output* coordinates too, because `A C0 A^T` can amplify a
  tolerated asymmetry in `C0`. Singular is valid; no floor and no jitter.
- **`path-geometry-v1` requires its curvature interpolation.** The field has no
  default in the dataclass, none in `from_dict`, and none in
  `artefact_from_envelope` -- a default there would only move the silent choice
  one layer up. On the saddle the monotone cubic and the piecewise-linear
  interpolant differ by a factor of 580 at the coarsest declared sampling, so
  an artefact that does not say which one it means has not said enough. The
  normal curvatures arrive three at a time with the mean curvature, and Euler's
  theorem is checked on arrival.
- **A validity envelope carries the fit that produced it.** The bound comes
  from an adaptive window over the probe ladder, so the coefficient, the probes
  kept, their observed slopes and every rejected probe with its reason are
  recorded: two ladders can give the same bound from different evidence, and an
  adaptive selection nobody can inspect is a number with a provenance of "trust
  me". The bound is then re-probed at 0.8, 1.0 and 1.2 times itself -- none of
  which took part in the fit -- and the 1.2 point must **exceed** the
  tolerance, or the bound is wherever the ladder stopped rather than where the
  linearisation fails.
- **The validity probe is an independent computational route, not an
  independent implementation.** It never touches the Jacobi equation, but it
  shares the surface model, the geodesic right-hand side and the integrator, so
  their error is common mode. `reference_method` and `reference_samples` are
  declared for that reason: with RK4 the bound is stable to a part in ten
  thousand under step refinement, and with a second-order method at the same
  steps it moves ~2% and does so non-monotonically, which is a noisy fit rather
  than a trend.
- **`Sigma_num` is a bound or a distribution, and it says which.** A Richardson
  estimate of a truncation error is deterministic; calling it a variance
  implies a sampling story that does not exist. It rides in the same matrix
  arithmetic because that is the only way to add it, and `numerical_basis`
  records which it is -- a chi-square against a total containing a
  deterministic bound is conservative by an unknown amount, and `nis()` reports
  `calibrated: false` rather than letting a reader assume otherwise.
- **The two Gramian forms are different objects.** The integral form treats `R`
  as a noise *density* and needs independent samples; the stacked form treats
  it as the covariance of the measurements taken and is the only one that
  admits a correlated `R`. Each reports its `noise_convention`, its scaling and
  basis, the eigenvalues, the numerical rank and the least-observable
  direction -- a condition number alone cannot say how many directions the path
  constrains. A window on the stacked form inverts the submatrix of `R`, never
  a submatrix of `R^-1`.
- **CI runs each piece of work once.** The contract tests run on 3.11, 3.12 and
  3.13 because an interpreter can change their outcome; the `numerical` suite,
  the artefact regeneration, the wheel smoke test and the determinism cycles
  each run once, because an interpreter cannot. Mark a test `numerical` when it
  runs an experiment stage or a perturbation sweep. The marker is for placement,
  not for skipping.
- **The wheel is what gets tested, not the editable install.** The version is
  dynamic, so `pip install -e` and the built wheel exercise different metadata
  paths. CI builds the wheel, installs it into an empty environment and asks it
  what version it is.

- **A differential control states two hypotheses, not one.** The plate and the
  rolled cylinder have the same transfer map and *not* the same ambient-chord
  prediction -- the cylinder's transverse normal curvature is why its validity
  envelope is 15.7% tighter. So the intrinsic null is about `Phi` and the
  observation-space null is `r_D = (y_c - y_p) - (yhat_c - yhat_p)`. Testing
  the raw difference against zero would reject a correct runtime on a coupon
  pair it predicts perfectly.
- **Every pair carries its own prediction and covariance.** A campaign runs
  several perturbations, replicates and scales; a single predicted difference
  or covariance applied to all of them is broadcast across unlike conditions,
  agreeing with one pair and meaning nothing for the rest. `DifferentialCase`
  is per pair and names the prediction it came from.
- **A campaign-wide boolean is withheld unless every pair was tested.**
  `matched_pairs`, `tested_pairs`, `untested_pairs` and `covariance_status`
  are always reported; `all_tested_consistent` is `None` while anything is
  untested, because one tested pair among eight reading as a consistent
  campaign is worse than no result.
- **A verdict names the tail, not the cause.** A chi-square below the band can
  mean an overstated covariance, a prediction not independent of the
  observation, parameters fitted on the evaluated data, fewer effective
  degrees of freedom, or a wrong correlation structure. The result lists them;
  it does not pick one.
- **Cancellation is measured, never assumed.** A parameter shared by two
  coupons contributes `(J_c - J_p) C_theta (J_c - J_p)^T` to their difference,
  which is zero only where both felt it identically. Combining two scalar
  uncertainties in quadrature assumes independence, which is the opposite of
  the cancellation a differential control claims; with no declared joint
  covariance and no shared-parameter Jacobians, the control reports
  `differential_covariance_not_established` and computes no statistic.
  `differential_to_separate_variance_ratio` is what the subtraction did and is
  **not a fraction** -- it is two when the coupons felt the parameter
  oppositely, because differencing then amplifies it. Do not clip it. And each
  component declares its sources, so the same one inside a trial's covariance
  and inside the shared block is refused rather than counted twice. The
  independent parts are declared `IndependentCovariance` components, never a
  record's bare `measurement_covariance`: an undeclared matrix says how big it
  is and nothing about what is inside it, so adding it to a shared block
  leaves the overlap check with nothing to compare. A case carries either a
  complete covariance or the components, never both.
- **Provenance is verified, not carried.** `grid_digest` is checked against
  the trials' arclength grid, `calibration_ids` against the calibrations they
  ran under, `prediction_digest` against the report they were compared with,
  and duplicate run ids are refused before pairing. A field nothing verifies
  is a label.
- **Pair on achieved, compare on whitened.** Trials pair on achieved
  perturbations inside their own declared uncertainty, never on the command,
  and every field that must agree before two trials can be differenced --
  observation mode and version, units, coordinate and datum frames, filter
  identity, operator digest and causality, calibration relationship, arclength
  grid -- is checked first. The comparison then uses the whitened residual and
  the two-sided chi-square band, not a maximum over a combined scalar.

## Licence

- **MPL-2.0, and every hand-authored file says so.** File-level copyleft is
  the licence that matches the boundary this repository already enforces:
  improvements to the runtime's own files come back, and an adapter or bench
  that *consumes* the transfer record is a separate work under whatever
  licence its author chooses. `boundary.LAYERS` draws that line in the syntax
  tree; the licence draws it legally.
- **The identifier is the first line of every `.py` under `src`, `tests`,
  `tools` and `examples`**, and `tests/test_licensing.py` fails a file that
  arrives without one. Under file-level copyleft an unheaded file is one whose
  terms a reader has to guess, and the file that gets missed is always the new
  one.
- **Generated artefacts are not headed.** The committed reports, the figures
  and `uv.lock` are output, not source: heading them would claim authorship of
  something a program emitted and would make every regeneration a licence
  edit. The exclusion is asserted, so a future mechanical pass cannot quietly
  widen it.
- **No `License ::` classifier.** PEP 639 forbids it alongside a licence
  expression and PyPI rejects the combination -- but hatchling emits both
  without complaint, so the failure would arrive at upload, after the tag
  exists. The SPDX expression is the declaration.
