# The boundary

This repository is a parallel computational substrate, not a module inside an
instrument workbench. It answers one question — given a surface and a path, how
does a starting-pose error propagate — and answers it from geometry alone.
Calibration, sensing, filtering, uncertainty budgets, physical trials,
observability and operational decisions are a different kind of work and belong
to a different system. The two are adjacent:

```text
geometry / path artefact                <- path-geometry-v1, the inbound contract
        |
        v
geodesic sensitivity runtime            <- this repository
        |
        v
transfer / path-sensitivity record      <- the shared contract
        |
        v
instrument calibration + measurement tooling
```

One thing crosses: the record. `geodesic_testbed.boundary` is that contract on
its own — import it and you have everything a consumer is entitled to, and
nothing else.

```python
from geodesic_testbed.boundary import read_record, write_record

record = envelope.as_transfer_record(units=UNITS)
write_record(record, "out/path-sensitivity-record.json")
```

`uv run python examples/emit_boundary_record.py --out out` writes one and reads
it back.

## What the record carries

A boundary is worth having only if it is **narrow** and **complete**. Complete
means a consumer never has to infer anything about the numbers it received, and
never has to reach back into the runtime to get something the runtime already
knew.

### What the map means

| field | why it cannot be inferred |
|---|---|
| `units` | A tolerance in millimetres against a record in metres is a thousandfold error that agrees in shape. `PathTolerance` and `compare` both check it. |
| `frame` | Two correct records in different frames differ by a rotation neither can see. Only registered frames are accepted; an unknown one is refused at construction. |
| `arclength` | The grid itself, strictly increasing, carried as a vector rather than a start/stop/count triple: a non-uniform grid is legal, and a resampled one is a different record. `record.grid` summarises it; the vector stays authoritative. |
| `observation_mode` | An in-surface distance and an ambient chord differ at the order the campaign is trying to resolve, so an untagged comparison is not evidence. Checked **against the domain**: the pair is what can be false. |
| `path_type` | `j'' + K j = 0` has no first-derivative term *because* the curve is a geodesic. A record that says `non-geodesic` is telling a consumer its map was computed under an assumption the path does not satisfy. |

### Where the path is

`geometry` carries the path itself, and a consumer that must point an
instrument at it, register a measurement to it, or turn a transverse deviation
into a coordinate cannot recompute it without carrying this runtime's solver.

| field | what it is |
|---|---|
| `position` | `(n, 3)` ambient points. |
| `tangent`, `transverse`, `surface_normal` | The Darboux triad at every sample, as **vectors**. `frame` says the transverse direction is parallel-transported; these are what it actually is, so the claim can be checked rather than trusted. `transverse = normal × tangent` fixes the handedness — a frame right in every respect but orientation flips the sign of every heading error it carries. |
| `normal_curvature_along` | How the surface bends in the direction of travel. |
| `normal_curvature_transverse` | The one that sets how far an ambient chord falls short of an in-surface separation — the `kappa_n^2 sn_K(s)^2` term — so the one a comparison against reconstructed 3-D points needs. Both are called `kappa_n` in the literature, which is why both are named here. |
| `coordinate_frame`, `datum_frame` | What the positions are expressed in, and what the part is fixtured against. |
| `uncertainty` | How well the surface and the path are known: position, normal and curvature sigmas, with a basis. `analytic` — zero, and meant — is a stronger statement than `not-declared`. |

The triad is checked on construction: unit norms, mutual orthogonality, and
`normal × tangent` reproducing the carried transverse direction. The surfaces
experiment checks it further, against **Euler's theorem** — the normal
curvatures in any two orthogonal tangent directions sum to `2H` — on every
surface, to machine precision, which ties the frame the record publishes to the
surface it claims to be on with no closed form required.

`geometry` is `None` for a record built from a declared curvature profile, and
correctly so: a profile is not an embedding, so there are no points to sample
and no ambient frame to write down.

### How well it was solved, and how far it got

| field | why it cannot be inferred |
|---|---|
| `resolution` | Method, samples, step — and `convergence`, a per-quantity step-doubling error estimate. "rk4 at h = 0.005" is a recipe; a consumer deciding whether a focus at `s = 3.1416` is good enough to plan against needs the error. Reported separately for position, transfer, curvature, focus and covariance, because they do not converge together. `not-established` is the default, because the estimate costs a second integration and a producer that did not pay for it must not appear to have. |
| `validity` | The perturbation range over which the linear map is declared to hold, and **on what basis**: a bound measured against an exact separation is a different claim from one a caller asserted. |
| `chart` | How much of the *requested* path the parameterisation could carry. A record never holds an invalid sample, so this is not a mask over the samples in hand — it is the other half of the story, that the path is shorter than the one asked for and why. A consumer asking whether a route covers a part has to tell a route that ended because it finished from one that ended because the chart ran out. |

### Who made it, from what, under what

| field | why it cannot be inferred |
|---|---|
| `covariance` | `C0`, the starting-pose distribution. The runtime can propagate one and cannot know one. |
| `provenance` | Producer, producer version, and the upstream artefacts the path came from, each with a `kind` from `ARTEFACT_KINDS` — `cad-model`, `as-built-scan`, `mesh`, `path-artefact`. A prediction that cannot be traced to its geometry cannot be re-derived when that geometry turns out to be wrong. |
| `calibration` | Opaque identifiers, carried and never interpreted here. |

### The fields that are allowed to say *no*

`covariance`, `calibration`, `validity`, `convergence` and the geometry's
`uncertainty` each have an undeclared state, and each undeclared state is the
honest default for a record computed from an analytic surface. Nothing was
calibrated, so nothing claims to be. No step-doubling run was paid for, so no
error budget is claimed.

The discipline is that an undeclared field is never silently replaced:

- `propagate_declared_covariance()` raises when no `C0` was declared. A
  consumer needing a distributional answer must not receive one computed from
  a covariance nobody stated.
- `CalibrationBinding.agrees_with` returns `False` for two *unbound* records.
  Absence of a calibration is not evidence of a shared one, and the default-pass
  is precisely the case the check exists to catch.
- `compare(..., prediction_source=record)` refuses a prediction bound to a
  different instrument state, and reports `calibration.agreed = None` when the
  prediction is unbound — so the reader sees that no tie was established rather
  than assuming one.

- `ConvergenceEstimate` refuses to carry numbers while its basis says
  `not-established`, so a budget cannot be half-filled.

`StartingCovariance.from_tolerance_box` exists because a box is a bound and a
covariance is a distribution: converting one to the other needs a coverage
convention, and the convention is recorded as `coverage_factor` so a downstream
resolvability figure reads as "at 3 sigma" instead of as an unqualified claim
about the hardware.

## The record is complete, and here is the proof

A contract is complete when the thing on the other side can be built from it
alone. `engine/prediction.py` is that thing: it takes a record and walks it all
the way to something a sensor could have reported, and it imports no solver.

```text
geometric transfer map        Phi(s) dz0        -- a tangent vector
        |   first-order -> finite separation
        v
intrinsic surface distance    d(s)              -- a distance in the surface
        |   in-surface arc -> ambient chord
        v
ambient euclidean chord       c(s)              -- what 3-D points give
        |   H(s), then the filter F
        v
instrument output             y(s), Cov(y)      -- what the sensor reports
        |
        v
comparison statistic          whitened residual, chi-square
```

Each stage is a `Prediction` that carries **the transformation that produced
it**, not just a name for what it now is. A prediction that reached
`ambient-euclidean-chord` by applying the chord correction and one that got
there by being relabelled carry the same mode and different `chain`s, and only
one of them is evidence. The chain runs forward only; stepping back is refused,
because a chord relabelled an intrinsic distance is the single confusion the
module exists to prevent.

Every step is second order in the perturbation — the same order as the
first-order model's own failure. The surfaces experiment measures what that is
worth: against an independently computed finite-difference chord, applying the
two transformations drops the disagreement from the size of the effect to the
numerical floor, three to seven orders of magnitude, on every surface where
both are available. Where the curvature varies, only the chord correction is
computable; `chord_from_tangent` applies it and **declares that the other was
not applied**, in `chain`, in `note` and in `extra["intrinsic_correction"]`,
because a consumer comparing at second order has to know which second-order
terms are in the number it was handed.

The comparison is not a scalar. `residual_statistics` returns the full residual
covariance, the whitened residual `L^-1 r` and the chi-square, because two
residuals of the same size are different evidence when one lies in a direction
the instrument resolves well. A filter correlates arc lengths, so a filtered
comparison gets one covariance over every scalar residual at once rather than a
per-sample stack — keeping only the diagonal blocks would let a comparison
treat as independent exactly the samples the filter made dependent. What counts
as too far is still not decided here; that belongs to the instrument's
protocol.

## The inbound half: `path-geometry-v1`

The diagram has two arrows and the repository had only implemented one. A path
computed upstream now arrives as a declared artefact rather than as whatever a
caller happened to have in memory:

```python
from geodesic_testbed.boundary import read_artefact, transfer_record_from_artefact

artefact = read_artefact("path-geometry-v1.json")
record = transfer_record_from_artefact(artefact)
```

`engine/path_artefact.py` is the schema and `engine/imported_path.py` is the
adapter. The adapter interpolates `K(s)` under the policy the artefact
declares, calls the same `transfer_rhs` and `integrate_on_grid` every other
producer here calls, and wraps the result as an ordinary `TransferRecord`.
There is no second solver and there is no mesh tracing: the artefact's
curvature samples are the whole of the input.

Nine things cross inbound, each because it cannot be recovered on this side.

| field | why it cannot be recomputed here |
| --- | --- |
| `arclength` | a producer that resampled knows what it resampled along; chord lengths would lose exactly the curvature-dependent difference being computed |
| `position`, `tangent` | where the path is, in the producer's frame |
| `transverse` | parallel transport needs the connection, which needs the surface, which stayed upstream |
| `gaussian_curvature` | the one coefficient of the equation |
| `units`, `frame`, `coordinate_frame`, `datum_frame` | there is no safe default for any of them |
| `surface_digest`, `path_digest` | "which surface was this?" has to be answerable later |
| `sampling`, `curvature_interpolation` | see below |
| `uncertainty` | how well the surface and path are actually known, with its basis |
| `validity`, `upstream_status` | how much of the path the producer could produce, and whether the run succeeded |

**The curvature between samples is a declared choice, not a convention.** A
producer samples `K` where it chose to; a consumer has to evaluate it
everywhere the integrator steps. Which interpolant fills the gap is a numerical
decision with an order, and the report measures it: on the saddle the monotone
cubic converges at order 3.8 and the piecewise-linear one at 1.9997 -- second
order, as it must be -- and at the coarsest sampling in the ladder they differ
by a factor of 580. A field that
changes the answer by 580 is not a field to default quietly, so the artefact
requires it.

**Only the monotone cubic is offered as the default.** A natural spline through
a curvature profile that is flat and then bends overshoots at the corner and
dips `K` below zero, which a Jacobi solver reads as a patch of hyperbolic
surface that is not there — a conjugate point in a region where the surface is
flat. Fritsch–Carlson slopes cost the same and cannot do that.

**What the artefact refuses.** Construction validates and does not repair: a
tangent that is not unit (the grid is then not arclength), a triad that is not
orthogonal (the transverse direction has left the tangent plane), an arclength
that repeats or steps backwards, a length unit nobody declared, a missing
surface or path digest, an unknown frame or interpolation policy, and an
`upstream_status` of `failed` — which is refused at integration rather than at
construction, because a failed artefact is still a real artefact worth being
able to hold and inspect.

**Normal curvatures are declared three at a time.** `normal_curvature_along`,
`normal_curvature_transverse` and `mean_curvature` arrive together or not at
all, and Euler's theorem is checked on arrival. Two of the three would put a
number into the contract that nothing here could check, and a producer that has
none says so — the record then carries no `PathGeometry` rather than a
fabricated one.

**A mode the domain supports is not a mode this artefact earned.** Records from
an imported path live in the `imported-path-artefact` domain, where the
intrinsic distance is unavailable for the same reason it is unavailable on a
parametric surface, and the ambient chord is available — but only to an
artefact that actually declared the transverse normal curvature the chord
correction needs. The first is a statement about the repository; the second is
a statement about one file, and they fail differently.

## Where the linear map stops holding

`Phi dz0` is the first term of a series, and the second term is the same order
as the effect most campaigns here are trying to resolve. So "over what range of
starting errors is this the answer" is a number the record carries, not a
caveat in a docstring. `ValidityEnvelope` is that number and the four things
that make it readable:

| field | why it is not optional |
| --- | --- |
| `directions` | a probe that perturbs only the heading measures `b` and says nothing about `a`, and the two focus in different places |
| `reference`, `reference_digest` | what the linear map was compared *against*. On a parametric surface it is the geodesic flow itself, central-differenced — that never touches the Jacobi equation, so it is an independent computation of the same quantity |
| `relative_tolerance`, `tolerance_basis` | a tolerance chosen after seeing the residuals is not a tolerance |
| `convergence` | an envelope whose own numerics are unresolved is a bound on the solver, not on the linearisation |

`pointwise_error` and `route_error` are kept apart: a path can be linear at
every arc length and still accumulate over the route, and a campaign planning
to one while measuring the other is comparing two different numbers.

**The bound is fitted, then clipped.** The relative error goes as `C eps^2`, so
`C` is fitted over the probes still in that regime and the bound is
`sqrt(tolerance / C)` — reading off the largest rung that happened to pass
would quantise the answer to the ladder and round it upwards, admitting a
perturbation that was never tested. It is then clipped to the end of the
ladder, per direction, and `probe_limited_directions` says which. On a plane
the lateral column is exact, so the fitted coefficient is roundoff and the
extrapolated bound comes out in the hundreds of radians; what was established
there is that it held out to the largest perturbation tested.

**The cylinder is the result worth stating.** A rolled sheet has the plate's
transfer map to 1e-13 and an envelope 15.7% tighter, because the probe measures
an ambient chord and a cylinder has a transverse normal curvature the plate
does not. Where `K` is constant the measured bound reproduces
`sqrt(24 tol / max(a^2 + kappa_n^2 b^2))` to 0.72% or better; where it varies
that formula simply does not apply — it is 43% out on the saddle — so the
declared check there is a different one, which needs no closed form: re-probe
at exactly the bound the fit chose and confirm it costs the declared tolerance.
That holds to 1.3% on every surface.

**An imported artefact leaves it `not-established`.** Measuring the envelope
means flowing neighbouring paths, and the surface they would be flowed on
stayed upstream. A producer that measured it on its own side passes one in;
nobody here invents it. An envelope that was never established admits nothing,
including the nominal path — the alternative reading turns "we never checked"
into "it always holds".

## Two forms of the observability Gramian

`W = int Phi^T H^T R^-1 H Phi ds` treats `R` as a noise *density* and needs the
samples independent. `W = A^T R^-1 A` treats `R` as the covariance of the
measurements actually taken, and is the only form that admits a correlated one.
A filter correlates arc lengths, so that is not an exotic case: a filtered
campaign ranking routes by the integral form counts information it does not
have, because the samples it averaged were already averages of each other. In
the report a correlated `R` carries 6.8% of the information an independent one
of the same variance does.

On a uniform grid with a stationary `R` the two agree to first order in the
spacing, and the declared check is that the discrepancy *halves when the
sampling doubles* — 1.991 and 1.996 over the ladder — because a fixed tolerance
would only describe one grid. The conversion is stated rather than left
implicit: it is exactly the factor that decides whether two campaigns at
different sampling rates are comparable.

A window on the stacked form inverts the *submatrix of `R`*, not a submatrix of
`R^-1`. Those differ whenever the noise is correlated, and only the first is
the information those measurements carry on their own — the second is what they
carry given the ones outside the window, which a planner cannot act on before
they have been taken.

## What does not cross, in either direction

Narrow means everything below stays on exactly one side.

**Solver internals.** Integrators, charts, Christoffel symbols, the
eight-component system, step control. A consumer that depended on any of these
would be pinned to this runtime's numerics. What it is entitled to is `Phi`
sampled on a declared grid with the resolution that produced it stated.

**Mesh processing.** Triangulated surfaces, discrete curvature estimators, mesh
path convergence — the Intrinsic Surface Geodesics Testbed's work. It arrives
here as a `path-geometry-v1` artefact named in `UpstreamArtefact`, and this
repository does not grow a second mesh solver. `artefact_from_envelope` runs
the other way and looks like an exporter; it is not one. It reads what a
parametric surface already computed and changes the container, so that the
adapter can be anchored against the closed forms — it estimates nothing and
traces nothing.

**Filters.** A filter is part of an observation instrument, not of a solver.
Smoothing inside `engine/flows.py` or `engine/transfer.py` would tune the model
to the data through the model's own machinery. A filter changes the observation
model — compare `y_f = F H Phi dz0` against `F R F^T`, never a filtered
measurement against an unfiltered prediction with the original `R`.

**Hardware behaviour.** Servo error, material mechanics, tow compaction,
weld-pool behaviour, probability of detection. None of it is modelled here, and
the record carries calibration identifiers precisely so that it does not have
to.

**Route policy.** What counts as resolved, what margin is acceptable, which
route is chosen. The runtime reports amplification, resolvability and clearance;
a threshold that decides comes from an instrument protocol. A constant like
`SNR >= 3` compiled into a comparison is a declared limit smuggled into
arithmetic.

## The rule has a direction, and the direction is checked

`geodesic_testbed.boundary` names five layers, in `LAYERS`:

| layer | modules | what it is |
|---|---|---|
| `CONTRACT` | `contract`, `canonical`, `observation`, `record` | the shared vocabulary and the record |
| `SUBSTRATE` | `spaceforms`, `surfaces`, `integrators`, `flows`, `transfer`, `envelope`, `analysis`, `jacobi` | geometry in, transfer map out |
| `INSTRUMENT_FACING` | `observation_model`, `prediction`, `planning`, `uncertainty`, `measurement`, `campaign`, `tracking`, `routing` | everything that exists to meet an instrument |
| `CONSUMERS` | `applications`, `tolerances`, `reports` | the application contracts, which speak the record and not the instrument |
| `HARNESS` | `boundary`, `experiment`, `experiment_surfaces`, `figure`, `figure_surfaces`, `cli` | the experiment stages and entry points |

The rules:

- the **substrate** may not import the instrument-facing side;
- the **contract** may import neither — `record` reaches `transfer` for the map
  it wraps and nothing else, and `contract.py` imports nothing but the standard
  library and NumPy, so a vocabulary that computes nothing cannot drift from
  what it promises;
- the **consumers** may not import the instrument-facing side either: a
  tolerance assessment that reached for an observation model would be deciding
  with a threshold that belongs to a protocol;
- the **harness** may import anything, which is exactly why it is named. A
  layer allowed to reach everywhere has to be listed, or every module could
  quietly claim to be it.

Note what is on the instrument-facing side. The whole prediction chain, the
uncertainty budget, the route planner and the coupon programme are built from
the record and import no solver — which is the strongest statement available
that the record is complete.

`tests/test_boundary.py` reads the imports out of the syntax tree and asserts
this, rather than importing the modules — a rule about what a module is allowed
to depend on should not be enforced by a mechanism that depends on it. It also
asserts that **every** module in the package belongs to exactly one layer: a
module in none of them is a module the rule does not reach, and a new one that
imports across the boundary would otherwise pass by not being listed. The
separation is therefore a property of the code, not an intention in a document.

## Versioning

Two version strings, and they move for different reasons.

`RECORD_SCHEMA` (`path-transfer-record-v2`) names the payload shape. Readers
accept every schema in `SUPPORTED_RECORD_SCHEMAS`; a `v1` payload — written
before covariance, provenance, calibration, geometry, chart validity, path type
and the convergence estimate existed — reads back with all of those undeclared
or absent, which is what they in fact were. An unrecognised schema is refused
rather than partially understood, and a payload written without samples comes
back as an error rather than as a record with the metadata right and the
numbers missing.

`BOUNDARY_CONTRACT` (`path-sensitivity-boundary-v1`) names the *shape of the
boundary*: which fields exist and what they mean. It moves when a field is
added, removed or reinterpreted, and stays put when a record's contents change.

`RUNTIME_VERSION` is declared once, in `engine/contract.py`. The distribution
metadata, `geodesic_testbed.__version__` and the `producer_version` in every
record's provenance all read it, and a test holds the three together:
provenance naming a version the package does not have is worse than provenance
naming none.

There is no timestamp and no hostname in a record. Both would be useful and
both would make two runs of the same computation differ, which would make the
committed reports unreproducible — and a report that cannot be regenerated and
compared is the thing this repository is least willing to ship. A caller who
needs wall-clock provenance puts it in `Provenance.extra`, knowingly.
