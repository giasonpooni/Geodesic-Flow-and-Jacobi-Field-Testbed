# The boundary

This repository is a parallel computational substrate, not a module inside an
instrument workbench. It answers one question — given a surface and a path, how
does a starting-pose error propagate — and answers it from geometry alone.
Calibration, sensing, filtering, uncertainty budgets, physical trials,
observability and operational decisions are a different kind of work and belong
to a different system. The two are adjacent:

```text
geometry / path artefact
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

## What does not cross, in either direction

Narrow means everything below stays on exactly one side.

**Solver internals.** Integrators, charts, Christoffel symbols, the
eight-component system, step control. A consumer that depended on any of these
would be pinned to this runtime's numerics. What it is entitled to is `Phi`
sampled on a declared grid with the resolution that produced it stated.

**Mesh processing.** Triangulated surfaces, discrete curvature estimators, mesh
path convergence — the Intrinsic Surface Geodesics Testbed's work. It arrives
here as a versioned path artefact named in `UpstreamArtefact`, and this
repository does not grow a second mesh solver.

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

`geodesic_testbed.boundary` names three layers:

| layer | modules |
|---|---|
| `SUBSTRATE` | `spaceforms`, `surfaces`, `integrators`, `flows`, `transfer`, `envelope`, `analysis`, `jacobi` |
| `CONTRACT` | `contract`, `observation`, `record` |
| `INSTRUMENT_FACING` | `observation_model`, `measurement`, `tracking`, `routing` |

The substrate may not import the instrument-facing side. The contract may
import neither — `record` reaches `transfer` for the map it wraps and nothing
else, and `contract.py` imports nothing but the standard library and NumPy, so
a vocabulary that computes nothing cannot drift from what it promises.

`tests/test_boundary.py` reads the imports out of the syntax tree and asserts
this, rather than importing the modules — a rule about what a module is allowed
to depend on should not be enforced by a mechanism that depends on it. The
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
