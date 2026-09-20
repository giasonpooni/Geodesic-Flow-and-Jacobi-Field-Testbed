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
means a consumer never has to infer anything about the numbers it received.
These are the seven things that cannot be reconstructed from the samples, so
they travel with them.

| field | why it cannot be inferred |
|---|---|
| `units` | A tolerance in millimetres against a record in metres is a thousandfold error that agrees in shape. `PathTolerance` and `compare` both check it. |
| `frame` | Two correct records in different frames differ by a rotation neither can see. Only registered frames are accepted; an unknown one is refused at construction. |
| `arclength` | The grid itself, strictly increasing, carried as a vector rather than a start/stop/count triple: a non-uniform grid is legal, and a resampled one is a different record. `record.grid` summarises it; the vector stays authoritative. |
| `covariance` | `C0`, the starting-pose distribution. The runtime can propagate one and cannot know one. |
| `provenance` | Producer, producer version, and the upstream artefacts the path came from. A prediction that cannot be traced to its geometry cannot be re-derived when that geometry turns out to be wrong. |
| `calibration` | Opaque identifiers, carried and never interpreted here. |
| `observation_mode` | An in-surface distance and an ambient chord differ at the order the campaign is trying to resolve, so an untagged comparison is not evidence. |

Alongside them the record carries `resolution` (method, samples, step) and
`validity` (the perturbation range over which the linear map is declared to
hold, and **on what basis**), because "rk4" alone does not say whether a focus
is located to 1e-3 or 1e-12, and a bound measured against an exact separation
is a different claim from one a caller asserted.

### Three fields that are allowed to say *no*

`covariance`, `calibration` and `validity` each have an undeclared state, and
each undeclared state is the honest default for a record computed from an
analytic surface. Nothing was calibrated, so nothing claims to be.

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
before covariance, provenance and calibration existed — reads back with those
fields undeclared, which is what they in fact were. An unrecognised schema is
refused rather than partially understood.

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
