# Workflows

`ci.yml` is the gate for this repository.

`verify` lints, runs the test suite on Python 3.11-3.13, re-runs both
experiment stages and the application reference report end to end from a clean
checkout, and uploads the regenerated reports and figures as build artefacts.

`determinism` runs `tools/e2e.py`, which drives every shipped entry point
several times over and changes something each cycle: the interpreter's hash
seed, the BLAS thread count, and the directory the command is run from. It then
requires the artefacts to be identical across the cycles. A dependence on
dictionary ordering, on how a threaded reduction happened to split, or on the
working directory is invisible to a single run, because a single run agrees
with itself.

It compares against the **first cycle's own output**, not against the tracked
reports. Those were written by a different numpy build, and the last digits of
`sin`, `cosh`, `svd` and every BLAS reduction belong to the platform's libm --
four builds give four content hashes, and no rounding rule aligns them. What is
asserted against the tracked reports, in the test suite, is that every reported
*value* agrees to a declared tolerance, which is true and says how far the
artefact may drift.

`locked` repeats the lint and the tests through `uv sync --locked`, so a change
to `pyproject.toml` that was never locked fails the build rather than silently
leaving `uv.lock` stale.

`examples/run_experiment.py` exits non-zero when any declared check in either
report fails, so a numerical regression fails the build rather than quietly
producing a report that says so. The test suite additionally compares the
committed reports with freshly computed ones by check identity, so an artefact
that no longer describes the code fails too.
