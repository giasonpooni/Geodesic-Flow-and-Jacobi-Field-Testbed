# Workflows

`ci.yml` is the gate for this repository.

`verify` lints, runs the test suite on Python 3.11-3.13, re-runs both
experiment stages and the application reference report end to end from a clean
checkout, and uploads the regenerated reports and figures as build artefacts.

`locked` repeats the lint and the tests through `uv sync --locked`, so a change
to `pyproject.toml` that was never locked fails the build rather than silently
leaving `uv.lock` stale.

`examples/run_experiment.py` exits non-zero when any declared check in either
report fails, so a numerical regression fails the build rather than quietly
producing a report that says so. The test suite additionally compares the
committed reports with freshly computed ones by check identity, so an artefact
that no longer describes the code fails too.
