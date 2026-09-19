# Workflows

`ci.yml` is the gate for this repository. On every push it lints, runs the test
suite on Python 3.10-3.13, re-runs both experiment stages end to end from a
clean checkout, and uploads the regenerated reports and figures as build
artefacts.

`examples/run_experiment.py` exits non-zero when any declared check in either
report fails, so a numerical regression fails the build rather than quietly
producing a report that says so. The test suite additionally compares the
committed reports with freshly computed ones by check identity, so an artefact
that no longer describes the code fails too.
