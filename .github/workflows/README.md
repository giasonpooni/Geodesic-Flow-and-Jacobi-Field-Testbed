# Workflows

`ci.yml` is the gate for this repository. On every push it lints, runs the test
suite on Python 3.10-3.13, re-runs the experiment end to end from a clean
checkout, and uploads the regenerated `report-v1.json` and
`jacobi-testbed-v1.png` as build artefacts.

`examples/run_experiment.py` exits non-zero when any declared check in the
report fails, so a numerical regression fails the build rather than quietly
producing a report that says so.
