"""End-to-end verification, run repeatedly, with something different each time.

Running a deterministic pipeline twice proves nothing. Running it a hundred
times proves nothing *more*, unless something varies between the runs -- and
then it proves the one thing the committed artefacts depend on: that the same
inputs give the same bytes, whatever the interpreter happened to do with its
hash seed, whatever order the tests ran in, and wherever the process was
started from.

So each cycle here changes what it is allowed to change and asserts that
nothing downstream moved:

``PYTHONHASHSEED``
    A different value every cycle. Any dependence on dictionary or set
    iteration order shows up as an artefact that differs from the committed
    one -- the failure mode a single run cannot see, because a single run has
    one ordering and agrees with itself.
working directory
    Every cycle runs from a different directory. An entry point that only
    works from the repository root is a broken entry point.
test order
    The full profile shuffles the test files, so a test that passes only
    because another ran first fails here instead of in six months.

And each cycle checks the things CI does not:

* the shipped entry points exit zero and their reports declare zero failed
  checks -- which CI does check -- **and** reproduce the committed
  ``content_hash`` exactly, which it does not;
* ``examples/emit_boundary_record.py`` runs, and the record it writes reads
  back with its samples and its contract fields intact;
* the whole pipeline is *idempotent*: run twice into two different
  directories, the two results are byte-for-byte identical.

Usage::

    python tools/e2e.py --runs 100              # the fast profile
    python tools/e2e.py --runs 10 --profile full  # adds stage two and pytest
    python tools/e2e.py --runs 100 --jobs 4     # cycles in parallel

Exit code is zero only if every cycle passed. The summary names the first
failing step of the first failing cycle, because that is the one worth looking
at.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: Committed artefacts, and the entry point that rebuilds each one.
COMMITTED_REPORTS = {
    "constant-curvature": ROOT / "validation" / "report-v1.json",
    "surfaces": ROOT / "validation" / "report-v2-surfaces.json",
}

#: Fields that legitimately differ between machines and say nothing about the
#: computation. Everything else is compared exactly.
VOLATILE = ("environment",)


@dataclass
class StepResult:
    name: str
    passed: bool
    seconds: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "passed": self.passed,
            "seconds": round(self.seconds, 3),
            "detail": self.detail,
        }


@dataclass
class CycleResult:
    index: int
    seed: int
    profile: str
    threads: int = 1
    steps: list[StepResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(step.passed for step in self.steps)

    @property
    def seconds(self) -> float:
        return sum(step.seconds for step in self.steps)

    @property
    def first_failure(self) -> StepResult | None:
        return next((step for step in self.steps if not step.passed), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.index,
            "hash_seed": self.seed,
            "threads": self.threads,
            "profile": self.profile,
            "passed": self.passed,
            "seconds": round(self.seconds, 3),
            "steps": [step.to_dict() for step in self.steps],
        }


#: Thread counts to rotate through. A threaded BLAS splits a reduction across
#: workers and adds the partial sums in whatever order they finish, so the last
#: bits of a dot product can depend on the thread count -- which is exactly the
#: kind of drift the twelve-significant-digit canonicalisation exists to
#: absorb. Rotating the count is how that claim gets tested rather than
#: assumed, and it is not a claim a single-threaded run can make.
THREAD_COUNTS: tuple[int, ...] = (1, 2, 4)


def _run(
    command: list[str], *, seed: int, cwd: Path, threads: int = 1,
    timeout: float = 3600.0,
) -> tuple[int, str]:
    """A subprocess with the cycle's hash seed, threads and directory.

    Subprocesses rather than imports: the claim being tested is that the
    *shipped commands* work, and a command imported into this process would
    share its interpreter, its hash seed and its working directory -- three of
    the four things the cycle is varying.
    """
    environment = os.environ | {
        "PYTHONHASHSEED": str(seed),
        "MPLBACKEND": "Agg",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
    }
    try:
        finished = subprocess.run(
            command,
            cwd=str(cwd),
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:g}s"
    return finished.returncode, (finished.stdout + finished.stderr)


def _stable(document: dict[str, Any]) -> str:
    """A report's canonical text, with the per-machine fields dropped.

    The interpreter version and the platform string belong in the artefact and
    say nothing about the arithmetic, so they are excluded here rather than
    from the artefact.
    """
    payload = {key: value for key, value in document.items() if key not in VOLATILE}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


class Baseline:
    """What every cycle has to agree with, and where that expectation comes from.

    Two modes, because "reproducible" means two different things.

    ``committed`` compares each cycle with the tracked artefact. That is the
    right question on the machine the artefact was generated on, and it is the
    wrong question anywhere else: the last digits of ``sin``, ``cosh``, ``svd``
    and every BLAS reduction belong to the platform's libm, and no rounding
    rule aligns two builds of it. Asking for a matching hash across builds is
    asking two libraries to agree bit for bit.

    ``self`` takes the first cycle's own output as the expectation and holds
    every later cycle to it. That claim is true anywhere, and it is the one
    worth making a hundred times: same machine, same code, same bytes --
    whatever the hash seed, the thread count or the working directory.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.lock = threading.Lock()
        self.expected: dict[str, bytes] = {}
        if mode == "committed":
            for stage, path in COMMITTED_REPORTS.items():
                # Through the same canonical form the cycles are compared in,
                # not the file's own pretty-printed bytes: the two differ by
                # whitespace and key order, which is not what is being tested.
                self.expected[f"report/{stage}"] = _stable(
                    json.loads(path.read_text())
                ).encode()
            for name in ("reference-report.json", "reference-report.md"):
                self.expected[f"reference/{name}"] = (ROOT / "results" / name).read_bytes()

    def agree(self, key: str, produced: bytes) -> str:
        """Record the first answer, or hold this one to it."""
        with self.lock:
            if key not in self.expected:
                self.expected[key] = produced
                return f"baseline set ({len(produced)} bytes)"
            expected = self.expected[key]
        assert produced == expected, (
            f"{key} differs from the {self.mode} baseline: {len(produced)} bytes "
            f"against {len(expected)}"
        )
        return f"agrees with the {self.mode} baseline ({len(produced)} bytes)"


class Cycle:
    """One complete pass over the shipped entry points."""

    def __init__(
        self, index: int, seed: int, profile: str, workspace: Path, baseline: Baseline
    ) -> None:
        self.baseline = baseline
        self.index = index
        self.seed = seed
        self.profile = profile
        self.workspace = workspace
        self.threads = THREAD_COUNTS[index % len(THREAD_COUNTS)]
        self.result = CycleResult(
            index=index, seed=seed, profile=profile, threads=self.threads
        )
        self.python = [sys.executable]

    def spawn(self, command: list[str], cwd: Path) -> tuple[int, str]:
        return _run(command, seed=self.seed, cwd=cwd, threads=self.threads)

    # -- plumbing ---------------------------------------------------------
    def step(self, name: str, function) -> bool:
        started = time.monotonic()
        try:
            detail = function() or ""
            passed = True
        except AssertionError as failure:
            detail, passed = str(failure), False
        except Exception as failure:  # noqa: BLE001 - the harness reports, not raises
            detail = f"{type(failure).__name__}: {failure}"
            passed = False
        self.result.steps.append(
            StepResult(name, passed, time.monotonic() - started, detail)
        )
        return passed

    def out(self, name: str) -> Path:
        path = self.workspace / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -- the steps --------------------------------------------------------
    def check_import(self) -> str:
        code, output = self.spawn(
            [
                *self.python,
                "-c",
                "import geodesic_testbed as g, tomllib, pathlib;"
                "meta = tomllib.loads(pathlib.Path(r'%s').read_text());"
                "assert g.__version__ == g.RUNTIME_VERSION == meta['project']['version'],"
                " (g.__version__, meta['project']['version']);"
                "print(g.__version__)" % (ROOT / "pyproject.toml"),
            ],
            cwd=self.workspace,
        )
        assert code == 0, f"import failed:\n{_tail(output)}"
        return output.strip()

    def check_lint(self) -> str:
        code, output = self.spawn(
            [*self.python, "-m", "ruff", "check", "."], ROOT
        )
        assert code == 0, f"ruff exited {code}:\n{_tail(output)}"
        return "clean"

    def check_boundary_record(self) -> str:
        destination = self.out("boundary")
        code, output = self.spawn(
            [
                *self.python,
                str(ROOT / "examples" / "emit_boundary_record.py"),
                "--out",
                str(destination),
            ],
            cwd=self.workspace,
        )
        assert code == 0, f"the boundary example exited {code}:\n{_tail(output)}"
        written = destination / "path-sensitivity-record.json"
        assert written.is_file(), "the boundary example wrote no record"

        payload = json.loads(written.read_text())
        for key in (
            "schema", "contract", "units", "frame", "grid", "covariance",
            "provenance", "calibration", "observation_mode", "geometry",
            "chart", "path_type", "arclength", "a", "b",
        ):
            assert key in payload, f"the record is missing {key!r}"
        assert payload["geometry"] is not None, "the record carries no geometry"
        assert len(payload["arclength"]) == payload["samples"], "sample count disagrees"
        return f"{payload['samples']} samples, {payload['schema']}"

    def check_reference_report(self) -> str:
        """Rebuild it elsewhere and compare the bytes with the tracked copy.

        Elsewhere rather than in place: overwriting the file being checked
        would make the check vacuous on the second run, and two cycles doing it
        at once would race for the same path. Byte-identity against the tracked
        copy is the stronger statement anyway -- it is what "regenerate the
        tracked artefacts" is supposed to mean.
        """
        destination = self.out("reference")
        code, output = self.spawn(
            [
                *self.python,
                str(ROOT / "examples" / "write_reference_report.py"),
                "--out", str(destination),
            ],
            cwd=self.workspace,
        )
        assert code == 0, f"the reference report exited {code}:\n{_tail(output)}"
        notes = [
            self.baseline.agree(f"reference/{name}", (destination / name).read_bytes())
            for name in ("reference-report.json", "reference-report.md")
        ]
        return "; ".join(notes)

    def check_experiment(self, stage: str) -> str:
        destination = self.out(f"experiment-{stage}")
        code, output = self.spawn(
            [
                *self.python,
                str(ROOT / "examples" / "run_experiment.py"),
                "--stage", stage, "--out", str(destination), "--no-figure",
            ],
            cwd=self.workspace,
        )
        assert code == 0, f"stage {stage!r} exited {code}:\n{_tail(output)}"

        name = COMMITTED_REPORTS[stage].name
        produced = (destination / name).read_bytes()
        fresh = json.loads(produced.decode())

        assert fresh["summary"]["n_failed"] == 0, (
            f"stage {stage!r} reports {fresh['summary']['n_failed']} failed checks: "
            f"{fresh['summary']['failed']}"
        )
        # Compared on the stable payload rather than the raw bytes: the report
        # names its own interpreter and platform, which differ legitimately
        # between the baseline's process and this one only if the baseline came
        # from somewhere else -- and then the comparison should not fail on it.
        note = self.baseline.agree(f"report/{stage}", _stable(fresh).encode())
        return (
            f"{fresh['summary']['n_checks']} checks, "
            f"hash {fresh['content_hash'][:12]}, {note}"
        )

    def check_idempotent(self, stage: str) -> str:
        """Twice into two directories, and the two must agree byte for byte."""
        texts = []
        for attempt in ("a", "b"):
            destination = self.out(f"idempotent-{stage}-{attempt}")
            code, output = self.spawn(
                [
                    *self.python,
                    str(ROOT / "examples" / "run_experiment.py"),
                    "--stage", stage, "--out", str(destination), "--no-figure",
                ],
                cwd=self.workspace,
            )
            assert code == 0, f"idempotence run {attempt} exited {code}:\n{_tail(output)}"
            texts.append((destination / COMMITTED_REPORTS[stage].name).read_bytes())
        assert texts[0] == texts[1], (
            f"stage {stage!r} produced different bytes on two runs in one process "
            "generation -- the pipeline is not idempotent"
        )
        return f"{len(texts[0])} bytes, twice"

    def check_tests(self) -> str:
        """The suite, with the files handed to pytest in a shuffled order."""
        files = sorted(path.name for path in (ROOT / "tests").glob("test_*.py"))
        shuffled = list(files)
        random.Random(self.seed).shuffle(shuffled)
        code, output = self.spawn(
            [
                *self.python, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                *[str(ROOT / "tests" / name) for name in shuffled],
            ],
            cwd=ROOT,
        )
        assert code == 0, f"pytest exited {code}:\n{_tail(output, 25)}"
        return f"{len(shuffled)} files, order seed {self.seed}"

    # -- the cycle --------------------------------------------------------
    def run(self) -> CycleResult:
        steps: list[tuple[str, Any]] = [
            ("import", self.check_import),
            ("lint", self.check_lint),
            ("boundary-record", self.check_boundary_record),
            ("reference-report", self.check_reference_report),
            ("experiment/constant-curvature", lambda: self.check_experiment(
                "constant-curvature"
            )),
            ("idempotence/constant-curvature", lambda: self.check_idempotent(
                "constant-curvature"
            )),
        ]
        if self.profile == "full":
            steps += [
                ("experiment/surfaces", lambda: self.check_experiment("surfaces")),
                ("tests", self.check_tests),
            ]
        for name, function in steps:
            if not self.step(name, function):
                break  # a broken cycle's later steps say nothing new
        return self.result


def run_cycle(
    index: int, seed: int, profile: str, keep: bool, baseline: Baseline
) -> CycleResult:
    workspace = Path(tempfile.mkdtemp(prefix=f"e2e-{index:03d}-"))
    try:
        return Cycle(index, seed, profile, workspace, baseline).run()
    finally:
        if not keep:
            shutil.rmtree(workspace, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--profile", choices=("fast", "full"), default="fast")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument(
        "--baseline",
        choices=("committed", "self"),
        default="committed",
        help=(
            "what every cycle must agree with: the tracked artefacts "
            "(only meaningful where they were generated) or the first cycle's "
            "own output (true anywhere, and what CI should use)"
        ),
    )
    parser.add_argument("--keep", action="store_true", help="keep the workspaces")
    parser.add_argument("--report", type=Path, help="write a JSON summary here")
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.runs < 1:
        parser.error("--runs must be at least 1")

    started = time.monotonic()
    baseline = Baseline(arguments.baseline)
    plan = [
        (index, arguments.seed_base + index, arguments.profile, arguments.keep, baseline)
        for index in range(1, arguments.runs + 1)
    ]
    results: list[CycleResult] = []
    streak = 0
    best_streak = 0

    def record(result: CycleResult) -> None:
        nonlocal streak, best_streak
        results.append(result)
        if result.passed:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0
        if not arguments.quiet:
            failure = result.first_failure
            mark = "ok  " if result.passed else "FAIL"
            note = "" if result.passed else f"  <- {failure.name}: {failure.detail}"
            print(
                f"[{len(results):3d}/{arguments.runs}] {mark} "
                f"seed={result.seed} threads={result.threads} "
                f"{result.seconds:6.1f}s "
                f"streak={streak}{note}",
                flush=True,
            )

    if arguments.jobs > 1:
        # The first cycle runs alone when it is the one setting the baseline:
        # several cycles racing to write it would each compare against
        # whichever got there first, which is a weaker claim than they look.
        if arguments.baseline == "self" and plan:
            record(run_cycle(*plan[0]))
            plan = plan[1:]
        with ThreadPoolExecutor(max_workers=arguments.jobs) as pool:
            for result in pool.map(lambda item: run_cycle(*item), plan):
                record(result)
    else:
        for item in plan:
            record(run_cycle(*item))

    elapsed = time.monotonic() - started
    passed = sum(1 for result in results if result.passed)
    summary = {
        "profile": arguments.profile,
        "baseline": arguments.baseline,
        "runs": arguments.runs,
        "passed": passed,
        "failed": arguments.runs - passed,
        "longest_streak": best_streak,
        "consecutive": best_streak == arguments.runs,
        "wall_seconds": round(elapsed, 1),
        "cycles": [result.to_dict() for result in results],
    }
    if arguments.report:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        f"\n{passed}/{arguments.runs} cycles passed "
        f"({arguments.profile} profile, {elapsed:.0f}s, "
        f"longest consecutive streak {best_streak})"
    )
    if passed != arguments.runs:
        first = next(result for result in results if not result.passed)
        failure = first.first_failure
        print(f"\nfirst failure, cycle {first.index} (seed {first.seed}), "
              f"step {failure.name}:\n{failure.detail}")
    return 0 if passed == arguments.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
