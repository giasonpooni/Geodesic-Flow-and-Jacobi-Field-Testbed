# SPDX-License-Identifier: MPL-2.0
"""The release manifest: what a tag names, stated once and checkable twice.

A release is an identity chain. A tag names a commit; a commit declares a
version; a build turns that commit into files; and the evidence that the
commit was worth releasing is a set of reports whose content hashes were
computed before any of this started. Break any link and the tag stops meaning
anything: the wheel on the release page is not obviously the wheel that was
built, and the reports quoted in the notes are not obviously the reports that
ran.

This module writes that chain down, and verifies it. It does nothing else, and
the *nothing else* is the point. Generation and publication are separate
authorities:

generation
    pure, offline, deterministic. It reads declared files, hashes them and
    emits JSON. It has no network, no credentials, no ability to create a
    release, and no clock.
publication
    a workflow with narrowly scoped write permission, which re-verifies every
    byte against the manifest immediately before it uploads anything.

A generator that could also publish would be a generator whose output nobody
needs to check, because the same process produced the claim and the artefact.
Keeping them apart is what makes ``verify`` a real gate rather than a
formality: the publishing job can refuse a payload it did not build.

**No timestamps.** Nothing in the content-addressed payload records when it
was made. A wall clock would make two byte-identical releases produce two
different manifests, which destroys the one property worth having -- that the
same commit and the same inputs give the same manifest, on any machine, in any
order, a year apart. The tag object already carries the date, and git signs it.

**No implicit discovery.** Every file in the manifest was named on the command
line. Globbing a directory means the manifest depends on what happened to be
lying in ``dist/`` -- a stale wheel from a previous version, an editor backup,
an unrelated download -- and a release that ships whatever it finds is a
release nobody declared.

Usage::

    python tools/release_manifest.py generate \
        --version 0.3.0 --tag v0.3.0 --commit "$GITHUB_SHA" \
        --distribution dist/curved_surface_geodesic_sensitivity-0.3.0-py3-none-any.whl \
        --distribution dist/curved_surface_geodesic_sensitivity-0.3.0.tar.gz \
        --evidence validation/report-v1.json \
        --evidence validation/report-v2-surfaces.json \
        --out dist/release-manifest.json

    python tools/release_manifest.py verify \
        --manifest dist/release-manifest.json \
        --expect-tag v0.3.0 --expect-commit "$GITHUB_SHA"

``generate`` exits non-zero and writes nothing if any link in the chain fails.
``verify`` exits non-zero if any file on disk differs from what the manifest
says, naming every one of them rather than the first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

#: The schema this tool writes and the only one it reads. A ``v1`` manifest is
#: frozen: fields may be added in a ``v2``, but what ``v1`` means does not
#: change, because a verifier from last year has to keep agreeing with it.
SCHEMA = "geodesic-runtime-release.v1"

#: What a listed file is *for*. A distribution is uploaded to the release; a
#: piece of evidence is the artefact whose content hash the notes quote.
ROLES = ("distribution", "evidence")

DIGEST_ALGORITHM = "sha256"

#: Where the version lives. One source of truth, read the way hatchling reads
#: it at build time, so the manifest cannot disagree with the wheel.
DEFAULT_CONTRACT_SOURCE = "src/geodesic_testbed/engine/contract.py"

_RUNTIME_VERSION = re.compile(r'^RUNTIME_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")
_VERSION = re.compile(r"\A[0-9]+\.[0-9]+\.[0-9]+(?:[.\-+a-z0-9]*)\Z")
_WHEEL = re.compile(r"\A(?P<name>[^-]+)-(?P<version>[^-]+)-.+\.whl\Z")
_SDIST = re.compile(r"\A(?P<name>.+)-(?P<version>[^-]+)\.tar\.gz\Z")

_READ_BLOCK = 1 << 20


class ManifestError(Exception):
    """A link in the chain does not hold. Never repaired, only reported."""


def file_digest(path: Path) -> str:
    """The ``sha256`` of a file, read in blocks so a wheel does not land in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(_READ_BLOCK)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def declared_version(contract_source: Path) -> str:
    """``RUNTIME_VERSION`` as the build backend reads it, from the source text.

    Read rather than imported: the generator must work against a checkout that
    was never installed, and importing the package to learn its version makes
    the manifest depend on which copy is on ``sys.path``.
    """
    if not contract_source.is_file():
        raise ManifestError(f"version source not found: {contract_source}")
    match = _RUNTIME_VERSION.search(contract_source.read_text(encoding="utf-8"))
    if match is None:
        raise ManifestError(f"no RUNTIME_VERSION assignment in {contract_source}")
    return match.group(1)


def distribution_version(name: str) -> str:
    """The version a distribution filename claims, or a refusal.

    A wheel built from a different checkout than the one being tagged is the
    failure this catches, and it is not hypothetical: ``dist/`` survives
    between builds.
    """
    for pattern in (_WHEEL, _SDIST):
        match = pattern.match(name)
        if match is not None:
            return match.group("version")
    raise ManifestError(f"not a wheel or sdist filename: {name}")


def _normalised(version: str) -> str:
    """Compare ``0.3.0`` with the ``0_3_0``/``0-3-0`` a filename may carry."""
    return version.replace("_", "-").replace("-", ".").lower()


def _entry(base: Path, declared: str, role: str) -> dict[str, object]:
    path = (base / declared).resolve()
    if not path.is_file():
        raise ManifestError(f"{role} not found: {declared}")
    relative = Path(declared).as_posix()
    entry: dict[str, object] = {
        "role": role,
        "path": relative,
        "asset_name": path.name,
        "size_bytes": path.stat().st_size,
        DIGEST_ALGORITHM: file_digest(path),
    }
    if role == "evidence":
        report = _report_summary(path)
        if report is not None:
            entry["report"] = report
    return entry


def _report_summary(path: Path) -> dict[str, object] | None:
    """The three numbers that make a report evidence rather than a file.

    A manifest that named a report but not its ``content_hash`` would prove the
    bytes were uploaded, not that they were the bytes the checks ran against.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    summary = document.get("summary")
    if not isinstance(summary, dict):
        return None
    declared_checks = summary.get("n_checks")
    failed_checks = summary.get("n_failed")
    if not isinstance(declared_checks, int) or not isinstance(failed_checks, int):
        return None
    if failed_checks:
        raise ManifestError(
            f"{path.name} declares {failed_checks} failed check(s); "
            "a release manifest does not cite failing evidence"
        )
    block: dict[str, object] = {
        "declared_checks": declared_checks,
        "failed_checks": failed_checks,
    }
    for key in ("schema", "content_hash"):
        value = document.get(key)
        if isinstance(value, str):
            block[key] = value
    return block


def build_payload(
    *,
    version: str,
    tag: str,
    commit: str,
    distributions: Sequence[str],
    evidence: Sequence[str],
    base: Path,
    contract_source: Path,
) -> dict[str, object]:
    """The content-addressed half of the manifest, or a refusal.

    Every consistency rule fires here, before anything is written, because a
    manifest that exists is a manifest something downstream will trust.
    """
    if not _VERSION.match(version):
        raise ManifestError(f"version is not a release version: {version!r}")
    if not _COMMIT.match(commit):
        raise ManifestError(f"commit is not a full lowercase sha1: {commit!r}")
    if tag != f"v{version}":
        raise ManifestError(f"tag {tag!r} does not name version {version!r} (expected v{version})")

    source_version = declared_version(contract_source)
    if source_version != version:
        raise ManifestError(
            f"version {version!r} disagrees with RUNTIME_VERSION {source_version!r} "
            f"in {contract_source.as_posix()}"
        )
    if not distributions:
        raise ManifestError("a release with no distribution is not a release")

    entries = [_entry(base, declared, "distribution") for declared in distributions]
    entries += [_entry(base, declared, "evidence") for declared in evidence]

    for entry in entries:
        if entry["role"] != "distribution":
            continue
        claimed = distribution_version(str(entry["asset_name"]))
        if _normalised(claimed) != _normalised(version):
            raise ManifestError(
                f"{entry['asset_name']} is version {claimed!r}, not {version!r}"
            )

    _refuse_collisions(entries)
    entries.sort(key=lambda item: (ROLES.index(str(item["role"])), str(item["path"])))

    return {
        "schema": SCHEMA,
        "digest_algorithm": DIGEST_ALGORITHM,
        "release": {"version": version, "tag": tag, "commit": commit},
        "artefacts": entries,
    }


def _refuse_collisions(entries: Iterable[dict[str, object]]) -> None:
    """Two files cannot share one release asset name, and two paths cannot repeat.

    The upload would silently keep one of them, and the manifest would describe
    a release that does not exist.
    """
    for field in ("path", "asset_name"):
        seen: set[str] = set()
        for entry in entries:
            value = str(entry[field])
            if value in seen:
                raise ManifestError(f"duplicate {field}: {value}")
            seen.add(value)


def canonical_json(payload: object) -> str:
    """The one serialisation the digest is taken over.

    Sorted keys so assembly order is not identity, no whitespace so formatting
    is not either, and ``allow_nan=False`` so nothing leaves here that a JSON
    parser would refuse.
    """
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def payload_digest(payload: object) -> str:
    """``sha256:`` over :func:`canonical_json`, which is the manifest's own identity."""
    return DIGEST_ALGORITHM + ":" + hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


def render(payload: dict[str, object]) -> str:
    """The manifest as it is written: the payload, and the digest of the payload.

    Indented, because a release manifest is read by people as well -- the
    digest is taken over the canonical form, so the file's own formatting is
    free to be legible without touching identity.
    """
    manifest = {"payload": payload, "digest": payload_digest(payload)}
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def load(manifest_path: Path) -> dict[str, object]:
    """Read a manifest and check it against itself before anyone acts on it."""
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ManifestError(f"manifest is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise ManifestError("manifest is not an object")
    payload = document.get("payload")
    digest = document.get("digest")
    if not isinstance(payload, dict) or not isinstance(digest, str):
        raise ManifestError("manifest has no payload/digest pair")
    recomputed = payload_digest(payload)
    if recomputed != digest:
        raise ManifestError(
            f"manifest digest does not match its payload: recorded {digest}, computed {recomputed}"
        )
    if payload.get("schema") != SCHEMA:
        raise ManifestError(f"unsupported schema {payload.get('schema')!r}, expected {SCHEMA}")
    return payload


def verify(
    payload: dict[str, object],
    *,
    base: Path,
    expect_tag: str | None = None,
    expect_commit: str | None = None,
    expect_version: str | None = None,
) -> list[str]:
    """Every problem, not the first one.

    A publisher that stopped at the first mismatch would be re-run once per bad
    file, and each re-run would look like a different failure.
    """
    problems: list[str] = []
    release = payload.get("release")
    release = release if isinstance(release, dict) else {}

    for name, expected in (
        ("tag", expect_tag),
        ("commit", expect_commit),
        ("version", expect_version),
    ):
        if expected is None:
            continue
        recorded = release.get(name)
        if recorded != expected:
            problems.append(f"manifest {name} is {recorded!r}, expected {expected!r}")

    artefacts = payload.get("artefacts")
    if not isinstance(artefacts, list) or not artefacts:
        problems.append("manifest lists no artefacts")
        return problems

    for entry in artefacts:
        if not isinstance(entry, dict):
            problems.append(f"artefact entry is not an object: {entry!r}")
            continue
        declared = str(entry.get("path", ""))
        path = base / declared
        if not path.is_file():
            problems.append(f"missing: {declared}")
            continue
        size = path.stat().st_size
        recorded_size = entry.get("size_bytes")
        if size != recorded_size:
            problems.append(
                f"size differs: {declared} is {size}, manifest says {recorded_size}"
            )
        actual = file_digest(path)
        recorded_digest = entry.get(DIGEST_ALGORITHM)
        if actual != recorded_digest:
            problems.append(
                f"digest differs: {declared} is {actual}, manifest says {recorded_digest}"
            )
    return problems


def _generate(arguments: argparse.Namespace) -> int:
    base = arguments.base.resolve()
    payload = build_payload(
        version=arguments.version,
        tag=arguments.tag,
        commit=arguments.commit,
        distributions=arguments.distribution,
        evidence=arguments.evidence,
        base=base,
        contract_source=base / arguments.contract_source,
    )
    text = render(payload)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(text, encoding="utf-8")
    artefacts = payload["artefacts"]
    assert isinstance(artefacts, list)
    print(f"{arguments.out}: {payload_digest(payload)}")
    for entry in artefacts:
        print(f"  {entry['role']:12s} {entry[DIGEST_ALGORITHM][:16]}  {entry['path']}")
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    payload = load(arguments.manifest)
    problems = verify(
        payload,
        base=arguments.base.resolve(),
        expect_tag=arguments.expect_tag,
        expect_commit=arguments.expect_commit,
        expect_version=arguments.expect_version,
    )
    if problems:
        print(f"{arguments.manifest}: {len(problems)} problem(s)", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    artefacts = payload["artefacts"]
    assert isinstance(artefacts, list)
    release = payload["release"]
    assert isinstance(release, dict)
    print(
        f"{arguments.manifest}: {len(artefacts)} artefact(s) match, "
        f"{release['tag']} at {release['commit'][:12]}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="write a release manifest from declared files")
    generate.add_argument("--version", required=True)
    generate.add_argument("--tag", required=True)
    generate.add_argument("--commit", required=True, help="the full 40-character commit sha")
    generate.add_argument(
        "--distribution", action="append", default=[], metavar="PATH",
        help="a wheel or sdist to publish; repeat for each",
    )
    generate.add_argument(
        "--evidence", action="append", default=[], metavar="PATH",
        help="a report whose content hash the release cites; repeat for each",
    )
    generate.add_argument("--base", type=Path, default=Path("."))
    generate.add_argument("--contract-source", type=Path, default=Path(DEFAULT_CONTRACT_SOURCE))
    generate.add_argument("--out", type=Path, required=True)
    generate.set_defaults(handler=_generate)

    check = sub.add_parser("verify", help="check every listed file against the manifest")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--base", type=Path, default=Path("."))
    check.add_argument("--expect-tag", default=None)
    check.add_argument("--expect-commit", default=None)
    check.add_argument("--expect-version", default=None)
    check.set_defaults(handler=_verify)

    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
