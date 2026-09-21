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

**Evidence fails closed.** A file offered as evidence must read as a report and
state a schema, a content hash, a positive check count and zero failures. A
file that cannot be parsed, or that states none of this, is refused rather
than recorded as evidence with the claim quietly missing. And the claim is
recomputed on the way back in: ``verify`` re-derives the report block from the
file rather than trusting the number the manifest states, because a hash over
the file does not notice a check count edited only in the manifest.

**The reader is frozen too.** :func:`load` validates the whole ``v1`` shape --
fields, types, roles, digest formats, relative paths, uniqueness -- rather
than reading the fields it happens to want. A schema that only its own
producer enforces is not frozen; it is a habit.

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

#: The exact shape of a ``v1`` payload, enforced on read as well as on write.
PAYLOAD_FIELDS = frozenset({"schema", "digest_algorithm", "release", "artefacts"})
RELEASE_FIELDS = frozenset({"version", "tag", "commit"})
ARTEFACT_FIELDS = frozenset({"role", "path", "asset_name", "size_bytes", DIGEST_ALGORITHM})
REPORT_FIELDS = frozenset({"schema", "content_hash", "declared_checks", "failed_checks"})

#: Where the version lives. One source of truth, read the way hatchling reads
#: it at build time, so the manifest cannot disagree with the wheel.
DEFAULT_CONTRACT_SOURCE = "src/geodesic_testbed/engine/contract.py"

_RUNTIME_VERSION = re.compile(r'^RUNTIME_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)
_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_VERSION = re.compile(r"\A[0-9]+\.[0-9]+\.[0-9]+(?:[.\-+a-z0-9]*)\Z")
_WHEEL = re.compile(r"\A(?P<name>[^-]+)-(?P<version>[^-]+)-.+\.whl\Z")
_SDIST = re.compile(r"\A(?P<name>.+)-(?P<version>[^-]+)\.tar\.gz\Z")

#: What an evidence file's ``schema`` must look like. A *shape* rather than a
#: fixed list: the report schemas are versioned separately and will reach v4
#: without this tool changing, and pinning the list here would make a routine
#: report revision a release-tooling edit. What it refuses is an arbitrary
#: string, which is the failure that matters -- "evidence" that is not a report
#: of this repository at all.
_EVIDENCE_SCHEMA = re.compile(r"\Ageodesic-jacobi-[a-z-]+-v[0-9]+\Z")

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


def _counted(value: object, name: str, *, minimum: int) -> int:
    """An integer that is not a bool. ``True`` is an ``int`` in Python, and a
    check count of ``True`` is not a check count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{name} is not an integer: {value!r}")
    if value < minimum:
        raise ManifestError(f"{name} is {value}, expected at least {minimum}")
    return value


def _read_json(path: Path, what: str) -> object:
    def _refuse_constant(token: str) -> object:
        raise ManifestError(f"{what} contains {token}, which is not JSON")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_refuse_constant)
    except (ValueError, UnicodeDecodeError) as error:
        raise ManifestError(f"{what} is not readable JSON: {error}") from error


def evidence_report(path: Path) -> dict[str, object]:
    """The four things that make a report evidence, or a refusal. Never ``None``.

    A manifest that named a report but not its ``content_hash`` would prove the
    bytes were uploaded, not that they were the bytes the checks ran against --
    so a file that cannot supply one is not evidence, and is refused here
    rather than recorded as evidence with the claim quietly missing.
    """
    document = _read_json(path, path.name)
    if not isinstance(document, dict):
        raise ManifestError(f"{path.name} is not a report object")

    schema = document.get("schema")
    if not isinstance(schema, str) or not _EVIDENCE_SCHEMA.match(schema):
        raise ManifestError(f"{path.name} does not declare a report schema: {schema!r}")

    content_hash = document.get("content_hash")
    if not isinstance(content_hash, str) or not _HEX64.match(content_hash):
        raise ManifestError(f"{path.name} has no valid content_hash: {content_hash!r}")

    summary = document.get("summary")
    if not isinstance(summary, dict):
        raise ManifestError(f"{path.name} has no summary block")

    declared_checks = _counted(summary.get("n_checks"), f"{path.name} n_checks", minimum=1)
    failed_checks = _counted(summary.get("n_failed"), f"{path.name} n_failed", minimum=0)
    if failed_checks:
        raise ManifestError(
            f"{path.name} declares {failed_checks} failed check(s); "
            "a release manifest does not cite failing evidence"
        )

    return {
        "schema": schema,
        "content_hash": content_hash,
        "declared_checks": declared_checks,
        "failed_checks": failed_checks,
    }


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
        entry["report"] = evidence_report(path)
    return entry


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

    payload = {
        "schema": SCHEMA,
        "digest_algorithm": DIGEST_ALGORITHM,
        "release": {"version": version, "tag": tag, "commit": commit},
        "artefacts": entries,
    }
    # The writer is held to the reader's rules rather than to its own: a
    # payload this tool emits but its own `load` would refuse is a schema
    # break that nothing else would catch until a consumer hit it.
    validate_payload(payload)
    return payload


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


# --------------------------------------------------------------------------
# The reader. `v1` is frozen for consumers, not only for this producer.
# --------------------------------------------------------------------------

def _exact_fields(value: object, expected: frozenset[str], what: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"{what} is not an object")
    present = set(value)
    missing = expected - present
    unknown = present - expected
    if missing:
        raise ManifestError(f"{what} is missing {sorted(missing)}")
    if unknown:
        raise ManifestError(f"{what} has fields no v1 manifest carries: {sorted(unknown)}")
    return value


def _text(value: object, what: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{what} is not a non-empty string: {value!r}")
    if pattern is not None and not pattern.match(value):
        raise ManifestError(f"{what} is malformed: {value!r}")
    return value


def _relative_path(value: object, what: str) -> str:
    """A path inside the payload directory, and nothing else.

    A manifest is consumed by a verifier that joins these onto a base
    directory. An absolute path or a ``..`` segment would make ``verify``
    read, and a downloader write, outside the tree it was pointed at.
    """
    text = _text(value, what)
    if "\\" in text or ":" in text or text.startswith("/"):
        raise ManifestError(f"{what} is not a relative POSIX path: {text!r}")
    segments = text.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ManifestError(f"{what} has an empty or traversing segment: {text!r}")
    return text


def validate_payload(payload: object) -> dict[str, object]:
    """The whole of ``v1``, enforced. Raises :class:`ManifestError` on anything else.

    Freezing a schema means a consumer written today keeps agreeing with it,
    which is a property of what the *reader* accepts. A reader that takes the
    fields it wants and ignores the rest accepts a hundred documents the
    schema does not describe, and the first producer to emit one of them has
    changed ``v1`` without anyone noticing.
    """
    document = _exact_fields(payload, PAYLOAD_FIELDS, "payload")

    if document["schema"] != SCHEMA:
        raise ManifestError(f"unsupported schema {document['schema']!r}, expected {SCHEMA}")
    if document["digest_algorithm"] != DIGEST_ALGORITHM:
        raise ManifestError(
            f"unsupported digest_algorithm {document['digest_algorithm']!r}, "
            f"expected {DIGEST_ALGORITHM}"
        )

    release = _exact_fields(document["release"], RELEASE_FIELDS, "release")
    version = _text(release["version"], "release.version", _VERSION)
    tag = _text(release["tag"], "release.tag")
    _text(release["commit"], "release.commit", _COMMIT)
    if tag != f"v{version}":
        raise ManifestError(f"release.tag {tag!r} does not name version {version!r}")

    artefacts = document["artefacts"]
    if not isinstance(artefacts, list) or not artefacts:
        raise ManifestError("artefacts is not a non-empty list")

    seen: dict[str, set[str]] = {"path": set(), "asset_name": set()}
    for index, item in enumerate(artefacts):
        where = f"artefacts[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{where} is not an object")
        role = item.get("role")
        if role not in ROLES:
            raise ManifestError(f"{where}.role is not one of {list(ROLES)}: {role!r}")
        expected = ARTEFACT_FIELDS | ({"report"} if role == "evidence" else frozenset())
        entry = _exact_fields(item, expected, where)

        path = _relative_path(entry["path"], f"{where}.path")
        asset_name = _text(entry["asset_name"], f"{where}.asset_name")
        if asset_name != path.rsplit("/", 1)[-1]:
            raise ManifestError(
                f"{where}.asset_name {asset_name!r} is not the basename of {path!r}"
            )
        _counted(entry["size_bytes"], f"{where}.size_bytes", minimum=0)
        _text(entry[DIGEST_ALGORITHM], f"{where}.{DIGEST_ALGORITHM}", _HEX64)

        for field, value in (("path", path), ("asset_name", asset_name)):
            if value in seen[field]:
                raise ManifestError(f"duplicate {field}: {value}")
            seen[field].add(value)

        if role == "evidence":
            report = _exact_fields(entry["report"], REPORT_FIELDS, f"{where}.report")
            _text(report["schema"], f"{where}.report.schema", _EVIDENCE_SCHEMA)
            _text(report["content_hash"], f"{where}.report.content_hash", _HEX64)
            _counted(report["declared_checks"], f"{where}.report.declared_checks", minimum=1)
            failed = _counted(report["failed_checks"], f"{where}.report.failed_checks", minimum=0)
            if failed:
                raise ManifestError(f"{where}.report declares {failed} failed check(s)")

    return document


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
    """Read a manifest, check it against itself, and hold it to the whole schema."""
    if not manifest_path.is_file():
        raise ManifestError(f"manifest not found: {manifest_path}")
    document = _read_json(manifest_path, "manifest")
    if not isinstance(document, dict):
        raise ManifestError("manifest is not an object")
    if set(document) != {"payload", "digest"}:
        raise ManifestError(f"manifest has fields no v1 manifest carries: {sorted(document)}")
    payload = document["payload"]
    digest = document["digest"]
    if not isinstance(payload, dict) or not isinstance(digest, str):
        raise ManifestError("manifest has no payload/digest pair")
    recomputed = payload_digest(payload)
    if recomputed != digest:
        raise ManifestError(
            f"manifest digest does not match its payload: recorded {digest}, computed {recomputed}"
        )
    return validate_payload(payload)


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
        if entry.get("role") == "evidence":
            problems.extend(_evidence_problems(path, declared, entry))
    return problems


def _evidence_problems(path: Path, declared: str, entry: dict[str, object]) -> list[str]:
    """The report block, recomputed from the file rather than believed.

    A file digest does not notice a check count edited only in the manifest:
    the bytes on disk still hash to what is recorded, and the claim the notes
    quote is now someone else's. So the claim is derived again here and
    compared field by field.
    """
    try:
        recomputed = evidence_report(path)
    except ManifestError as error:
        return [f"evidence no longer reads as a report: {declared}: {error}"]
    recorded = entry.get("report")
    if not isinstance(recorded, dict):
        return [f"evidence has no report block: {declared}"]
    return [
        f"report {field} differs: {declared} is {recomputed[field]!r}, "
        f"manifest says {recorded.get(field)!r}"
        for field in sorted(recomputed)
        if recorded.get(field) != recomputed[field]
    ]


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
