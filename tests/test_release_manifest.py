# SPDX-License-Identifier: MPL-2.0
"""The release manifest, and the properties that make it worth trusting.

Three things are being tested here, and only the first is about JSON.

**Purity.** The generator must not be able to publish. That is not a promise
in a docstring -- it is read out of the syntax tree: the modules it imports
and the attributes it calls. A generator that grew a ``urlopen`` or a
``subprocess.run`` would be a generator that could ship something nobody
verified, and the failure would be invisible in its output.

**Determinism.** The same commit and the same files give the same bytes, from
a different working directory, under a different hash seed. Without that, a
manifest digest is not an identity and re-generating one proves nothing.

**Refusal.** Every consistency rule is exercised by breaking it. A rule that
has never been seen to fire is a rule nobody knows is wired up.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "release_manifest.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_manifest", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_manifest = _load()
ManifestError = release_manifest.ManifestError


COMMIT = "0123456789abcdef0123456789abcdef01234567"
VERSION = "9.9.9"


def _repo(tmp_path: Path, *, version: str = VERSION, failed_checks: int = 0) -> Path:
    """A minimal checkout: a version source, two distributions, one report."""
    base = tmp_path / "checkout"
    (base / "src" / "geodesic_testbed" / "engine").mkdir(parents=True)
    (base / "src" / "geodesic_testbed" / "engine" / "contract.py").write_text(
        f'RUNTIME_VERSION = "{version}"\n', encoding="utf-8"
    )
    dist = base / "dist"
    dist.mkdir()
    stem = "curved_surface_geodesic_sensitivity"
    (dist / f"{stem}-{version}-py3-none-any.whl").write_bytes(b"wheel-content")
    (dist / f"{stem}-{version}.tar.gz").write_bytes(b"sdist-content")
    validation = base / "validation"
    validation.mkdir()
    (validation / "report-v1.json").write_text(
        json.dumps(
            {
                "schema": "geodesic-jacobi-report-v3",
                "content_hash": "f" * 64,
                "summary": {"n_checks": 143, "n_failed": failed_checks},
            }
        ),
        encoding="utf-8",
    )
    return base


def _generate(base: Path, **overrides) -> int:
    stem = "curved_surface_geodesic_sensitivity"
    version = overrides.pop("version", VERSION)
    arguments = [
        "generate",
        "--base", str(base),
        "--version", version,
        "--tag", overrides.pop("tag", f"v{version}"),
        "--commit", overrides.pop("commit", COMMIT),
        "--out", str(overrides.pop("out", base / "dist" / "release-manifest.json")),
    ]
    distributions = overrides.pop(
        "distributions",
        [
            f"dist/{stem}-{VERSION}-py3-none-any.whl",
            f"dist/{stem}-{VERSION}.tar.gz",
        ],
    )
    for item in distributions:
        arguments += ["--distribution", item]
    for item in overrides.pop("evidence", ["validation/report-v1.json"]):
        arguments += ["--evidence", item]
    assert not overrides, overrides
    return release_manifest.main(arguments)


def _manifest(base: Path) -> dict:
    return json.loads((base / "dist" / "release-manifest.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Purity: what the generator is structurally incapable of doing.
# --------------------------------------------------------------------------

#: Everything the generator is allowed to import. An allowlist rather than a
#: blocklist, because the interesting failure is the import nobody thought to
#: forbid -- ``http.client`` arriving under a name the blocklist did not know.
ALLOWED_IMPORTS = frozenset(
    {"__future__", "argparse", "collections", "hashlib", "json", "pathlib", "re", "sys"}
)

#: Attribute names that would give it a capability it must not have: a clock, a
#: subprocess, a socket, an environment read, or a directory walk.
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "check_output", "environ", "getenv", "glob", "iterdir", "listdir", "now",
        "Popen", "rglob", "run", "scandir", "system", "time", "today", "urlopen",
        "utcnow", "walk",
    }
)

TOOL_TREE = ast.parse(TOOL.read_text(encoding="utf-8"))


def test_generator_imports_nothing_that_could_publish():
    imported = set()
    for node in ast.walk(TOOL_TREE):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= ALLOWED_IMPORTS, sorted(imported - ALLOWED_IMPORTS)


def test_generator_calls_no_clock_no_process_no_socket_no_directory_walk():
    used = {
        node.attr
        for node in ast.walk(TOOL_TREE)
        if isinstance(node, ast.Attribute)
    } | {
        node.id for node in ast.walk(TOOL_TREE) if isinstance(node, ast.Name)
    }
    assert not used & FORBIDDEN_ATTRIBUTES, sorted(used & FORBIDDEN_ATTRIBUTES)


def test_no_timestamp_anywhere_in_the_payload(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    text = json.dumps(_manifest(base)["payload"])
    for token in ("generated_at", "timestamp", "created", "date", "utc", "epoch"):
        assert token not in text.lower(), token


# --------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------

def test_two_generations_from_different_directories_agree_byte_for_byte(tmp_path, monkeypatch):
    base = _repo(tmp_path)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"

    monkeypatch.chdir(tmp_path)
    assert _generate(base, out=first) == 0
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert _generate(base, out=second) == 0

    assert first.read_bytes() == second.read_bytes()


def test_a_different_hash_seed_does_not_move_the_manifest(tmp_path):
    base = _repo(tmp_path)
    digests = []
    for seed in ("0", "1", "12345"):
        out = tmp_path / f"manifest-{seed}.json"
        environment = dict(os.environ, PYTHONHASHSEED=seed)
        stem = "curved_surface_geodesic_sensitivity"
        result = subprocess.run(
            [
                sys.executable, str(TOOL), "generate",
                "--base", str(base),
                "--version", VERSION, "--tag", f"v{VERSION}", "--commit", COMMIT,
                "--distribution", f"dist/{stem}-{VERSION}-py3-none-any.whl",
                "--distribution", f"dist/{stem}-{VERSION}.tar.gz",
                "--evidence", "validation/report-v1.json",
                "--out", str(out),
            ],
            capture_output=True, text=True, env=environment, cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        digests.append(hashlib.sha256(out.read_bytes()).hexdigest())
    assert len(set(digests)) == 1, digests


def test_the_recorded_digest_is_the_digest_of_the_canonical_payload(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    document = _manifest(base)
    assert document["digest"] == release_manifest.payload_digest(document["payload"])


# --------------------------------------------------------------------------
# Declared inputs only.
# --------------------------------------------------------------------------

def test_an_undeclared_file_in_the_same_directory_is_not_picked_up(tmp_path):
    base = _repo(tmp_path)
    stale = base / "dist" / "curved_surface_geodesic_sensitivity-0.1.0-py3-none-any.whl"
    stale.write_bytes(b"a wheel from a previous release")
    assert _generate(base) == 0
    paths = {entry["path"] for entry in _manifest(base)["payload"]["artefacts"]}
    assert "dist/curved_surface_geodesic_sensitivity-0.1.0-py3-none-any.whl" not in paths
    assert len(paths) == 3


def test_a_declared_file_that_is_absent_is_refused(tmp_path):
    base = _repo(tmp_path)
    with pytest.raises(ManifestError, match="not found"):
        release_manifest.build_payload(
            version=VERSION, tag=f"v{VERSION}", commit=COMMIT,
            distributions=["dist/nothing-9.9.9-py3-none-any.whl"], evidence=[],
            base=base, contract_source=base / release_manifest.DEFAULT_CONTRACT_SOURCE,
        )


# --------------------------------------------------------------------------
# The identity chain: tag, version, commit, filename.
# --------------------------------------------------------------------------

def _payload(base: Path, **overrides):
    stem = "curved_surface_geodesic_sensitivity"
    arguments = {
        "version": VERSION,
        "tag": f"v{VERSION}",
        "commit": COMMIT,
        "distributions": [f"dist/{stem}-{VERSION}-py3-none-any.whl"],
        "evidence": ["validation/report-v1.json"],
        "base": base,
        "contract_source": base / release_manifest.DEFAULT_CONTRACT_SOURCE,
    }
    arguments.update(overrides)
    return release_manifest.build_payload(**arguments)


def test_a_tag_that_does_not_name_the_version_is_refused(tmp_path):
    base = _repo(tmp_path)
    with pytest.raises(ManifestError, match="does not name version"):
        _payload(base, tag="v9.9.8")


def test_a_version_the_source_does_not_declare_is_refused(tmp_path):
    base = _repo(tmp_path, version="9.9.8")
    with pytest.raises(ManifestError, match="disagrees with RUNTIME_VERSION"):
        _payload(base, distributions=["dist/curved_surface_geodesic_sensitivity-9.9.8.tar.gz"])


def test_a_distribution_from_another_version_is_refused(tmp_path):
    base = _repo(tmp_path)
    stale = base / "dist" / "curved_surface_geodesic_sensitivity-0.1.0-py3-none-any.whl"
    stale.write_bytes(b"stale")
    older = "dist/curved_surface_geodesic_sensitivity-0.1.0-py3-none-any.whl"
    with pytest.raises(ManifestError, match="is version '0.1.0', not '9.9.9'"):
        _payload(base, distributions=[older])


def test_an_abbreviated_commit_is_refused(tmp_path):
    base = _repo(tmp_path)
    with pytest.raises(ManifestError, match="full lowercase sha1"):
        _payload(base, commit=COMMIT[:12])


def test_a_release_with_no_distribution_is_refused(tmp_path):
    base = _repo(tmp_path)
    with pytest.raises(ManifestError, match="no distribution"):
        _payload(base, distributions=[])


def test_two_files_cannot_share_one_asset_name(tmp_path):
    base = _repo(tmp_path)
    (base / "other").mkdir()
    (base / "other" / "report-v1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ManifestError, match="duplicate asset_name"):
        _payload(base, evidence=["validation/report-v1.json", "other/report-v1.json"])


def test_failing_evidence_is_refused_and_nothing_is_written(tmp_path):
    base = _repo(tmp_path, failed_checks=2)
    out = base / "dist" / "release-manifest.json"
    assert _generate(base, out=out) == 2
    assert not out.exists()


def test_the_report_block_carries_the_content_hash_and_the_check_counts(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    evidence = [
        entry for entry in _manifest(base)["payload"]["artefacts"]
        if entry["role"] == "evidence"
    ]
    assert len(evidence) == 1
    assert evidence[0]["report"] == {
        "schema": "geodesic-jacobi-report-v3",
        "content_hash": "f" * 64,
        "declared_checks": 143,
        "failed_checks": 0,
    }


# --------------------------------------------------------------------------
# Tamper detection: every way the bytes can stop matching the claim.
# --------------------------------------------------------------------------

def test_a_modified_distribution_is_caught_by_name(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    wheel = base / "dist" / f"curved_surface_geodesic_sensitivity-{VERSION}-py3-none-any.whl"
    wheel.write_bytes(b"wheel-contenT")  # same length, one bit different
    payload = release_manifest.load(base / "dist" / "release-manifest.json")
    problems = release_manifest.verify(payload, base=base)
    assert len(problems) == 1
    assert problems[0].startswith("digest differs: dist/")


def test_a_truncated_file_is_caught_on_both_size_and_digest(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    (base / "validation" / "report-v1.json").write_text("{}", encoding="utf-8")
    payload = release_manifest.load(base / "dist" / "release-manifest.json")
    problems = release_manifest.verify(payload, base=base)
    assert any(problem.startswith("size differs") for problem in problems)
    assert any(problem.startswith("digest differs") for problem in problems)


def test_a_missing_file_is_reported_once_not_as_a_digest_failure(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    (base / "dist" / f"curved_surface_geodesic_sensitivity-{VERSION}.tar.gz").unlink()
    payload = release_manifest.load(base / "dist" / "release-manifest.json")
    problems = release_manifest.verify(payload, base=base)
    assert problems == ["missing: dist/curved_surface_geodesic_sensitivity-9.9.9.tar.gz"]


def test_every_mismatch_is_reported_not_only_the_first(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    for name in (
        f"dist/curved_surface_geodesic_sensitivity-{VERSION}-py3-none-any.whl",
        f"dist/curved_surface_geodesic_sensitivity-{VERSION}.tar.gz",
    ):
        (base / name).write_bytes(b"tampered")
    payload = release_manifest.load(base / "dist" / "release-manifest.json")
    problems = release_manifest.verify(payload, base=base)
    assert sum(problem.startswith("digest differs") for problem in problems) == 2


def test_an_edited_payload_fails_before_any_file_is_looked_at(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    path = base / "dist" / "release-manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["artefacts"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(ManifestError, match="digest does not match its payload"):
        release_manifest.load(path)


def test_a_manifest_re_signed_after_tampering_still_fails_its_commit_binding(tmp_path):
    """Recomputing the digest is not forgery-proof on its own, and is not asked to be.

    Someone who edits the payload can recompute the digest over it -- the digest
    binds the manifest to itself, not to the repository. What it cannot do is
    make the manifest name the commit the publishing workflow was handed, which
    is why ``--expect-commit`` exists and why the publisher passes it.
    """
    base = _repo(tmp_path)
    assert _generate(base) == 0
    path = base / "dist" / "release-manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["release"]["commit"] = "a" * 40
    document["digest"] = release_manifest.payload_digest(document["payload"])
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    payload = release_manifest.load(path)  # self-consistent, so it loads
    problems = release_manifest.verify(payload, base=base, expect_commit=COMMIT)
    assert problems == [f"manifest commit is {'a' * 40!r}, expected {COMMIT!r}"]


def test_an_unknown_schema_is_refused(tmp_path):
    base = _repo(tmp_path)
    assert _generate(base) == 0
    path = base / "dist" / "release-manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["schema"] = "geodesic-runtime-release.v2"
    document["digest"] = release_manifest.payload_digest(document["payload"])
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(ManifestError, match="unsupported schema"):
        release_manifest.load(path)


# --------------------------------------------------------------------------
# The frozen shape of v1.
# --------------------------------------------------------------------------

def test_v1_payload_and_artefact_fields_are_exactly_these(tmp_path):
    """``v1`` is frozen. New fields belong in a ``v2``; a verifier written
    against this schema has to keep agreeing with it."""
    base = _repo(tmp_path)
    assert _generate(base) == 0
    payload = _manifest(base)["payload"]
    assert set(payload) == {"schema", "digest_algorithm", "release", "artefacts"}
    assert set(payload["release"]) == {"version", "tag", "commit"}
    common = {"role", "path", "asset_name", "size_bytes", "sha256"}
    for entry in payload["artefacts"]:
        assert set(entry) <= common | {"report"}
        assert common <= set(entry)
        assert entry["role"] in release_manifest.ROLES


def test_the_default_version_source_is_the_repository_one():
    source = ROOT / release_manifest.DEFAULT_CONTRACT_SOURCE
    assert source.is_file()
    from geodesic_testbed.engine.contract import RUNTIME_VERSION

    assert release_manifest.declared_version(source) == RUNTIME_VERSION


# --------------------------------------------------------------------------
# The other authority: publication.
# --------------------------------------------------------------------------

WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
WORKFLOW_TEXT = WORKFLOW.read_text(encoding="utf-8") if WORKFLOW.is_file() else ""


def test_the_publication_workflow_exists_and_starts_from_read_only_permissions():
    assert WORKFLOW.is_file()
    head = WORKFLOW_TEXT.split("jobs:", 1)[0]
    assert "permissions:\n  contents: read" in head


def test_only_the_publishing_job_can_write_contents():
    build, _, publish = WORKFLOW_TEXT.partition("  publish:")
    assert publish, "no publish job"
    assert "contents: write" not in build
    assert "contents: write" in publish


def test_the_manifest_is_verified_immediately_before_the_upload():
    _, _, publish = WORKFLOW_TEXT.partition("  publish:")
    verify_at = publish.find("release_manifest.py verify")
    upload_at = publish.find("createRelease")
    assert verify_at != -1, "the publish job does not verify the manifest"
    assert upload_at != -1, "the publish job does not create a release"
    assert verify_at < upload_at, "the release is created before the manifest is verified"


def test_the_publisher_binds_the_tag_and_the_commit_it_was_handed():
    assert "--expect-tag" in WORKFLOW_TEXT
    assert "--expect-commit" in WORKFLOW_TEXT
    assert "--expect-version" in WORKFLOW_TEXT
