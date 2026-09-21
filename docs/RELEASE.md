# Releasing

A release is an identity chain, and every link in it is checkable:

```
tag  ->  commit  ->  RUNTIME_VERSION  ->  distributions  ->  evidence
v0.3.0   bbc535a…    0.3.0               wheel + sdist     two reports,
                                         by sha256         by content_hash
```

`tools/release_manifest.py` writes that chain down as
`geodesic-runtime-release.v1`, and `.github/workflows/release.yml` refuses to
upload anything that no longer matches it.

## Four authorities, kept apart

| authority | what it may do | what it may not do |
| --- | --- | --- |
| `tools/release_manifest.py` | read declared files, hash them, emit JSON | reach the network, hold a credential, create a release, read a clock |
| `.github/workflows/ci.yml` | run the suites and regenerate the artefacts on every push to `main` and every pull request | publish anything |
| `.github/workflows/release.yml` | build, verify, and attach assets to a **draft** release; replace its own earlier draft | decide that a draft is ready; touch a release someone has already published |
| the maintainer | publish the draft | hand any of the above the others' powers |

Every action the publishing workflow uses is pinned to a full commit sha, with
the release it was reviewed at in a trailing comment. A major tag is a pointer
its owner can move: `@v7` promises compatibility, not bytes, and this is the
one workflow in the repository that can write.

The separation is what makes verification meaningful. A generator that could
also publish would be checking its own work, and `verify` would be a
formality. So the generator is *structurally* incapable of publishing, and
that is asserted rather than promised: `tests/test_release_manifest.py` parses
its syntax tree and fails on an import outside a fixed allowlist, or a call to
a clock, a subprocess, a socket, an environment read or a directory walk.

## Two properties the manifest depends on

**No timestamps in the content-addressed payload.** A wall clock would make
two byte-identical releases produce two different manifests, which destroys
the only property worth having: the same commit and the same inputs give the
same manifest, on any machine, in any order, a year apart. The tag object
carries the date already, and git signs it.

**No implicit discovery.** Every file in a manifest was named on the command
line. Globbing `dist/` means the manifest describes whatever happened to be
lying there — a stale wheel from the previous version, an editor backup, an
unrelated download — and a release that ships what it finds is a release
nobody declared. A stale `0.1.0` wheel sitting next to the `0.3.0` one is a
test case, not a hypothetical.

## Evidence fails closed

A file offered as `--evidence` is not evidence until it proves it is one. It
must parse as a report object and state a recognised report schema, a
64-character content hash, a positive declared-check count and exactly zero
failures. Anything else is refused, and nothing is written — rather than being
recorded as evidence with the claim quietly missing, which is the fail-open
this repository refuses everywhere else.

The claim is then derived *again* on the way back in. `verify` recomputes each
report block from the file rather than reading the number the manifest states,
because a file hash does not notice a check count edited only in the manifest:
the bytes on disk still hash to what is recorded, and the figure the release
notes quote is now someone else's.

## The reader is frozen, not only the writer

`load()` holds a manifest to the whole of `v1` — exact fields at every level,
types, permitted roles, `digest_algorithm`, digest formatting, relative paths
with no `..` or drive letter, an `asset_name` that is the basename of its own
path, unique paths and asset names, a report block on every evidence entry and
on no distribution entry.

A schema only its own producer enforces is not frozen; it is a habit. Every
one of those rules is tested against a manifest whose payload was edited *and*
whose digest was recomputed, so the self-digest cannot be what catches it.
What catches it is the schema. `build_payload` runs the reader's validator over
its own output for the same reason: a payload this tool emits but its own
`load` would refuse is a schema break nothing else would catch until a
consumer hit it.

## The procedure

1. `main` is green: all eight jobs of `verify` on the merge commit, including
   the five-cycle end-to-end determinism run that only the default branch
   gets.
2. `RUNTIME_VERSION` in `src/geodesic_testbed/engine/contract.py` is the
   version being released. It is the only place a version is written;
   `pyproject.toml` reads it at build time.
3. Tag that commit `vX.Y.Z` and push the tag. `release.yml` fires.
4. The `build` job checks tag against version *before* building, builds the
   wheel and sdist, installs the wheel into an empty environment and asks it
   what version and licence it thinks it carries, writes the manifest over the
   distributions and the two committed reports, and verifies it in place.
5. The `publish` job — the only one in the repository with `contents: write` —
   re-hashes every asset against the manifest, refuses the whole release if
   any one of them differs, and only then creates the draft and attaches the
   assets together with the manifest itself.
6. Read the draft and publish it.

### What the publisher guarantees, exactly

- **No incomplete release is ever published.** Nothing is uploaded until every
  file has been read and checked, and what is created is a draft that a person
  has to publish.
- A draft is created *before* its assets are uploaded, so a failure part-way
  through does leave an incomplete draft. The workflow deletes its own draft
  when an upload raises; a run killed outright — cancelled, runner lost —
  cannot run cleanup at all, and will leave one behind.
- A rerun is therefore idempotent rather than careful: it deletes an existing
  **draft** for the tag and starts again. It refuses outright if a
  **published** release already exists for the tag, because that is a release
  someone has already stood behind.

To rebuild a release's claim from a clean checkout of the tag, run the same
`generate` command and compare the digest. It is the same bytes or it is not.

## Verifying a release as a consumer

```
python tools/release_manifest.py verify --manifest release-manifest.json --base .
```

with the downloaded assets laid out at the paths the manifest names. It
reports every mismatch, not the first, and exits non-zero if there is one.

The manifest digest binds the manifest to itself, not to the repository: a
payload can be edited and re-hashed. What that cannot do is make the manifest
name the tag and commit the publisher was handed, which is why
`--expect-tag`, `--expect-commit` and `--expect-version` exist and why the
workflow always passes all three.

## Schema freeze

From `0.3.0` the following are frozen at `v1`. Their meaning does not change;
a field may be *added* only in a successor schema with a new identifier, so a
consumer written against `v1` keeps agreeing with a `v1` artefact forever.

| schema | where |
| --- | --- |
| `path-geometry-v1` | `engine/path_artefact.py` — the inbound contract |
| `transfer-record-v1` family | `engine/record.py` — the outbound contract |
| `geodesic-runtime-release.v1` | `tools/release_manifest.py` — the release manifest |

What "frozen" forbids, concretely:

- removing a field, or renaming one;
- changing a field's type, units, or the basis a covariance is expressed in;
- widening an enumeration that a consumer switches on exhaustively;
- changing a digest's algorithm or its coverage;
- making an optional field required, or a required field optional.

What it permits — and the distinction that makes the permission coherent:

- adding a field to a **new** schema identifier;
- **repairing a validator**, where the input was *already invalid under the
  written `v1` contract* and the code admitted it by accident. The contract did
  not change; the code caught up with it. That stays `v1`. It is still a
  breaking change for anyone who relied on the defect, so it belongs in a minor
  version with the refusal named in the release notes — but the schema
  identifier does not move, because a consumer reading the written contract was
  never entitled to the input in the first place.
- **not** rejecting input that the written `v1` contract permits. That is a
  change of meaning, however desirable, and it needs `v2`. "Frozen" would say
  nothing at all if the set of documents `v1` accepts could shrink whenever a
  rule seemed like a good idea.

The test for which one you have is written, not felt: read the `v1` contract as
it stands and ask whether it admits the input. If it does not, the code was
wrong. If it does, the contract is what you are proposing to change.

The report schemas (`geodesic-jacobi-report-v3`,
`geodesic-jacobi-surfaces-v3`) are versioned separately and are not frozen:
they are evidence, they grow checks, and their `content_hash` is what a
release cites. A release manifest records the schema identifier alongside the
hash so the two can never be read apart.

## Licence

MPL-2.0. The wheel carries `License-Expression: MPL-2.0` in
Metadata-Version 2.5 and no legacy `License ::` classifier — PEP 639 forbids
the pair and PyPI rejects it, while hatchling will happily emit both. The
release build asserts the absence, because the failure would otherwise arrive
at upload, after a tag already exists.
