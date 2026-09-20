# License policy: Curved-Surface Geodesic Sensitivity Runtime

Status: **proposed transition; licensing counsel review is required before adoption**. This review-branch draft does not authorize changing the default-branch license, creating a release or tag, uploading a package, or replacing an existing distribution.

Proposed default: **MPL-2.0**. Classification: Public mathematical and scientific runtime. The complete unmodified standard license text is in `LICENSE`.

## Preserved MIT baseline

Default-branch review baseline: [`3fced3eeda1a4d2545f1d057e9c17d43d597e606`](https://github.com/giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime/commit/3fced3eeda1a4d2545f1d057e9c17d43d597e606). The published default branch remains MIT pending review.

The audited baseline is [`af92d42b9e5f6a2cac809fe582ee2f17cb29dcfa`](https://github.com/giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime/commit/af92d42b9e5f6a2cac809fe582ee2f17cb29dcfa), where the root license and package metadata declare MIT. Its original copyright and complete permission notice are copied byte-for-byte to `LICENSES/MIT-legacy.txt`.

Earlier copies received under MIT retain those terms; this proposal does not revoke, rewrite or retroactively replace that grant. Existing commits, tags, releases and distributed packages must remain intact. The legacy notice is retained as attribution and historical licensing evidence, not as an offer to apply MIT to all future additions.

The audit found no git tags or GitHub releases for this repository. This is an observation at the baseline, not a claim that no package or other copy has ever been distributed. Do not overwrite an existing package version under different terms; allocate an unused version before any package publication.

## Scope

After approval, MPL-2.0 is intended to cover project-owned source, tests, examples, build/configuration files and public documentation in this revision unless a file or directory carries separate terms. Documentation follows the source license; this proposal adds no separate Creative Commons grant. Mathematical facts and independent third-party rights are not relicensed by this policy.

Preserve all accurate existing copyright, patent, attribution and license notices. External dependencies keep their own licenses. Referenced or linked projects are not relicensed by changing this repository. A retained copy, generated record, third-party extract or vendor file must not be assigned new rights merely because it is present here.

## Review required before publication

- Confirm authority for project-owned contributions, including the relationship between Git author aliases and rights holders. Git authorship and automated co-author trailers are not proof of assignment or exclusive ownership.
- Confirm the scope and terms of any externally sourced material, preserve its notices, and record any exclusions before distribution.
- Review the complete standard license, its grants and obligations, and the proposed historical boundary. No custom changes are made to the standard license text.
- Keep package metadata, bundled license files and the README consistent, while preserving the provenance/version fields of historical scientific reports. No new physical-validation claim follows from this license proposal.

## Repository-specific boundaries

- The audited main-line package version 0.2.0 remains MIT. Any existing 0.2.x copies, including work on the separate 0.2.1 development branch, are not relabeled by this proposal.
- Future 0.3.0 is the proposed MPL-2.0 boundary. Package metadata, runtime `__version__`, and the local project entry in `uv.lock` are set consistently to 0.3.0 in this review patch. No 0.3.0 release, tag or upload has been created.
- The baseline had package metadata 0.2.0 while runtime metadata was 0.1.0. Historical generated reports truthfully retain the version they recorded; their provenance is not changed to pretend they were produced by 0.3.0.
- A separate branch contains 0.2.1 runtime/version-contract work. Reconcile that work before an eventual 0.3.0 release; this license proposal imports none of its numerical changes.
- NumPy, optional plotting/build/test dependencies and linked geometry projects retain their own terms. No vendored dependency license file or submodule was found in the audited tree.

## Standard text source

- https://www.mozilla.org/media/MPL/2.0/index.txt
- Retrieved standard-text SHA-256: `3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04`.
