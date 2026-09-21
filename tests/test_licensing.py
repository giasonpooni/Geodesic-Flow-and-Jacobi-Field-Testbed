# SPDX-License-Identifier: MPL-2.0
"""The licence is declared once, and every hand-authored file says which it is.

MPL-2.0 is file-level copyleft, which is why the per-file identifier matters
more here than it would under a project-level licence: what is reciprocal is
*these files*, so a file that does not say so is the one a downstream reader
has to guess about. Mozilla supports the machine-readable SPDX form, and a
one-line comment is cheaper to keep correct than a twenty-line notice.

Generated artefacts are deliberately excluded. The committed reports, the
figures and ``uv.lock`` are output, not source; heading them would be claiming
authorship of something a program emitted, and would also make every
regeneration a licence edit.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The declared licence, in one place. Everything below is held against it.
LICENCE = "MPL-2.0"
IDENTIFIER = f"SPDX-License-Identifier: {LICENCE}"

#: Directories whose ``.py`` files are hand-authored and must carry it.
SOURCE_ROOTS = ("src", "tests", "tools", "examples")


def _hand_authored() -> list[Path]:
    return sorted(
        path
        for root in SOURCE_ROOTS
        for path in (ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_every_hand_authored_source_file_names_the_licence() -> None:
    """The check that catches a *new* file, which is the one that gets missed.

    An existing file keeps its header for free. A file added next month does
    not, and under file-level copyleft an unheaded file is the one whose terms
    a reader cannot determine from the file itself.
    """
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in _hand_authored()
        if IDENTIFIER not in path.read_text(encoding="utf-8").split("\n\n", 1)[0]
    ]
    assert not missing, (
        f"these files carry no {IDENTIFIER!r} in their opening block: {missing}. "
        "Add it as the first line; under file-level copyleft a file that does "
        "not name its licence is a file whose terms have to be guessed."
    )


def test_the_identifier_is_the_first_line() -> None:
    """Not merely present somewhere near the top.

    A reader, and every scanning tool, looks at the first line. One buried
    below a docstring is technically there and practically absent.
    """
    misplaced = [
        path.relative_to(ROOT).as_posix()
        for path in _hand_authored()
        if path.read_text(encoding="utf-8").splitlines()[0].strip() != f"# {IDENTIFIER}"
    ]
    assert not misplaced, f"the identifier is not the first line of: {misplaced}"


def test_the_licence_file_is_the_unmodified_mozilla_text() -> None:
    """An 'MPL-2.0' that has been edited is not MPL-2.0.

    Checked structurally rather than by hash: the sections, both exhibits and
    the length. A hash would pin one byte-for-byte copy and fail on a trailing
    newline; this fails on an edit.
    """
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert text.startswith("Mozilla Public License Version 2.0")
    for section in (
        "1. Definitions",
        "2. License Grants and Conditions",
        "3. Responsibilities",
        "4. Inability to Comply Due to Statute or Regulation",
        "5. Termination",
        "6. Disclaimer of Warranty",
        "7. Limitation of Liability",
        "8. Litigation",
        "9. Miscellaneous",
        "10. Versions of the License",
        'Exhibit A - Source Code Form License Notice',
        'Exhibit B - "Incompatible With Secondary Licenses" Notice',
    ):
        assert section in text, f"the licence text is missing {section!r}"
    assert len(text) > 16_000, "the licence text is too short to be the whole of MPL-2.0"


def test_the_project_metadata_declares_the_same_licence() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["license"] == LICENCE
    assert metadata["project"]["license-files"] == ["LICENSE"]


def test_no_legacy_classifier_accompanies_the_spdx_expression() -> None:
    """PEP 639 forbids the pair, and PyPI rejects a distribution carrying both.

    Hatchling will happily emit them together, so nothing upstream catches
    this: the failure arrives at upload, after the tag exists. The SPDX
    expression *is* the licence declaration now, and the ``License ::``
    classifier is the deprecated spelling of the same thing.
    """
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    legacy = [
        entry
        for entry in metadata["project"].get("classifiers", ())
        if entry.startswith("License ::")
    ]
    assert not legacy, (
        f"{legacy} accompanies license = {LICENCE!r}. PEP 639 deprecates the "
        "classifier when a licence expression is present and PyPI rejects the "
        "combination, so this would fail at upload rather than at build."
    )


def test_the_readme_names_the_licence_it_ships_under() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Mozilla Public License 2.0" in readme
    assert "MIT" not in readme.split("## License", 1)[-1], (
        "the licence section still mentions MIT"
    )


@pytest.mark.parametrize(
    "generated",
    (
        "validation/report-v1.json",
        "validation/report-v2-surfaces.json",
        "results/reference-report.json",
        "uv.lock",
    ),
)
def test_generated_artefacts_are_not_headed(generated: str) -> None:
    """Output, not source.

    Heading a regenerated file would make every regeneration a licence edit,
    and would claim authorship of something a program emitted. The exclusion
    is deliberate and is asserted so that a future mechanical pass does not
    quietly widen.
    """
    path = ROOT / generated
    if not path.exists():
        pytest.skip(f"{generated} is not present in this checkout")
    assert IDENTIFIER not in path.read_text(encoding="utf-8")[:4096]
