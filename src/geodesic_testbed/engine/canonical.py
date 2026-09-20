"""Canonical JSON for artefacts that are committed, compared and hashed.

A report in this repository is evidence: it is checked in, regenerated in CI
and compared against the working tree, and its content hash is quoted as an
identity. None of that survives full ``repr`` precision. The last two or three
digits of a BLAS reduction, an ``svd`` or a ``trapezoid`` move between numpy
builds, between CPUs and between vectorisation widths, while the quantity being
measured has not changed at all -- so an artefact serialised at full precision
reports a difference where there is none, and a content hash stops meaning
"the same computation".

Rounding is *relative*, so it does not flatten small numbers: an error of
2.22e-16 is still recorded as 2.22e-16. What it removes is the trailing noise
below the level at which any threshold here is declared.

**What this buys, and what it does not.** Within one environment it makes the
content hash an identity: the same code on the same machine gives the same
bytes whatever the hash seed, the BLAS thread count or the working directory,
and ``tools/e2e.py`` asserts exactly that, a hundred cycles at a time.

Across environments it buys nothing, and no rounding rule could. ``sin``,
``cosh``, ``exp`` and every BLAS reduction are the platform's, they differ in
their last bits between builds, and an ODE integrated over two thousand steps
carries that difference upward. Four numpy builds here produce four different
content hashes. So the cross-environment claim is not a hash at all: it is that
every *reported value* agrees to a declared tolerance, which is both true and
more informative, and ``tests/test_committed_report.py`` is where it is made.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Significant digits every float in an artefact is rounded to before it is
#: serialised or hashed. Twelve is far below the level at which any threshold
#: in this repository is declared and far above the level at which platforms
#: disagree.
CANONICAL_DIGITS = 12


def canonical_float(value: float, digits: int = CANONICAL_DIGITS) -> float:
    """One float, rounded to ``digits`` significant figures, with no signed zero.

    ``-0.0`` and ``0.0`` are the same number and serialise differently, and
    which one a reduction produces depends on the order it summed in. They are
    the same here.
    """
    rounded = float(f"{float(value):.{digits}g}")
    return 0.0 if rounded == 0.0 else rounded


def jsonable(value: Any) -> Any:
    """Normalise to strict JSON: numpy scalars become Python, non-finite becomes null.

    ``NaN`` and ``Infinity`` are not JSON, and a report that only some parsers
    can read is not machine readable.  They arise here legitimately -- an order
    fit has nothing to fit when a method is already exact -- so they are
    recorded as ``null``.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return canonical_float(value)
    return value




def content_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a canonicalised payload.

    ``sort_keys`` so that the order a report was assembled in does not change
    its identity, and ``allow_nan=False`` so that a non-finite value that
    escaped :func:`jsonable` fails loudly rather than being written as a token
    no JSON parser accepts.
    """
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
