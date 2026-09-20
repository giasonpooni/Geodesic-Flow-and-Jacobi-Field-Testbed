"""The boundary: what crosses it, and the one-way rule that keeps it narrow.

Two kinds of check live here. The first reads the imports out of the source and
asserts the direction of dependency: the modules that compute the answer may
not know that an instrument exists. The second exercises the contract itself --
units, frame, arclength grid, covariance, provenance, calibration identifiers
and observation mode -- across a write and a read, because a boundary that has
never survived leaving the process is a type, not a boundary.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from geodesic_testbed import constant_curvature_trace
from geodesic_testbed.boundary import (
    CONSUMERS,
    CONTRACT,
    INSTRUMENT_FACING,
    LAYERS,
    RUNTIME_VERSION,
    SUBSTRATE,
    CalibrationBinding,
    ConvergenceEstimate,
    GeometryUncertainty,
    Provenance,
    StartingCovariance,
    TransferRecord,
    Units,
    UpstreamArtefact,
    read_record,
    write_record,
)
from geodesic_testbed.engine.envelope import estimate_convergence, integrate_path
from geodesic_testbed.engine.record import RECORD_SCHEMA, SUPPORTED_RECORD_SCHEMAS
from geodesic_testbed.engine.surfaces import torus

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SOURCE = (ROOT / "src" / "geodesic_testbed" / "engine" / "contract.py").read_text()
SOURCE = ROOT / "src"
GRID = np.linspace(0.0, 1.25, 126)


def _record():
    return constant_curvature_trace(GRID, 1.0).as_transfer_record()


def _surface_record():
    return integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=400
    ).as_transfer_record()


# -- the one-way rule ------------------------------------------------------


def _module_path(dotted: str) -> Path:
    return SOURCE / (dotted.replace(".", "/") + ".py")


def _imported_modules(dotted: str) -> set[str]:
    """Every module the named module imports, resolved to dotted form.

    Read from the syntax tree rather than by importing: importing would run the
    module, and a rule about what a module is *allowed* to depend on should not
    be enforced by a mechanism that depends on it.
    """
    path = _module_path(dotted)
    package = dotted.rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                parts = package.split(".")
                base_parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join([*base_parts, node.module] if node.module else base_parts)
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def test_every_declared_module_exists() -> None:
    for layer in LAYERS.values():
        for dotted in layer:
            assert _module_path(dotted).is_file(), dotted


def test_every_module_belongs_to_exactly_one_layer() -> None:
    """A module in no layer is a module the one-way rule does not reach.

    Without this, the rule is enforced only against the modules that thought to
    declare themselves, which is the same as not enforcing it: a new module
    that imports across the boundary passes by not being listed.
    """
    on_disk = {
        ".".join(path.relative_to(SOURCE).with_suffix("").parts)
        for path in SOURCE.rglob("*.py")
        if path.name != "__init__.py"
    }
    declared = [dotted for layer in LAYERS.values() for dotted in layer]
    assert len(declared) == len(set(declared)), "a module is in two layers"
    assert on_disk == set(declared), {
        "unlisted": sorted(on_disk - set(declared)),
        "listed but absent": sorted(set(declared) - on_disk),
    }


def test_the_substrate_does_not_import_the_instrument_side() -> None:
    """Geometry in, transfer map out. Nothing in between knows a sensor exists.

    This is the whole separation, stated as something a machine can check. A
    filter, a tracking schedule or a route criterion reached from inside the
    solver would make the numerics depend on a policy, and the dependency would
    be invisible in the reports the solver produces.
    """
    for dotted in SUBSTRATE:
        leaked = sorted(_imported_modules(dotted) & set(INSTRUMENT_FACING))
        assert not leaked, f"{dotted} imports instrument-facing {leaked}"


def test_the_contract_depends_on_neither_side() -> None:
    """The record is shared, so it may not carry either side's machinery."""
    for dotted in CONTRACT:
        imported = _imported_modules(dotted)
        assert not sorted(imported & set(INSTRUMENT_FACING)), dotted
        # ``record`` needs ``transfer`` for the map it wraps; nothing else in
        # the contract may reach into the solver at all.
        allowed = {"geodesic_testbed.engine.transfer"} if dotted.endswith("record") else set()
        assert not sorted((imported & set(SUBSTRATE)) - allowed), dotted


def test_the_application_contracts_do_not_import_the_instrument_side() -> None:
    """They answer a tolerance question from a record, not from a protocol."""
    for dotted in CONSUMERS:
        leaked = sorted(_imported_modules(dotted) & set(INSTRUMENT_FACING))
        assert not leaked, f"{dotted} imports instrument-facing {leaked}"


def test_the_contract_module_carries_no_numerics() -> None:
    """``contract.py`` states the vocabulary and computes nothing with it."""
    imported = _imported_modules("geodesic_testbed.engine.contract")
    assert imported <= {"__future__", "dataclasses", "typing", "numpy"} | {
        name for name in imported if name.startswith(("dataclasses.", "typing.", "__future__."))
    }


def test_the_version_is_declared_in_one_place() -> None:
    """Provenance naming a version the package does not have is worse than none.

    ``pyproject.toml`` used to carry its own literal, and the two drifted: it
    said 0.2.0 while the package and both reports said 0.1.0.  It now declares
    the version dynamic and reads ``RUNTIME_VERSION`` out of ``contract.py``,
    so there is one string and nothing to hold together by hand.  What is left
    to check is that the arrangement is still in force.
    """
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    import geodesic_testbed

    assert "version" not in metadata["project"], (
        "pyproject declares a literal version again; that is the second source "
        "of truth this test exists to prevent"
    )
    assert metadata["project"]["dynamic"] == ["version"]
    source = metadata["tool"]["hatch"]["version"]
    assert source["path"] == "src/geodesic_testbed/engine/contract.py"
    assert re.search(source["pattern"], CONTRACT_SOURCE).group("version") == RUNTIME_VERSION
    assert geodesic_testbed.__version__ == RUNTIME_VERSION


def test_the_built_distribution_carries_the_declared_version() -> None:
    """The dynamic hook is what the build backend runs, so run it.

    A pattern that matched nothing would fail the build rather than the test
    suite, which is the wrong place to find out.
    """
    metadata = version("curved-surface-geodesic-sensitivity")
    assert metadata == RUNTIME_VERSION, (
        f"the installed distribution reports {metadata}, the source declares "
        f"{RUNTIME_VERSION}; reinstall or fix [tool.hatch.version]"
    )


# -- the contract ----------------------------------------------------------


def test_the_record_carries_every_field_the_boundary_promises() -> None:
    payload = _surface_record().to_dict(include_samples=False)
    for key in (
        "units",
        "frame",
        "grid",
        "covariance",
        "provenance",
        "calibration",
        "observation_mode",
    ):
        assert key in payload, key
    assert payload["schema"] == RECORD_SCHEMA
    assert payload["grid"]["samples"] == payload["samples"]


def test_a_record_survives_a_round_trip_through_a_file(tmp_path) -> None:
    record = (
        _surface_record()
        .with_covariance(StartingCovariance.from_tolerance_box(1.0e-3, 1.0e-3))
        .with_calibration(
            CalibrationBinding(
                calibration_ids=("cal-2026-03-11",),
                registration_id="reg-7",
                reconstruction_version="recon-2.1",
            )
        )
        .with_provenance(
            Provenance(note="round trip").with_upstream(
                UpstreamArtefact(kind="path-artefact", identifier="coupon-4", version="v1")
            )
        )
    )
    path = write_record(record, tmp_path / "record.json")
    restored = read_record(path)

    for name in ("arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate"):
        assert np.array_equal(getattr(restored, name), getattr(record, name))
    assert restored.to_dict(include_samples=False) == record.to_dict(include_samples=False)
    assert restored.calibration.calibration_ids == ("cal-2026-03-11",)
    assert restored.provenance.upstream[0].identifier == "coupon-4"
    assert np.allclose(
        restored.propagate_declared_covariance(), record.propagate_declared_covariance()
    )


def test_a_summary_is_not_a_record(tmp_path) -> None:
    """A payload written without samples must not come back as a record."""
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(_record().to_dict(include_samples=False)), encoding="utf-8")
    with pytest.raises(ValueError, match="no samples"):
        read_record(path)


def test_an_unknown_schema_is_refused(tmp_path) -> None:
    payload = _record().to_dict()
    payload["schema"] = "path-transfer-record-v99"
    path = tmp_path / "future.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown record schema"):
        read_record(path)


def test_a_v1_payload_reads_back_with_the_new_fields_undeclared(tmp_path) -> None:
    """The fields did not exist in v1, and they come back saying exactly that."""
    payload = _record().to_dict()
    payload["schema"] = "path-transfer-record-v1"
    for key in ("covariance", "provenance", "calibration", "grid", "contract"):
        payload.pop(key, None)
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = read_record(path)
    assert "path-transfer-record-v1" in SUPPORTED_RECORD_SCHEMAS
    assert not restored.covariance.declared
    assert not restored.calibration.bound
    assert restored.grid["samples"] == GRID.size


# -- the fields that must not be guessed -----------------------------------


def test_a_missing_covariance_is_an_error_and_not_a_default() -> None:
    with pytest.raises(ValueError, match="declares no starting covariance"):
        _record().propagate_declared_covariance()


def test_a_covariance_from_a_tolerance_box_records_its_coverage_factor() -> None:
    covariance = StartingCovariance.from_tolerance_box(3.0e-3, 6.0e-3, coverage_factor=3.0)
    assert covariance.basis == "declared-tolerance-box"
    assert covariance.coverage_factor == 3.0
    assert np.allclose(np.sqrt(np.diag(covariance.require())), [1.0e-3, 2.0e-3])


def test_a_covariance_in_another_frame_or_unit_is_refused() -> None:
    record = _record()
    millimetres = Units(length="millimetre", angle="radian")
    with pytest.raises(ValueError, match="millimetre"):
        record.with_covariance(
            StartingCovariance.from_tolerance_box(1.0, 1.0, units=millimetres)
        )


def test_a_covariance_must_be_symmetric_and_positive_semi_definite() -> None:
    for matrix in ([[1.0, 2.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]]):
        with pytest.raises(ValueError):
            StartingCovariance(matrix=np.array(matrix), basis="measured")


def test_a_declared_matrix_must_declare_a_basis() -> None:
    with pytest.raises(ValueError, match="not-declared"):
        StartingCovariance(matrix=np.eye(2))


def test_an_unregistered_frame_is_refused() -> None:
    record = _record()
    fields = {
        name: getattr(record, name)
        for name in ("arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate")
    }
    with pytest.raises(KeyError, match="unknown frame"):
        TransferRecord(**fields, frame="whatever-the-camera-was-bolted-to")


def test_two_unbound_records_do_not_agree_on_calibration() -> None:
    """Absence of a calibration is not evidence of a shared one."""
    assert not CalibrationBinding.unbound().agrees_with(CalibrationBinding.unbound())
    bound = CalibrationBinding(calibration_ids=("cal-1",), reconstruction_version="r1")
    assert bound.agrees_with(
        CalibrationBinding(calibration_ids=("cal-1",), reconstruction_version="r1")
    )
    assert not bound.agrees_with(
        CalibrationBinding(calibration_ids=("cal-1",), reconstruction_version="r2")
    )


def test_a_record_from_geometry_alone_declares_no_instrument() -> None:
    """The honest default: nothing was calibrated, so nothing claims to be."""
    for record in (_record(), _surface_record()):
        assert not record.calibration.bound
        assert not record.covariance.declared
        assert record.provenance.producer_version == RUNTIME_VERSION


# -- the path the record describes ----------------------------------------


def test_the_record_carries_the_path_and_not_only_the_map() -> None:
    """A consumer that must point at the path cannot recompute it."""
    record = _surface_record()
    geometry = record.geometry
    assert geometry is not None
    assert geometry.samples == record.arclength.size
    assert geometry.position.shape == (record.arclength.size, 3)
    for name in ("tangent", "transverse", "surface_normal"):
        vectors = getattr(geometry, name)
        assert np.allclose(np.linalg.norm(vectors, axis=-1), 1.0, atol=1e-12)


def test_the_frame_is_carried_as_vectors_so_the_frame_name_can_be_checked() -> None:
    """``frame`` says parallel-transported; these are what it actually is."""
    geometry = _surface_record().geometry
    assert geometry.orientation_residual() < 1e-12
    for left, right in (
        ("tangent", "surface_normal"),
        ("tangent", "transverse"),
        ("transverse", "surface_normal"),
    ):
        products = np.einsum(
            "ij,ij->i", getattr(geometry, left), getattr(geometry, right)
        )
        assert np.max(np.abs(products)) < 1e-12


def test_both_normal_curvatures_are_carried_and_satisfy_eulers_theorem() -> None:
    """Two quantities both called kappa_n, and only one sets the chord shortfall."""
    envelope = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=400
    )
    geometry = envelope.as_transfer_record().geometry
    mean = np.asarray(envelope.surface.mean_curvature(envelope.u, envelope.v), dtype=float)
    assert np.allclose(
        geometry.normal_curvature_along + geometry.normal_curvature_transverse,
        2.0 * mean,
        atol=1e-12,
    )
    assert not np.allclose(
        geometry.normal_curvature_along, geometry.normal_curvature_transverse
    )


def test_a_curvature_profile_declares_no_geometry_and_no_chart() -> None:
    """Both absences are the right answer: a profile is not an embedding."""
    record = _record()
    assert record.geometry is None
    assert record.chart is None
    assert record.path_type == "geodesic"
    assert "K(s) is given as the curvature along a geodesic" in record.path_type_basis


def test_a_truncated_path_says_it_is_shorter_than_the_one_requested() -> None:
    """A route that ends because the chart ran out is not a route that finished."""
    from geodesic_testbed.engine.surfaces import pseudosphere

    record = integrate_path(
        pseudosphere(), u0=0.6, v0=0.0, heading=1.2, length=6.0, n_steps=600
    ).as_transfer_record()
    chart = record.chart
    assert chart.truncated is True
    assert chart.complete is False
    assert chart.samples < chart.requested_samples
    assert chart.requested_length == 6.0
    assert chart.reason
    assert int(chart.mask().sum()) == chart.samples


def test_the_geometry_and_the_transfer_map_must_be_the_same_path() -> None:
    record = _surface_record()
    fields = {
        name: getattr(record, name)[:-1]
        for name in ("arclength", "gaussian_curvature", "a", "a_rate", "b", "b_rate")
    }
    with pytest.raises(ValueError, match="the same record"):
        TransferRecord(**fields, domain="parametric-surface",
                       observation_mode="ambient-euclidean-chord",
                       geometry=record.geometry)


def test_geometry_from_a_formula_declares_zero_uncertainty_and_means_it() -> None:
    """``analytic`` is a stronger statement than ``not-declared``, not a synonym."""
    uncertainty = _surface_record().geometry.uncertainty
    assert uncertainty.declared
    assert uncertainty.basis == "analytic"
    assert uncertainty.position == 0.0


def test_a_geometry_uncertainty_can_be_attached_later() -> None:
    """How well a surface is known belongs to whoever measured it."""
    record = _surface_record().with_geometry_uncertainty(
        GeometryUncertainty(
            position=0.015, normal=1.0e-4, curvature=2.0e-3,
            basis="as-built-scan", note="coupon 4, scan 2026-03-11",
        )
    )
    assert record.geometry.uncertainty.basis == "as-built-scan"
    assert record.geometry.uncertainty.position == 0.015
    with pytest.raises(ValueError, match="carries no geometry"):
        _record().with_geometry_uncertainty(GeometryUncertainty.analytic())


def test_a_record_with_geometry_survives_the_round_trip(tmp_path) -> None:
    record = _surface_record()
    restored = read_record(write_record(record, tmp_path / "geometry.json"))
    for name in (
        "position", "tangent", "transverse", "surface_normal",
        "normal_curvature_along", "normal_curvature_transverse",
    ):
        assert np.array_equal(
            getattr(restored.geometry, name), getattr(record.geometry, name)
        )
    assert restored.chart.to_dict() == record.chart.to_dict()
    assert restored.path_type == record.path_type


def test_the_record_carries_what_the_solve_cost_per_quantity() -> None:
    """An order is a statement about the limit; this is about the run in hand."""
    envelope = integrate_path(
        torus(2.0, 1.0), u0=0.3, v0=0.2, heading=0.6, length=2.0, n_steps=400
    )
    bare = envelope.as_transfer_record()
    assert not bare.resolution.convergence.established

    measured = envelope.as_transfer_record(convergence=estimate_convergence(envelope))
    convergence = measured.resolution.convergence
    assert convergence.established
    assert convergence.order == 4
    for name in ("position", "transfer", "curvature", "covariance"):
        assert getattr(convergence, name) is not None
        assert getattr(convergence, name) >= 0.0
    assert convergence.worst == max(
        convergence.position, convergence.transfer,
        convergence.curvature, convergence.covariance,
    )


def test_an_unestablished_convergence_may_not_carry_numbers() -> None:
    with pytest.raises(ValueError, match="not-established"):
        ConvergenceEstimate(transfer=1e-9)
