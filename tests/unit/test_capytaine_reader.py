"""Unit tests for ``floatsim.hydro.readers.capytaine`` (M5 PR2).

Two committed fixtures drive these tests:

* ``synthetic_simple.nc`` -- hand-authored Capytaine-schema NetCDF with
  two finite omegas + one ``omega = +inf`` row, two wave directions,
  and a known excitation phase. Lets us assert exact (rtol=1e-12)
  values for every transformation the reader performs (DOF reorder,
  complex merge, lags->leads conjugation, infinite-frequency split,
  rad->deg heading).
* ``synthetic_sphere.nc`` -- hand-authored Capytaine-schema NetCDF
  encoding the deeply-submerged sphere case (Q4): ``A_ii``
  identically equal to ``(2/3) pi rho R^3`` for the three translational
  DOFs, ``B = 0``, ``C = 0``, ``F_exc = 0``. Confirms the analytical
  reference round-trips through the reader unchanged.

Both fixtures are produced by
``scripts/build_capytaine_synthetic_fixtures.py``. The real
Capytaine-generated sphere case (built by
``scripts/build_sphere_capytaine_fixture.py``) lands in PR3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.readers import load_hydro_database, read_capytaine

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "bem" / "capytaine"
_FIXTURE_SIMPLE = _FIXTURE_DIR / "synthetic_simple.nc"
_FIXTURE_SPHERE = _FIXTURE_DIR / "synthetic_sphere.nc"


# ---------------------------------------------------------------------------
# Fixture availability sanity check
# ---------------------------------------------------------------------------


def test_synthetic_fixtures_committed() -> None:
    """Both synthetic NetCDF files must exist on disk."""
    assert _FIXTURE_SIMPLE.is_file(), (
        f"Missing fixture {_FIXTURE_SIMPLE}; run " "scripts/build_capytaine_synthetic_fixtures.py"
    )
    assert _FIXTURE_SPHERE.is_file(), (
        f"Missing fixture {_FIXTURE_SPHERE}; run " "scripts/build_capytaine_synthetic_fixtures.py"
    )


# ---------------------------------------------------------------------------
# read_capytaine on synthetic_simple.nc
# ---------------------------------------------------------------------------


def test_simple_returns_hydrodatabase_with_correct_shapes() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    assert isinstance(db, HydroDatabase)
    assert db.A.shape == (6, 6, 2)
    assert db.B.shape == (6, 6, 2)
    assert db.A_inf.shape == (6, 6)
    assert db.C.shape == (6, 6)
    assert db.RAO.shape == (6, 2, 2)
    assert db.reference_point.shape == (3,)


def test_simple_omega_grid_excludes_infinite_row_and_is_sorted() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    np.testing.assert_array_equal(db.omega, np.array([0.5, 1.0]))
    assert np.all(np.isfinite(db.omega))


def test_simple_heading_converted_rad_to_deg() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    np.testing.assert_allclose(db.heading_deg, np.array([0.0, 45.0]), rtol=0, atol=1.0e-12)


def test_simple_a_inf_extracted_from_inf_omega_row() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    # Fixture script wrote diag_at_inf = 0.5 * diag_at_finite
    expected_diag = np.array([5.0e4, 5.5e4, 6.0e4, 5.0e6, 5.5e6, 6.0e6])
    np.testing.assert_allclose(np.diag(db.A_inf), expected_diag, rtol=1.0e-12, atol=0)


def test_simple_added_mass_diagonal_at_first_omega() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    # Fixture script: diag(A[..., k=0]) = diag_at_finite * (1 + 0.1 * 0)
    expected = np.array([1.0e5, 1.1e5, 1.2e5, 1.0e7, 1.1e7, 1.2e7])
    np.testing.assert_allclose(np.diag(db.A[..., 0]), expected, rtol=1.0e-12, atol=0)


def test_simple_radiation_damping_diagonal_at_second_omega() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    # Fixture script: diag(B[..., k=1]) = 5e3 * (1 + 0.05 * 1) * [1.0, 1.1, 1.2, 100, 110, 120]
    expected = 5.0e3 * 1.05 * np.array([1.0, 1.1, 1.2, 100.0, 110.0, 120.0])
    np.testing.assert_allclose(np.diag(db.B[..., 1]), expected, rtol=1.0e-12, atol=0)


def test_simple_hydrostatic_stiffness_diagonal_only_heave_roll_pitch() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    # Fixture: C[2,2]=4e5, C[3,3]=C[4,4]=7e6, all else zero.
    np.testing.assert_allclose(np.diag(db.C), np.array([0, 0, 4e5, 7e6, 7e6, 0]), atol=0)
    off = db.C - np.diag(np.diag(db.C))
    assert np.all(off == 0.0)


def test_simple_excitation_lags_to_leads_conjugation() -> None:
    """RAO must conjugate Capytaine's complex value (lags -> leads)."""
    db = read_capytaine(_FIXTURE_SIMPLE)
    # Fixture wrote F_capy[heave, omega=0.5, h=0] = 1e6 * exp(-i * 30 deg).
    # Conjugated: RAO_floatsim = 1e6 * exp(+i * 30 deg).
    rao_heave = db.RAO[2, 0, 0]  # (DOF=heave=2, omega index=0, heading index=0)
    expected = 1.0e6 * np.exp(+1j * np.deg2rad(30.0))
    np.testing.assert_allclose(rao_heave, expected, rtol=1.0e-12, atol=0)


def test_simple_excitation_frequency_dependent_phase() -> None:
    """Reader preserves the (k * 10 deg + h * 5 deg) phase progression."""
    db = read_capytaine(_FIXTURE_SIMPLE)
    # Fixture: phase at (k=1, h=1) is 30 + 10 + 5 = 45 deg (Capytaine arg = -45 deg)
    # Magnitude: 1e6 * (1 - 0.1 * 1) * 0.5 = 4.5e5 N
    # Reader conjugates -> RAO arg = +45 deg.
    rao = db.RAO[2, 1, 1]
    expected = 4.5e5 * np.exp(+1j * np.deg2rad(45.0))
    np.testing.assert_allclose(rao, expected, rtol=1.0e-12, atol=0)


def test_simple_surge_excitation_reflects_dof_reorder() -> None:
    """Surge column was written into Capytaine DOF index 0; FloatSim DOF=0 surge."""
    db = read_capytaine(_FIXTURE_SIMPLE)
    # Surge magnitude: 1e6 * 0.3 * 1.0 (k=0, h=0)
    # Phase: -(30 + 20) deg in Capytaine -> +50 deg leads.
    rao_surge = db.RAO[0, 0, 0]
    expected = 3.0e5 * np.exp(+1j * np.deg2rad(50.0))
    np.testing.assert_allclose(rao_surge, expected, rtol=1.0e-12, atol=0)


def test_simple_default_reference_point_is_origin() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    np.testing.assert_array_equal(db.reference_point, np.zeros(3))


def test_simple_explicit_reference_point_overrides_default() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE, reference_point=(1.5, -2.5, 3.0))
    np.testing.assert_array_equal(db.reference_point, np.array([1.5, -2.5, 3.0]))


def test_simple_metadata_includes_source_format_and_attrs() -> None:
    db = read_capytaine(_FIXTURE_SIMPLE)
    assert db.metadata["source_format"] == "capytaine"
    assert "synthetic_simple" in db.metadata.get("body_name", "")
    assert "rho" in db.metadata


# ---------------------------------------------------------------------------
# Sphere analytical match
# ---------------------------------------------------------------------------


def test_sphere_added_mass_matches_lamb_formula() -> None:
    """A_ii must equal (2/3) pi rho R^3 to machine precision in the synthetic case."""
    db = read_capytaine(_FIXTURE_SPHERE)
    rho = 1025.0
    radius = 1.0
    a_analytical = (2.0 / 3.0) * np.pi * rho * radius**3
    np.testing.assert_allclose(db.A[0, 0, :], a_analytical, rtol=1.0e-12, atol=0)
    np.testing.assert_allclose(db.A[1, 1, :], a_analytical, rtol=1.0e-12, atol=0)
    np.testing.assert_allclose(db.A[2, 2, :], a_analytical, rtol=1.0e-12, atol=0)


def test_sphere_added_mass_off_diagonal_zero() -> None:
    db = read_capytaine(_FIXTURE_SPHERE)
    A = db.A.copy()
    for k in range(A.shape[-1]):
        np.fill_diagonal(A[..., k], 0.0)
    assert np.max(np.abs(A)) == 0.0


def test_sphere_radiation_damping_zero() -> None:
    db = read_capytaine(_FIXTURE_SPHERE)
    assert np.max(np.abs(db.B)) == 0.0


def test_sphere_hydrostatic_stiffness_zero() -> None:
    db = read_capytaine(_FIXTURE_SPHERE)
    assert np.max(np.abs(db.C)) == 0.0


def test_sphere_excitation_zero() -> None:
    db = read_capytaine(_FIXTURE_SPHERE)
    assert np.max(np.abs(db.RAO)) == 0.0


def test_sphere_a_inf_matches_finite_added_mass() -> None:
    """For a deeply submerged sphere, A is frequency-independent so A_inf == A(omega)."""
    db = read_capytaine(_FIXTURE_SPHERE)
    np.testing.assert_allclose(db.A_inf, db.A[..., 0], rtol=1.0e-12, atol=0)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_read_capytaine_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        read_capytaine(_FIXTURE_DIR / "does_not_exist.nc")


def test_read_capytaine_missing_radiation_dim_raises(tmp_path: Path) -> None:
    """A NetCDF file without the required schema dims must raise ValueError."""
    bad = tmp_path / "bad_dims.nc"
    ds = xr.Dataset(
        data_vars={"foo": (("x",), np.zeros(3))},
        coords={"x": np.arange(3)},
    )
    ds.to_netcdf(bad)
    with pytest.raises(ValueError, match="missing required dimensions"):
        read_capytaine(bad)


def test_read_capytaine_inf_omega_AND_a_inf_kwarg_raises() -> None:
    """Specifying both an inf-frequency row and a_inf is ambiguous."""
    a_inf_dummy = np.eye(6) * 1.0e3
    with pytest.raises(ValueError, match="already contains an omega=inf"):
        read_capytaine(_FIXTURE_SIMPLE, a_inf=a_inf_dummy)


def test_read_capytaine_no_inf_omega_and_no_a_inf_kwarg_raises(tmp_path: Path) -> None:
    """If neither source for A_inf is available, the reader must say so."""
    src = xr.open_dataset(_FIXTURE_SIMPLE).load()
    no_inf = src.isel(omega=src.omega != np.inf)
    out = tmp_path / "no_inf.nc"
    encoding = {
        "radiating_dof": {"dtype": "S16"},
        "influenced_dof": {"dtype": "S16"},
        "complex": {"dtype": "S2"},
    }
    no_inf.to_netcdf(out, encoding=encoding)
    src.close()
    with pytest.raises(ValueError, match="no omega=inf sample"):
        read_capytaine(out)


def test_read_capytaine_caller_supplied_a_inf_used_when_no_inf_row(tmp_path: Path) -> None:
    """When the file has no omega=inf row, an a_inf kwarg must be honoured."""
    src = xr.open_dataset(_FIXTURE_SIMPLE).load()
    no_inf = src.isel(omega=src.omega != np.inf)
    out = tmp_path / "no_inf.nc"
    encoding = {
        "radiating_dof": {"dtype": "S16"},
        "influenced_dof": {"dtype": "S16"},
        "complex": {"dtype": "S2"},
    }
    no_inf.to_netcdf(out, encoding=encoding)
    src.close()

    a_inf = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    db = read_capytaine(out, a_inf=a_inf)
    np.testing.assert_array_equal(db.A_inf, a_inf)


def test_read_capytaine_rejects_unknown_dof_labels(tmp_path: Path) -> None:
    """If radiating_dof contains labels other than the canonical six, reject."""
    src = xr.open_dataset(_FIXTURE_SIMPLE).load()
    bogus = np.array(["Surge", "Sway", "Heave", "Roll", "Pitch", "WeirdMode"], dtype="U16")
    src = src.assign_coords(radiating_dof=bogus, influenced_dof=bogus)
    out = tmp_path / "bogus_dofs.nc"
    encoding = {
        "radiating_dof": {"dtype": "S16"},
        "influenced_dof": {"dtype": "S16"},
        "complex": {"dtype": "S2"},
    }
    src.to_netcdf(out, encoding=encoding)
    src.close()
    with pytest.raises(ValueError, match="rigid-body names"):
        read_capytaine(out)


def test_read_capytaine_handles_dof_order_permutation(tmp_path: Path) -> None:
    """Reader must reorder when Capytaine writes DOFs in a non-canonical order."""
    src = xr.open_dataset(_FIXTURE_SIMPLE).load()
    # Permute Capytaine DOF order: put Heave first.
    perm = [2, 0, 1, 3, 4, 5]
    src = src.isel(radiating_dof=perm, influenced_dof=perm)
    out = tmp_path / "permuted.nc"
    encoding = {
        "radiating_dof": {"dtype": "S16"},
        "influenced_dof": {"dtype": "S16"},
        "complex": {"dtype": "S2"},
    }
    src.to_netcdf(out, encoding=encoding)
    src.close()

    db = read_capytaine(out)
    # FloatSim canonical order should still place Surge at index 0 with 1e5 added mass.
    expected_diag = np.array([1.0e5, 1.1e5, 1.2e5, 1.0e7, 1.1e7, 1.2e7])
    np.testing.assert_allclose(np.diag(db.A[..., 0]), expected_diag, rtol=1.0e-12)


def test_read_capytaine_falls_back_to_FK_plus_diffraction(tmp_path: Path) -> None:
    """When excitation_force is absent, sum FK + diffraction with same conventions."""
    src = xr.open_dataset(_FIXTURE_SIMPLE).load()
    # Move excitation_force to FK; add a separate diffraction = 0 variable.
    fk = src["excitation_force"].copy()
    diff = src["excitation_force"].copy() * 0.0
    no_exc = src.drop_vars("excitation_force")
    no_exc["Froude_Krylov_force"] = fk
    no_exc["diffraction_force"] = diff
    out = tmp_path / "fk_plus_diff.nc"
    encoding = {
        "radiating_dof": {"dtype": "S16"},
        "influenced_dof": {"dtype": "S16"},
        "complex": {"dtype": "S2"},
    }
    no_exc.to_netcdf(out, encoding=encoding)
    src.close()

    db_fk = read_capytaine(out)
    db_exc = read_capytaine(_FIXTURE_SIMPLE)
    np.testing.assert_allclose(db_fk.RAO, db_exc.RAO, rtol=1.0e-12, atol=0)


# ---------------------------------------------------------------------------
# load_hydro_database dispatch
# ---------------------------------------------------------------------------


def test_dispatcher_routes_capytaine_format_to_read_capytaine() -> None:
    db = load_hydro_database(_FIXTURE_SPHERE, format="capytaine")
    assert db.metadata["source_format"] == "capytaine"
    rho = 1025.0
    a = (2.0 / 3.0) * np.pi * rho * 1.0**3
    np.testing.assert_allclose(np.diag(db.A_inf)[:3], np.full(3, a), rtol=1.0e-12)
