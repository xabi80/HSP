"""Milestone 1.5 — OrcaFlex VesselType YAML reader unit tests.

Fixture: ``tests/fixtures/bem/orcaflex/platform_small.yml`` — a 10-frequency,
2-heading OrcaFlex 11.2c demo VesselType export of an OC4 DeepCwind-shaped
semi-submersible. See ARCHITECTURE.md §8 M1.5 for the format contract.

Numerical spot-checks reference raw fixture values converted from OrcaFlex
"SI" (mass in tonnes, force in kN) to pure SI (kg, N) — a uniform factor of
1000 on every mass/force-related entry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.readers.orcaflex_vessel_yaml import read_orcaflex_vessel_yaml

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "bem" / "orcaflex" / "platform_small.yml"
)

# Tonnes → kg and kN → N share the same factor in OrcaFlex's "SI" system.
OFX_SI_SCALE = 1000.0


@pytest.fixture(scope="module")
def hdb() -> HydroDatabase:
    return read_orcaflex_vessel_yaml(FIXTURE_PATH)


# ---------- basic contract ----------


def test_reader_returns_hydro_database(hdb: HydroDatabase) -> None:
    assert isinstance(hdb, HydroDatabase)


def test_omega_grid_is_sorted_ascending_rad_per_s(hdb: HydroDatabase) -> None:
    # File lists frequencies as 1.0, 0.9, ..., 0.1 (descending); reader must sort.
    assert hdb.omega.ndim == 1
    assert hdb.omega.size == 10
    assert np.all(np.diff(hdb.omega) > 0.0)
    np.testing.assert_allclose(hdb.omega[0], 0.1, rtol=1e-12)
    np.testing.assert_allclose(hdb.omega[-1], 1.0, rtol=1e-12)


def test_heading_grid_matches_fixture(hdb: HydroDatabase) -> None:
    np.testing.assert_allclose(hdb.heading_deg, [0.0, 90.0])


def test_reference_point_from_reference_origin(hdb: HydroDatabase) -> None:
    # File sets ReferenceOrigin: [0, 0, 0] inside the Draught block.
    np.testing.assert_allclose(hdb.reference_point, [0.0, 0.0, 0.0])


# ---------- A_inf: Infinity entry in FrequencyDependentAddedMassAndDamping ----------


def test_a_inf_heave_heave_is_tonnes_times_1000(hdb: HydroDatabase) -> None:
    # Fixture row 3 col 3 at AMDPeriodOrFrequency: Infinity = 14.474094985810829e3 tonne.
    expected_kg = 14.474094985810829e3 * OFX_SI_SCALE
    np.testing.assert_allclose(hdb.A_inf[2, 2], expected_kg, rtol=1e-12)


def test_a_inf_pitch_pitch_is_tonne_m2_times_1000(hdb: HydroDatabase) -> None:
    # Row 5 col 5 at Infinity = 7.266623108528852e6 tonne*m^2.
    expected_kgm2 = 7.266623108528852e6 * OFX_SI_SCALE
    np.testing.assert_allclose(hdb.A_inf[4, 4], expected_kgm2, rtol=1e-12)


def test_a_inf_is_symmetric(hdb: HydroDatabase) -> None:
    np.testing.assert_allclose(hdb.A_inf, hdb.A_inf.T, atol=1e-6)


# ---------- A(omega), B(omega): frequency-dependent blocks ----------


def test_added_mass_at_omega_1_rad_per_s_heave_heave(hdb: HydroDatabase) -> None:
    # Fixture row 3 col 3 at AMDPeriodOrFrequency: 1 = 14.700313567584539e3 tonne.
    idx = int(np.argmin(np.abs(hdb.omega - 1.0)))
    np.testing.assert_allclose(hdb.omega[idx], 1.0, rtol=1e-12)
    expected = 14.700313567584539e3 * OFX_SI_SCALE
    np.testing.assert_allclose(hdb.A[2, 2, idx], expected, rtol=1e-12)


def test_damping_at_omega_1_rad_per_s_heave_heave(hdb: HydroDatabase) -> None:
    # Fixture row 3 col 3 Damping at omega=1 = 586.0986919308609 kN*s/m.
    idx = int(np.argmin(np.abs(hdb.omega - 1.0)))
    expected = 586.0986919308609 * OFX_SI_SCALE
    np.testing.assert_allclose(hdb.B[2, 2, idx], expected, rtol=1e-12)


# ---------- C: hydrostatic restoring ----------


def test_hydrostatic_c_heave(hdb: HydroDatabase) -> None:
    # HydrostaticStiffnessz row: [3648.231..., 0, 9.377...] — first entry is C_zz.
    expected = 3648.231299171783 * OFX_SI_SCALE
    np.testing.assert_allclose(hdb.C[2, 2], expected, rtol=1e-12)


def test_hydrostatic_c_roll(hdb: HydroDatabase) -> None:
    # HydrostaticStiffnessRx row: [0, 997.3393...e3, 0] — middle entry is C_RxRx.
    expected = 997.3393359600629e3 * OFX_SI_SCALE
    np.testing.assert_allclose(hdb.C[3, 3], expected, rtol=1e-12)


def test_hydrostatic_c_pitch(hdb: HydroDatabase) -> None:
    # HydrostaticStiffnessRy row: [9.377..., 0, 997.1690...e3] — last entry is C_RyRy.
    expected = 997.1690152668243e3 * OFX_SI_SCALE
    np.testing.assert_allclose(hdb.C[4, 4], expected, rtol=1e-12)


def test_hydrostatic_c_surge_sway_yaw_are_zero(hdb: HydroDatabase) -> None:
    # Only heave/roll/pitch carry hydrostatic restoring for an unmoored body.
    assert hdb.C[0, 0] == 0.0
    assert hdb.C[1, 1] == 0.0
    assert hdb.C[5, 5] == 0.0


# ---------- RAO: complex wave-excitation force per unit amplitude ----------


def test_rao_is_complex(hdb: HydroDatabase) -> None:
    assert np.issubdtype(hdb.RAO.dtype, np.complexfloating)


def test_rao_heave_at_omega_1_heading_0_matches_amp_and_phase(
    hdb: HydroDatabase,
) -> None:
    """LoadRAO heave at freq=1 rad/s, heading=0:
    amp = 1301.1292559517626 kN/m wave-amp, phase = 60.62380058043383 deg (leads).
    Complex representation with 'leads/degrees' convention:
        X = amp * exp(1j * phase_rad)
    """
    idx_w = int(np.argmin(np.abs(hdb.omega - 1.0)))
    idx_h = int(np.argmin(np.abs(hdb.heading_deg - 0.0)))
    amp = 1301.1292559517626 * OFX_SI_SCALE
    phase_rad = np.deg2rad(60.62380058043383)
    expected = amp * np.exp(1j * phase_rad)
    np.testing.assert_allclose(hdb.RAO[2, idx_w, idx_h], expected, rtol=1e-12)


def test_rao_pitch_at_omega_1_heading_0_matches_amp_and_phase(
    hdb: HydroDatabase,
) -> None:
    """LoadRAO pitch (index 4) at freq=1 rad/s, heading=0:
    amp = 19.595707457085602e3 kN*m/m, phase = 164.04262785507987 deg.
    """
    idx_w = int(np.argmin(np.abs(hdb.omega - 1.0)))
    idx_h = int(np.argmin(np.abs(hdb.heading_deg - 0.0)))
    amp = 19.595707457085602e3 * OFX_SI_SCALE
    phase_rad = np.deg2rad(164.04262785507987)
    expected = amp * np.exp(1j * phase_rad)
    np.testing.assert_allclose(hdb.RAO[4, idx_w, idx_h], expected, rtol=1e-12)


# ---------- metadata ----------


def test_metadata_records_source_and_file(hdb: HydroDatabase) -> None:
    assert "source" in hdb.metadata
    assert "orcaflex" in hdb.metadata["source"].lower()
    assert hdb.metadata.get("file", "").endswith("platform_small.yml")


# ---------- input validation ----------


def _load_fixture_text() -> str:
    return FIXTURE_PATH.read_text(encoding="utf-8")


def _write_modified_fixture(tmp_path: Path, replacements: list[tuple[str, str]]) -> Path:
    text = _load_fixture_text()
    for old, new in replacements:
        assert old in text, f"substring not in fixture: {old!r}"
        text = text.replace(old, new, 1)
    p = tmp_path / "modified.yml"
    p.write_text(text, encoding="utf-8")
    return p


def test_rejects_non_si_units_system(tmp_path: Path) -> None:
    modified = _write_modified_fixture(tmp_path, [("UnitsSystem: SI", "UnitsSystem: User")])
    with pytest.raises(ValueError, match="UnitsSystem"):
        read_orcaflex_vessel_yaml(modified)


def test_rejects_unsupported_waves_referred_to_by(tmp_path: Path) -> None:
    modified = _write_modified_fixture(
        tmp_path,
        [("WavesReferredToBy: frequency (rad/s)", "WavesReferredToBy: period (s)")],
    )
    with pytest.raises(ValueError, match="WavesReferredToBy"):
        read_orcaflex_vessel_yaml(modified)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_orcaflex_vessel_yaml(tmp_path / "does_not_exist.yml")


# ---------- sanity: raw YAML load works (guards fixture integrity) ----------


def test_fixture_is_valid_yaml() -> None:
    with FIXTURE_PATH.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert "VesselTypes" in data
    assert len(data["VesselTypes"]) >= 1
