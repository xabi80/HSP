"""Unit tests for the WAMIT plain-text reader.

Each test asserts exact values against the hand-authored synthetic fixture
``tests/fixtures/bem/wamit/synthetic_simple.{1,3,hst,4}``. Period values in
the fixture round-trip cleanly to omega = {0.5, 1.0} rad/s — preserving
this invariant matters because the parser keys excitation rows on
``omega = 2*pi/PER`` matched against the .1 grid.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.readers import load_hydro_database
from floatsim.hydro.readers.wamit import (
    read_added_mass_and_damping,
    read_excitation_force,
    read_hydrostatic_stiffness,
    read_motion_rao,
    read_wamit,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "bem" / "wamit"
_STEM = _FIXTURE_DIR / "synthetic_simple"


# ---------------------------------------------------------------------------
# .1 parser
# ---------------------------------------------------------------------------


def test_dot1_omega_grid_is_sorted_ascending() -> None:
    omega, _A, _B, _A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"))
    np.testing.assert_allclose(omega, [0.5, 1.0], rtol=1e-12)


def test_dot1_A_inf_diagonal_matches_fixture() -> None:
    _omega, _A, _B, A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"))
    expected_diag = np.array([5.0e4, 5.0e4, 1.0e5, 5.0e6, 5.0e6, 1.0e7])
    np.testing.assert_allclose(np.diag(A_inf), expected_diag, rtol=1e-12)


def test_dot1_A_inf_off_diagonals_are_zero() -> None:
    _omega, _A, _B, A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"))
    off = A_inf - np.diag(np.diag(A_inf))
    assert np.all(off == 0.0)


def test_dot1_A_at_first_frequency_includes_heave_pitch_coupling() -> None:
    omega, A, _B, _A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"))
    # omega=0.5 is index 0 (sorted ascending).
    assert omega[0] == pytest.approx(0.5, rel=1e-12)
    # diagonal
    expected_diag = np.array([8.0e4, 8.0e4, 1.5e5, 8.0e6, 8.0e6, 1.5e7])
    np.testing.assert_allclose(np.diag(A[..., 0]), expected_diag, rtol=1e-12)
    # heave-pitch coupling (DOF 3 <-> DOF 5, zero-indexed 2 <-> 4)
    assert A[2, 4, 0] == pytest.approx(1.0e5, rel=1e-12)
    assert A[4, 2, 0] == pytest.approx(1.0e5, rel=1e-12)


def test_dot1_B_at_first_frequency_matches_fixture() -> None:
    omega, _A, B, _A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"))
    assert omega[0] == pytest.approx(0.5, rel=1e-12)
    expected_diag = np.array([1.0e4, 1.0e4, 2.0e4, 1.0e5, 1.0e5, 3.0e5])
    np.testing.assert_allclose(np.diag(B[..., 0]), expected_diag, rtol=1e-12)
    assert B[2, 4, 0] == pytest.approx(5.0e3, rel=1e-12)
    assert B[4, 2, 0] == pytest.approx(5.0e3, rel=1e-12)


def test_dot1_A_at_second_frequency_matches_fixture() -> None:
    omega, A, _B, _A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"))
    assert omega[1] == pytest.approx(1.0, rel=1e-12)
    expected_diag = np.array([6.5e4, 6.5e4, 1.2e5, 6.5e6, 6.5e6, 1.3e7])
    np.testing.assert_allclose(np.diag(A[..., 1]), expected_diag, rtol=1e-12)
    assert A[2, 4, 1] == pytest.approx(8.0e4, rel=1e-12)


def test_dot1_zero_frequency_row_is_silently_discarded() -> None:
    _omega, _A, _B, A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"))
    # The PER=0 row in the fixture has A(3,3)=2e5, distinct from A_inf(3,3)=1e5.
    # If the reader confused PER=0 with PER=-1, A_inf(3,3) would be 2e5 instead.
    assert A_inf[2, 2] == pytest.approx(1.0e5, rel=1e-12)


def test_dot1_each_slice_is_symmetric() -> None:
    _omega, A, B, A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"))
    np.testing.assert_allclose(A_inf, A_inf.T, rtol=1e-12, atol=1e-12)
    for k in range(A.shape[2]):
        np.testing.assert_allclose(A[..., k], A[..., k].T, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(B[..., k], B[..., k].T, rtol=1e-12, atol=1e-12)


def test_dot1_without_infinite_freq_row_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no_inf.1"
    bad.write_text("12.5663706143592   1  1   1.0E+05  0.0\n")
    with pytest.raises(ValueError, match="PER == -1"):
        read_added_mass_and_damping(bad)


def test_dot1_off_diagonal_solver_noise_is_silently_averaged(tmp_path: Path) -> None:
    """Real WAMIT panel-method output produces tiny asymmetric off-diagonals
    (e.g. for marin_semi at T=12.57 s, M[4,6]≈92 vs M[6,4]≈48 against a
    diagonal scale of ~8e6). The reader must accept these and average
    them — they are physically zero coupling polluted by panel noise, not
    file bugs.
    """
    bad = tmp_path / "noisy.1"
    # Diagonal is ~1e7; off-diagonal noise of 50 vs 92 is at solver-noise
    # level (~1e-5 of diagonal). Should be accepted and averaged.
    bad.write_text(
        "  -1.0   1  1   1.0E+07\n"
        "  -1.0   2  2   1.0E+07\n"
        "  -1.0   3  3   1.0E+07\n"
        "  -1.0   4  4   1.0E+07\n"
        "  -1.0   5  5   1.0E+07\n"
        "  -1.0   6  6   1.0E+07\n"
        "  -1.0   4  6   9.224319E+01\n"
        "  -1.0   6  4   4.751778E+01\n"
        "   12.566   1  1   1.0E+07   1.0E+05\n"
        "   12.566   2  2   1.0E+07   1.0E+05\n"
        "   12.566   3  3   1.0E+07   1.0E+05\n"
        "   12.566   4  4   1.0E+07   1.0E+05\n"
        "   12.566   5  5   1.0E+07   1.0E+05\n"
        "   12.566   6  6   1.0E+07   1.0E+05\n"
    )
    _omega, _A, _B, A_inf = read_added_mass_and_damping(bad)
    # Averaged value lands between the two inputs.
    expected_avg = 0.5 * (9.224319e1 + 4.751778e1)
    assert A_inf[3, 5] == pytest.approx(expected_avg)
    assert A_inf[5, 3] == pytest.approx(expected_avg)


def test_dot1_duplicate_row_disagreement_raises(tmp_path: Path) -> None:
    """Two rows for the same (PER, I, J) with different values is a corrupt
    file — distinguish from solver-noise off-diagonal asymmetry."""
    bad = tmp_path / "dup.1"
    bad.write_text(
        "  -1.0   1  1   1.0E+05\n"
        "  -1.0   1  1   2.0E+05\n"  # duplicate, disagrees
        "  -1.0   2  2   1.0E+05\n"
        "  -1.0   3  3   1.0E+05\n"
        "  -1.0   4  4   1.0E+05\n"
        "  -1.0   5  5   1.0E+05\n"
        "  -1.0   6  6   1.0E+05\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_added_mass_and_damping(bad)


def test_dot1_nondimensional_emits_warning(tmp_path: Path) -> None:
    bad = tmp_path / "nondim.1"
    bad.write_text(
        "  -1.0   1  1   0.5\n"
        "  -1.0   2  2   0.5\n"
        "  -1.0   3  3   0.5\n"
        "  -1.0   4  4   0.5\n"
        "  -1.0   5  5   0.5\n"
        "  -1.0   6  6   0.5\n"
        "   12.566   1  1   0.4   0.05\n"
        "   12.566   2  2   0.4   0.05\n"
        "   12.566   3  3   0.4   0.05\n"
        "   12.566   4  4   0.4   0.05\n"
        "   12.566   5  5   0.4   0.05\n"
        "   12.566   6  6   0.4   0.05\n"
    )
    with pytest.warns(UserWarning, match="nondimensional"):
        read_added_mass_and_damping(bad)


def test_dot1_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_added_mass_and_damping(tmp_path / "missing.1")


def test_dot1_malformed_row_count_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad_cols.1"
    bad.write_text("  -1.0   1  1\n")  # 3 cols, parser wants 4 or 5
    with pytest.raises(ValueError, match="must have 4 or 5"):
        read_added_mass_and_damping(bad)


# ---------------------------------------------------------------------------
# .hst parser
# ---------------------------------------------------------------------------


def test_dot_hst_full_matrix_matches_fixture() -> None:
    C = read_hydrostatic_stiffness(_STEM.with_suffix(".hst"))
    expected = np.zeros((6, 6))
    expected[2, 2] = 1.0e6
    expected[3, 3] = 1.0e7
    expected[4, 4] = 1.2e7
    expected[2, 4] = -1.0e5
    expected[4, 2] = -1.0e5
    np.testing.assert_allclose(C, expected, rtol=1e-12, atol=1e-12)


def test_dot_hst_is_symmetric() -> None:
    C = read_hydrostatic_stiffness(_STEM.with_suffix(".hst"))
    np.testing.assert_allclose(C, C.T, rtol=1e-12, atol=1e-12)


def test_dot_hst_unrestored_dofs_are_zero() -> None:
    C = read_hydrostatic_stiffness(_STEM.with_suffix(".hst"))
    # Surge / sway / yaw must be exactly zero (not restored hydrostatically).
    assert C[0, 0] == 0.0
    assert C[1, 1] == 0.0
    assert C[5, 5] == 0.0


def test_dot_hst_solver_noise_asymmetry_is_averaged(tmp_path: Path) -> None:
    """Asymmetric C entries are averaged (consistent with the .1 reader's
    handling of solver-noise asymmetry). Truly different values for a
    physically symmetric coupling are at most a panel-noise effect."""
    bad = tmp_path / "asym.hst"
    bad.write_text(
        "3 3  1.0E+06\n"
        "3 5  1.0E+06\n"
        "5 3  1.000001E+06\n"  # solver noise: ~1e-6 relative
        "5 5  1.0E+06\n"
    )
    C = read_hydrostatic_stiffness(bad)
    expected = 0.5 * (1.0e6 + 1.000001e6)
    assert C[2, 4] == pytest.approx(expected)
    assert C[4, 2] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# .3 parser
# ---------------------------------------------------------------------------


def test_dot3_heading_axis_is_sorted() -> None:
    omega = np.asarray([0.5, 1.0])
    heading, _F = read_excitation_force(_STEM.with_suffix(".3"), omega=omega)
    np.testing.assert_allclose(heading, [0.0, 90.0])


def test_dot3_excitation_at_omega_0p5_beta_0_matches_fixture() -> None:
    omega = np.asarray([0.5, 1.0])
    _heading, F = read_excitation_force(_STEM.with_suffix(".3"), omega=omega)
    # surge: 1e6 ∠ 0deg
    assert F[0, 0, 0] == pytest.approx(1.0e6 + 0.0j, rel=1e-6, abs=1.0)
    # heave: 5e5 ∠ 45deg
    assert F[2, 0, 0] == pytest.approx(5.0e5 * np.exp(1j * np.pi / 4.0), rel=1e-4)
    # pitch: 2e6 ∠ 90deg → pure imaginary
    assert F[4, 0, 0].imag == pytest.approx(2.0e6, rel=1e-4)
    assert abs(F[4, 0, 0].real) < 1.0e-3
    # sway must be zero at beta=0
    assert F[1, 0, 0] == 0j


def test_dot3_excitation_at_omega_0p5_beta_90_matches_fixture() -> None:
    omega = np.asarray([0.5, 1.0])
    _heading, F = read_excitation_force(_STEM.with_suffix(".3"), omega=omega)
    # sway: 1e6 ∠ 0deg at beta=90
    assert F[1, 0, 1] == pytest.approx(1.0e6 + 0.0j, rel=1e-6, abs=1.0)
    # roll: 2e6 ∠ 90deg
    assert F[3, 0, 1].imag == pytest.approx(2.0e6, rel=1e-4)


def test_dot3_inconsistent_re_im_vs_mod_pha_raises(tmp_path: Path) -> None:
    bad = tmp_path / "inconsistent.3"
    bad.write_text(
        "  12.5663706143592    0.0   1   1.000000E+06    0.0   " "5.000000E+05   0.000000E+00\n"
    )
    omega = np.asarray([0.5, 1.0])
    with pytest.raises(ValueError, match="disagrees"):
        read_excitation_force(bad, omega=omega)


def test_dot3_omega_not_in_grid_raises(tmp_path: Path) -> None:
    bad = tmp_path / "wrong_omega.3"
    bad.write_text("  10.0    0.0   1   1.000000E+06    0.0    1.000000E+06    0.000000E+00\n")
    omega = np.asarray([0.5, 1.0])
    with pytest.raises(ValueError, match="omega"):
        read_excitation_force(bad, omega=omega)


def test_dot3_incomplete_grid_raises(tmp_path: Path) -> None:
    bad = tmp_path / "partial.3"
    # Only one row -- far from a full (6 modes x 1 freq x 1 heading) grid.
    bad.write_text(
        "  12.5663706143592    0.0   1   1.000000E+06    0.0   " "1.000000E+06    0.000000E+00\n"
    )
    omega = np.asarray([0.5, 1.0])
    with pytest.raises(ValueError, match="incompletely populated"):
        read_excitation_force(bad, omega=omega)


# ---------------------------------------------------------------------------
# .4 parser (cross-check infrastructure)
# ---------------------------------------------------------------------------


def test_dot4_returns_correctly_shaped_complex_array() -> None:
    omega = np.asarray([0.5, 1.0])
    heading, rao = read_motion_rao(_STEM.with_suffix(".4"), omega=omega)
    assert rao.shape == (6, 2, 2)
    assert rao.dtype == np.complex128
    np.testing.assert_allclose(heading, [0.0, 90.0])


def test_dot4_surge_rao_at_omega_0p5_beta_0_matches_fixture() -> None:
    omega = np.asarray([0.5, 1.0])
    _heading, rao = read_motion_rao(_STEM.with_suffix(".4"), omega=omega)
    assert rao[0, 0, 0] == pytest.approx(1.5 + 0.0j, rel=1e-6)


def test_dot4_pitch_rao_at_omega_1_beta_0_matches_fixture() -> None:
    omega = np.asarray([0.5, 1.0])
    _heading, rao = read_motion_rao(_STEM.with_suffix(".4"), omega=omega)
    expected = 5.0e-2 * np.exp(1j * np.deg2rad(30.0))
    assert rao[4, 1, 0] == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# read_wamit composite
# ---------------------------------------------------------------------------


def test_read_wamit_returns_valid_HydroDatabase() -> None:
    db = read_wamit(_STEM)
    assert isinstance(db, HydroDatabase)
    np.testing.assert_allclose(db.omega, [0.5, 1.0], rtol=1e-12)
    np.testing.assert_allclose(db.heading_deg, [0.0, 90.0])
    assert db.A.shape == (6, 6, 2)
    assert db.B.shape == (6, 6, 2)
    assert db.A_inf.shape == (6, 6)
    assert db.C.shape == (6, 6)
    assert db.RAO.shape == (6, 2, 2)


def test_read_wamit_metadata_records_source() -> None:
    db = read_wamit(_STEM)
    assert db.metadata["source"] == "wamit"
    assert db.metadata["stem"] == "synthetic_simple"


def test_read_wamit_accepts_path_with_suffix() -> None:
    db = read_wamit(_STEM.with_suffix(".1"))  # accepts a stem-with-suffix
    assert db.A.shape == (6, 6, 2)


def test_read_wamit_propagates_reference_point() -> None:
    db = read_wamit(_STEM, reference_point=(1.0, 2.0, 3.0))
    np.testing.assert_allclose(db.reference_point, [1.0, 2.0, 3.0])


def test_read_wamit_default_reference_point_is_origin() -> None:
    db = read_wamit(_STEM)
    np.testing.assert_allclose(db.reference_point, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# load_hydro_database dispatch
# ---------------------------------------------------------------------------


def test_dispatch_routes_wamit_format_to_read_wamit() -> None:
    db = load_hydro_database(_STEM, format="wamit")
    assert isinstance(db, HydroDatabase)
    assert db.metadata["source"] == "wamit"


def test_dispatch_unknown_format_raises() -> None:
    with pytest.raises(ValueError, match="Unknown BEM format"):
        load_hydro_database(_STEM, format="floatation")  # type: ignore[arg-type]


def test_dispatch_capytaine_raises_not_implemented(tmp_path: Path) -> None:
    fake = tmp_path / "fake.nc"
    fake.write_bytes(b"")  # contents irrelevant — the dispatcher rejects first
    with pytest.raises(NotImplementedError, match="PR2"):
        load_hydro_database(fake, format="capytaine")


def test_dispatch_orcaflex_format_routes_to_yaml_reader() -> None:
    # Re-uses the existing OrcaFlex M1.5 fixture as a smoke test that the
    # dispatch wires the right reader.
    fx = (
        Path(__file__).resolve().parents[1] / "fixtures" / "bem" / "orcaflex" / "platform_small.yml"
    )
    if not fx.is_file():
        pytest.skip("OrcaFlex demo fixture not present in this checkout")
    db = load_hydro_database(fx, format="orcaflex")
    assert isinstance(db, HydroDatabase)


# ---------------------------------------------------------------------------
# warning hygiene: dimensional fixture must not warn
# ---------------------------------------------------------------------------


def test_dot1_dimensional_fixture_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # promote any UserWarning to error
        read_added_mass_and_damping(_STEM.with_suffix(".1"))


# ---------------------------------------------------------------------------
# real-fixture integration: trimmed marin_semi (OC4 DeepCwind, OpenFAST
# r-test source — see docs/wamit-fixture-attribution.md)
# ---------------------------------------------------------------------------


_MARIN = _FIXTURE_DIR / "marin_semi_trimmed"


def test_marin_semi_trimmed_loads_into_HydroDatabase() -> None:
    db = read_wamit(_MARIN)
    assert db.A.shape == (6, 6, 3)
    assert db.B.shape == (6, 6, 3)
    assert db.A_inf.shape == (6, 6)
    assert db.RAO.shape == (6, 3, 1)


def test_marin_semi_trimmed_omega_grid_is_ascending() -> None:
    db = read_wamit(_MARIN)
    assert np.all(np.diff(db.omega) > 0.0)


def test_marin_semi_trimmed_A_inf_diagonal_is_platform_scale() -> None:
    """A_inf surge/sway should be ~8.5e3 kg, heave ~1.5e4 kg (tiny —
    semi-sub heave is mostly mass + waterplane), roll/pitch ~7.5e6 kg*m^2.
    Just spot-check the order of magnitude as a sanity gate; exact values
    are part of the OpenFAST regression-test reference."""
    db = read_wamit(_MARIN)
    surge_aa = db.A_inf[0, 0]
    pitch_aa = db.A_inf[4, 4]
    assert 1.0e3 < surge_aa < 1.0e5, surge_aa
    assert 1.0e6 < pitch_aa < 1.0e8, pitch_aa


def test_marin_semi_trimmed_C_heave_is_positive() -> None:
    """WAMIT .hst writes the BUOYANCY-only restoring (no gravity term).
    Heave is purely waterplane, should be positive (~3.8e2 here for the
    OC4 semi). Roll/pitch may be negative — that is expected, because
    the gravity restoring contribution m*g*z_G must be added by the
    body assembly downstream."""
    db = read_wamit(_MARIN)
    assert db.C[2, 2] > 0.0


def test_marin_semi_trimmed_F_exc_at_lowest_omega_is_finite() -> None:
    db = read_wamit(_MARIN)
    F0 = db.RAO[:, 0, 0]
    assert np.all(np.isfinite(F0.real))
    assert np.all(np.isfinite(F0.imag))
    # surge force must be nonzero at BETA=0 (waves traveling +X push surge)
    assert abs(F0[0]) > 0.0
