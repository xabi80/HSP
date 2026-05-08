"""Unit tests for the WAMIT plain-text reader.

Each test asserts exact values against the hand-authored synthetic fixture
``tests/fixtures/bem/wamit/synthetic_simple.{1,3,hst,4}``. Period values in
the fixture round-trip cleanly to omega = {0.5, 1.0} rad/s — preserving
this invariant matters because the parser keys excitation rows on
``omega = 2*pi/PER`` matched against the .1 grid.
"""

from __future__ import annotations

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

# The ``synthetic_simple`` fixture is hand-crafted at sub-platform-scale
# (rotational A_inf = 1e7 kg*m^2) and therefore trips the strengthened
# ``_maybe_warn_nondimensional`` heuristic on every assume_dimensional=True
# call. A dedicated test
# (``test_dot1_synthetic_fixture_does_warn_post_heuristic_strengthening``)
# pins that the warning fires; the rest of the suite silences it for
# signal hygiene.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*magnitude expected of a non-dimensional.*:UserWarning"
)

# Synthetic fixture is dimensional by construction; pass this kwarg to the
# reader to bypass the (default) WAMIT non-dim → dim rescaling.
_AS_DIM: dict[str, bool] = {"assume_dimensional": True}


# ---------------------------------------------------------------------------
# .1 parser
# ---------------------------------------------------------------------------


def test_dot1_omega_grid_is_sorted_ascending() -> None:
    omega, _A, _B, _A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)
    np.testing.assert_allclose(omega, [0.5, 1.0], rtol=1e-12)


def test_dot1_A_inf_diagonal_matches_fixture() -> None:
    _omega, _A, _B, A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)
    expected_diag = np.array([5.0e4, 5.0e4, 1.0e5, 5.0e6, 5.0e6, 1.0e7])
    np.testing.assert_allclose(np.diag(A_inf), expected_diag, rtol=1e-12)


def test_dot1_A_inf_off_diagonals_are_zero() -> None:
    _omega, _A, _B, A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)
    off = A_inf - np.diag(np.diag(A_inf))
    assert np.all(off == 0.0)


def test_dot1_A_at_first_frequency_includes_heave_pitch_coupling() -> None:
    omega, A, _B, _A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)
    # omega=0.5 is index 0 (sorted ascending).
    assert omega[0] == pytest.approx(0.5, rel=1e-12)
    # diagonal
    expected_diag = np.array([8.0e4, 8.0e4, 1.5e5, 8.0e6, 8.0e6, 1.5e7])
    np.testing.assert_allclose(np.diag(A[..., 0]), expected_diag, rtol=1e-12)
    # heave-pitch coupling (DOF 3 <-> DOF 5, zero-indexed 2 <-> 4)
    assert A[2, 4, 0] == pytest.approx(1.0e5, rel=1e-12)
    assert A[4, 2, 0] == pytest.approx(1.0e5, rel=1e-12)


def test_dot1_B_at_first_frequency_matches_fixture() -> None:
    omega, _A, B, _A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)
    assert omega[0] == pytest.approx(0.5, rel=1e-12)
    expected_diag = np.array([1.0e4, 1.0e4, 2.0e4, 1.0e5, 1.0e5, 3.0e5])
    np.testing.assert_allclose(np.diag(B[..., 0]), expected_diag, rtol=1e-12)
    assert B[2, 4, 0] == pytest.approx(5.0e3, rel=1e-12)
    assert B[4, 2, 0] == pytest.approx(5.0e3, rel=1e-12)


def test_dot1_A_at_second_frequency_matches_fixture() -> None:
    omega, A, _B, _A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)
    assert omega[1] == pytest.approx(1.0, rel=1e-12)
    expected_diag = np.array([6.5e4, 6.5e4, 1.2e5, 6.5e6, 6.5e6, 1.3e7])
    np.testing.assert_allclose(np.diag(A[..., 1]), expected_diag, rtol=1e-12)
    assert A[2, 4, 1] == pytest.approx(8.0e4, rel=1e-12)


def test_dot1_zero_frequency_row_is_silently_discarded() -> None:
    _omega, _A, _B, A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)
    # The PER=0 row in the fixture has A(3,3)=2e5, distinct from A_inf(3,3)=1e5.
    # If the reader confused PER=0 with PER=-1, A_inf(3,3) would be 2e5 instead.
    assert A_inf[2, 2] == pytest.approx(1.0e5, rel=1e-12)


def test_dot1_each_slice_is_symmetric() -> None:
    _omega, A, B, A_inf = read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)
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
    _omega, _A, _B, A_inf = read_added_mass_and_damping(bad, **_AS_DIM)
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
    with pytest.warns(
        UserWarning,
        match="nondimensional|non-dimensional|expected for a real floating platform",
    ):
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
    C = read_hydrostatic_stiffness(_STEM.with_suffix(".hst"), **_AS_DIM)
    expected = np.zeros((6, 6))
    expected[2, 2] = 1.0e6
    expected[3, 3] = 1.0e7
    expected[4, 4] = 1.2e7
    expected[2, 4] = -1.0e5
    expected[4, 2] = -1.0e5
    np.testing.assert_allclose(C, expected, rtol=1e-12, atol=1e-12)


def test_dot_hst_is_symmetric() -> None:
    C = read_hydrostatic_stiffness(_STEM.with_suffix(".hst"), **_AS_DIM)
    np.testing.assert_allclose(C, C.T, rtol=1e-12, atol=1e-12)


def test_dot_hst_unrestored_dofs_are_zero() -> None:
    C = read_hydrostatic_stiffness(_STEM.with_suffix(".hst"), **_AS_DIM)
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
    C = read_hydrostatic_stiffness(bad, **_AS_DIM)
    expected = 0.5 * (1.0e6 + 1.000001e6)
    assert C[2, 4] == pytest.approx(expected)
    assert C[4, 2] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# .3 parser
# ---------------------------------------------------------------------------


def test_dot3_heading_axis_is_sorted() -> None:
    omega = np.asarray([0.5, 1.0])
    heading, _F = read_excitation_force(_STEM.with_suffix(".3"), omega=omega, **_AS_DIM)
    np.testing.assert_allclose(heading, [0.0, 90.0])


def test_dot3_excitation_at_omega_0p5_beta_0_matches_fixture() -> None:
    omega = np.asarray([0.5, 1.0])
    _heading, F = read_excitation_force(_STEM.with_suffix(".3"), omega=omega, **_AS_DIM)
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
    _heading, F = read_excitation_force(_STEM.with_suffix(".3"), omega=omega, **_AS_DIM)
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
    db = read_wamit(_STEM, **_AS_DIM)
    assert isinstance(db, HydroDatabase)
    np.testing.assert_allclose(db.omega, [0.5, 1.0], rtol=1e-12)
    np.testing.assert_allclose(db.heading_deg, [0.0, 90.0])
    assert db.A.shape == (6, 6, 2)
    assert db.B.shape == (6, 6, 2)
    assert db.A_inf.shape == (6, 6)
    assert db.C.shape == (6, 6)
    assert db.RAO.shape == (6, 2, 2)


def test_read_wamit_metadata_records_source() -> None:
    db = read_wamit(_STEM, **_AS_DIM)
    assert db.metadata["source"] == "wamit"
    assert db.metadata["stem"] == "synthetic_simple"


def test_read_wamit_accepts_path_with_suffix() -> None:
    db = read_wamit(_STEM.with_suffix(".1"), **_AS_DIM)  # accepts a stem-with-suffix
    assert db.A.shape == (6, 6, 2)


def test_read_wamit_propagates_reference_point() -> None:
    db = read_wamit(_STEM, reference_point=(1.0, 2.0, 3.0), **_AS_DIM)
    np.testing.assert_allclose(db.reference_point, [1.0, 2.0, 3.0])


def test_read_wamit_default_reference_point_is_origin() -> None:
    db = read_wamit(_STEM, **_AS_DIM)
    np.testing.assert_allclose(db.reference_point, [0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# load_hydro_database dispatch
# ---------------------------------------------------------------------------


def test_dispatch_routes_wamit_format_to_read_wamit() -> None:
    db = load_hydro_database(_STEM, format="wamit")
    assert isinstance(db, HydroDatabase)
    assert db.metadata["source"] == "wamit"


# ---------------------------------------------------------------------------
# Dimensional rescaling regression tests (M6 PR4 Pre-3 finding)
# ---------------------------------------------------------------------------
#
# Surfaced in M6 PR4 Pre-3: the FloatSim WAMIT reader was returning
# non-dimensional WAMIT output as-is, missing the rho * ULEN^k /
# rho * g * ULEN^k dimensionalisation factors documented in the
# WAMIT v7 manual §4.2 (and HydroDyn user guide §6 which references
# the same scheme).
#
# The bug was latent through M5-M6 PR3 because free-decay periods
# are dominated by the dimensional rigid-body M and Robertson C;
# non-dim A is ~ 0.1 % of dim M for OC4 in the natural-period band,
# so the missing 1000x factor on A doesn't perturb T = 2*pi*sqrt((M+A)/C)
# measurably. Surfaces immediately at PR4 because RAO at the long-wave
# limit goes as F_exc / C, and the missing rho*g*ULEN^2 factor on F_exc
# makes the heave RAO come out ~1e4 too small.
#
# The regression tests below pin the dimensional output against
# Robertson 2014 Table 3-1 (NREL/TP-5000-60601) published OC4
# values. See conventions doc Items 22 + 23 and
# docs/post-mortems/m6-pr4-wamit-dim-bug.md.

_MARIN_SEMI_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "openfast"
    / "oc4_deepcwind"
    / "baseline"
    / "5MW_Baseline"
    / "HydroData"
    / "marin_semi.1"
)

# Robertson 2014 NREL/TP-5000-60601 Table 3-1 reports:
#   A_inf_heave (added mass at infinite frequency, heave): ~ 1.45e7 kg
#   A_inf_pitch (added mass at infinite frequency, pitch about CoG):
#        ~ 7.27e9 kg*m^2
# The marin_semi BEM produces values close to these after the WAMIT
# dimensional rescaling. We allow 5% slack because the published
# values are quoted to 3 significant figures.
_OC4_AINF_HEAVE_PUBLISHED_KG: float = 1.45e7
_OC4_AINF_PITCH_PUBLISHED_KG_M2: float = 7.27e9
_PUBLISHED_RTOL: float = 5.0e-2


def test_dot1_marin_semi_dimensional_A_inf_heave_matches_robertson() -> None:
    """Post-fix WAMIT reader must dimensionalise A_inf_heave to ~ Robertson 2014.

    The marin_semi.1 file is the as-shipped OpenFAST r-test BEM output
    in WAMIT's default non-dimensional form. After applying rho * ULEN^3
    (rho = 1025 kg/m^3, ULEN = 1.0 m), A_inf_heave should match the
    Robertson 2014 Table 3-1 published OC4 value to within rtol = 5%.
    """
    _omega, _A, _B, A_inf = read_added_mass_and_damping(_MARIN_SEMI_PATH)
    a_inf_heave = float(A_inf[2, 2])
    rel_err = abs(a_inf_heave - _OC4_AINF_HEAVE_PUBLISHED_KG) / _OC4_AINF_HEAVE_PUBLISHED_KG
    assert rel_err < _PUBLISHED_RTOL, (
        f"A_inf_heave = {a_inf_heave:.4e} kg vs Robertson 2014 published "
        f"{_OC4_AINF_HEAVE_PUBLISHED_KG:.3e} kg; rel-err {rel_err:.3%} "
        f"(limit {_PUBLISHED_RTOL:.1%}). The reader is likely returning "
        "WAMIT's non-dimensional values without applying rho * ULEN^3 "
        "dimensionalisation. See conventions doc Item 22 + "
        "docs/post-mortems/m6-pr4-wamit-dim-bug.md."
    )


def test_dot1_marin_semi_dimensional_A_inf_pitch_matches_robertson() -> None:
    """Post-fix WAMIT reader must dimensionalise A_inf_pitch to ~ Robertson 2014.

    The non-dim -> dim factor for rotational diagonal is rho * ULEN^5
    (= 1025 for ULEN = 1). Robertson 2014 Table 3-1 reports OC4
    A_inf_pitch ~ 7.27e9 kg*m^2 (about the CoG; the marin_semi BEM
    is at the BEM reference at SWL, so the value is slightly different
    via parallel axis -- 5% slack accommodates this).
    """
    _omega, _A, _B, A_inf = read_added_mass_and_damping(_MARIN_SEMI_PATH)
    a_inf_pitch = float(A_inf[4, 4])
    rel_err = abs(a_inf_pitch - _OC4_AINF_PITCH_PUBLISHED_KG_M2) / _OC4_AINF_PITCH_PUBLISHED_KG_M2
    assert rel_err < _PUBLISHED_RTOL, (
        f"A_inf_pitch = {a_inf_pitch:.4e} kg*m^2 vs Robertson 2014 published "
        f"{_OC4_AINF_PITCH_PUBLISHED_KG_M2:.3e} kg*m^2; rel-err {rel_err:.3%} "
        f"(limit {_PUBLISHED_RTOL:.1%}). Likely missing rho * ULEN^5 "
        "dimensionalisation. See conventions doc Item 22."
    )


def test_dimensionality_heuristic_catches_high_amplitude_nondim(tmp_path: Path) -> None:
    """The pre-fix _maybe_warn_nondimensional heuristic missed the marin_semi
    case (surge A_inf = 8527 non-dim > the threshold of 10).

    Post-fix: when the caller asserts assume_dimensional=True but the
    values look non-dim (e.g., they would scale up to absurd dimensional
    magnitudes given a typical rho), the reader should raise rather
    than silently accept. This test pins that behaviour.
    """
    bad = tmp_path / "high_nondim.1"
    # Mimic the marin_semi situation: surge A_inf = 8527 (non-dim); after
    # rho*ULEN^3 = 1025 the dimensional value would be 8.74e6 kg -- the
    # correct OC4 surge added mass. But the pre-fix heuristic with
    # threshold 10 fires only when max|A| < 10; 8527 > 10 so it DIDN'T
    # fire on marin_semi.
    bad.write_text(
        "  -1.0   1  1   8.527E+03\n"  # surge A_inf (non-dim, magnitude ~ 8500)
        "  -1.0   2  2   8.527E+03\n"
        "  -1.0   3  3   1.462E+04\n"  # heave A_inf (non-dim, ~ 14600)
        "  -1.0   4  4   7.441E+06\n"  # roll A_inf (non-dim, ~ 7.4e6)
        "  -1.0   5  5   7.441E+06\n"
        "  -1.0   6  6   6.273E+06\n"
        "   12.566   1  1   8.0E+03   1.0E+02\n"
        "   12.566   2  2   8.0E+03   1.0E+02\n"
        "   12.566   3  3   1.4E+04   1.0E+02\n"
        "   12.566   4  4   7.0E+06   1.0E+05\n"
        "   12.566   5  5   7.0E+06   1.0E+05\n"
        "   12.566   6  6   6.0E+06   1.0E+05\n"
    )
    # When `assume_dimensional=True`, the reader must emit a warning
    # (or raise) on input that fails the strengthened check. The
    # strengthened check uses dimensional reasonableness rather than
    # the pre-fix max-magnitude heuristic.
    with pytest.warns(UserWarning, match="nondimensional|non-dimensional"):
        read_added_mass_and_damping(bad, **_AS_DIM)


def test_dispatch_unknown_format_raises() -> None:
    with pytest.raises(ValueError, match="Unknown BEM format"):
        load_hydro_database(_STEM, format="floatation")  # type: ignore[arg-type]


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
# warning hygiene: synthetic_simple fixture is below platform-scale
# ---------------------------------------------------------------------------


def test_dot1_synthetic_fixture_does_warn_post_heuristic_strengthening() -> None:
    """Pre-fix-wamit-dimensionalisation behaviour: synthetic_simple was
    flagged dimensional (max|A| > 10) and didn't warn. Post-fix
    (M6 PR4 Pre-3): the heuristic checks rotational A_inf > 1e8 kg*m^2,
    a platform-realistic threshold. synthetic_simple's diagonal is 1e7
    (hand-crafted small for parser tests), so the heuristic now fires
    even when ``assume_dimensional=True``.

    This is correct behaviour: the heuristic is a sanity backstop for
    real platform decks, and synthetic_simple is intentionally below
    that scale. Real OC4 marin_semi reads with the default
    ``assume_dimensional=False`` and the rescaled rotational A_inf
    (~7.4e9 kg*m^2) is comfortably above the 1e8 threshold.
    """
    with pytest.warns(UserWarning, match="magnitude expected of a non-dimensional"):
        read_added_mass_and_damping(_STEM.with_suffix(".1"), **_AS_DIM)


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
    """Post-fix-wamit-dimensionalisation magnitudes: surge ~ 8.5e6 kg,
    pitch ~ 7.5e9 kg*m^2 (the WAMIT default non-dim values are 1000x
    smaller; the reader now applies rho * ULEN^k rescaling per
    conventions doc Item 22). Just spot-check the order of magnitude.
    Exact values are part of the OpenFAST regression reference."""
    db = read_wamit(_MARIN)
    surge_aa = db.A_inf[0, 0]
    pitch_aa = db.A_inf[4, 4]
    # Semi-sub surge added mass ~ 1e7 kg (Robertson Table 3-1).
    assert 1.0e6 < surge_aa < 1.0e8, surge_aa
    # Pitch added mass at SWL ~ 7-8e9 kg*m^2.
    assert 1.0e9 < pitch_aa < 1.0e11, pitch_aa


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
