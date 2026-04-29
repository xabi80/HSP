"""Unit tests for RAO interpolation and regular-wave excitation."""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.excitation import interpolate_rao, make_regular_wave_force
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave
from tests.support.synthetic_bem import diagonal_6x6

# ---------------------------------------------------------------------------
# small helpers to build an HDB with a specific RAO
# ---------------------------------------------------------------------------


def _build_hdb(
    *,
    omega: np.ndarray,
    heading_deg: np.ndarray,
    rao: np.ndarray,
) -> HydroDatabase:
    """HDB with trivial A, B, A_inf, C, and the supplied RAO."""
    n_w = omega.size
    A_diag = diagonal_6x6([1.0] * 6)
    A = np.stack([A_diag for _ in range(n_w)], axis=-1)
    B = np.zeros_like(A)
    return HydroDatabase(
        omega=omega.astype(np.float64),
        heading_deg=heading_deg.astype(np.float64),
        A=A,
        B=B,
        A_inf=A_diag,
        C=diagonal_6x6([1.0] * 6),
        RAO=rao.astype(np.complex128),
        reference_point=np.zeros(3, dtype=np.float64),
        C_source="full",
    )


# ---------------------------------------------------------------------------
# interpolate_rao
# ---------------------------------------------------------------------------


def test_interpolate_rao_returns_grid_value_at_grid_point() -> None:
    omega = np.array([0.2, 0.5, 1.0])
    heading = np.array([0.0, 90.0])
    # RAO: dof=heave (index 2) has distinct values per (omega, heading).
    rao = np.zeros((6, 3, 2), dtype=np.complex128)
    rao[2, 0, 0] = 1.0 + 2.0j
    rao[2, 1, 0] = 3.0 + 0.5j
    rao[2, 2, 0] = -1.0 - 1.0j
    rao[2, 0, 1] = 0.5 + 0.0j
    rao[2, 1, 1] = -2.0 + 1.5j
    rao[2, 2, 1] = 4.0 - 3.0j
    hdb = _build_hdb(omega=omega, heading_deg=heading, rao=rao)

    got = interpolate_rao(hdb, omega=0.5, heading_deg=0.0)
    assert got[2] == pytest.approx(3.0 + 0.5j, rel=1e-12)
    got2 = interpolate_rao(hdb, omega=1.0, heading_deg=90.0)
    assert got2[2] == pytest.approx(4.0 - 3.0j, rel=1e-12)


def test_interpolate_rao_midpoint_is_bilinear_average() -> None:
    omega = np.array([0.0, 1.0])
    heading = np.array([0.0, 90.0])
    rao = np.zeros((6, 2, 2), dtype=np.complex128)
    rao[0, 0, 0] = 1.0 + 0.0j
    rao[0, 1, 0] = 3.0 + 2.0j
    rao[0, 0, 1] = -1.0 + 1.0j
    rao[0, 1, 1] = 5.0 - 3.0j
    hdb = _build_hdb(omega=omega, heading_deg=heading, rao=rao)

    got = interpolate_rao(hdb, omega=0.5, heading_deg=45.0)
    expected = 0.25 * (rao[0, 0, 0] + rao[0, 1, 0] + rao[0, 0, 1] + rao[0, 1, 1])
    assert got[0] == pytest.approx(expected, rel=1e-12)


def test_interpolate_rao_rejects_out_of_range_omega() -> None:
    omega = np.array([0.1, 1.0])
    heading = np.array([0.0])
    rao = np.zeros((6, 2, 1), dtype=np.complex128)
    hdb = _build_hdb(omega=omega, heading_deg=heading, rao=rao)
    with pytest.raises(ValueError, match="omega"):
        interpolate_rao(hdb, omega=0.05, heading_deg=0.0)
    with pytest.raises(ValueError, match="omega"):
        interpolate_rao(hdb, omega=1.01, heading_deg=0.0)


def test_interpolate_rao_rejects_out_of_range_heading() -> None:
    omega = np.array([0.1, 1.0])
    heading = np.array([0.0, 90.0])
    rao = np.zeros((6, 2, 2), dtype=np.complex128)
    hdb = _build_hdb(omega=omega, heading_deg=heading, rao=rao)
    with pytest.raises(ValueError, match="heading"):
        interpolate_rao(hdb, omega=0.5, heading_deg=-1.0)
    with pytest.raises(ValueError, match="heading"):
        interpolate_rao(hdb, omega=0.5, heading_deg=90.5)


def test_interpolate_rao_single_heading_must_match_exactly() -> None:
    omega = np.array([0.1, 1.0])
    heading = np.array([30.0])
    rao = np.zeros((6, 2, 1), dtype=np.complex128)
    rao[2, 0, 0] = 2.0 + 1.0j
    rao[2, 1, 0] = 4.0 - 1.0j
    hdb = _build_hdb(omega=omega, heading_deg=heading, rao=rao)

    got = interpolate_rao(hdb, omega=0.55, heading_deg=30.0)  # midpoint
    assert got[2] == pytest.approx(3.0 + 0.0j, rel=1e-12)
    with pytest.raises(ValueError, match="heading"):
        interpolate_rao(hdb, omega=0.5, heading_deg=45.0)


# ---------------------------------------------------------------------------
# make_regular_wave_force — phasor algebra
# ---------------------------------------------------------------------------


def test_force_at_t0_is_real_part_of_rao_times_amplitude() -> None:
    omega = np.array([0.5, 1.0])
    heading = np.array([0.0])
    rao = np.zeros((6, 2, 1), dtype=np.complex128)
    # heave DOF has RAO = 2 + i at omega=0.5.
    rao[2, 0, 0] = 2.0 + 1.0j
    rao[2, 1, 0] = 0.0 + 0.0j
    hdb = _build_hdb(omega=omega, heading_deg=heading, rao=rao)

    wave = RegularWave(amplitude=1.5, omega=0.5, heading_deg=0.0)
    force = make_regular_wave_force(hdb=hdb, wave=wave)
    f0 = force(0.0)
    # At the origin, zero phase: F(0) = Re{RAO * A} = A * Re{RAO} = 1.5 * 2 = 3
    assert f0.shape == (6,)
    assert f0[2] == pytest.approx(3.0, rel=1e-12)
    # Other DOFs zero.
    assert np.all(f0[[0, 1, 3, 4, 5]] == 0.0)


def test_force_time_series_matches_phasor_formula() -> None:
    omega_grid = np.array([0.3, 0.6, 1.2])
    heading = np.array([0.0])
    rao = np.zeros((6, 3, 1), dtype=np.complex128)
    # surge RAO at omega=0.6: 1 + 2i; heave: 3 - i; others zero.
    rao[0, 1, 0] = 1.0 + 2.0j
    rao[2, 1, 0] = 3.0 - 1.0j
    hdb = _build_hdb(omega=omega_grid, heading_deg=heading, rao=rao)

    wave = RegularWave(amplitude=2.0, omega=0.6, heading_deg=0.0, phase=0.25)
    force = make_regular_wave_force(hdb=hdb, wave=wave)

    A = 2.0
    w = 0.6
    phi = 0.25
    ts = np.linspace(0.0, 10.0, 101)
    f_surge = np.array([force(t)[0] for t in ts])
    f_heave = np.array([force(t)[2] for t in ts])
    eta_hat = A * np.exp(1j * phi)  # body at origin
    expected_surge = np.real((1.0 + 2.0j) * eta_hat * np.exp(-1j * w * ts))
    expected_heave = np.real((3.0 - 1.0j) * eta_hat * np.exp(-1j * w * ts))
    np.testing.assert_allclose(f_surge, expected_surge, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(f_heave, expected_heave, rtol=1e-12, atol=1e-14)


def test_body_offset_introduces_expected_phase_shift() -> None:
    """A body at x_b along heading 0 sees a phase shift of +k * x_b in eta_hat,
    i.e. the time series is time-advanced by (k * x_b)/omega seconds."""
    omega_grid = np.array([0.5, 1.0, 2.0])
    heading = np.array([0.0])
    rao = np.zeros((6, 3, 1), dtype=np.complex128)
    rao[2, 1, 0] = 1.0 + 0.0j  # real heave RAO of 1 at omega=1
    hdb = _build_hdb(omega=omega_grid, heading_deg=heading, rao=rao)

    wave = RegularWave(amplitude=1.0, omega=1.0, heading_deg=0.0)
    k = wave.wavenumber
    x_b = 5.0
    f_origin = make_regular_wave_force(hdb=hdb, wave=wave)
    f_offset = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(x_b, 0.0, 0.0))
    dt_shift = k * x_b / wave.omega
    ts = np.linspace(0.0, 20.0, 201)
    f_off = np.array([f_offset(t)[2] for t in ts])
    f_ref = np.array([f_origin(t - dt_shift)[2] for t in ts])
    np.testing.assert_allclose(f_off, f_ref, rtol=1e-12, atol=1e-14)


def test_heading_90_body_offset_in_y_shifts_phase() -> None:
    omega_grid = np.array([0.5, 1.0])
    heading = np.array([0.0, 90.0])
    rao = np.zeros((6, 2, 2), dtype=np.complex128)
    rao[2, 1, 1] = 1.0 + 0.0j  # heave RAO at (omega=1, heading=90)
    hdb = _build_hdb(omega=omega_grid, heading_deg=heading, rao=rao)

    wave = RegularWave(amplitude=1.0, omega=1.0, heading_deg=90.0)
    k = wave.wavenumber
    y_b = 3.0
    # Offset in x must NOT matter for a 90-deg wave.
    f_off_y = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, y_b, 0.0))
    f_off_x = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(y_b, 0.0, 0.0))
    f_origin = make_regular_wave_force(hdb=hdb, wave=wave)

    dt_shift = k * y_b / wave.omega
    ts = np.linspace(0.0, 10.0, 51)
    np.testing.assert_allclose(
        [f_off_y(t)[2] for t in ts],
        [f_origin(t - dt_shift)[2] for t in ts],
        rtol=1e-12,
        atol=1e-14,
    )
    # An x offset for a y-propagating wave should not shift the signal.
    np.testing.assert_allclose(
        [f_off_x(t)[2] for t in ts],
        [f_origin(t)[2] for t in ts],
        rtol=1e-12,
        atol=1e-14,
    )


# ---------------------------------------------------------------------------
# ramp
# ---------------------------------------------------------------------------


def test_ramp_scales_force_linearly_in_r_t() -> None:
    omega_grid = np.array([0.5, 1.0])
    heading = np.array([0.0])
    rao = np.zeros((6, 2, 1), dtype=np.complex128)
    rao[2, 1, 0] = 2.0 + 0.0j
    hdb = _build_hdb(omega=omega_grid, heading_deg=heading, rao=rao)

    wave = RegularWave(amplitude=1.0, omega=1.0)
    ramp = HalfCosineRamp(duration=20.0)
    f_unramped = make_regular_wave_force(hdb=hdb, wave=wave)
    f_ramped = make_regular_wave_force(hdb=hdb, wave=wave, ramp=ramp)

    for t in [0.0, 5.0, 10.0, 20.0, 30.0]:
        r = ramp.value(t)
        np.testing.assert_allclose(f_ramped(t), r * f_unramped(t), rtol=1e-12, atol=1e-14)


def test_ramp_is_zero_at_t_zero_and_full_after_duration() -> None:
    omega_grid = np.array([0.5, 1.0])
    heading = np.array([0.0])
    rao = np.zeros((6, 2, 1), dtype=np.complex128)
    rao[0, 1, 0] = 5.0 + 1.0j  # surge
    rao[2, 1, 0] = 3.0 + 0.0j  # heave
    hdb = _build_hdb(omega=omega_grid, heading_deg=heading, rao=rao)

    wave = RegularWave(amplitude=1.0, omega=1.0)
    ramp = HalfCosineRamp(duration=20.0)
    f_ramped = make_regular_wave_force(hdb=hdb, wave=wave, ramp=ramp)
    np.testing.assert_array_equal(f_ramped(0.0), np.zeros(6))
    # After duration, must equal unramped values
    f_unramped = make_regular_wave_force(hdb=hdb, wave=wave)
    np.testing.assert_allclose(f_ramped(25.0), f_unramped(25.0), rtol=1e-12)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_body_position_shape_validated() -> None:
    omega_grid = np.array([0.5, 1.0])
    heading = np.array([0.0])
    rao = np.zeros((6, 2, 1), dtype=np.complex128)
    hdb = _build_hdb(omega=omega_grid, heading_deg=heading, rao=rao)
    wave = RegularWave(amplitude=1.0, omega=1.0)
    with pytest.raises(ValueError, match="body_position"):
        make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, 0.0))
