"""Unit tests for :mod:`floatsim.waves.kinematics` (M5 PR4; sign fix STEP 5 PR1).

Linear-Airy fluid velocity and acceleration at arbitrary points,
clipped at MWL (no Wheeler stretching in Phase 1; see module docstring
for the TODO).

The SIGN of the vertical components is pinned by physical laws, not by
asserted numbers (section "Physical laws" below): continuity, irrotationality
and the linear kinematic free-surface condition ``w = d(eta)/dt`` at z = 0,
against ``RegularWave.elevation``. The M5 PR4 version asserted
``u_z = +A*omega`` at t = T/4 (and ``a_z = +A*omega^2`` at t = 0). That
encoded the defect: the shipped field had ``u_z = -d(eta)/dt`` and failed all
three laws by exactly 2x. A negative control keeps the law checks
non-vacuous: they must reject that sign-flipped field.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from floatsim.waves.kinematics import airy_acceleration, airy_velocity
from floatsim.waves.regular import RegularWave

_Field = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# ---------------------------------------------------------------------------
# Velocity field at the still water level (z = 0, decay factor = 1)
# ---------------------------------------------------------------------------


def test_velocity_at_z0_t0_origin_is_horizontal_with_amplitude_omega() -> None:
    """At ``(x=0, y=0, z=0, t=0, beta=0, phi=0)``: psi=0, cos psi = 1, sin psi = 0.

    -> u_x = A * omega, u_y = 0, u_z = 0.
    """
    A, omega = 1.5, 0.8
    wave = RegularWave(amplitude=A, omega=omega, heading_deg=0.0)
    u = airy_velocity(wave, np.zeros(3), t=0.0)
    assert u[0] == pytest.approx(A * omega, rel=1e-12)
    assert u[1] == pytest.approx(0.0, abs=1e-12)
    assert u[2] == pytest.approx(0.0, abs=1e-12)


def test_velocity_quarter_period_is_purely_vertical() -> None:
    """At ``t = T/4``: psi = pi/2, cos psi = 0 -> u purely vertical, |u_z| = A*omega.

    Magnitude only: the SIGN of u_z is pinned by the free-surface law
    (``test_kinematic_free_surface_condition``), not by an asserted number.
    """
    A, omega = 1.0, 0.5
    wave = RegularWave(amplitude=A, omega=omega)
    u = airy_velocity(wave, np.zeros(3), t=wave.period / 4.0)
    assert u[0] == pytest.approx(0.0, abs=1e-12)
    assert u[1] == pytest.approx(0.0, abs=1e-12)
    assert abs(u[2]) == pytest.approx(A * omega, rel=1e-12)


def test_velocity_decays_exponentially_with_depth() -> None:
    """At z = -1/k: ``e^{kz} = e^{-1}`` regardless of x, y, t."""
    wave = RegularWave(amplitude=1.0, omega=0.6)
    k = wave.wavenumber
    u_surface = airy_velocity(wave, np.array([0.0, 0.0, 0.0]), t=0.0)
    u_deep = airy_velocity(wave, np.array([0.0, 0.0, -1.0 / k]), t=0.0)
    assert u_deep[0] == pytest.approx(u_surface[0] * np.exp(-1.0), rel=1e-12)


def test_velocity_clipped_above_mwl() -> None:
    """For z > 0, the depth-decay factor is clamped to 1 (no stretching)."""
    wave = RegularWave(amplitude=1.0, omega=0.6)
    u_at_surface = airy_velocity(wave, np.array([0.0, 0.0, 0.0]), t=0.0)
    u_above_surface = airy_velocity(wave, np.array([0.0, 0.0, 0.5]), t=0.0)
    np.testing.assert_allclose(u_above_surface, u_at_surface, rtol=1e-12)


def test_velocity_horizontal_aligned_with_heading() -> None:
    """Heading = 90 deg: horizontal velocity aligned with +Y, none in +X."""
    wave = RegularWave(amplitude=1.0, omega=0.5, heading_deg=90.0)
    u = airy_velocity(wave, np.zeros(3), t=0.0)
    assert u[0] == pytest.approx(0.0, abs=1e-12)
    assert u[1] == pytest.approx(wave.amplitude * wave.omega, rel=1e-12)
    assert u[2] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Acceleration field
# ---------------------------------------------------------------------------


def test_acceleration_leads_velocity_by_quarter_period_horizontal() -> None:
    """At t=0 the horizontal velocity is at peak (cos 0 = 1) and the
    horizontal acceleration is at zero (-sin 0 = 0); |a_z| peaks at A*omega^2.

    Magnitude only for a_z: its SIGN is pinned by
    ``test_vertical_acceleration_matches_free_surface``.
    """
    wave = RegularWave(amplitude=1.0, omega=0.5)
    a = airy_acceleration(wave, np.zeros(3), t=0.0)
    assert a[0] == pytest.approx(0.0, abs=1e-12)
    assert abs(a[2]) == pytest.approx(wave.amplitude * wave.omega**2, rel=1e-12)


def test_acceleration_quarter_period_horizontal_minus_amplitude_omega_squared() -> None:
    """At t = T/4: psi = pi/2, sin psi = 1, cos psi = 0 ->
    a_x = -A omega^2, a_z = 0.
    """
    A, omega = 1.0, 0.5
    wave = RegularWave(amplitude=A, omega=omega)
    a = airy_acceleration(wave, np.zeros(3), t=wave.period / 4.0)
    assert a[0] == pytest.approx(-A * omega**2, rel=1e-12)
    assert a[1] == pytest.approx(0.0, abs=1e-12)
    assert a[2] == pytest.approx(0.0, abs=1e-12)


def test_acceleration_is_time_derivative_of_velocity_finite_diff() -> None:
    """For a smooth Airy field, ``(u(t+h) - u(t-h)) / (2h)`` matches a(t)
    to O(h^2)."""
    wave = RegularWave(amplitude=1.5, omega=0.8, heading_deg=30.0, phase=0.4)
    point = np.array([2.0, -1.0, -0.5])
    t, h = 0.7, 1.0e-5
    u_plus = airy_velocity(wave, point, t=t + h)
    u_minus = airy_velocity(wave, point, t=t - h)
    a_fd = (u_plus - u_minus) / (2.0 * h)
    a_an = airy_acceleration(wave, point, t=t)
    np.testing.assert_allclose(a_fd, a_an, rtol=1e-7, atol=1e-9)


def test_acceleration_decays_exponentially_with_depth() -> None:
    wave = RegularWave(amplitude=1.0, omega=0.6)
    k = wave.wavenumber
    a_surface = airy_acceleration(wave, np.array([0.0, 0.0, 0.0]), t=0.5)
    a_deep = airy_acceleration(wave, np.array([0.0, 0.0, -1.0 / k]), t=0.5)
    np.testing.assert_allclose(a_deep, a_surface * np.exp(-1.0), rtol=1e-12)


# ---------------------------------------------------------------------------
# Physical laws -- these pin the SIGN of the vertical components
# ---------------------------------------------------------------------------
#
# Central differences on the function itself (black-box). With a step of
# 1e-4 wavelengths/2pi the truncation error is ~2e-9 and the round-off
# ~2e-12 of the scale A*omega*k*e^{kz}; the tolerance 1e-6 of scale leaves
# ~3 orders of margin, while a sign error violates each law by O(1) of
# scale.

_TOL = 1.0e-6

_waves = st.builds(
    RegularWave,
    amplitude=st.floats(0.01, 3.0),
    omega=st.floats(0.3, 6.0),
    heading_deg=st.floats(-180.0, 180.0),
    phase=st.floats(-np.pi, np.pi),
)


def _jacobian(vel: _Field, p: NDArray[np.float64], h: float) -> NDArray[np.float64]:
    """J[i, j] = d u_i / d x_j by central differences."""
    J = np.empty((3, 3))
    for j in range(3):
        e = np.zeros(3)
        e[j] = h
        J[:, j] = (vel(p + e) - vel(p - e)) / (2.0 * h)
    return J


def _law_residuals(
    vel: _Field, wave: RegularWave, x: float, y: float, z: float, t: float
) -> tuple[float, float, float]:
    """(divergence, |curl|, KBC residual) of ``vel`` at (x, y, z) and at the MWL, each
    normalised by its scale. ``vel`` maps a point to the velocity at time ``t``."""
    k, A, w = wave.wavenumber, wave.amplitude, wave.omega
    scale = A * w * k * np.exp(k * z)
    J = _jacobian(vel, np.array([x, y, z]), 1.0e-4 / k)
    div = float(np.trace(J)) / scale
    curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    ht = 1.0e-5 / w
    deta_dt = (float(wave.elevation(t + ht, x, y)) - float(wave.elevation(t - ht, x, y))) / (2 * ht)
    kbc = (float(vel(np.array([x, y, 0.0]))[2]) - deta_dt) / (A * w)
    return div, float(np.linalg.norm(curl)) / scale, kbc


@given(
    wave=_waves,
    x=st.floats(-50.0, 50.0),
    y=st.floats(-50.0, 50.0),
    kz=st.floats(0.01, 3.0),
    t=st.floats(0.0, 100.0),
)
def test_continuity_divergence_free(
    wave: RegularWave, x: float, y: float, kz: float, t: float
) -> None:
    """Incompressible flow below the MWL: div u = 0."""
    z = -kz / wave.wavenumber
    div, _, _ = _law_residuals(lambda p: airy_velocity(wave, p, t), wave, x, y, z, t)
    assert abs(div) < _TOL


@given(
    wave=_waves,
    x=st.floats(-50.0, 50.0),
    y=st.floats(-50.0, 50.0),
    kz=st.floats(0.01, 3.0),
    t=st.floats(0.0, 100.0),
)
def test_irrotational_curl_free(wave: RegularWave, x: float, y: float, kz: float, t: float) -> None:
    """Potential flow below the MWL: curl u = 0."""
    z = -kz / wave.wavenumber
    _, curl, _ = _law_residuals(lambda p: airy_velocity(wave, p, t), wave, x, y, z, t)
    assert curl < _TOL


@given(wave=_waves, x=st.floats(-50.0, 50.0), y=st.floats(-50.0, 50.0), t=st.floats(0.0, 100.0))
def test_kinematic_free_surface_condition(wave: RegularWave, x: float, y: float, t: float) -> None:
    """Linear kinematic free-surface condition: w(x, y, 0, t) = d(eta)/dt, against
    ``RegularWave.elevation``. This is what fixes the SIGN of u_z."""
    z = -0.5 / wave.wavenumber
    _, _, kbc = _law_residuals(lambda p: airy_velocity(wave, p, t), wave, x, y, z, t)
    assert abs(kbc) < _TOL


@given(wave=_waves, x=st.floats(-50.0, 50.0), y=st.floats(-50.0, 50.0), t=st.floats(0.0, 100.0))
def test_vertical_acceleration_matches_free_surface(
    wave: RegularWave, x: float, y: float, t: float
) -> None:
    """At the MWL the vertical fluid acceleration is d2(eta)/dt2 (time derivative of the
    kinematic condition). Pins the SIGN of a_z."""
    ht = 1.0e-3 / wave.omega
    e = [float(wave.elevation(t + d, x, y)) for d in (-ht, 0.0, ht)]
    d2eta = (e[0] - 2.0 * e[1] + e[2]) / ht**2
    a_z = float(airy_acceleration(wave, np.array([x, y, 0.0]), t)[2])
    assert abs(a_z - d2eta) < _TOL * wave.amplitude * wave.omega**2


def test_law_checks_reject_the_sign_flipped_field() -> None:
    """Negative control: the M5 PR4 field (u_z sign-flipped) must FAIL all three laws,
    so the law tests above cannot pass with the defect back in."""
    wave = RegularWave(amplitude=0.7, omega=1.3, heading_deg=25.0, phase=0.3)

    def flipped(p: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        u = airy_velocity(wave, p, t).copy()
        u[2] = -u[2]
        return u

    # a point / time where sin(psi) and cos(psi) are both far from zero
    t = (np.pi / 4.0 + wave.phase) / wave.omega
    div, curl, kbc = _law_residuals(
        lambda p: flipped(p, t), wave, 0.0, 0.0, -0.4 / wave.wavenumber, t
    )
    assert abs(div) > 0.5
    assert curl > 0.5
    assert abs(kbc) > 0.5


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_velocity_rejects_bad_point_shape() -> None:
    wave = RegularWave(amplitude=1.0, omega=0.5)
    with pytest.raises(ValueError, match=r"point must have shape \(3,\)"):
        airy_velocity(wave, np.array([0.0, 0.0]), t=0.0)


def test_acceleration_rejects_bad_point_shape() -> None:
    wave = RegularWave(amplitude=1.0, omega=0.5)
    with pytest.raises(ValueError, match=r"point must have shape \(3,\)"):
        airy_acceleration(wave, np.zeros(4), t=0.0)
