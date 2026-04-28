"""Unit tests for :mod:`floatsim.waves.kinematics` (M5 PR4).

Linear-Airy fluid velocity and acceleration at arbitrary points,
clipped at MWL (no Wheeler stretching in Phase 1; see module docstring
for the TODO).
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.waves.kinematics import airy_acceleration, airy_velocity
from floatsim.waves.regular import RegularWave

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
    """At ``t = T/4``: psi = pi/2, cos psi = 0, sin psi = 1 -> u purely vertical."""
    A, omega = 1.0, 0.5
    wave = RegularWave(amplitude=A, omega=omega)
    u = airy_velocity(wave, np.zeros(3), t=wave.period / 4.0)
    assert u[0] == pytest.approx(0.0, abs=1e-12)
    assert u[1] == pytest.approx(0.0, abs=1e-12)
    assert u[2] == pytest.approx(A * omega, rel=1e-12)


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
    horizontal acceleration is at zero (-sin 0 = 0)."""
    wave = RegularWave(amplitude=1.0, omega=0.5)
    a = airy_acceleration(wave, np.zeros(3), t=0.0)
    assert a[0] == pytest.approx(0.0, abs=1e-12)
    # Vertical acceleration peaks at t = 0 with cos 0 = 1 -> +A*omega^2
    assert a[2] == pytest.approx(wave.amplitude * wave.omega**2, rel=1e-12)


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
