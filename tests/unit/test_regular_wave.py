"""Unit tests for regular (monochromatic) Airy waves."""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.waves.regular import RegularWave

_G = 9.80665


# ---------------------------------------------------------------------------
# derived quantities: period, wavenumber, wavelength
# ---------------------------------------------------------------------------


def test_period_matches_two_pi_over_omega() -> None:
    wave = RegularWave(amplitude=1.0, omega=0.8)
    assert wave.period == pytest.approx(2.0 * np.pi / 0.8, rel=1e-15)


def test_deep_water_wavenumber_is_omega_squared_over_g() -> None:
    wave = RegularWave(amplitude=1.0, omega=1.0, gravity=_G)
    assert wave.wavenumber == pytest.approx(1.0 / _G, rel=1e-15)


def test_wavelength_consistent_with_wavenumber() -> None:
    wave = RegularWave(amplitude=1.0, omega=0.5, gravity=_G)
    assert wave.wavelength == pytest.approx(2.0 * np.pi / wave.wavenumber, rel=1e-15)


def test_custom_gravity_rescales_wavenumber() -> None:
    w_earth = RegularWave(amplitude=1.0, omega=1.0, gravity=9.80665)
    w_mars = RegularWave(amplitude=1.0, omega=1.0, gravity=3.721)
    assert w_mars.wavenumber == pytest.approx(w_earth.wavenumber * 9.80665 / 3.721, rel=1e-12)


# ---------------------------------------------------------------------------
# elevation at the origin
# ---------------------------------------------------------------------------


def test_elevation_at_origin_zero_time_equals_amplitude() -> None:
    """At t=0, x=y=0, phi=0: eta = A * cos(0) = A."""
    wave = RegularWave(amplitude=1.5, omega=0.6)
    assert wave.elevation(0.0) == pytest.approx(1.5, rel=1e-15)


def test_elevation_after_quarter_period_at_origin_is_zero() -> None:
    """eta(T/4) = A cos(pi/2) = 0."""
    wave = RegularWave(amplitude=1.0, omega=0.8)
    assert abs(wave.elevation(wave.period / 4.0)) < 1.0e-12


def test_elevation_after_half_period_at_origin_is_minus_amplitude() -> None:
    wave = RegularWave(amplitude=0.75, omega=1.2)
    assert wave.elevation(wave.period / 2.0) == pytest.approx(-0.75, rel=1e-12)


def test_elevation_matches_direct_formula_on_time_series() -> None:
    wave = RegularWave(amplitude=2.0, omega=0.9, phase=0.3)
    t = np.linspace(0.0, 20.0, 401)
    expected = 2.0 * np.cos(0.9 * t - 0.3)
    np.testing.assert_allclose(wave.elevation(t), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# spatial propagation (phase across the domain)
# ---------------------------------------------------------------------------


def test_crest_travels_in_heading_direction_at_phase_speed() -> None:
    """A wave at heading 0 has crests moving with +x at c = omega/k."""
    wave = RegularWave(amplitude=1.0, omega=1.0)
    # At t=0 the crest sits at x=0. At t=T/4 it should be at x = lambda/4.
    t = wave.period / 4.0
    x_crest = wave.wavelength / 4.0
    assert wave.elevation(t, x=x_crest) == pytest.approx(1.0, rel=1e-12)


def test_heading_90_propagates_in_positive_y() -> None:
    wave = RegularWave(amplitude=1.0, omega=0.8, heading_deg=90.0)
    # Same test as above, but the crest moves in +y.
    t = wave.period / 4.0
    y_crest = wave.wavelength / 4.0
    assert wave.elevation(t, x=0.0, y=y_crest) == pytest.approx(1.0, rel=1e-12)
    # And must NOT propagate in x:
    assert wave.elevation(t, x=y_crest, y=0.0) == pytest.approx(np.cos(wave.omega * t), rel=1e-12)


def test_oblique_heading_separates_into_components() -> None:
    """Heading 45 deg: phase advances with both x and y by the same factor."""
    wave = RegularWave(amplitude=1.0, omega=0.7, heading_deg=45.0)
    d = 10.0
    eta_x = wave.elevation(0.0, x=d, y=0.0)
    eta_y = wave.elevation(0.0, x=0.0, y=d)
    eta_xy = wave.elevation(0.0, x=d, y=d)
    # At t=0, eta = cos(-k (x cosβ + y sinβ)). At β=45°, cosβ=sinβ, so
    # eta_x == eta_y, and eta_xy uses twice the argument.
    assert eta_x == pytest.approx(eta_y, rel=1e-12)
    k = wave.wavenumber
    assert eta_xy == pytest.approx(np.cos(-k * 2 * d / np.sqrt(2.0)), rel=1e-12)


def test_elevation_broadcasts_over_spatial_grid() -> None:
    wave = RegularWave(amplitude=1.0, omega=0.5)
    x = np.linspace(-50.0, 50.0, 5)
    t = np.array([[0.0], [wave.period / 2.0]])  # column vector
    eta = wave.elevation(t, x=x[np.newaxis, :], y=0.0)
    assert eta.shape == (2, 5)


# ---------------------------------------------------------------------------
# phase offset
# ---------------------------------------------------------------------------


def test_phase_offset_shifts_in_time_correctly() -> None:
    """A phase phi shifts the signal by phi/omega seconds earlier (so
    eta_shifted(t) = eta_unshifted(t - phi/omega)) because the argument is
    (omega*t - phi)."""
    wave_a = RegularWave(amplitude=1.0, omega=1.0, phase=0.0)
    wave_b = RegularWave(amplitude=1.0, omega=1.0, phase=0.5)
    t = np.linspace(0.0, 20.0, 201)
    np.testing.assert_allclose(wave_b.elevation(t), wave_a.elevation(t - 0.5), rtol=1e-12)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_rejects_negative_amplitude() -> None:
    with pytest.raises(ValueError, match="amplitude"):
        RegularWave(amplitude=-1.0, omega=0.5)


def test_rejects_nonpositive_omega() -> None:
    with pytest.raises(ValueError, match="omega"):
        RegularWave(amplitude=1.0, omega=0.0)
    with pytest.raises(ValueError, match="omega"):
        RegularWave(amplitude=1.0, omega=-0.5)


def test_rejects_nonpositive_gravity() -> None:
    with pytest.raises(ValueError, match="gravity"):
        RegularWave(amplitude=1.0, omega=0.5, gravity=0.0)
