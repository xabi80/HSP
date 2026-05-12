"""Pin the +i wave-force convention end-to-end (F-WAVE-FORCE-CONV epilogue).

The WAMIT reader stores the excitation-force RAO ``X(omega, beta)`` such
that the time-domain force is::

    F(t) = Re[ X * A_wave * exp(+i * omega * t) ]

(see ``floatsim/hydro/readers/wamit.py`` module docstring + conventions
doc Item 24). FloatSim's central data structure ``HydroDatabase.RAO``
carries the WAMIT-convention values verbatim because the OrcaFlex,
Capytaine, and (forthcoming) OrcaWave readers all return RAOs under the
same +i convention.

``floatsim.hydro.excitation.make_regular_wave_force`` is the
WAMIT-data → time-domain-force consumer. Before
``fix-make-regular-wave-force-convention`` it used the ``exp(-i * omega * t)``
convention internally, which conjugates the imaginary part of every
WAMIT-derived RAO and produces a mirror-reflected motion (see
``docs/post-mortems/m6-epilogue-wave-force-convention-bug.md``). The
bug was tracked as F-WAVE-FORCE-CONV through M6 PR4.

This file is the discriminator. Each test predicts the time-domain force
from first principles under the WAMIT +i convention and asserts that
``make_regular_wave_force`` agrees. Synthetic RAOs with non-trivial
imaginary part are essential: a real-valued RAO is invariant under the
convention flip and would not separate +i from -i.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floatsim.hydro.excitation import make_regular_wave_force
from floatsim.hydro.readers.wamit import read_wamit
from floatsim.waves.regular import RegularWave

_FIXTURE_STEM = Path(__file__).parent.parent / "fixtures" / "bem" / "wamit" / "synthetic_simple"
# The synthetic_simple fixture is below the platform-realistic rotational
# A_inf threshold, so the dimensional rescaling pathway fires a warning
# we don't care about here. Pass assume_dimensional=True to bypass.
_AS_DIM: dict[str, object] = {"assume_dimensional": True}

# Fixture facts (from tests/fixtures/bem/wamit/synthetic_simple.3, PER=12.566,
# BETA=0):
#   heave (mode 3) RAO X_3 = (3.535534e5 + 3.535534e5j)  -- |X|=5e5, +45deg
#   pitch (mode 5) RAO X_5 = (1.224647e-10 + 2.000000e6j)  -- |X|=2e6, +90deg
# These are chosen so the +i and -i conventions are well-separated by
# Im(X).
_HEAVE_X: complex = 3.535534e5 + 3.535534e5j
_PITCH_X: complex = 1.224647e-10 + 2.000000e6j
_OMEGA: float = 0.5  # rad/s (matches PER = 4*pi)


def _wamit_plus_i_prediction(rao_value: complex, A: float, omega: float, t: float) -> float:
    """First-principles WAMIT-convention time-domain force at the origin.

    Under the +i convention, ``F(t) = Re[X * A * exp(+i * omega * t)]``.
    With the wave at the origin (body at (0,0,0), wave phase = 0), the
    complex wave amplitude at the body is just ``A``.
    """
    return float(np.real(rao_value * A * np.exp(+1j * omega * t)))


# ---------------------------------------------------------------------------
# discriminator at quarter-period -- the only way to separate +i from -i
# ---------------------------------------------------------------------------


def test_heave_force_at_quarter_period_matches_plus_i_convention() -> None:
    """Discriminator: at t = T/4, +i gives -Im(X) and -i gives +Im(X)."""
    hdb = read_wamit(_FIXTURE_STEM, **_AS_DIM)
    wave = RegularWave(amplitude=1.0, omega=_OMEGA, heading_deg=0.0)
    force = make_regular_wave_force(hdb=hdb, wave=wave)

    quarter_period = 0.5 * np.pi / _OMEGA  # T / 4
    actual = force(quarter_period)[2]  # heave DOF index 2

    expected = _wamit_plus_i_prediction(_HEAVE_X, A=1.0, omega=_OMEGA, t=quarter_period)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-6)
    # Sanity check: the value must be NEGATIVE (i.e. -Im(X)) at T/4 under
    # the +i convention. Under the -i convention it would be POSITIVE.
    assert actual < 0.0, (
        "heave force at T/4 must be negative under WAMIT +i convention "
        f"(got {actual:.3e}); positive would mean -i convention is in force"
    )


def test_pitch_force_at_quarter_period_matches_plus_i_convention() -> None:
    """Pitch RAO has |Im(X)/|X|| = 1.0 (pure +90deg), so the discriminator
    has maximum sensitivity here -- the entire force magnitude flips sign
    between the two conventions at t = T/4.
    """
    hdb = read_wamit(_FIXTURE_STEM, **_AS_DIM)
    wave = RegularWave(amplitude=1.0, omega=_OMEGA, heading_deg=0.0)
    force = make_regular_wave_force(hdb=hdb, wave=wave)

    quarter_period = 0.5 * np.pi / _OMEGA
    actual = force(quarter_period)[4]  # pitch DOF index 4

    expected = _wamit_plus_i_prediction(_PITCH_X, A=1.0, omega=_OMEGA, t=quarter_period)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-3)
    # +i: F_pitch(T/4) = Re[(eps + 2e6j) * j] ~= -2e6 (must be negative).
    assert actual < -1.0e6, (
        "pitch force at T/4 must be ~-2e6 N*m under WAMIT +i convention "
        f"(got {actual:.3e}); ~+2e6 would mean -i convention is in force"
    )


# ---------------------------------------------------------------------------
# whole-period time-series agreement
# ---------------------------------------------------------------------------


def test_heave_force_time_series_matches_plus_i_prediction_over_one_period() -> None:
    """Sample the force at 17 phases over one wave period and assert
    pointwise agreement against the WAMIT +i prediction.

    17 samples is enough to exercise both quadrants of cosine and sine
    (the discriminator is the sin term; on a 17-sample grid every t is
    a non-trivial mix of cos and sin).
    """
    hdb = read_wamit(_FIXTURE_STEM, **_AS_DIM)
    wave = RegularWave(amplitude=1.0, omega=_OMEGA, heading_deg=0.0)
    force = make_regular_wave_force(hdb=hdb, wave=wave)

    period = 2.0 * np.pi / _OMEGA
    ts = np.linspace(0.0, period, 17)
    actual = np.array([force(float(t))[2] for t in ts])
    expected = np.array(
        [_wamit_plus_i_prediction(_HEAVE_X, A=1.0, omega=_OMEGA, t=float(t)) for t in ts]
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-6)


# ---------------------------------------------------------------------------
# non-zero wave phase + non-zero body position -- exercises eta_hat
# ---------------------------------------------------------------------------


def test_force_with_nonzero_wave_phase_matches_plus_i_prediction() -> None:
    """``RegularWave.phase`` is a physical phase offset on the elevation
    cosine: ``eta(t) = A cos(omega*t - k*x - phi)``. Under +i convention
    the eta phasor at (0,0) is ``eta_hat = A * exp(-i * phi)`` (negate to
    keep the physical wave invariant under the convention flip).
    """
    hdb = read_wamit(_FIXTURE_STEM, **_AS_DIM)
    phi = 0.5  # 28.6 degrees -- not a special angle
    A = 2.5
    wave = RegularWave(amplitude=A, omega=_OMEGA, heading_deg=0.0, phase=phi)
    force = make_regular_wave_force(hdb=hdb, wave=wave)

    period = 2.0 * np.pi / _OMEGA
    ts = np.linspace(0.0, period, 11)
    # eta_hat_at_body = A * exp(-i * phi)  for body at origin, +i convention
    eta_hat_at_body = A * np.exp(-1j * phi)
    expected = np.array(
        [float(np.real(_HEAVE_X * eta_hat_at_body * np.exp(+1j * _OMEGA * float(t)))) for t in ts]
    )
    actual = np.array([force(float(t))[2] for t in ts])
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-6)


def test_force_with_body_offset_in_plus_x_matches_plus_i_prediction() -> None:
    """A wave traveling in +X with the body at +x_b experiences a delayed
    signal: ``F_body(t) = F_origin(t - k*x_b/omega)``. This invariant holds
    under BOTH conventions, but the underlying phasor at the body shifts
    sign-flipped between the two -- we verify by predicting from first
    principles in +i.
    """
    hdb = read_wamit(_FIXTURE_STEM, **_AS_DIM)
    A = 1.5
    x_b = 8.0  # metres along +X
    wave = RegularWave(amplitude=A, omega=_OMEGA, heading_deg=0.0)
    k = wave.wavenumber

    force = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(x_b, 0.0, 0.0))

    period = 2.0 * np.pi / _OMEGA
    ts = np.linspace(0.0, period, 13)
    # +i convention: eta_hat_at_body = A * exp(-i * k * x_b)
    eta_hat_at_body = A * np.exp(-1j * k * x_b)
    expected = np.array(
        [float(np.real(_HEAVE_X * eta_hat_at_body * np.exp(+1j * _OMEGA * float(t)))) for t in ts]
    )
    actual = np.array([force(float(t))[2] for t in ts])
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-6)


# ---------------------------------------------------------------------------
# elevation-vs-force time-alignment cross-check
# ---------------------------------------------------------------------------


def test_force_aligns_with_wave_elevation_under_plus_i_convention() -> None:
    """A real-valued RAO ``X = |X|`` should produce a force that is in-phase
    with the elevation at the body (cosine peaks at the same time). Use the
    surge mode (X_1 = 1e6 + 0j at PER=12.566 / beta=0) -- pure real, so
    the force is just ``|X| * eta_at_body(t)``.

    This is the physical-consistency check: regardless of internal phase
    convention, a real-valued RAO must produce in-phase force.
    """
    hdb = read_wamit(_FIXTURE_STEM, **_AS_DIM)
    A = 2.0
    wave = RegularWave(amplitude=A, omega=_OMEGA, heading_deg=0.0)
    force = make_regular_wave_force(hdb=hdb, wave=wave)

    period = 2.0 * np.pi / _OMEGA
    ts = np.linspace(0.0, period, 15)
    f_surge = np.array([force(float(t))[0] for t in ts])
    # Surge X = 1e6 + 0j (purely real); expected force = 1e6 * eta_at_body(t)
    eta_at_body = np.array([float(wave.elevation(float(t), x=0.0, y=0.0)) for t in ts])
    expected = 1.0e6 * eta_at_body
    np.testing.assert_allclose(f_surge, expected, rtol=1.0e-9, atol=1.0e-6)
