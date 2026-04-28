"""Fluid kinematics at arbitrary points — ARCHITECTURE.md §4 ``waves/kinematics.py``.

Computes the inertial-frame fluid velocity ``u_fluid(x, y, z, t)`` and
acceleration ``a_fluid(x, y, z, t)`` for a regular Airy wave, evaluated
at a point in space. These are the inputs the Morison drag/inertia
formulas (:mod:`floatsim.hydro.morison`) need at each member's midpoint.

Phase 1 scope (locked in `docs/milestone-5-plan.md` Q1)
-------------------------------------------------------
- **Linear Airy, deep water.** ``u`` and ``a`` are derived from the
  velocity-potential expressions in any standard reference (Faltinsen
  *Sea Loads on Ships and Offshore Structures*, eq. 2.34, or Newman
  *Marine Hydrodynamics* §6.3). The deep-water dispersion ``k = ω²/g``
  is consumed via :attr:`floatsim.waves.regular.RegularWave.wavenumber`.
- **Clipped at MWL.** For a query point above the still water level
  (``z > 0``), the velocity/acceleration is evaluated at ``z = 0``
  (i.e. ``e^{kz}`` is clamped to 1). This is the standard "stretching =
  0" baseline; it overestimates kinematics in the crest and
  underestimates in the trough, but is the correct linear-theory
  reference Phase 1 calibrates against.
- **TODO(phase-2): Wheeler stretching.** A position-mapping refinement
  ``z' = z · h / (h + η)`` (Wheeler 1970) that respects the
  instantaneous free surface. The two helpers in this module are the
  natural injection points; the stretching adds one line each before
  the ``np.exp(k*z)`` factor.

Sign / phase convention
-----------------------
Matches the ``RegularWave.elevation`` convention
(``η(x, y, t) = A·cos(ωt − k·(x·cos β + y·sin β) − φ)``). Time
derivative gives the surface vertical velocity::

    dη/dt = −A·ω·sin(ψ)        with ψ = ω·t − k·(x cos β + y sin β) − φ

For a deep-water Airy wave the velocity field is::

    u_x = A·ω · e^{kz} · cos(ψ) · cos β
    u_y = A·ω · e^{kz} · cos(ψ) · sin β
    u_z = A·ω · e^{kz} · sin(ψ)

and the acceleration field is the partial time derivative::

    a_x = −A·ω² · e^{kz} · sin(ψ) · cos β
    a_y = −A·ω² · e^{kz} · sin(ψ) · sin β
    a_z =  A·ω² · e^{kz} · cos(ψ)

The horizontal velocity is in phase with the elevation; the vertical
velocity leads it by ``π/2``; both decay exponentially with depth at
rate ``k``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from floatsim.waves.regular import RegularWave


def _phase_and_decay(
    wave: RegularWave, point: NDArray[np.float64], t: float
) -> tuple[float, float]:
    """Return ``(ψ, e^{k·z_clipped})`` for a single inertial-frame point at time ``t``.

    ``z_clipped = min(z, 0)`` clamps the depth-decay factor to ``e^0 = 1``
    above the still water level (linear-Airy "no stretching" baseline;
    see module docstring).
    """
    beta = np.radians(wave.heading_deg)
    k = wave.wavenumber
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    psi = wave.omega * float(t) - k * (x * np.cos(beta) + y * np.sin(beta)) - wave.phase
    z_clipped = z if z < 0.0 else 0.0
    decay = float(np.exp(k * z_clipped))
    return float(psi), decay


def airy_velocity(wave: RegularWave, point: NDArray[np.floating], t: float) -> NDArray[np.float64]:
    """Inertial-frame fluid velocity at ``point`` and time ``t`` (m/s).

    Parameters
    ----------
    wave
        :class:`floatsim.waves.regular.RegularWave` carrying amplitude,
        ``ω``, heading, phase, and gravity (for ``k = ω²/g``).
    point
        Length-3 inertial-frame coordinate ``(x, y, z)`` in metres. ``z``
        is positive up, MWL at ``z = 0``.
    t
        Time in seconds.

    Returns
    -------
    NDArray[np.float64]
        Length-3 velocity ``[u_x, u_y, u_z]`` in m/s.

    Notes
    -----
    Above MWL the depth-decay factor is clamped to ``e^0 = 1`` (no
    stretching). See module docstring for the Wheeler-stretching TODO.
    """
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"point must have shape (3,); got {p.shape}")
    psi, decay = _phase_and_decay(wave, p, t)
    beta = np.radians(wave.heading_deg)
    horiz = wave.amplitude * wave.omega * decay * np.cos(psi)
    vert = wave.amplitude * wave.omega * decay * np.sin(psi)
    return np.array([horiz * np.cos(beta), horiz * np.sin(beta), vert], dtype=np.float64)


def airy_acceleration(
    wave: RegularWave, point: NDArray[np.floating], t: float
) -> NDArray[np.float64]:
    """Inertial-frame fluid acceleration at ``point`` and time ``t`` (m/s²).

    Same conventions as :func:`airy_velocity`. The horizontal component
    lags the velocity by ``π/2`` (sin instead of cos); the vertical
    component leads (cos instead of sin), per the analytical
    differentiation in the module docstring.
    """
    p = np.asarray(point, dtype=np.float64)
    if p.shape != (3,):
        raise ValueError(f"point must have shape (3,); got {p.shape}")
    psi, decay = _phase_and_decay(wave, p, t)
    beta = np.radians(wave.heading_deg)
    horiz = -wave.amplitude * (wave.omega**2) * decay * np.sin(psi)
    vert = wave.amplitude * (wave.omega**2) * decay * np.cos(psi)
    return np.array([horiz * np.cos(beta), horiz * np.sin(beta), vert], dtype=np.float64)
