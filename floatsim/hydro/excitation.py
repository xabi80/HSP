"""First-order wave-excitation force from BEM RAOs — ARCHITECTURE.md §2, §M3.

For a regular wave with complex elevation phasor ``eta_hat`` at the body,
the linear wave-excitation force in the ``exp(-i omega t)`` phasor
convention is::

    F_hat(omega, beta) = RAO(omega, beta) * eta_hat

Its time-domain realization is ``F(t) = Re{F_hat * exp(-i omega t)}``
multiplied by an optional smooth ramp ``r(t)`` (ARCHITECTURE.md §9.3).

RAO interpolation
-----------------
The BEM database stores ``RAO`` on a rectangular grid
``(dof, omega, heading)``. For arbitrary ``(omega, beta)`` we use bilinear
interpolation *on the complex values directly* (equivalent to separately
interpolating real and imaginary parts). That is the OrcaFlex convention
for VesselType RAOs in linear mode (ARCHITECTURE.md §M1.5).

Body location
-------------
A body at inertial horizontal position ``(x_b, y_b)`` experiences an
elevation phasor shifted by ``exp(i k (x_b cos beta + y_b sin beta))``
relative to the origin. Milestone-3 validation places the body at the
origin so this factor is unity; the factor is included here for
completeness (§3.1, §M3 steady-state cases with body offset).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave


def interpolate_rao(hdb: HydroDatabase, omega: float, heading_deg: float) -> NDArray[np.complex128]:
    """Bilinear interpolation of the complex RAO at ``(omega, heading_deg)``.

    Parameters
    ----------
    hdb
        Source hydrodynamic database.
    omega
        Angular frequency in rad/s. Must lie within the database grid
        ``[hdb.omega[0], hdb.omega[-1]]``.
    heading_deg
        Wave heading in degrees. Must lie within
        ``[hdb.heading_deg[0], hdb.heading_deg[-1]]`` when the grid has
        more than one heading; must match the single entry exactly when
        it does not.

    Returns
    -------
    ndarray of shape ``(6,)``, dtype ``complex128`` — the interpolated
    force RAO per DOF, per unit wave amplitude.
    """
    omegas = np.asarray(hdb.omega, dtype=np.float64)
    headings = np.asarray(hdb.heading_deg, dtype=np.float64)

    if omega < omegas[0] or omega > omegas[-1]:
        raise ValueError(f"omega={omega} rad/s is outside BEM grid " f"[{omegas[0]}, {omegas[-1]}]")

    i_w = int(np.searchsorted(omegas, omega) - 1)
    i_w = max(0, min(i_w, omegas.size - 2))
    w_lo = omegas[i_w]
    w_hi = omegas[i_w + 1]
    t_w = 0.0 if w_hi == w_lo else (omega - w_lo) / (w_hi - w_lo)

    rao = np.asarray(hdb.RAO, dtype=np.complex128)

    if headings.size == 1:
        if heading_deg != headings[0]:
            raise ValueError(
                f"heading_deg={heading_deg} does not match the single " f"BEM heading {headings[0]}"
            )
        col = rao[:, :, 0]  # (6, n_w)
        return (1.0 - t_w) * col[:, i_w] + t_w * col[:, i_w + 1]

    if heading_deg < headings[0] or heading_deg > headings[-1]:
        raise ValueError(
            f"heading_deg={heading_deg} is outside BEM grid " f"[{headings[0]}, {headings[-1]}]"
        )

    i_h = int(np.searchsorted(headings, heading_deg) - 1)
    i_h = max(0, min(i_h, headings.size - 2))
    h_lo = headings[i_h]
    h_hi = headings[i_h + 1]
    t_h = 0.0 if h_hi == h_lo else (heading_deg - h_lo) / (h_hi - h_lo)

    r00 = rao[:, i_w, i_h]
    r10 = rao[:, i_w + 1, i_h]
    r01 = rao[:, i_w, i_h + 1]
    r11 = rao[:, i_w + 1, i_h + 1]
    return (
        (1.0 - t_w) * (1.0 - t_h) * r00
        + t_w * (1.0 - t_h) * r10
        + (1.0 - t_w) * t_h * r01
        + t_w * t_h * r11
    )


def make_regular_wave_force(
    *,
    hdb: HydroDatabase,
    wave: RegularWave,
    body_position: Sequence[float] = (0.0, 0.0, 0.0),
    ramp: HalfCosineRamp | None = None,
) -> Callable[[float], NDArray[np.float64]]:
    """Build the 6-DOF wave-excitation force callable ``F(t)``.

    The returned function evaluates::

        F(t) = r(t) * Re{ RAO(omega, beta) * eta_hat * exp(-i omega t) }

    where ``eta_hat = A * exp(i (k * (x_b cos beta + y_b sin beta) + phi))``
    is the complex elevation phasor at the body, consistent with
    ``eta(x_b, y_b, t) = Re{ eta_hat * exp(-i omega t) }``.

    Parameters
    ----------
    hdb
        Hydrodynamic database providing the RAO grid.
    wave
        Incident regular wave (frequency, heading, amplitude, phase).
    body_position
        Inertial horizontal position ``(x_b, y_b, z_b)`` of the body
        reference point in metres. Only ``x_b, y_b`` affect the phase;
        ``z_b`` is accepted for symmetry with the inertial frame and
        ignored here (the RAO is evaluated at the mean waterline).
    ramp
        Optional half-cosine ramp. When ``None`` the force is applied
        without envelope (equivalent to ``r(t) = 1``); production runs
        must pass a non-zero ramp per §9.3.

    Returns
    -------
    Callable mapping a scalar time ``t`` in seconds to a
    ``(6,)`` ``float64`` array of wave-excitation force/moment components
    in the DOF order ``(surge, sway, heave, roll, pitch, yaw)``.
    """
    rao = interpolate_rao(hdb, wave.omega, wave.heading_deg)

    body = np.asarray(body_position, dtype=np.float64)
    if body.shape != (3,):
        raise ValueError(f"body_position must have shape (3,); got {body.shape}")
    x_b, y_b, _z_b = body
    beta = np.radians(wave.heading_deg)
    k = wave.wavenumber
    k_dot_x = k * (x_b * float(np.cos(beta)) + y_b * float(np.sin(beta)))
    eta_hat = wave.amplitude * np.exp(1j * (k_dot_x + wave.phase))
    F_hat = rao * eta_hat  # (6,) complex

    omega = wave.omega
    captured_ramp = ramp

    def force(t: float) -> NDArray[np.float64]:
        phasor = F_hat * np.exp(-1j * omega * t)
        f: NDArray[np.float64] = np.real(phasor).astype(np.float64)
        if captured_ramp is not None:
            f = captured_ramp.value(float(t)) * f
        return f

    return force
