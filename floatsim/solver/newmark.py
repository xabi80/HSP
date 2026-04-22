"""Generalized-alpha integrator for the single-body Cummins equation.

ARCHITECTURE.md §4 names this module (``floatsim/solver/newmark.py``)
and §8 Milestone 2 requires it to advance the linear Cummins ODE

    [M + A_inf] xi_ddot(t) + mu(t) + C xi(t) = F_ext(t)

    mu(t) = integral_{0}^{t} K(t - tau) xi_dot(tau) dtau

with the trailing convolution supplied by
:class:`floatsim.hydro.retardation.RadiationConvolution`.

Integration scheme
------------------
Chung & Hulbert (1993) generalized-alpha method, a one-parameter family
of second-order-accurate, unconditionally stable (for linear problems)
integrators with tunable high-frequency dissipation via
``rho_inf in [0, 1]``.

Parameters (derived from ``rho_inf``)::

    alpha_m = (2 rho_inf - 1) / (rho_inf + 1)
    alpha_f =       rho_inf   / (rho_inf + 1)
    gamma   = 0.5 - alpha_m + alpha_f
    beta    = 0.25 (1 - alpha_m + alpha_f) ** 2

Generalized-alpha balance at step ``n -> n+1``::

    (1-alpha_m) M_eff xi_ddot_{n+1} + alpha_m M_eff xi_ddot_n
      + (1-alpha_f) C xi_{n+1}      + alpha_f C xi_n
      + mu_{n+1-alpha_f}
      = (1-alpha_f) F_{n+1} + alpha_f F_n

Newmark-beta updates for position and velocity::

    xi_{n+1}     = xi_n + h xi_dot_n + h^2 [(1/2 - beta) xi_ddot_n + beta xi_ddot_{n+1}]
    xi_dot_{n+1} = xi_dot_n + h [(1 - gamma) xi_ddot_n + gamma xi_ddot_{n+1}]

Convolution coupling
--------------------
``mu`` is treated explicitly: ``mu_{n+1-alpha_f} ~= mu_n`` (the value from
the end of the previous step). This preserves the 2nd-order accuracy of
the M-C-F parts of the integrator at the cost of an O(h) lag in the
radiation-damping term — negligible for the free-decay timescales that
M2 targets, and consistent with the explicit-convolution treatment used
in OrcaFlex, Fossen (2011), and most marine time-domain codes.

The "startup" boundary condition from ARCHITECTURE.md §9.3 (``xi_dot(tau)
= 0`` for ``tau < 0``) gives ``mu(0) = 0``, so the first step uses ``mu_n
= 0`` for its RHS. The buffer is then loaded with ``xi_dot_0`` at lag 0
(so ``mu_1`` correctly picks up ``xi_dot_0`` at lag ``h`` after the first
push of ``xi_dot_1``).

References
----------
Chung, J. & Hulbert, G.M., 1993. "A time integration algorithm for
structural dynamics with improved numerical dissipation: the
generalized-alpha method." Journal of Applied Mechanics 60 (2), 371-375.

Fossen, T.I., 2011. "Handbook of Marine Craft Hydrodynamics and Motion
Control." Wiley. Chapter 5 (time-domain vs. frequency-domain
representation).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.radiation import CumminsLHS
from floatsim.hydro.retardation import RadiationConvolution, RetardationKernel

_DT_MATCH_RTOL: Final[float] = 1.0e-12


@dataclass(frozen=True)
class IntegrationResult:
    """Time-series output of :func:`integrate_cummins`.

    Attributes
    ----------
    t
        ``(N+1,)`` float64. Uniform time grid ``0, dt, 2*dt, ..., N*dt``
        where ``N = round(duration / dt)``.
    xi
        ``(N+1, 6)`` float64. Generalized position history in body-frame
        DOF order ``(surge, sway, heave, roll, pitch, yaw)``.
    xi_dot
        ``(N+1, 6)`` float64. Generalized velocity history.
    xi_ddot
        ``(N+1, 6)`` float64. Generalized acceleration history.
    """

    t: NDArray[np.float64]
    xi: NDArray[np.float64]
    xi_dot: NDArray[np.float64]
    xi_ddot: NDArray[np.float64]


def _zero_force(_t: float) -> NDArray[np.float64]:
    return np.zeros(6, dtype=np.float64)


def _generalized_alpha_coefficients(rho_inf: float) -> tuple[float, float, float, float]:
    """Return ``(alpha_m, alpha_f, gamma, beta)`` for Chung-Hulbert 1993."""
    alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    gamma = 0.5 - alpha_m + alpha_f
    beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
    return alpha_m, alpha_f, gamma, beta


def integrate_cummins(
    *,
    lhs: CumminsLHS,
    kernel: RetardationKernel,
    xi0: NDArray[np.floating],
    xi_dot0: NDArray[np.floating],
    duration: float,
    dt: float | None = None,
    external_force: Callable[[float], NDArray[np.float64]] | None = None,
    rho_inf: float = 0.9,
) -> IntegrationResult:
    """Integrate the linear Cummins ODE with generalized-alpha.

    Parameters
    ----------
    lhs
        Assembled :class:`CumminsLHS` carrying ``M + A_inf`` and ``C``.
    kernel
        :class:`RetardationKernel` from
        :func:`floatsim.hydro.retardation.compute_retardation_kernel`.
        Its ``dt`` defines the integrator step; ``dt`` below must match
        or be left ``None``.
    xi0, xi_dot0
        Length-6 initial generalized position and velocity.
    duration
        Total simulation duration in seconds. Must be positive.
    dt
        Integration step. Defaults to ``kernel.dt``; if provided, must
        equal ``kernel.dt`` within floating-point tolerance (the
        convolution buffer is sampled at the kernel's grid).
    external_force
        Optional callable ``t -> F(t)`` returning a length-6 force/moment
        vector in N / N*m. Defaults to zero (free response).
    rho_inf
        Spectral radius at infinite step size in ``[0, 1]``; tunes the
        integrator's high-frequency numerical damping. ``1`` is the
        energy-conserving trapezoidal limit, ``0`` is maximum damping.
        Default ``0.9`` matches the marine/offshore convention.

    Returns
    -------
    IntegrationResult
        Time grid and per-DOF position, velocity, and acceleration
        histories, each of shape ``(N+1, 6)`` with ``N = round(duration / dt)``.

    Raises
    ------
    ValueError
        If ``duration`` is non-positive, ``rho_inf`` is outside ``[0, 1]``,
        or ``dt`` (if explicitly provided) does not match ``kernel.dt``.

    Notes
    -----
    The linear-system matrix ``A_eff = (1-alpha_m)(M+A_inf) + (1-alpha_f) h^2 beta C``
    is constant across the run (linear problem) and factorized once via
    :func:`numpy.linalg.solve`'s implicit LU on each call — the factor
    reuse opportunity is left for later profiling.
    """
    if duration <= 0.0:
        raise ValueError(f"duration must be positive; got {duration}")
    if rho_inf < 0.0 or rho_inf > 1.0:
        raise ValueError(f"rho_inf must be in [0, 1]; got {rho_inf}")
    if dt is None:
        dt = kernel.dt
    elif not np.isclose(dt, kernel.dt, rtol=_DT_MATCH_RTOL, atol=0.0):
        raise ValueError(
            f"integrator dt ({dt}) must equal kernel dt ({kernel.dt}); "
            "resample the kernel to the integrator step before calling."
        )

    xi_0 = np.asarray(xi0, dtype=np.float64).copy()
    xi_dot_0 = np.asarray(xi_dot0, dtype=np.float64).copy()
    if xi_0.shape != (6,) or xi_dot_0.shape != (6,):
        raise ValueError(
            f"xi0 and xi_dot0 must have shape (6,); got {xi_0.shape}, {xi_dot_0.shape}"
        )

    force = external_force if external_force is not None else _zero_force

    alpha_m, alpha_f, gamma, beta = _generalized_alpha_coefficients(rho_inf)
    h = float(dt)
    M_eff = lhs.M_plus_Ainf
    C = lhs.C
    A_eff = (1.0 - alpha_m) * M_eff + (1.0 - alpha_f) * (h**2) * beta * C

    n_steps = round(duration / h)
    n_samples = n_steps + 1
    t = h * np.arange(n_samples, dtype=np.float64)

    xi_hist = np.empty((n_samples, 6), dtype=np.float64)
    xi_dot_hist = np.empty((n_samples, 6), dtype=np.float64)
    xi_ddot_hist = np.empty((n_samples, 6), dtype=np.float64)

    # Initial acceleration from the instantaneous EOM at t = 0 with mu(0) = 0
    # (continuous-form startup, ARCHITECTURE.md §9.3).
    F0 = np.asarray(force(0.0), dtype=np.float64)
    if F0.shape != (6,):
        raise ValueError(f"external_force(t) must return shape (6,); got {F0.shape}")
    xi_ddot_0 = np.linalg.solve(M_eff, F0 - C @ xi_0)

    xi_hist[0] = xi_0
    xi_dot_hist[0] = xi_dot_0
    xi_ddot_hist[0] = xi_ddot_0

    buffer = RadiationConvolution(kernel)
    buffer.push(xi_dot_0)

    # mu at t_0: continuous value is 0; the buffer-evaluated artifact
    # K_0 * xi_dot_0 * dt is O(dt) and skipped here to match the §9.3
    # startup convention exactly at the first RHS.
    mu_n = np.zeros(6, dtype=np.float64)

    xi_n = xi_0
    xi_dot_n = xi_dot_0
    xi_ddot_n = xi_ddot_0
    F_n = F0

    for n in range(n_steps):
        t_np1 = t[n + 1]
        F_np1 = np.asarray(force(t_np1), dtype=np.float64)

        # Predictor terms that depend only on step-n state.
        xi_pred = xi_n + h * xi_dot_n + (h**2) * (0.5 - beta) * xi_ddot_n

        rhs = (
            (1.0 - alpha_f) * F_np1
            + alpha_f * F_n
            - alpha_m * (M_eff @ xi_ddot_n)
            - (1.0 - alpha_f) * (C @ xi_pred)
            - alpha_f * (C @ xi_n)
            - mu_n
        )
        xi_ddot_np1 = np.linalg.solve(A_eff, rhs)

        xi_np1 = xi_pred + (h**2) * beta * xi_ddot_np1
        xi_dot_np1 = xi_dot_n + h * ((1.0 - gamma) * xi_ddot_n + gamma * xi_ddot_np1)

        buffer.push(xi_dot_np1)
        mu_np1 = buffer.evaluate()

        xi_hist[n + 1] = xi_np1
        xi_dot_hist[n + 1] = xi_dot_np1
        xi_ddot_hist[n + 1] = xi_ddot_np1

        xi_n = xi_np1
        xi_dot_n = xi_dot_np1
        xi_ddot_n = xi_ddot_np1
        F_n = F_np1
        mu_n = mu_np1

    return IntegrationResult(
        t=t,
        xi=xi_hist,
        xi_dot=xi_dot_hist,
        xi_ddot=xi_ddot_hist,
    )
