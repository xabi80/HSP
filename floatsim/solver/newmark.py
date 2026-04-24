"""Generalized-alpha integrator for the linear Cummins equation.

ARCHITECTURE.md §4 names this module (``floatsim/solver/newmark.py``)
and §8 Milestone 2 requires it to advance the linear Cummins ODE

    [M + A_inf] xi_ddot(t) + mu(t) + C xi(t) = F_ext(t)

    mu(t) = integral_{0}^{t} K(t - tau) xi_dot(tau) dtau

with the trailing convolution supplied by
:class:`floatsim.hydro.retardation.RadiationConvolution`.

DOF count
---------
The integrator is agnostic to the number of bodies: ``n_dof = 6`` for a
single body (M2 path) or ``n_dof = 6N`` for ``N``-body runs
(ARCHITECTURE.md §2.2, M4). The size is read off the supplied
:class:`CumminsLHS` and retardation kernel, which must agree.

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

State-dependent force (connectors, mooring)
-------------------------------------------
Forces that depend on generalized position or velocity — 6-DOF linear
spring-damper connectors (M4), analytical catenary tension (M4), drag
elements (M5) — are passed in via ``state_force``. That callable is
evaluated at the **previous step's state** ``(t_n, xi_n, xi_dot_n)``,
same explicit treatment as ``mu_n``. The O(h) lag is acceptable for the
free-decay/periodic-response regimes FloatSim targets and matches the
treatment used in OrcaFlex and Fossen's formulation. For very stiff
penalty springs this imposes the usual explicit-stability floor
``dt < 2 / omega_max`` — connectors expose a diagnostic helper to check
this before the run (see :func:`floatsim.bodies.connector.check_connector_stability`).

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
        ``(N+1, n_dof)`` float64. Generalized position history. Per body
        the DOF order is ``(surge, sway, heave, roll, pitch, yaw)``;
        multi-body globals concatenate bodies in order.
    xi_dot
        ``(N+1, n_dof)`` float64. Generalized velocity history.
    xi_ddot
        ``(N+1, n_dof)`` float64. Generalized acceleration history.
    """

    t: NDArray[np.float64]
    xi: NDArray[np.float64]
    xi_dot: NDArray[np.float64]
    xi_ddot: NDArray[np.float64]


def _zero_force(n_dof: int) -> Callable[[float], NDArray[np.float64]]:
    """Factory for the default ``F(t) = 0`` force callable at ``n_dof`` DOFs."""
    zeros = np.zeros(n_dof, dtype=np.float64)

    def _f(_t: float) -> NDArray[np.float64]:
        return zeros

    return _f


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
    state_force: (
        Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None
    ) = None,
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
        Length-``n_dof`` initial generalized position and velocity, with
        ``n_dof`` read from ``lhs`` (``6`` single-body, ``6N`` multi-body).
    duration
        Total simulation duration in seconds. Must be positive.
    dt
        Integration step. Defaults to ``kernel.dt``; if provided, must
        equal ``kernel.dt`` within floating-point tolerance (the
        convolution buffer is sampled at the kernel's grid).
    external_force
        Optional callable ``t -> F(t)`` returning a length-``n_dof``
        force/moment vector in N / N*m. Defaults to zero (free response).
        Used for time-only forcing such as wave excitation.
    state_force
        Optional callable ``(t, xi, xi_dot) -> F`` returning a length-
        ``n_dof`` force vector that depends on the instantaneous generalized
        state. Evaluated at the **previous step's state** (explicit, lagged
        one step — consistent with the treatment of ``mu_n``). Use for
        connectors, mooring lines, drag, or any other state-dependent load.
        Defaults to zero. If supplied, summed with ``external_force`` on the
        RHS. The O(h) lag imposes an explicit-stability floor
        ``dt < 2 / omega_max`` for very stiff elements — callers should use
        :func:`floatsim.bodies.connector.check_connector_stability` (or an
        analogous helper) to validate ``dt`` before calling.
    rho_inf
        Spectral radius at infinite step size in ``[0, 1]``; tunes the
        integrator's high-frequency numerical damping. ``1`` is the
        energy-conserving trapezoidal limit, ``0`` is maximum damping.
        Default ``0.9`` matches the marine/offshore convention.

    Returns
    -------
    IntegrationResult
        Time grid and per-DOF position, velocity, and acceleration
        histories, each of shape ``(N+1, n_dof)`` with
        ``N = round(duration / dt)``.

    Raises
    ------
    ValueError
        If ``duration`` is non-positive, ``rho_inf`` is outside
        ``[0, 1]``, the DOF counts of ``lhs`` and ``kernel`` disagree,
        the state vectors have the wrong shape, or ``dt`` (if explicitly
        provided) does not match ``kernel.dt``.

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
    if lhs.n_dof != kernel.n_dof:
        raise ValueError(
            f"lhs has {lhs.n_dof} DOFs but kernel has {kernel.n_dof}; "
            "global LHS and retardation kernel must share the same DOF count."
        )
    if dt is None:
        dt = kernel.dt
    elif not np.isclose(dt, kernel.dt, rtol=_DT_MATCH_RTOL, atol=0.0):
        raise ValueError(
            f"integrator dt ({dt}) must equal kernel dt ({kernel.dt}); "
            "resample the kernel to the integrator step before calling."
        )

    n_dof = lhs.n_dof
    xi_0 = np.asarray(xi0, dtype=np.float64).copy()
    xi_dot_0 = np.asarray(xi_dot0, dtype=np.float64).copy()
    if xi_0.shape != (n_dof,) or xi_dot_0.shape != (n_dof,):
        raise ValueError(
            f"xi0 and xi_dot0 must have shape ({n_dof},); got " f"{xi_0.shape}, {xi_dot_0.shape}"
        )

    force = external_force if external_force is not None else _zero_force(n_dof)

    def _eval_state_force(
        t_eval: float, xi_eval: NDArray[np.float64], xi_dot_eval: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if state_force is None:
            return np.zeros(n_dof, dtype=np.float64)
        F_sd = np.asarray(state_force(t_eval, xi_eval, xi_dot_eval), dtype=np.float64)
        if F_sd.shape != (n_dof,):
            raise ValueError(
                f"state_force(t, xi, xi_dot) must return shape ({n_dof},); got {F_sd.shape}"
            )
        return F_sd

    alpha_m, alpha_f, gamma, beta = _generalized_alpha_coefficients(rho_inf)
    h = float(dt)
    M_eff = lhs.M_plus_Ainf
    C = lhs.C
    A_eff = (1.0 - alpha_m) * M_eff + (1.0 - alpha_f) * (h**2) * beta * C

    n_steps = round(duration / h)
    n_samples = n_steps + 1
    t = h * np.arange(n_samples, dtype=np.float64)

    xi_hist = np.empty((n_samples, n_dof), dtype=np.float64)
    xi_dot_hist = np.empty((n_samples, n_dof), dtype=np.float64)
    xi_ddot_hist = np.empty((n_samples, n_dof), dtype=np.float64)

    # Initial acceleration from the instantaneous EOM at t = 0 with mu(0) = 0
    # (continuous-form startup, ARCHITECTURE.md §9.3).
    F0_time = np.asarray(force(0.0), dtype=np.float64)
    if F0_time.shape != (n_dof,):
        raise ValueError(f"external_force(t) must return shape ({n_dof},); got {F0_time.shape}")
    F0_sd = _eval_state_force(0.0, xi_0, xi_dot_0)
    F0 = F0_time + F0_sd
    xi_ddot_0 = np.linalg.solve(M_eff, F0 - C @ xi_0)

    xi_hist[0] = xi_0
    xi_dot_hist[0] = xi_dot_0
    xi_ddot_hist[0] = xi_ddot_0

    buffer = RadiationConvolution(kernel)
    buffer.push(xi_dot_0)

    # mu at t_0: continuous value is 0; the buffer-evaluated artifact
    # K_0 * xi_dot_0 * dt is O(dt) and skipped here to match the §9.3
    # startup convention exactly at the first RHS.
    mu_n = np.zeros(n_dof, dtype=np.float64)

    xi_n = xi_0
    xi_dot_n = xi_dot_0
    xi_ddot_n = xi_ddot_0
    F_n = F0

    for n in range(n_steps):
        t_np1 = t[n + 1]
        F_np1_time = np.asarray(force(t_np1), dtype=np.float64)
        if F_np1_time.shape != (n_dof,):
            raise ValueError(
                f"external_force(t) must return shape ({n_dof},); got {F_np1_time.shape}"
            )
        # State-dependent force evaluated at step-n state — explicit, lagged
        # one step, same treatment as mu_n. This keeps the step-n+1 RHS
        # linear in xi_{n+1}.
        F_np1_sd = _eval_state_force(t[n], xi_n, xi_dot_n)
        F_np1 = F_np1_time + F_np1_sd

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
