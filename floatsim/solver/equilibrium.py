"""Static equilibrium solver — ARCHITECTURE.md §9.4.

Pre-step before any dynamic run: find ``xi_eq`` such that the sum of
static forces vanishes,

.. math::

    F_{hydrostatic}(\\xi_{eq}) + F_{mooring}(\\xi_{eq}) = 0.

For the Phase-1 linearization ``F_{hydrostatic}(\\xi) = -C \\xi`` (the
Cummins hydrostatic restoring matrix, assembled in
:func:`floatsim.hydro.radiation.assemble_cummins_lhs`) and mooring /
connector forces are supplied by a state-force closure identical in
shape to the one consumed by
:func:`floatsim.solver.newmark.integrate_cummins`. The residual is

.. math::

    r(\\xi) = C\\,\\xi - F_{state}(t = 0, \\xi, \\dot\\xi = 0),

and we solve ``r(xi_eq) = 0`` with ``scipy.optimize.root``. Sign
convention: ``F_state`` is the force applied **to** the body (positive
pushes the body in the +DOF direction); the hydrostatic reaction is
``-C \\xi``, so the net generalized force on the body at static
equilibrium is ``-C xi + F_state = 0``, i.e. ``C xi = F_state``.

Numerical method: hybrid Powell (``method="hybr"``) by default; it
converges in a handful of iterations for the smooth mooring models in
Phase 1 and does not require a user-supplied Jacobian (a finite
difference is computed internally).

Rank-deficient ``C``
--------------------
Many offshore deck configurations leave some DOFs without hydrostatic
restoring (e.g. surge of a free-floating body, yaw of an axisymmetric
hull) — their rows/columns of ``C`` are zero. If ``F_state`` also does
not depend on such a DOF, the residual is identically zero in that
direction with a zero Jacobian column, and scipy's ``hybr`` can take
unbounded steps before terminating. To keep the solve well-posed for
all deck topologies, we add a small diagonal regularisation
``lambda_reg * I`` to ``C`` inside the residual (see ``regularization``
parameter). Its default is ``1e-8 * max(diag(C))`` — small enough that
a body with ``C_zz = 1e7 N/m`` sees a ``1e-1 N/m`` perturbation to the
surge stiffness, i.e. a ~10 m excursion would be damped by a 1 N fake
restoring. The real physics is unaffected to many decimal places, but
scipy's Jacobian is full-rank.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from floatsim.hydro.radiation import CumminsLHS

_StateForce = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class EquilibriumResult:
    """Outcome of a static-equilibrium solve.

    Attributes
    ----------
    xi_eq
        Generalized position at equilibrium, length ``6N``.
    residual_norm
        ``inf``-norm of ``r(xi_eq)`` in newtons. Scales with ``C`` and
        ``xi`` and is reported for diagnostics; ``converged`` is scipy's
        own flag, not a threshold on this quantity.
    iterations
        Number of residual evaluations reported by ``scipy.optimize.root``.
    converged
        Whether ``scipy.optimize.root`` reported success. A ``False``
        value raises on :func:`solve_static_equilibrium` call unless
        ``allow_failure`` was set.
    """

    xi_eq: NDArray[np.float64]
    residual_norm: float
    iterations: int
    converged: bool


def solve_static_equilibrium(
    *,
    lhs: CumminsLHS,
    state_force: _StateForce | None = None,
    xi0: NDArray[np.floating] | None = None,
    tol: float = 1.0e-6,
    regularization: float | None = None,
    allow_failure: bool = False,
) -> EquilibriumResult:
    """Solve ``C xi = F_state(0, xi, 0)`` for the equilibrium generalized position.

    Parameters
    ----------
    lhs
        Assembled :class:`CumminsLHS` carrying the ``6N x 6N`` global
        hydrostatic matrix ``C`` (and unused-here ``M + A_inf``).
    state_force
        State-dependent force closure, identical in signature to the
        argument of :func:`floatsim.solver.newmark.integrate_cummins`.
        When ``None`` the system reduces to ``C xi = 0`` — trivially
        ``xi = 0`` if ``C`` is positive-definite, singular otherwise.
    xi0
        Initial guess, length ``6N``. Defaults to zeros (often a good
        starting point for small mooring pretensions).
    tol
        Absolute tolerance on the residual ``inf``-norm in newtons. The
        solver also passes this to scipy's own convergence criterion.
    regularization
        Small diagonal regularisation ``lambda_reg * I`` added to ``C``
        inside the residual to keep the scipy Jacobian full-rank when
        some DOFs have no hydrostatic restoring and no ``F_state``
        coupling. Defaults to ``1e-8 * max(diag(C))`` (or ``1e-3`` if
        ``C`` has zero diagonal throughout). Pass ``0.0`` to disable.
    allow_failure
        If ``False`` (default), raise ``RuntimeError`` when scipy reports
        non-convergence. If ``True``, return the best-effort result with
        ``converged=False`` so the caller can inspect the residual.

    Returns
    -------
    EquilibriumResult

    Raises
    ------
    ValueError
        If ``lhs.C`` is not square, ``xi0`` has the wrong shape, or
        ``regularization`` is negative.
    RuntimeError
        On solver failure when ``allow_failure`` is ``False``.
    """
    n_dof = int(lhs.C.shape[0])
    if lhs.C.shape != (n_dof, n_dof):
        raise ValueError(f"lhs.C must be square; got shape {lhs.C.shape}")
    if xi0 is None:
        xi0_arr = np.zeros(n_dof, dtype=np.float64)
    else:
        xi0_arr = np.asarray(xi0, dtype=np.float64)
        if xi0_arr.shape != (n_dof,):
            raise ValueError(f"xi0 must have shape ({n_dof},); got {xi0_arr.shape}")

    if regularization is None:
        max_c_diag = float(np.max(np.abs(np.diag(lhs.C))))
        lambda_reg = 1.0e-8 * max_c_diag if max_c_diag > 0.0 else 1.0e-3
    else:
        if regularization < 0.0:
            raise ValueError(f"regularization must be non-negative; got {regularization}")
        lambda_reg = float(regularization)

    zero_vel = np.zeros(n_dof, dtype=np.float64)

    def residual(xi: NDArray[np.float64]) -> NDArray[np.float64]:
        hydrostatic = lhs.C @ xi + lambda_reg * xi
        f_state = (
            state_force(0.0, xi, zero_vel)
            if state_force is not None
            else np.zeros(n_dof, dtype=np.float64)
        )
        return np.asarray(hydrostatic - f_state, dtype=np.float64)

    sol = root(residual, xi0_arr, method="hybr", tol=tol)
    xi_eq = np.asarray(sol.x, dtype=np.float64)
    r_final = residual(xi_eq)
    residual_norm = float(np.max(np.abs(r_final)))
    iterations = int(getattr(sol, "nfev", -1))
    # Either scipy's own ``success`` flag is set, or the residual is
    # already within ``tol`` newtons — the latter catches rank-deficient
    # C (e.g. DOFs with no hydrostatic restoring) where hybr reports "no
    # good progress" while actually sitting on the solution.
    converged = bool(sol.success) or residual_norm <= tol

    if not converged and not allow_failure:
        raise RuntimeError(
            f"static equilibrium failed to converge: "
            f"message={sol.message!r}, residual_inf_norm={residual_norm:.3e} N"
        )
    return EquilibriumResult(
        xi_eq=xi_eq,
        residual_norm=residual_norm,
        iterations=iterations,
        converged=converged,
    )
