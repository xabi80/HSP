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
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from floatsim.hydro.radiation import CumminsLHS

if TYPE_CHECKING:
    from floatsim.bodies.joints import JointSet

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


@dataclass(frozen=True)
class ConstrainedEquilibriumResult:
    """Outcome of a joint-constrained static-equilibrium solve
    (docs/weight-preload-plan.md §2.2).

    Attributes
    ----------
    xi_eq
        Generalized position at equilibrium, length ``6N``.
    lam_eq
        Joint multipliers at equilibrium, length ``m``: the physical force on each
        joint's body A (N on translational rows, N·m on rotational rows), in the M9
        convention where at rest ``F + Gᵀ λ = C ξ`` (docs/m9-joints-plan.md,
        λ-units derivation).
    residual_norm
        ``inf``-norm of the stacked residual: N / N·m on the balance rows, m / rad
        on the constraint rows.
    iterations
        Residual evaluations reported by ``scipy.optimize.root``.
    converged
        ``residual_norm <= tol``: judged on the residual itself, not on scipy's flag
        (tracker STATIC-SOLVE-FALSE-CONVERGENCE).
    """

    xi_eq: NDArray[np.float64]
    lam_eq: NDArray[np.float64]
    residual_norm: float
    iterations: int
    converged: bool


def solve_constrained_static_equilibrium(
    *,
    lhs: CumminsLHS,
    constraints: JointSet,
    state_force: _StateForce | None = None,
    xi0: NDArray[np.floating] | None = None,
    tol: float = 1.0e-6,
    regularization: float | None = None,
    allow_failure: bool = False,
) -> ConstrainedEquilibriumResult:
    """Solve the joint-constrained static equilibrium (ARCHITECTURE.md §9.4;
    docs/weight-preload-plan.md §2.2)::

        C ξ - G(ξ)ᵀ λ = F_state(0, ξ, 0)
        φ(ξ) = 0

    ``F_state`` carries the static loads (the weight-preload vector that
    ``build_system`` composes into the state force) and any state-dependent
    connector or mooring force at rest. ``G`` and ``φ`` are the M9 joint
    Jacobian and residual (:class:`floatsim.bodies.joints.JointSet`); ``λ`` is
    the physical joint force on each joint's body A.

    Structural bodies have no ``C``: the joints fix their position. Rigid-body
    DOFs with no restoring at all (the unmoored surge, sway and yaw of the whole
    assembly) are kept well-posed by the same small diagonal regularisation
    ``lambda_reg * I`` as :func:`solve_static_equilibrium`.

    Parameters
    ----------
    lhs
        Assembled :class:`CumminsLHS`; only ``C`` is used.
    constraints
        The deck's joint set (``n_dof`` must match ``lhs``).
    state_force
        State-dependent force closure, as in :func:`solve_static_equilibrium`.
    xi0
        Initial guess for ``ξ``, length ``6N`` (default zeros); ``λ`` starts at zero.
    tol
        Convergence threshold on the residual ``inf``-norm.
    regularization
        Diagonal regularisation on ``C`` (default ``1e-8 * max|diag(C)|``, or
        ``1e-3`` if ``C`` is zero throughout).
    allow_failure
        Return a non-converged result instead of raising.

    Returns
    -------
    ConstrainedEquilibriumResult

    Raises
    ------
    ValueError
        On shape mismatches or a negative ``regularization``.
    RuntimeError
        If the residual stays above ``tol`` and ``allow_failure`` is ``False``.
    """
    n_dof = int(lhs.C.shape[0])
    if constraints.n_dof != n_dof:
        raise ValueError(f"constraints have n_dof = {constraints.n_dof}; lhs has {n_dof}")
    m = constraints.n_constraints
    if xi0 is None:
        xi_init = np.zeros(n_dof, dtype=np.float64)
    else:
        xi_init = np.asarray(xi0, dtype=np.float64)
        if xi_init.shape != (n_dof,):
            raise ValueError(f"xi0 must have shape ({n_dof},); got {xi_init.shape}")
    C = np.asarray(lhs.C, dtype=np.float64)
    if regularization is None:
        max_c_diag = float(np.max(np.abs(np.diag(C))))
        lambda_reg = 1.0e-8 * max_c_diag if max_c_diag > 0.0 else 1.0e-3
    else:
        if regularization < 0.0:
            raise ValueError(f"regularization must be non-negative; got {regularization}")
        lambda_reg = float(regularization)
    zero_vel = np.zeros(n_dof, dtype=np.float64)
    c_reg = C + lambda_reg * np.eye(n_dof)

    def residual(z: NDArray[np.float64]) -> NDArray[np.float64]:
        xi, lam = z[:n_dof], z[n_dof:]
        f_state = (
            state_force(0.0, xi, zero_vel)
            if state_force is not None
            else np.zeros(n_dof, dtype=np.float64)
        )
        balance = c_reg @ xi - constraints.jacobian(xi).T @ lam - f_state
        return np.asarray(np.concatenate([balance, constraints.phi(xi)]), dtype=np.float64)

    def jacobian(z: NDArray[np.float64]) -> NDArray[np.float64]:
        # Bordered KKT matrix; the geometric (dGᵀλ/dξ) and state-force terms are
        # left to the solver's quasi-Newton updates.
        g = constraints.jacobian(z[:n_dof])
        jac = np.zeros((n_dof + m, n_dof + m), dtype=np.float64)
        jac[:n_dof, :n_dof] = c_reg
        jac[:n_dof, n_dof:] = -g.T
        jac[n_dof:, :n_dof] = g
        return jac

    z0 = np.concatenate([xi_init, np.zeros(m, dtype=np.float64)])
    sol = root(residual, z0, jac=jacobian, method="hybr", tol=tol)
    z = np.asarray(sol.x, dtype=np.float64)
    residual_norm = float(np.max(np.abs(residual(z))))
    converged = residual_norm <= tol
    if not converged and not allow_failure:
        raise RuntimeError(
            f"constrained static equilibrium failed to converge: message={sol.message!r}, "
            f"residual_inf_norm={residual_norm:.3e}"
        )
    return ConstrainedEquilibriumResult(
        xi_eq=z[:n_dof],
        lam_eq=z[n_dof:],
        residual_norm=residual_norm,
        iterations=int(getattr(sol, "nfev", -1)),
        converged=converged,
    )
