"""Static equilibrium solver — ARCHITECTURE.md §9.4.

Unit-level verification of :func:`floatsim.solver.equilibrium.solve_static_equilibrium`.
The residual is ``C xi - F_state(0, xi, 0)`` and we check it on three
closed-form cases plus the failure path.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.radiation import CumminsLHS
from floatsim.solver.equilibrium import EquilibriumResult, solve_static_equilibrium


def _spd_diag(diag: list[float]) -> np.ndarray:
    return np.diag(np.asarray(diag, dtype=np.float64))


def _trivial_lhs(n_dof: int = 6, c_diag: float = 1.0e6) -> CumminsLHS:
    """Diagonal LHS with unit-scaled mass and a simple positive-definite C."""
    return CumminsLHS(
        M_plus_Ainf=_spd_diag([1.0] * n_dof),
        C=_spd_diag([c_diag] * n_dof),
    )


def _two_body_lhs() -> CumminsLHS:
    """12-DOF diagonal LHS for a two-body sanity case."""
    return CumminsLHS(
        M_plus_Ainf=_spd_diag([1.0] * 12),
        C=_spd_diag([1.0e6] * 12),
    )


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_zero_force_gives_zero_equilibrium() -> None:
    """No external load + positive-definite C -> xi_eq = 0."""
    lhs = _trivial_lhs()
    result = solve_static_equilibrium(lhs=lhs)
    assert isinstance(result, EquilibriumResult)
    assert result.converged
    assert result.residual_norm < 1.0e-8
    np.testing.assert_allclose(result.xi_eq, np.zeros(6), atol=1.0e-10)


def test_constant_force_matches_closed_form_single_body() -> None:
    """Constant F -> xi_eq = C^{-1} F for a diagonal linear system.

    The 1e-8 regularisation shifts the answer by ``~eps/C = 1e-8``, so we
    check with ``rtol=1e-6`` — still five orders of magnitude below the
    N-level force tolerance the solver actually targets.
    """
    lhs = _trivial_lhs(c_diag=2.5e6)
    F_const = np.array([1.0e5, 0.0, -3.0e5, 0.0, 0.0, 0.0])

    def state_force(_t: float, _xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        return F_const

    result = solve_static_equilibrium(lhs=lhs, state_force=state_force)
    expected = F_const / 2.5e6
    assert result.converged
    np.testing.assert_allclose(result.xi_eq, expected, rtol=1.0e-6, atol=1.0e-10)


def test_linear_spring_to_earth_gives_closed_form_heave_offset() -> None:
    """Body with hydrostatic C_33 and a spring-to-earth with rest offset z0.

    ``F_state(xi) = -k (xi - r0)``. Equilibrium: ``C xi + k xi = k r0`` ->
    ``xi = k r0 / (C + k)``.
    """
    C_33 = 2.0e6
    k = 1.5e6
    r0_heave = -0.30  # 30 cm rest offset (typical mooring pretension)
    C = np.zeros((6, 6), dtype=np.float64)
    C[2, 2] = C_33
    lhs = CumminsLHS(M_plus_Ainf=_spd_diag([1.0] * 6), C=C)
    rest = np.zeros(6)
    rest[2] = r0_heave

    def state_force(_t: float, xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        F = np.zeros(6)
        F[2] = -k * (xi[2] - rest[2])
        return F

    result = solve_static_equilibrium(lhs=lhs, state_force=state_force)
    expected_heave = k * r0_heave / (C_33 + k)
    assert result.converged
    assert result.xi_eq[2] == pytest.approx(expected_heave, rel=1.0e-6)
    # All other DOFs are pinned at zero by the regularisation alone.
    for dof in (0, 1, 3, 4, 5):
        assert abs(result.xi_eq[dof]) < 1.0e-6


def test_two_body_independent_equilibria_solve_simultaneously() -> None:
    """Two uncoupled bodies with different loads -> each converges to its own offset."""
    lhs = _two_body_lhs()
    F_body0 = np.array([1.0e4, 0.0, 2.0e5, 0.0, 0.0, 0.0])
    F_body1 = np.array([-5.0e3, 0.0, 3.0e5, 0.0, 0.0, 0.0])

    def state_force(_t: float, _xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        F = np.zeros(12)
        F[0:6] = F_body0
        F[6:12] = F_body1
        return F

    result = solve_static_equilibrium(lhs=lhs, state_force=state_force)
    assert result.converged
    np.testing.assert_allclose(result.xi_eq[0:6], F_body0 / 1.0e6, rtol=1.0e-6, atol=1.0e-10)
    np.testing.assert_allclose(result.xi_eq[6:12], F_body1 / 1.0e6, rtol=1.0e-6, atol=1.0e-10)


def test_nonlinear_state_force_converges_to_fixed_point() -> None:
    """F_state(xi) = F0 - alpha * xi^3 -> C xi + alpha xi^3 = F0 (smooth scalar fixed point).

    Exercises that hybr handles a non-linear closure, not just constants.
    """
    C_33 = 1.0e6
    alpha = 1.0e9
    F0 = 1.0e5
    C = np.zeros((6, 6), dtype=np.float64)
    C[2, 2] = C_33
    lhs = CumminsLHS(M_plus_Ainf=_spd_diag([1.0] * 6), C=C)

    def state_force(_t: float, xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        F = np.zeros(6)
        F[2] = F0 - alpha * xi[2] ** 3
        return F

    result = solve_static_equilibrium(lhs=lhs, state_force=state_force)
    assert result.converged
    x = result.xi_eq[2]
    # Direct residual check: C*x + alpha*x^3 - F0 ~ 0
    r = C_33 * x + alpha * x**3 - F0
    assert abs(r) < 1.0e-3  # force tolerance (regularisation adds ~eps*x)


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_xi0_wrong_shape_rejected() -> None:
    lhs = _trivial_lhs()
    with pytest.raises(ValueError, match="shape"):
        solve_static_equilibrium(lhs=lhs, xi0=np.zeros(5))


def test_xi0_overrides_default_zero_guess() -> None:
    """A non-zero xi0 should be accepted and still solve to the correct xi_eq."""
    lhs = _trivial_lhs(c_diag=1.0e6)
    F = np.zeros(6)
    F[0] = 1.0e3

    def state_force(_t: float, _xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        return F

    xi0 = 1.0e-2 * np.ones(6)  # far-from-zero guess
    result = solve_static_equilibrium(lhs=lhs, state_force=state_force, xi0=xi0)
    assert result.converged
    np.testing.assert_allclose(result.xi_eq, F / 1.0e6, atol=1.0e-6)


# ---------------------------------------------------------------------------
# convergence / failure paths
# ---------------------------------------------------------------------------


def test_state_force_closure_sees_zero_velocity() -> None:
    """Per §9.4, the equilibrium residual evaluates F_state(t=0, xi, xi_dot=0)."""
    seen: dict[str, np.ndarray] = {}

    def state_force(t: float, xi: np.ndarray, xi_dot: np.ndarray) -> np.ndarray:
        seen["t"] = np.array([t])
        seen["xi_dot"] = xi_dot.copy()
        return np.zeros_like(xi)

    lhs = _trivial_lhs()
    solve_static_equilibrium(lhs=lhs, state_force=state_force)
    assert seen["t"][0] == 0.0
    np.testing.assert_array_equal(seen["xi_dot"], np.zeros(6))


def test_non_convergence_raises_by_default() -> None:
    """A discontinuous closure that never balances must raise RuntimeError."""
    lhs = _trivial_lhs(c_diag=1.0e-20)  # essentially no hydrostatic restoring

    def bad_force(_t: float, xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        # sign-flip around 0 with a huge constant offset -> no fixed point
        # within the scipy step where the derivative is zero.
        F = np.zeros_like(xi)
        F[0] = 1.0e6 if xi[0] >= 0.0 else -1.0e6
        F[0] += 2.0e6  # shifts so both branches have the same sign -> no root
        return F

    with pytest.raises(RuntimeError, match="static equilibrium failed to converge"):
        solve_static_equilibrium(lhs=lhs, state_force=bad_force)


def test_rank_deficient_c_with_loaded_dof_solves_via_regularization() -> None:
    """C has zero row for surge but a state-force pulls in surge.

    Without regularisation scipy hybr can take wild steps in directions
    where the Jacobian column is otherwise zero. The internal default
    ``regularization = 1e-8 * max(diag(C))`` keeps the Jacobian
    full-rank without measurably perturbing the equilibrium.
    """
    C_33 = 2.0e6
    k_surge = 5.0e4  # state-force surge stiffness toward x = 0.10
    rest_x = 0.10
    C = np.zeros((6, 6), dtype=np.float64)
    C[2, 2] = C_33  # heave-only hydrostatic
    lhs = CumminsLHS(M_plus_Ainf=_spd_diag([1.0] * 6), C=C)

    def state_force(_t: float, xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        F = np.zeros(6)
        F[0] = -k_surge * (xi[0] - rest_x)  # surge spring to earth
        return F

    result = solve_static_equilibrium(lhs=lhs, state_force=state_force)
    assert result.converged
    # Surge equilibrium: 0 * xi - (-k(xi - r)) = 0 -> xi = r
    assert result.xi_eq[0] == pytest.approx(rest_x, rel=1.0e-4)
    # Heave with zero force stays at zero (regularization pins it).
    assert abs(result.xi_eq[2]) < 1.0e-6


def test_regularization_zero_disables_pinning() -> None:
    """``regularization=0.0`` removes the diagonal regularisation entirely."""
    lhs = _trivial_lhs()
    F = np.zeros(6)
    F[0] = 1.0e3

    def state_force(_t: float, _xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        return F

    result = solve_static_equilibrium(lhs=lhs, state_force=state_force, regularization=0.0)
    # Even with reg disabled, this PD-C case still solves cleanly.
    np.testing.assert_allclose(result.xi_eq, F / 1.0e6, rtol=1.0e-8, atol=1.0e-12)


def test_negative_regularization_rejected() -> None:
    lhs = _trivial_lhs()
    with pytest.raises(ValueError, match="non-negative"):
        solve_static_equilibrium(lhs=lhs, regularization=-1.0)


def test_allow_failure_returns_result_without_raising() -> None:
    """allow_failure=True lets the caller inspect a non-convergent solve."""
    lhs = _trivial_lhs(c_diag=1.0e-20)

    def bad_force(_t: float, xi: np.ndarray, _xi_dot: np.ndarray) -> np.ndarray:
        F = np.zeros_like(xi)
        F[0] = 1.0e6 if xi[0] >= 0.0 else -1.0e6
        F[0] += 2.0e6
        return F

    result = solve_static_equilibrium(lhs=lhs, state_force=bad_force, allow_failure=True)
    # Whether scipy flags success on this ill-posed problem is an
    # implementation detail; we only require the function to return
    # rather than raise, and the residual is reported honestly.
    assert isinstance(result, EquilibriumResult)
    assert result.residual_norm > 0.0
    assert result.iterations >= 1
