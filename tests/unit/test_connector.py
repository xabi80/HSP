"""Unit tests for :mod:`floatsim.bodies.connector` (M4 PR3).

Covers the 6-DOF linear spring-damper primitive, the heave-rigid-link
factory, the state-force closure builder consumed by ``integrate_cummins``,
the per-DOF drift diagnostic, and the explicit-stability check.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.bodies.connector import (
    LinearConnector,
    check_connector_stability,
    connector_drift,
    heave_rigid_link,
    make_connector_state_force,
)
from floatsim.hydro.radiation import CumminsLHS

# ---------------------------------------------------------------------------
# LinearConnector dataclass
# ---------------------------------------------------------------------------


def _identity_K(scale: float = 1.0) -> np.ndarray:
    return scale * np.eye(6, dtype=np.float64)


def test_linear_connector_requires_distinct_endpoints() -> None:
    with pytest.raises(ValueError, match="distinct endpoints"):
        LinearConnector(body_a=0, body_b=0, K=_identity_K(), B=np.zeros((6, 6)))


def test_linear_connector_rejects_indices_below_minus_one() -> None:
    with pytest.raises(ValueError, match=">= -1"):
        LinearConnector(body_a=-2, body_b=0, K=_identity_K(), B=np.zeros((6, 6)))


def test_linear_connector_rejects_non_symmetric_K() -> None:
    K = _identity_K()
    K[0, 1] = 5.0  # break symmetry
    with pytest.raises(ValueError, match="K must be symmetric"):
        LinearConnector(body_a=0, body_b=1, K=K, B=np.zeros((6, 6)))


def test_linear_connector_rejects_wrong_shape_K() -> None:
    with pytest.raises(ValueError, match=r"K must have shape \(6, 6\)"):
        LinearConnector(body_a=0, body_b=1, K=np.eye(5), B=np.zeros((6, 6)))


def test_linear_connector_rejects_wrong_shape_rest_offset() -> None:
    with pytest.raises(ValueError, match="rest_offset must have shape"):
        LinearConnector(
            body_a=0,
            body_b=1,
            K=_identity_K(),
            B=np.zeros((6, 6)),
            rest_offset=np.zeros(5, dtype=np.float64),
        )


# ---------------------------------------------------------------------------
# heave_rigid_link factory
# ---------------------------------------------------------------------------


def test_heave_rigid_link_sets_only_the_heave_diagonal() -> None:
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1.0e11)
    expected_K = np.zeros((6, 6))
    expected_K[2, 2] = 1.0e11
    np.testing.assert_array_equal(c.K, expected_K)
    np.testing.assert_array_equal(c.B, np.zeros((6, 6)))
    np.testing.assert_array_equal(c.rest_offset, np.zeros(6))


def test_heave_rigid_link_accepts_damping() -> None:
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1.0e11, penalty_damping=1.0e8)
    assert c.B[2, 2] == 1.0e8
    # No off-diagonal leakage.
    off_diag = c.B.copy()
    off_diag[2, 2] = 0.0
    assert np.all(off_diag == 0.0)


def test_heave_rigid_link_rejects_non_positive_stiffness() -> None:
    with pytest.raises(ValueError, match="penalty_stiffness"):
        heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=0.0)


def test_heave_rigid_link_rejects_negative_damping() -> None:
    with pytest.raises(ValueError, match="penalty_damping"):
        heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1e11, penalty_damping=-1.0)


# ---------------------------------------------------------------------------
# make_connector_state_force — force assembly
# ---------------------------------------------------------------------------


def test_state_force_is_zero_at_rest_offset_with_no_velocity() -> None:
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1.0e10)
    f = make_connector_state_force([c], n_dof=12)
    xi = np.zeros(12)
    xi_dot = np.zeros(12)
    F = f(0.0, xi, xi_dot)
    np.testing.assert_array_equal(F, np.zeros(12))


def test_state_force_heave_displacement_produces_opposite_forces_on_two_bodies() -> None:
    """Heave rigid link: xi_a heave = +1 -> F_a = -k, F_b = +k on heave DOF only."""
    k = 1.0e10
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=k)
    f = make_connector_state_force([c], n_dof=12)
    xi = np.zeros(12)
    xi[2] = 1.0  # body 0 heave = 1
    F = f(0.0, xi, np.zeros(12))
    assert F[2] == -k
    assert F[8] == +k  # body 1 heave slot
    # Every other component zero.
    mask = np.ones(12, dtype=bool)
    mask[[2, 8]] = False
    np.testing.assert_array_equal(F[mask], 0.0)


def test_state_force_obeys_newton_third_law_on_arbitrary_state() -> None:
    """Sum of forces on the two bodies cancels, for any state."""
    rng = np.random.default_rng(0)
    K_raw = rng.standard_normal((6, 6))
    K = 0.5 * (K_raw + K_raw.T) + 10.0 * np.eye(6)  # symmetric
    B_raw = rng.standard_normal((6, 6))
    B = 0.5 * (B_raw + B_raw.T) + 5.0 * np.eye(6)  # symmetric
    c = LinearConnector(
        body_a=0,
        body_b=1,
        K=K,
        B=B,
        rest_offset=np.array([0.1, 0.0, 0.3, 0.0, 0.0, 0.0]),
    )
    f = make_connector_state_force([c], n_dof=12)
    xi = rng.standard_normal(12)
    xi_dot = rng.standard_normal(12)
    F = f(0.0, xi, xi_dot)
    F_a = F[0:6]
    F_b = F[6:12]
    np.testing.assert_allclose(F_a + F_b, 0.0, atol=1e-12)


def test_state_force_damping_couples_velocity_only() -> None:
    """Zero displacement, pure relative velocity -> force is -B @ (xi_dot_a - xi_dot_b)."""
    B = np.zeros((6, 6))
    B[2, 2] = 1.0e7
    c = LinearConnector(body_a=0, body_b=1, K=np.zeros((6, 6)), B=B)
    f = make_connector_state_force([c], n_dof=12)
    xi_dot = np.zeros(12)
    xi_dot[2] = 0.5  # body 0 heave velocity
    F = f(0.0, np.zeros(12), xi_dot)
    assert F[2] == -0.5 * 1.0e7
    assert F[8] == +0.5 * 1.0e7


def test_state_force_earth_endpoint_deposits_force_only_on_live_body() -> None:
    """body_a=-1 (earth): force goes into body_b's slot only."""
    k = 1.0e9
    K = np.zeros((6, 6))
    K[0, 0] = k  # surge spring
    c = LinearConnector(body_a=-1, body_b=0, K=K, B=np.zeros((6, 6)))
    f = make_connector_state_force([c], n_dof=12)
    xi = np.zeros(12)
    xi[0] = 0.0  # body 0 surge
    # Body 0 displaced by 0.3 m in surge relative to earth (0 - 0 - 0 with offset = 0);
    # force on earth is F_a = -K @ delta = -k * 0.3 (in surge), but earth has no slot.
    xi[0] = 0.3
    F = f(0.0, xi, np.zeros(12))
    # Body 0 is endpoint B: F_b = -F_a. Delta = xi_a - xi_b = 0 - 0.3 = -0.3 in surge.
    # F_a = -K @ delta = +k*0.3 in surge; F_b = -F_a = -k*0.3 -> body 0 surge slot.
    assert F[0] == -k * 0.3
    # Second body untouched.
    assert np.all(F[6:] == 0.0)


def test_make_state_force_rejects_bad_body_index() -> None:
    c = heave_rigid_link(body_a=0, body_b=5, penalty_stiffness=1e10)
    with pytest.raises(ValueError, match="outside valid range"):
        make_connector_state_force([c], n_dof=12)  # only 2 bodies -> idx 5 invalid


def test_make_state_force_rejects_n_dof_not_multiple_of_six() -> None:
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1e10)
    with pytest.raises(ValueError, match="multiple of 6"):
        make_connector_state_force([c], n_dof=11)


def test_state_force_multiple_connectors_superpose() -> None:
    k = 1.0e9
    c_ab = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=k)
    c_bc = heave_rigid_link(body_a=1, body_b=2, penalty_stiffness=k)
    f = make_connector_state_force([c_ab, c_bc], n_dof=18)
    xi = np.zeros(18)
    xi[2] = 1.0  # body 0 heave +1
    # c_ab: F_a = -k on body 0, +k on body 1.
    # c_bc: delta = 0 - 0 = 0, contributes nothing.
    F = f(0.0, xi, np.zeros(18))
    assert F[2] == -k
    assert F[8] == +k
    assert F[14] == 0.0


# ---------------------------------------------------------------------------
# connector_drift diagnostic
# ---------------------------------------------------------------------------


def test_connector_drift_is_zero_when_bodies_move_together() -> None:
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1.0e11)
    t = np.linspace(0.0, 10.0, 1001)
    xi_hist = np.zeros((t.size, 12))
    xi_hist[:, 2] = np.sin(t)
    xi_hist[:, 8] = np.sin(t)  # body 1 heave mirrors body 0
    drift = connector_drift(xi_hist, c)
    assert drift.shape == (6,)
    np.testing.assert_allclose(drift, np.zeros(6), atol=1e-15)


def test_connector_drift_returns_peak_delta_per_dof() -> None:
    K = np.eye(6)
    c = LinearConnector(body_a=0, body_b=1, K=K, B=np.zeros((6, 6)))
    xi_hist = np.zeros((3, 12))
    xi_hist[0, 2] = 0.2  # body 0 heave
    xi_hist[1, 8] = -0.3  # body 1 heave
    xi_hist[2, :] = 0.0
    # Heave delta: [0.2, 0.3, 0.0] -> peak 0.3. All other DOFs 0.
    drift = connector_drift(xi_hist, c)
    assert drift[2] == pytest.approx(0.3)
    mask = np.ones(6, dtype=bool)
    mask[2] = False
    np.testing.assert_array_equal(drift[mask], 0.0)


def test_connector_drift_rejects_earth_endpoint() -> None:
    K = np.eye(6)
    c = LinearConnector(body_a=-1, body_b=0, K=K, B=np.zeros((6, 6)))
    with pytest.raises(ValueError, match="earth"):
        connector_drift(np.zeros((10, 6)), c)


# ---------------------------------------------------------------------------
# check_connector_stability diagnostic
# ---------------------------------------------------------------------------


def _two_body_diagonal_lhs(mass: float, stiffness: float) -> CumminsLHS:
    """Helper: block-diagonal 12x12 LHS with diag(mass) and diag(stiffness)."""
    M = mass * np.eye(12, dtype=np.float64)
    C = stiffness * np.eye(12, dtype=np.float64)
    return CumminsLHS(M_plus_Ainf=M, C=C)


def test_check_connector_stability_silent_when_dt_is_safe() -> None:
    """Heave rigid link at k=1e8 on mass=1e7 -> omega=sqrt(k/mu)=sqrt(2e8/1e7)=sqrt(20)
    ~4.47 rad/s, dt_stable ~0.36 s. dt=0.01 s is well below."""
    lhs = _two_body_diagonal_lhs(mass=1.0e7, stiffness=1.0e4)
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1.0e8)
    msgs = check_connector_stability(lhs=lhs, connectors=[c], dt=0.01)
    assert msgs == []


def test_check_connector_stability_warns_when_dt_exceeds_bound() -> None:
    """Extreme stiffness: k=1e15, m=1e7 -> omega = sqrt(k*(m1+m2)/(m1*m2)) ~ 4.47e4 rad/s.
    dt_stable ~3.6e-5 s. dt=0.01 s is far above the bound."""
    lhs = _two_body_diagonal_lhs(mass=1.0e7, stiffness=1.0e4)
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1.0e15)
    msgs = check_connector_stability(lhs=lhs, connectors=[c], dt=0.01)
    assert len(msgs) == 1
    assert "dof 2" in msgs[0]
    assert "dt_stable" in msgs[0]


def test_check_connector_stability_handles_earth_attachment() -> None:
    """body-to-earth: reduced mass is just the live body mass."""
    lhs = _two_body_diagonal_lhs(mass=1.0e7, stiffness=1.0e4)
    K = np.zeros((6, 6))
    K[0, 0] = 1.0e15  # surge mooring spring
    c = LinearConnector(body_a=-1, body_b=0, K=K, B=np.zeros((6, 6)))
    msgs = check_connector_stability(lhs=lhs, connectors=[c], dt=0.01)
    assert len(msgs) == 1
    assert "dof 0" in msgs[0]


def test_check_connector_stability_rejects_non_positive_dt() -> None:
    lhs = _two_body_diagonal_lhs(mass=1e7, stiffness=1e4)
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1e10)
    with pytest.raises(ValueError, match="dt must be finite and positive"):
        check_connector_stability(lhs=lhs, connectors=[c], dt=0.0)


def test_check_connector_stability_rejects_bad_safety_factor() -> None:
    lhs = _two_body_diagonal_lhs(mass=1e7, stiffness=1e4)
    c = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=1e10)
    with pytest.raises(ValueError, match="safety_factor"):
        check_connector_stability(lhs=lhs, connectors=[c], dt=0.01, safety_factor=1.5)
