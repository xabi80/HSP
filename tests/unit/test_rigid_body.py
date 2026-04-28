"""Unit tests for rigid-body kinematics and Newton-Euler assembly.

ARCHITECTURE.md §3.2 locks the orientation representation: unit-norm
quaternion ``q = [q0, q1, q2, q3]`` (scalar first, Hamilton convention).
§3.3 fixes the per-body state vector. This file exercises the primitives
that live in :mod:`floatsim.bodies.rigid_body`:

* quaternion algebra (identity, axis-angle, Hamilton product, normalize);
* body-frame to inertial rotation matrix ``R(q)``;
* exponential-map quaternion step with renormalization;
* Newton-Euler body-frame accelerations (reference point at CoG).

The "gate" test is the analytical torque-free symmetric-top precession:
for ``I1 = I2 != I3`` under zero external torque, body-frame angular
velocity precesses around the body 3-axis at rate::

    n = (I3 - I1) / I1 * Omega_3                    (Goldstein §5-7)

with ``Omega_3`` the (conserved) component along the body 3-axis. The
test runs a Runge-Kutta 4 step loop over Euler's equations and fits the
precession period from ``omega_body[0](t)`` zero crossings.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.bodies.rigid_body import (
    integrate_quaternion,
    normalize_quaternion,
    quaternion_from_axis_angle,
    quaternion_from_euler_zyx,
    quaternion_identity,
    quaternion_multiply,
    rigid_body_accelerations,
    rotation_matrix,
)

# ---------------------------------------------------------------------------
# quaternion primitives
# ---------------------------------------------------------------------------


def test_quaternion_identity_is_unit_scalar_one() -> None:
    q = quaternion_identity()
    np.testing.assert_array_equal(q, np.array([1.0, 0.0, 0.0, 0.0]))


def test_rotation_matrix_of_identity_is_3x3_identity() -> None:
    R = rotation_matrix(quaternion_identity())
    np.testing.assert_allclose(R, np.eye(3), atol=1e-15)


def test_axis_angle_zero_rotation_is_identity() -> None:
    q = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.0)
    np.testing.assert_allclose(q, quaternion_identity(), atol=1e-15)


def test_axis_angle_180_about_x_flips_y_and_z() -> None:
    q = quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.pi)
    R = rotation_matrix(q)
    np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-14)
    np.testing.assert_allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-14)


def test_hamilton_product_with_identity_is_no_op() -> None:
    q = quaternion_from_axis_angle(np.array([0.3, 0.4, 0.5]), 0.7)
    q = normalize_quaternion(q)
    np.testing.assert_allclose(quaternion_multiply(q, quaternion_identity()), q, atol=1e-15)
    np.testing.assert_allclose(quaternion_multiply(quaternion_identity(), q), q, atol=1e-15)


def test_rotation_matrix_composes_under_hamilton_product() -> None:
    """R(q1 * q2) == R(q1) @ R(q2) — composition of rotations."""
    q1 = quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.3)
    q2 = quaternion_from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5)
    R_prod = rotation_matrix(quaternion_multiply(q1, q2))
    R_manual = rotation_matrix(q1) @ rotation_matrix(q2)
    np.testing.assert_allclose(R_prod, R_manual, atol=1e-14)


def test_rotation_matrix_is_orthogonal_for_arbitrary_q() -> None:
    """R(q) R(q)^T = I and det R(q) = +1 for any unit q."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = normalize_quaternion(rng.standard_normal(4))
        R = rotation_matrix(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-14)
        assert abs(float(np.linalg.det(R)) - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# quaternion_from_euler_zyx (M5 PR4: ZYX-intrinsic = yaw-pitch-roll)
# ---------------------------------------------------------------------------


def test_euler_zyx_zero_angles_is_identity() -> None:
    q = quaternion_from_euler_zyx(0.0, 0.0, 0.0)
    np.testing.assert_allclose(q, quaternion_identity(), atol=1e-15)


def test_euler_zyx_yaw_only_matches_axis_angle_about_z() -> None:
    angle = 0.6
    q = quaternion_from_euler_zyx(0.0, 0.0, angle)
    q_ref = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), angle)
    np.testing.assert_allclose(q, q_ref, atol=1e-14)


def test_euler_zyx_pitch_only_matches_axis_angle_about_y() -> None:
    angle = -0.4
    q = quaternion_from_euler_zyx(0.0, angle, 0.0)
    q_ref = quaternion_from_axis_angle(np.array([0.0, 1.0, 0.0]), angle)
    np.testing.assert_allclose(q, q_ref, atol=1e-14)


def test_euler_zyx_roll_only_matches_axis_angle_about_x() -> None:
    angle = 0.25
    q = quaternion_from_euler_zyx(angle, 0.0, 0.0)
    q_ref = quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), angle)
    np.testing.assert_allclose(q, q_ref, atol=1e-14)


def test_euler_zyx_composition_matches_Rz_Ry_Rx() -> None:
    """ZYX-intrinsic: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = 0.3, -0.2, 0.5
    q = quaternion_from_euler_zyx(roll, pitch, yaw)
    R_q = rotation_matrix(q)
    R_x = rotation_matrix(quaternion_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll))
    R_y = rotation_matrix(quaternion_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch))
    R_z = rotation_matrix(quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw))
    np.testing.assert_allclose(R_q, R_z @ R_y @ R_x, atol=1e-14)


def test_euler_zyx_returns_unit_norm() -> None:
    q = quaternion_from_euler_zyx(0.7, -0.4, 1.1)
    assert abs(float(np.linalg.norm(q)) - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# quaternion integration
# ---------------------------------------------------------------------------


def test_integrate_quaternion_with_zero_omega_is_no_op() -> None:
    q = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.7)
    np.testing.assert_allclose(integrate_quaternion(q, np.zeros(3), 0.1), q, atol=1e-15)


def test_integrate_quaternion_constant_spin_matches_axis_angle() -> None:
    """Spin at omega = e_z for dt seconds -> rotation of angle |omega| dt about z."""
    omega = np.array([0.0, 0.0, 2.0])  # 2 rad/s about body z
    dt = 0.25
    q_end = integrate_quaternion(quaternion_identity(), omega, dt)
    q_ref = quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0]), 2.0 * dt)
    np.testing.assert_allclose(q_end, q_ref, atol=1e-14)


def test_quaternion_norm_preserved_over_many_steps() -> None:
    """Q3 gate: |q|-1 stays within 1e-12 over 10_000 renormalized steps."""
    rng = np.random.default_rng(42)
    q = normalize_quaternion(np.array([1.0, 0.1, -0.2, 0.05]))
    dt = 1e-3
    max_dev = 0.0
    for _ in range(10_000):
        omega = rng.standard_normal(3) * 0.5  # bounded random body-frame spin
        q = integrate_quaternion(q, omega, dt)
        max_dev = max(max_dev, abs(float(np.linalg.norm(q)) - 1.0))
    assert max_dev < 1.0e-12, f"max |q|-1 = {max_dev:.3e}"


# ---------------------------------------------------------------------------
# Newton-Euler accelerations (reference point = CoG)
# ---------------------------------------------------------------------------


_I_BODY = np.diag([1.0, 1.0, 3.0])  # symmetric top: I1 = I2 = 1, I3 = 3
_MASS = 2.0


def test_zero_state_and_zero_force_gives_zero_acceleration() -> None:
    a, alpha = rigid_body_accelerations(
        mass=_MASS,
        inertia_body=_I_BODY,
        v_body=np.zeros(3),
        omega_body=np.zeros(3),
        force_body=np.zeros(3),
        torque_body=np.zeros(3),
    )
    np.testing.assert_array_equal(a, np.zeros(3))
    np.testing.assert_array_equal(alpha, np.zeros(3))


def test_pure_force_at_rest_gives_linear_accel_F_over_m() -> None:
    F = np.array([5.0, -7.0, 11.0])
    a, alpha = rigid_body_accelerations(
        mass=_MASS,
        inertia_body=_I_BODY,
        v_body=np.zeros(3),
        omega_body=np.zeros(3),
        force_body=F,
        torque_body=np.zeros(3),
    )
    np.testing.assert_allclose(a, F / _MASS, atol=1e-15)
    np.testing.assert_array_equal(alpha, np.zeros(3))


def test_pure_torque_at_rest_gives_angular_accel_tau_over_I() -> None:
    tau = np.array([0.3, -0.5, 1.2])
    a, alpha = rigid_body_accelerations(
        mass=_MASS,
        inertia_body=_I_BODY,
        v_body=np.zeros(3),
        omega_body=np.zeros(3),
        force_body=np.zeros(3),
        torque_body=tau,
    )
    np.testing.assert_array_equal(a, np.zeros(3))
    np.testing.assert_allclose(alpha, np.linalg.solve(_I_BODY, tau), atol=1e-15)


def test_linear_accel_includes_minus_omega_cross_v() -> None:
    """m a = F - m omega x v  ->  a = F/m - omega x v  (body frame)."""
    v = np.array([1.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 2.0])
    # Pure coriolis: F = 0 so a = -omega x v = -(-2, 0, 0) = (... let's compute)
    # omega x v = (0,0,2) x (1,0,0) = (0*0 - 2*0, 2*1 - 0*0, 0*0 - 0*1) = (0, 2, 0)
    # -> a = -(0, 2, 0) = (0, -2, 0)
    a, _alpha = rigid_body_accelerations(
        mass=_MASS,
        inertia_body=_I_BODY,
        v_body=v,
        omega_body=omega,
        force_body=np.zeros(3),
        torque_body=np.zeros(3),
    )
    np.testing.assert_allclose(a, np.array([0.0, -2.0, 0.0]), atol=1e-15)


def test_angular_accel_includes_minus_I_inverse_omega_cross_I_omega() -> None:
    """Euler's equation: I alpha = tau - omega x I omega ; tau = 0."""
    omega = np.array([1.0, 0.0, 2.0])  # body frame
    _a, alpha = rigid_body_accelerations(
        mass=_MASS,
        inertia_body=_I_BODY,
        v_body=np.zeros(3),
        omega_body=omega,
        force_body=np.zeros(3),
        torque_body=np.zeros(3),
    )
    # I omega = (1, 0, 6) ;  omega x I omega = (1,0,2) x (1,0,6) = (0*6 - 2*0, 2*1 - 1*6, 1*0 - 0*1)
    #                     = (0, -4, 0)
    # alpha = -I^-1 (0, -4, 0) = -(0, -4, 0) = (0, 4, 0)
    np.testing.assert_allclose(alpha, np.array([0.0, 4.0, 0.0]), atol=1e-14)


# ---------------------------------------------------------------------------
# torque-free symmetric top precession — the M4 Q3 gate
# ---------------------------------------------------------------------------


def _rk4_step_euler_equations(omega: np.ndarray, inertia: np.ndarray, dt: float) -> np.ndarray:
    """One RK4 step of d(omega)/dt = -I^-1 (omega x I omega)."""

    def f(w: np.ndarray) -> np.ndarray:
        _a, alpha = rigid_body_accelerations(
            mass=1.0,  # unused: only alpha is consumed
            inertia_body=inertia,
            v_body=np.zeros(3),
            omega_body=w,
            force_body=np.zeros(3),
            torque_body=np.zeros(3),
        )
        return alpha

    k1 = f(omega)
    k2 = f(omega + 0.5 * dt * k1)
    k3 = f(omega + 0.5 * dt * k2)
    k4 = f(omega + dt * k3)
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _fit_period_zero_crossings(t: np.ndarray, x: np.ndarray) -> float:
    sign = np.sign(x)
    crossings = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if crossings.size < 3:
        raise AssertionError(f"need >= 3 zero crossings; got {crossings.size}")
    t_cross = t[crossings] + (t[crossings + 1] - t[crossings]) * (
        -x[crossings] / (x[crossings + 1] - x[crossings])
    )
    return float(np.mean(np.diff(t_cross)))


def test_torque_free_symmetric_top_precession_rate_matches_goldstein() -> None:
    """For I1=I2=1, I3=3, Omega_3=2: n = (I3-I1)/I1 * Omega_3 = 4 rad/s."""
    I_body = np.diag([1.0, 1.0, 3.0])
    omega = np.array([1.0, 0.0, 2.0])  # Omega_perp = 1, Omega_3 = 2
    n_expected = (I_body[2, 2] - I_body[0, 0]) / I_body[0, 0] * omega[2]
    T_expected = 2.0 * np.pi / n_expected

    dt = 1.0e-4
    n_steps = 50_000  # 5 s -> ~ 3 precession periods (T ~ 1.57 s)
    t_arr = np.zeros(n_steps + 1)
    omega_hist = np.zeros((n_steps + 1, 3))
    omega_hist[0] = omega

    w = omega.copy()
    for k in range(n_steps):
        w = _rk4_step_euler_equations(w, I_body, dt)
        t_arr[k + 1] = (k + 1) * dt
        omega_hist[k + 1] = w

    T_fit = _fit_period_zero_crossings(t_arr, omega_hist[:, 0])
    rel_err = abs(T_fit - T_expected) / T_expected
    assert rel_err < 1.0e-3, (
        f"precession period {T_fit:.5f} s deviates from analytical "
        f"{T_expected:.5f} s by {rel_err:.3%} (limit 0.1%)"
    )


def test_torque_free_angular_momentum_conserved_in_inertial() -> None:
    """L_inertial = R(q) I_body omega_body must be conserved under zero torque."""
    I_body = np.diag([1.0, 1.5, 3.0])  # asymmetric (no symmetry axis)
    omega = np.array([0.4, 0.7, 1.1])
    q = quaternion_identity()

    dt = 1.0e-4
    n_steps = 20_000  # 2 s

    L0 = rotation_matrix(q) @ (I_body @ omega)
    L0_norm = float(np.linalg.norm(L0))

    w = omega.copy()
    for _ in range(n_steps):
        # Mid-point quaternion advance (trapezoidal on omega) for higher-order conservation.
        w_next = _rk4_step_euler_equations(w, I_body, dt)
        q = integrate_quaternion(q, 0.5 * (w + w_next), dt)
        w = w_next

    L_end = rotation_matrix(q) @ (I_body @ w)
    drift = float(np.linalg.norm(L_end - L0)) / L0_norm
    assert drift < 1.0e-6, f"|Delta L| / |L0| = {drift:.3e} (expected ~ 0 under zero torque)"


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_rigid_body_accelerations_rejects_bad_inertia_shape() -> None:
    with pytest.raises(ValueError, match="inertia_body"):
        rigid_body_accelerations(
            mass=1.0,
            inertia_body=np.eye(4),
            v_body=np.zeros(3),
            omega_body=np.zeros(3),
            force_body=np.zeros(3),
            torque_body=np.zeros(3),
        )


def test_rigid_body_accelerations_rejects_non_positive_mass() -> None:
    with pytest.raises(ValueError, match="mass"):
        rigid_body_accelerations(
            mass=0.0,
            inertia_body=np.eye(3),
            v_body=np.zeros(3),
            omega_body=np.zeros(3),
            force_body=np.zeros(3),
            torque_body=np.zeros(3),
        )
