"""M9 PR1 -- joint constraint-Jacobian builders (finite-difference gate).

The velocity-level Jacobian ``G(xi)`` is verified against group-consistent
finite differences of the position residual ``phi(xi)`` -- perturbing the
configuration by an actual rigid-body twist (translation + a rotation
composed through :class:`scipy.spatial.transform.Rotation`, never added to
the rotation vector), so the test is parameterization-honest.

What is asserted, and why the tiers differ (audit finding ME + Q3 lock):

* **Translational rows are EXACT at any configuration** -- ``G_trans =
  [I, -(R r)~]`` is the true derivative of ``p_A - p_B`` for a rigid body.
  This is the geometric-stiffness part that makes a pendulum swing; a
  constant-``G`` bug (the ME failure) shows here immediately. Machine
  precision at finite random angles.
* **The full G is EXACT at zero angle** -- proves the structure and signs
  of every row.
* **The rotational rows are the angular-velocity form** ``P (w_A - w_B)``
  (Q3 lock): the physically-correct velocity-level constraint rate, which
  agrees with ``d(phi_rot)/dtwist`` only to first order in the relative
  angle (M9's small-angle scope). Verified two ways: small-angle
  agreement, and that the residual *vanishes linearly* with the base
  angle -- which a constant/wrong-structure bug cannot fake.
* **Null spaces encode the right freedoms** -- the pendulum's free swing
  is in ``ker(G)``, a locked twist is not; the yaw-locked joint frees
  roll/pitch and locks yaw.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from floatsim.bodies.joints import (
    Joint,
    JointSet,
    hinge_joint,
    rigid_joint,
    yaw_locked_joint,
)

_RNG = np.random.default_rng(20260725)


def _perturb(xi: np.ndarray, twist: np.ndarray, eps: float) -> np.ndarray:
    """Group-consistent config perturbation by ``eps * twist`` (per body
    (v, omega)): translations add; rotations compose through Rotation."""
    out = xi.copy()
    for k in range(xi.size // 6):
        out[6 * k : 6 * k + 3] += eps * twist[6 * k : 6 * k + 3]
        r_k = Rotation.from_rotvec(xi[6 * k + 3 : 6 * k + 6])
        r_new = Rotation.from_rotvec(eps * twist[6 * k + 3 : 6 * k + 6]) * r_k
        out[6 * k + 3 : 6 * k + 6] = r_new.as_rotvec()
    return out


def _fd_jacobian(js: JointSet, xi: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    n = js.n_dof
    cols = []
    for c in range(n):
        tw = np.zeros(n)
        tw[c] = 1.0
        cols.append((js.phi(_perturb(xi, tw, eps)) - js.phi(_perturb(xi, tw, -eps))) / (2 * eps))
    return np.stack(cols, axis=1)


def _random_config(n_bodies: int, angle_scale: float) -> np.ndarray:
    xi = np.zeros(6 * n_bodies)
    for k in range(n_bodies):
        xi[6 * k : 6 * k + 3] = _RNG.standard_normal(3)
        xi[6 * k + 3 : 6 * k + 6] = angle_scale * _RNG.standard_normal(3)
    return xi


_PENDULUM = JointSet(
    joints=(
        hinge_joint(0, -1, attach_a=[0.0, 0.0, 0.5], attach_b=[0.0, 0.0, 0.0], axis=[0, 1, 0]),
    ),
    n_bodies=1,
)
_YAW_BB = JointSet(
    joints=(yaw_locked_joint(0, 1, attach_a=[0.5, 0, 0], attach_b=[-0.5, 0, 0], axis=[0, 0, 1]),),
    n_bodies=2,
)
_RIGID_BB = JointSet(
    joints=(rigid_joint(0, 1, attach_a=[0.5, 0, 0], attach_b=[-0.5, 0, 0]),),
    n_bodies=2,
)


# ---------- the finite-difference gate ----------


@pytest.mark.parametrize("js,angle", [(_PENDULUM, 0.30), (_YAW_BB, 0.30)])
def test_translational_rows_exact_at_any_configuration(js: JointSet, angle: float) -> None:
    """G's translational rows equal FD to machine precision even at finite
    angle -- the geometric-stiffness (ME) part is exact."""
    worst = 0.0
    for _ in range(30):
        xi = _random_config(js.n_bodies, angle)
        g, gf = js.jacobian(xi), _fd_jacobian(js, xi)
        worst = max(worst, float(np.max(np.abs(g[0:3] - gf[0:3]))))
    assert worst < 1e-6, worst


@pytest.mark.parametrize("js", [_PENDULUM, _YAW_BB])
def test_full_jacobian_exact_at_zero_angle(js: JointSet) -> None:
    """At zero relative angle the whole G (translational + rotational)
    equals FD -- structure and signs of every row."""
    xi = np.zeros(js.n_dof)
    xi[0:3] = _RNG.standard_normal(3)  # arbitrary translation, still zero angle
    if js.n_bodies > 1:
        xi[6:9] = _RNG.standard_normal(3)
    assert float(np.max(np.abs(js.jacobian(xi) - _fd_jacobian(js, xi)))) < 1e-7


def test_rotational_rows_are_first_order_angular_velocity_form() -> None:
    """The rotational rows are the angular-velocity constraint-rate form:
    they agree with d(phi)/dtwist at small angle and the residual vanishes
    linearly with the base angle (a constant/wrong-structure G cannot)."""

    def worst_rot_err(angle: float) -> float:
        w = 0.0
        for _ in range(20):
            xi = _random_config(1, angle)
            g, gf = _PENDULUM.jacobian(xi), _fd_jacobian(_PENDULUM, xi)
            w = max(w, float(np.max(np.abs(g[3:] - gf[3:]))))
        return w

    e_small = worst_rot_err(0.02)
    e_big = worst_rot_err(0.20)
    assert e_small < 3e-2  # small-angle agreement (M9 regime)
    # vanishes ~linearly: 10x angle -> ~10x error (5..20x brackets O(theta))
    assert 5.0 < e_big / e_small < 20.0, (e_small, e_big)


# ---------- null-space physics ----------


def test_pendulum_free_swing_in_kernel_locked_twist_not() -> None:
    """Pendulum pinned at world origin, CoM at (0,0,-0.5): the free swing
    about y is in ker(G); a locked roll twist is not."""
    xi = np.zeros(6)
    xi[2] = -0.5
    g = _PENDULUM.jacobian(xi)
    # rigid rotation omega=e_y about the pivot: v_CoM = e_y x (r_com - pivot)
    u_free = np.array([-0.5, 0.0, 0.0, 0.0, 1.0, 0.0])
    u_lock = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])  # pure roll about x
    assert float(np.max(np.abs(g @ u_free))) < 1e-12
    assert float(np.max(np.abs(g @ u_lock))) > 0.5


def test_yaw_locked_frees_roll_pitch_locks_yaw() -> None:
    """Body-body yaw-locked joint at coincident attach points: equal roll
    or pitch of both bodies is free; differential yaw is locked."""
    xi = np.zeros(12)
    xi[0:3] = [0.5, 0.0, 0.0]  # body0 ref so attach (=+x0) sits at origin...
    xi[6:9] = [-0.5, 0.0, 0.0]  # body1 ref so attach (=-x1) sits at origin
    g = _YAW_BB.jacobian(xi)
    # both bodies share angular velocity omega about x (roll) -> free
    u_roll = np.zeros(12)
    u_roll[3], u_roll[9] = 1.0, 1.0
    assert float(np.max(np.abs(g @ u_roll))) < 1e-12
    # differential yaw (about z) -> locked
    u_yaw = np.zeros(12)
    u_yaw[5], u_yaw[11] = 1.0, -1.0
    assert float(np.max(np.abs(g @ u_yaw))) > 0.5


def test_rigid_locks_all_relative_motion_shared_twist_free() -> None:
    """Body-body rigid weld at coincident attach points: a shared rigid twist
    of the pair is free; every differential (relative) motion -- translation
    or rotation about ANY axis -- is locked (unlike yaw_locked, which frees
    roll/pitch)."""
    xi = np.zeros(12)
    xi[0:3] = [0.5, 0.0, 0.0]  # body0 ref so attach (=+x0) sits at origin
    xi[6:9] = [-0.5, 0.0, 0.0]  # body1 ref so attach (=-x1) sits at origin
    g = _RIGID_BB.jacobian(xi)
    # shared roll of both bodies about the common joint (refs on the x-axis
    # -> no reference-point translation) -> a free rigid-body mode of the pair
    u_shared = np.zeros(12)
    u_shared[3], u_shared[9] = 1.0, 1.0
    assert float(np.max(np.abs(g @ u_shared))) < 1e-12
    # differential rotation about x, y AND z -> all locked
    for a in (3, 4, 5):
        u = np.zeros(12)
        u[a], u[a + 6] = 1.0, -1.0
        assert float(np.max(np.abs(g @ u))) > 0.5, a
    # differential translation -> locked
    u_t = np.zeros(12)
    u_t[0], u_t[6] = 1.0, -1.0
    assert float(np.max(np.abs(g @ u_t))) > 0.5


# ---------- shape / raise paths ----------


def test_n_rows_and_n_constraints() -> None:
    assert hinge_joint(0, -1, attach_a=[0, 0, 0], attach_b=[0, 0, 0], axis=[0, 0, 1]).n_rows == 5
    assert yaw_locked_joint(0, 1, attach_a=[0, 0, 0], attach_b=[0, 0, 0]).n_rows == 4
    assert rigid_joint(0, 1, attach_a=[0, 0, 0], attach_b=[0, 0, 0]).n_rows == 6
    js = JointSet(
        joints=(
            hinge_joint(0, -1, attach_a=[0, 0, 0], attach_b=[0, 0, 0], axis=[0, 1, 0]),
            yaw_locked_joint(1, 2, attach_a=[0, 0, 0], attach_b=[0, 0, 0]),
        ),
        n_bodies=3,
    )
    assert js.n_constraints == 9
    assert js.jacobian(np.zeros(18)).shape == (9, 18)
    assert js.phi(np.zeros(18)).shape == (9,)


def test_bad_kind_raises() -> None:
    with pytest.raises(ValueError, match="joint kind"):
        Joint(kind="ball", body_a=0, body_b=-1)


def test_earth_as_body_a_raises() -> None:
    with pytest.raises(ValueError, match="body_a must be a real body"):
        Joint(kind="hinge", body_a=-1, body_b=0)


def test_self_joint_raises() -> None:
    with pytest.raises(ValueError, match="cannot connect a body to itself"):
        Joint(kind="hinge", body_a=2, body_b=2)


def test_zero_axis_raises() -> None:
    with pytest.raises(ValueError, match="axis must be non-zero"):
        Joint(kind="hinge", body_a=0, body_b=-1, axis=np.zeros(3))


def test_body_index_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="outside"):
        JointSet(
            joints=(hinge_joint(0, 5, attach_a=[0, 0, 0], attach_b=[0, 0, 0], axis=[0, 0, 1]),),
            n_bodies=2,
        )


def test_phi_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="xi must have shape"):
        _PENDULUM.phi(np.zeros(5))


# ---------- rigid (weld) joint FD gate ----------
# Appended last so its RNG draws do not shift the module RNG stream seen by the
# order-sensitive small-angle test above.


def test_rigid_translational_rows_exact_at_any_configuration() -> None:
    """Rigid weld: translational rows equal FD to machine precision at finite
    angle -- the geometric-stiffness part is exact, as for the other joints."""
    worst = 0.0
    for _ in range(30):
        xi = _random_config(_RIGID_BB.n_bodies, 0.30)
        g, gf = _RIGID_BB.jacobian(xi), _fd_jacobian(_RIGID_BB, xi)
        worst = max(worst, float(np.max(np.abs(g[0:3] - gf[0:3]))))
    assert worst < 1e-6, worst


def test_rigid_full_jacobian_exact_at_zero_angle() -> None:
    """At zero relative angle the whole 6-row rigid G equals FD -- structure
    and signs of every translational and rotational-lock row."""
    xi = np.zeros(_RIGID_BB.n_dof)
    xi[0:3] = _RNG.standard_normal(3)
    xi[6:9] = _RNG.standard_normal(3)
    assert float(np.max(np.abs(_RIGID_BB.jacobian(xi) - _fd_jacobian(_RIGID_BB, xi)))) < 1e-7
