"""Articulated-joint constraint Jacobians (M9 B1/B2, velocity-level KKT).

A joint imposes holonomic constraints between two bodies (or a body and
earth). M9 represents each joint by

* a position-level residual ``phi(xi)`` (used by the KKT projection
  step, PR2), and
* a velocity-level Jacobian ``G(xi)`` mapping generalized velocities
  ``u = (v, omega)`` per body to constraint rates: ``phi_dot = G u``.

**Angular-velocity form (plan Q3 lock).** ``G``'s rotational columns
multiply the body **angular velocity** ``omega`` (world frame), never a
parameterization time-derivative. This honours the program's
rotation-parameterization-agnostic rider structurally: a later LEVEL2
change of the rotational state representation touches only the
kinematic map ``omega = E(theta) theta_dot``, not these constraint
definitions.

**Configuration-dependent arm (audit finding ME — binding).** The
translational Jacobian uses the **rotated** attachment arm
``R(theta) r``; its configuration dependence *is* the geometric
stiffness that makes a pendulum swing. A constant-``G`` implementation
(``[I, -r~]`` with the un-rotated arm) produces a silently-dead
pendulum. The translational rows here are exact at any configuration;
the rotational rows are first-order in the relative angle (M9's
small-angle scope, plan nonlinearity-scope statement).

State conventions
-----------------
* ``xi`` (generalized position), ``(6N,)``: body ``k`` occupies
  ``[6k:6k+6] = (x, y, z, rx, ry, rz)`` where ``(rx, ry, rz)`` is the
  orientation **rotation vector** (axis-angle); ``R_k =
  Rotation.from_rotvec(xi[6k+3:6k+6])``. Small-angle rotation-vector
  coincides with the integrator's small-angle Euler to ``O(theta^2)``.
* generalized velocity ``u``, ``(6N,)``: body ``k`` =
  ``(vx, vy, vz, wx, wy, wz)`` with ``v`` the reference-point velocity
  and ``w`` the world-frame angular velocity.
* earth is body index ``-1``: a fixed world frame; a body-earth joint's
  earth side contributes no columns and a fixed world anchor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

_EARTH: int = -1
_FLOAT_EPS: float = 1.0e-12

_JOINT_KINDS = ("hinge", "yaw_locked", "rigid")
# rows = 3 translational + rotational-lock rows
_N_ROWS = {"hinge": 5, "yaw_locked": 4, "rigid": 6}
_Z_AXIS: NDArray[np.float64] = np.array([0.0, 0.0, 1.0])


def _skew(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Skew matrix ``v~`` with ``v~ w = v x w``."""
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def _orthonormal_perp(axis: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Two orthonormal vectors spanning the plane perpendicular to ``axis``."""
    a = axis / np.linalg.norm(axis)
    seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = seed - (seed @ a) * a
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(a, t1)
    return t1, t2


@dataclass(frozen=True)
class Joint:
    """One articulated joint.

    Parameters
    ----------
    kind
        ``"hinge"`` (3 translations + 2 rotations perpendicular to the
        axis locked; 1 free rotation about the axis -> 5 rows),
        ``"yaw_locked"`` (3 translations + rotation about the axis
        locked; 2 free rotations -> 4 rows; the 12-buoy joint with
        ``axis = z``), or ``"rigid"`` (all 6 relative DOF locked -- 3
        translations + all 3 rotations -> 6 rows; a weld, the two bodies
        move as one rigid body; ``axis`` unused).
    body_a, body_b
        Global body indices; ``-1`` is earth. ``body_a`` must be a real
        body; ``body_b`` may be earth.
    attach_a, attach_b
        ``(3,)`` body-frame offsets from each body's reference point to
        the joint point. When ``body_b == -1``, ``attach_b`` is the
        fixed **world** anchor point instead.
    axis
        ``(3,)`` joint axis in the assembly (world) frame: the free
        rotation axis for a hinge, the locked rotation axis for
        ``yaw_locked``. Carried in body A's frame and rotated by
        ``R_A`` at evaluation time.
    """

    kind: str
    body_a: int
    body_b: int
    attach_a: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    attach_b: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    axis: NDArray[np.float64] = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self) -> None:
        if self.kind not in _JOINT_KINDS:
            raise ValueError(f"joint kind must be one of {_JOINT_KINDS}; got {self.kind!r}")
        if self.body_a == _EARTH:
            raise ValueError("body_a must be a real body (>= 0); only body_b may be earth (-1)")
        if self.body_a == self.body_b:
            raise ValueError(f"a joint cannot connect a body to itself (index {self.body_a})")
        for name in ("attach_a", "attach_b", "axis"):
            v = np.asarray(getattr(self, name), dtype=np.float64)
            if v.shape != (3,):
                raise ValueError(f"{name} must have shape (3,); got {v.shape}")
        if float(np.linalg.norm(self.axis)) < _FLOAT_EPS:
            raise ValueError("axis must be non-zero")

    @property
    def n_rows(self) -> int:
        """Number of scalar constraints this joint imposes."""
        return _N_ROWS[self.kind]


def hinge_joint(
    body_a: int,
    body_b: int,
    *,
    attach_a: NDArray[np.float64],
    attach_b: NDArray[np.float64],
    axis: NDArray[np.float64],
) -> Joint:
    """Revolute joint: free rotation about ``axis``, all else locked (5 rows)."""
    return Joint(
        kind="hinge",
        body_a=body_a,
        body_b=body_b,
        attach_a=np.asarray(attach_a, dtype=np.float64),
        attach_b=np.asarray(attach_b, dtype=np.float64),
        axis=np.asarray(axis, dtype=np.float64),
    )


def yaw_locked_joint(
    body_a: int,
    body_b: int,
    *,
    attach_a: NDArray[np.float64],
    attach_b: NDArray[np.float64],
    axis: NDArray[np.float64] = _Z_AXIS,
) -> Joint:
    """12-buoy joint: 3 translations + rotation about ``axis`` (yaw) locked;
    roll/pitch free (4 rows)."""
    return Joint(
        kind="yaw_locked",
        body_a=body_a,
        body_b=body_b,
        attach_a=np.asarray(attach_a, dtype=np.float64),
        attach_b=np.asarray(attach_b, dtype=np.float64),
        axis=np.asarray(axis, dtype=np.float64),
    )


def rigid_joint(
    body_a: int,
    body_b: int,
    *,
    attach_a: NDArray[np.float64],
    attach_b: NDArray[np.float64],
    axis: NDArray[np.float64] = _Z_AXIS,
) -> Joint:
    """Rigid (weld) joint: all 6 relative DOF locked -- 3 translations + all
    three rotations (6 rows). The two bodies move as one rigid body. ``axis``
    is unused (kept for signature uniformity with the other joint builders)."""
    return Joint(
        kind="rigid",
        body_a=body_a,
        body_b=body_b,
        attach_a=np.asarray(attach_a, dtype=np.float64),
        attach_b=np.asarray(attach_b, dtype=np.float64),
        axis=np.asarray(axis, dtype=np.float64),
    )


def _body_pose(xi: NDArray[np.float64], idx: int) -> tuple[NDArray[np.float64], Rotation]:
    """Return ``(reference position, orientation Rotation)`` for body ``idx``."""
    x = xi[6 * idx : 6 * idx + 3]
    rot = Rotation.from_rotvec(xi[6 * idx + 3 : 6 * idx + 6])
    return x, rot


@dataclass(frozen=True)
class JointSet:
    """A collection of joints over an ``N``-body system (``n_dof = 6N``).

    Assembles the stacked position residual ``phi(xi)`` (shape
    ``(m,)``) and the velocity Jacobian ``G(xi)`` (shape ``(m, 6N)``),
    ``m = sum(joint.n_rows)``.

    ``body_references`` (M10 PR0.75, plan Amendment A2): the world
    reference position of each body (length ``n_bodies``), so the state
    ``xi[6k:6k+3]`` is interpreted as a **displacement from the
    reference** rather than an absolute position. The absolute attach
    point of body ``k`` is then ``ref_k + xi_k[0:3] + R_k @ attach``.
    This is required whenever the joint layer is combined with the
    coupled Cummins system, where ``xi`` is displacement-from-equilibrium
    (``C @ xi`` is the restoring about ``xi = 0``). ``None`` (default) is
    the M9 absolute-position convention (equivalent to all references at
    the origin) -- unchanged. Note: only ``phi`` uses the reference; the
    velocity Jacobian ``G`` is a derivative w.r.t. ``xi`` and so is
    reference-independent (adding a constant ``ref`` does not change
    ``dphi/dxi``).
    """

    joints: tuple[Joint, ...]
    n_bodies: int
    body_references: tuple[NDArray[np.float64], ...] | None = None

    def __post_init__(self) -> None:
        if self.n_bodies < 1:
            raise ValueError(f"n_bodies must be >= 1; got {self.n_bodies}")
        for j in self.joints:
            for idx in (j.body_a, j.body_b):
                if idx != _EARTH and not (0 <= idx < self.n_bodies):
                    raise ValueError(
                        f"joint body index {idx} outside [0, {self.n_bodies}) (earth = -1)"
                    )
        if self.body_references is not None and len(self.body_references) != self.n_bodies:
            raise ValueError(
                f"body_references has {len(self.body_references)} entries; expected "
                f"n_bodies = {self.n_bodies}"
            )

    def _ref(self, idx: int) -> NDArray[np.float64]:
        """World reference position of body ``idx`` (zero for earth or when
        no references were supplied -- the M9 absolute convention)."""
        if idx == _EARTH or self.body_references is None:
            return np.zeros(3, dtype=np.float64)
        return self.body_references[idx]

    @property
    def n_dof(self) -> int:
        return 6 * self.n_bodies

    @property
    def n_constraints(self) -> int:
        return sum(j.n_rows for j in self.joints)

    # -- per-joint kinematics ------------------------------------------------

    def _joint_phi_g(
        self, j: Joint, xi: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Residual and Jacobian rows (``n_rows``) for a single joint."""
        n_dof = self.n_dof
        m = j.n_rows
        phi = np.zeros(m, dtype=np.float64)
        G = np.zeros((m, n_dof), dtype=np.float64)

        xa, Ra = _body_pose(xi, j.body_a)
        arm_a = Ra.apply(j.attach_a)  # R_A r_A (world)
        pa = self._ref(j.body_a) + xa + arm_a
        if j.body_b == _EARTH:
            pb = j.attach_b  # fixed world anchor (earth reference = origin)
            Rb: Rotation | None = None
            arm_b = np.zeros(3)
        else:
            xb, Rb = _body_pose(xi, j.body_b)
            arm_b = Rb.apply(j.attach_b)
            pb = self._ref(j.body_b) + xb + arm_b

        # --- translational rows 0:3 (exact at any configuration) ---
        phi[0:3] = pa - pb
        a0 = 6 * j.body_a
        G[0:3, a0 : a0 + 3] = np.eye(3)
        G[0:3, a0 + 3 : a0 + 6] = -_skew(arm_a)
        if j.body_b != _EARTH:
            b0 = 6 * j.body_b
            G[0:3, b0 : b0 + 3] = -np.eye(3)
            G[0:3, b0 + 3 : b0 + 6] = _skew(arm_b)

        # --- rotational lock rows (first-order in the relative angle) ---
        # Relative rotation A-rel-B; axis carried in A's frame, rotated to world.
        R_rel = Ra if Rb is None else (Rb.inv() * Ra)
        rotvec_rel = R_rel.as_rotvec()
        axis_world = Ra.apply(j.axis)
        axis_world /= np.linalg.norm(axis_world)
        if j.kind == "hinge":
            t1, t2 = _orthonormal_perp(axis_world)
            proj = np.vstack([t1, t2])  # (2, 3): lock the two perpendicular components
        elif j.kind == "yaw_locked":
            proj = axis_world.reshape(1, 3)  # lock the axis component
        else:  # rigid -> lock all three rotational components
            proj = np.eye(3)
        phi[3:] = proj @ rotvec_rel
        # rate: proj (omega_A - omega_B); rotational columns only.
        G[3:, a0 + 3 : a0 + 6] = proj
        if j.body_b != _EARTH:
            b0 = 6 * j.body_b
            G[3:, b0 + 3 : b0 + 6] = -proj
        return phi, G

    def phi(self, xi: NDArray[np.floating]) -> NDArray[np.float64]:
        """Stacked position-level constraint residual, shape ``(m,)``."""
        x: NDArray[np.float64] = np.asarray(xi, dtype=np.float64)
        if x.shape != (self.n_dof,):
            raise ValueError(f"xi must have shape ({self.n_dof},); got {x.shape}")
        out = [self._joint_phi_g(j, x)[0] for j in self.joints]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)

    def jacobian(self, xi: NDArray[np.floating]) -> NDArray[np.float64]:
        """Velocity-level constraint Jacobian ``G(xi)``, shape ``(m, 6N)``."""
        x: NDArray[np.float64] = np.asarray(xi, dtype=np.float64)
        if x.shape != (self.n_dof,):
            raise ValueError(f"xi must have shape ({self.n_dof},); got {x.shape}")
        rows = [self._joint_phi_g(j, x)[1] for j in self.joints]
        if not rows:
            return np.zeros((0, self.n_dof), dtype=np.float64)
        return np.vstack(rows)
