"""Rigid-body kinematics: quaternion algebra + Newton-Euler — ARCHITECTURE.md §3.2, §3.3.

Scope (M4 minimum, per docs/milestone-4-plan.md §Q3):

* Hamilton-product quaternion algebra with per-step renormalization.
* Rotation matrix ``R(q)`` mapping **body-frame vectors to inertial**.
* Exponential-map quaternion step under body-frame angular velocity.
* Newton-Euler body-frame accelerations with the reference point at the
  centre of gravity (``r_CoG = 0``). Off-CoG reference points require the
  ``m r_tilde`` coupling terms assembled by
  :func:`floatsim.bodies.mass_properties.rigid_body_mass_matrix`; they are
  deliberately out of scope for this module.

Conventions (ARCHITECTURE.md §3.2)
----------------------------------
* Quaternion ordering: ``q = [q0, q1, q2, q3]``, scalar first, unit norm.
* Hamilton product — same as OrcaFlex, SPICE, Eigen. For quaternion pair
  ``a, b``::

      (a * b)_0 = a0*b0 - a1*b1 - a2*b2 - a3*b3
      (a * b)_1 = a0*b1 + a1*b0 + a2*b3 - a3*b2
      (a * b)_2 = a0*b2 - a1*b3 + a2*b0 + a3*b1
      (a * b)_3 = a0*b3 + a1*b2 - a2*b1 + a3*b0

* Active rotation (``v_inertial = R(q) v_body``) composition rule:
  ``R(a * b) = R(a) @ R(b)``.
* Body-frame angular velocity ``omega`` advances the quaternion by a
  right-multiplied increment ``q_new = q * dq`` with
  ``dq = [cos(|omega|*dt/2), sin(|omega|*dt/2) * omega_hat]``. This is
  exact for constant body-frame ``omega`` over ``[t, t+dt]``.

Newton-Euler (body frame, CoG reference)
----------------------------------------
With ``v`` the linear velocity of the CoG in the body frame, ``omega`` the
body-frame angular velocity, ``m`` the body mass, and ``I`` the body-frame
inertia tensor about the CoG::

    m * (dv/dt + omega x v) = F_body
    I * (domega/dt) + omega x (I omega) = tau_body

solving for the body-frame accelerations::

    a      = F / m - omega x v
    alpha  = I^-1 (tau - omega x (I omega))

The second line is Euler's equation; the first is the transport-theorem
correction that appears because ``v`` is expressed in a frame rotating at
``omega``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_UNIT_NORM_RTOL = 1.0e-8

# ---------------------------------------------------------------------------
# quaternion primitives
# ---------------------------------------------------------------------------


def quaternion_identity() -> NDArray[np.float64]:
    """Return the identity quaternion ``[1, 0, 0, 0]`` (scalar first)."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def normalize_quaternion(q: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return ``q / |q|``. Raises if ``|q|`` is not finite or is zero."""
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,); got {q_arr.shape}")
    norm = float(np.linalg.norm(q_arr))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError(f"cannot normalize quaternion with norm {norm}")
    return q_arr / norm


def quaternion_from_axis_angle(axis: NDArray[np.floating], angle_rad: float) -> NDArray[np.float64]:
    """Build a unit quaternion for rotation by ``angle_rad`` about ``axis``.

    ``axis`` need not be unit-normalized on input; it is normalized here.
    If ``|axis| == 0`` the rotation angle must also be zero, in which case
    the identity quaternion is returned.

    Result: ``q = [cos(angle/2), sin(angle/2) * axis_hat]``.
    """
    axis_arr = np.asarray(axis, dtype=np.float64)
    if axis_arr.shape != (3,):
        raise ValueError(f"axis must have shape (3,); got {axis_arr.shape}")
    n = float(np.linalg.norm(axis_arr))
    if n == 0.0:
        if angle_rad != 0.0:
            raise ValueError("zero axis with non-zero angle is undefined")
        return quaternion_identity()
    axis_hat = axis_arr / n
    half = 0.5 * float(angle_rad)
    c, s = float(np.cos(half)), float(np.sin(half))
    return np.array([c, s * axis_hat[0], s * axis_hat[1], s * axis_hat[2]], dtype=np.float64)


def quaternion_multiply(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.float64]:
    """Hamilton product ``a * b`` of scalar-first quaternions."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != (4,) or b_arr.shape != (4,):
        raise ValueError(f"quaternions must have shape (4,); got {a_arr.shape} and {b_arr.shape}")
    a0, a1, a2, a3 = a_arr
    b0, b1, b2, b3 = b_arr
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ],
        dtype=np.float64,
    )


def rotation_matrix(q: NDArray[np.floating]) -> NDArray[np.float64]:
    """Body-to-inertial rotation matrix ``R(q)`` for a unit quaternion.

    Acts on body-frame 3-vectors: ``v_inertial = R(q) @ v_body``. The
    convention satisfies ``R(a * b) = R(a) @ R(b)`` with the Hamilton
    product above. Input is asserted unit-norm (within ``rtol=1e-8``); if
    you are unsure, pass the result of :func:`normalize_quaternion` first.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,); got {q_arr.shape}")
    norm = float(np.linalg.norm(q_arr))
    if abs(norm - 1.0) > _UNIT_NORM_RTOL:
        raise ValueError(
            f"rotation_matrix requires a unit quaternion; got |q| = {norm:.6e} "
            "(call normalize_quaternion first)"
        )
    q0, q1, q2, q3 = q_arr
    return np.array(
        [
            [
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                1.0 - 2.0 * (q1 * q1 + q2 * q2),
            ],
        ],
        dtype=np.float64,
    )


def integrate_quaternion(
    q: NDArray[np.floating], omega_body: NDArray[np.floating], dt: float
) -> NDArray[np.float64]:
    """Advance a quaternion by body-frame angular velocity over ``dt``.

    Exponential-map update for constant body-frame ``omega_body`` over the
    step: ``q_new = (q * dq) / |q * dq|``, with
    ``dq = [cos(theta/2), sin(theta/2) * omega_hat]`` and
    ``theta = |omega_body| * dt``. A per-step renormalization is applied
    unconditionally so that round-off drift in ``|q|`` stays bounded over
    long runs (validated to < 1e-12 over 10_000 steps in tests).

    For ``|omega_body| == 0`` the update reduces to ``q`` (identity
    increment) and no normalization is required.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    if q_arr.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,); got {q_arr.shape}")
    w = np.asarray(omega_body, dtype=np.float64)
    if w.shape != (3,):
        raise ValueError(f"omega_body must have shape (3,); got {w.shape}")
    if not np.isfinite(dt):
        raise ValueError(f"dt must be finite; got {dt}")

    speed = float(np.linalg.norm(w))
    if speed == 0.0:
        return q_arr.copy()
    half_angle = 0.5 * speed * float(dt)
    c, s = float(np.cos(half_angle)), float(np.sin(half_angle))
    axis_hat = w / speed
    dq = np.array([c, s * axis_hat[0], s * axis_hat[1], s * axis_hat[2]], dtype=np.float64)
    q_new = quaternion_multiply(q_arr, dq)
    return q_new / float(np.linalg.norm(q_new))


# ---------------------------------------------------------------------------
# Newton-Euler accelerations (body frame, reference point at CoG)
# ---------------------------------------------------------------------------


def rigid_body_accelerations(
    *,
    mass: float,
    inertia_body: NDArray[np.floating],
    v_body: NDArray[np.floating],
    omega_body: NDArray[np.floating],
    force_body: NDArray[np.floating],
    torque_body: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Body-frame linear and angular accelerations, reference point = CoG.

    Governing equations (ARCHITECTURE.md §3.3; see module docstring for
    the transport-theorem derivation)::

        a      = F / m - omega x v
        alpha  = I^-1 (tau - omega x (I omega))

    Parameters
    ----------
    mass
        Body mass in kg. Must be strictly positive and finite.
    inertia_body
        3x3 symmetric positive-definite inertia tensor about the CoG,
        expressed in the body frame, in kg*m^2.
    v_body
        Linear velocity of the CoG in the body frame (3-vector, m/s).
    omega_body
        Body-frame angular velocity (3-vector, rad/s).
    force_body
        Net external force in the body frame (3-vector, N).
    torque_body
        Net external torque about the CoG in the body frame (3-vector, N*m).

    Returns
    -------
    a_body, alpha_body
        Each a length-3 float64 array. ``a_body`` is ``dv/dt`` in body
        components (m/s^2); ``alpha_body`` is ``domega/dt`` in body
        components (rad/s^2).

    Notes
    -----
    This function makes no assumption about the reference frame of
    ``force_body`` or ``torque_body`` other than that they are already
    expressed in the body frame at the CoG. Gravity, hydrostatics, and
    hydrodynamic loads that live in the inertial frame must be rotated
    to body via ``R(q).T`` before being summed into ``force_body``.
    """
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"mass must be finite and positive; got {mass}")

    I_body = np.asarray(inertia_body, dtype=np.float64)
    if I_body.shape != (3, 3):
        raise ValueError(f"inertia_body must have shape (3, 3); got {I_body.shape}")

    v = np.asarray(v_body, dtype=np.float64)
    w = np.asarray(omega_body, dtype=np.float64)
    F = np.asarray(force_body, dtype=np.float64)
    tau = np.asarray(torque_body, dtype=np.float64)
    for name, arr in (("v_body", v), ("omega_body", w), ("force_body", F), ("torque_body", tau)):
        if arr.shape != (3,):
            raise ValueError(f"{name} must have shape (3,); got {arr.shape}")

    a = F / float(mass) - np.cross(w, v)
    alpha = np.linalg.solve(I_body, tau - np.cross(w, I_body @ w)).astype(np.float64)
    return a, alpha
