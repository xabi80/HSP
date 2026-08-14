"""Strip- and patch-resolved drag export (module 2 of 3).

Schema: FloatFEA `docs/load-interchange-v1.md` v1.2, groups
``/loads/strips/<member>/<source>`` and ``/loads/patches/<body>/plate``.

No solve-path hook is required
------------------------------
`docs/hsp-coupling.md` anticipated that this module would need "the replay
hook", the one permitted touch on the solve loop. **It does not.**

``morison_element_force`` and ``plate_element_force`` are *pure functions* of the
body pose, the body velocity and the fluid field. Given the stored kinematics
they can be re-evaluated offline, element by element, recovering exactly the
per-element and per-patch values that ``make_morison_state_force`` computes and
then sums away at ``morison.py:848`` and ``:812``.

That keeps the whole export additive: no lines added to the integrator or the
force model, nothing imported during a solve, and FloatFEA gate **G1.5** stays
provable by construction rather than by measurement. The same approach was used
for ``mu`` in :mod:`floatsim.io.flr_export`, and verified there against an
instrumented run as **bit-identical**.

The verification obligation transfers with the method
-----------------------------------------------------
Re-evaluation is only faithful if it reproduces the solver's inputs exactly --
the same pose convention, the same velocity, the same fluid sample point, the
same lag. **"Same function, same arguments" is an argument, not a measurement.**
Confirm it the way ``mu`` was confirmed: instrument the summation on a throwaway
branch, diff, and require bit-identity before relying on the output.

Two conventions that must be reproduced, not assumed
----------------------------------------------------
* **Pose.** ``_body_pose_from_xi`` reads ``xi[3:6]`` as **ZYX-intrinsic Euler**
  (morison.py:609-627). ``joints.py`` reads the same slice as an axis-angle
  rotation vector. They agree only to first order, so the strip export declares
  ``rotation_parameterisation = "zyx_intrinsic_euler"`` -- the producing
  module's convention, per FloatFEA's "reconstruct as produced, not as correct".
* **Lag.** ``state_force`` is evaluated at ``(t[n], xi_n, xi_dot_n)`` and applied
  to the step-(n+1) right-hand side (newmark.py:409). Strip values therefore
  carry ``time_alignment = "state_n"``: written at the index of the state they
  were evaluated from, not the index they were applied at.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.morison import (
    MorisonElement,
    PlateDragElement,
    _body_pose_from_xi,
    _body_velocity_at,
    morison_element_force,
    plate_element_force,
)

# The producing module's rotation convention -- morison.py:609-627. Declared
# per channel group because upstream it is not global.
STRIP_ROTATION_PARAMETERISATION: Final[str] = "zyx_intrinsic_euler"

# state_force is evaluated at step n and applied to the step-(n+1) RHS.
STRIP_TIME_ALIGNMENT: Final[str] = "state_n"

FluidFieldFn = object  # (point, t) -> vec3; typed loosely to avoid importing the alias


def recompute_strip_loads(
    element: MorisonElement,
    *,
    xi: NDArray[np.floating],
    xi_dot: NDArray[np.floating],
    t: NDArray[np.floating],
    fluid_velocity_fn: FluidFieldFn,
    rho: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Re-evaluate one Morison element over a stored history.

    Reproduces ``make_morison_state_force``'s inner loop for a single element
    (morison.py:815-848): body-frame midpoint to inertial via the pose, member
    axis likewise, body material velocity at the midpoint, fluid sampled **at
    the deformed midpoint**, then the element force.

    Returns
    -------
    ``(f6, midpoint_inertial)`` -- ``(N, 6)`` generalized force in the inertial
    frame with moments about the body reference point, and ``(N, 3)`` the
    application point at each step. The application point moves, so it is
    exported per-sample rather than as a constant.
    """
    x = np.asarray(xi, dtype=np.float64)
    v = np.asarray(xi_dot, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    if x.shape != v.shape:
        raise ValueError(f"xi {x.shape} and xi_dot {v.shape} must have the same shape")
    if x.shape[0] != tt.size:
        raise ValueError(f"history has {x.shape[0]} samples but t has {tt.size}")

    b = element.body_index
    sl = slice(6 * b, 6 * b + 6)
    mid_body = 0.5 * (element.node_a_body + element.node_b_body)
    axis_body = element.node_b_body - element.node_a_body

    f6 = np.empty((tt.size, 6), dtype=np.float64)
    mid_hist = np.empty((tt.size, 3), dtype=np.float64)
    for i in range(tt.size):
        r_ref, R = _body_pose_from_xi(x[i, sl])
        mid_inertial = r_ref + R @ mid_body
        axis_inertial = R @ axis_body
        axis_hat = axis_inertial / float(np.linalg.norm(axis_inertial))
        arm = mid_inertial - r_ref
        v_body = _body_velocity_at(v[i, sl], R, arm)
        u_fluid = np.asarray(fluid_velocity_fn(mid_inertial, float(tt[i])), dtype=np.float64)
        f6[i] = morison_element_force(
            element,
            midpoint_inertial=mid_inertial,
            axis_hat_inertial=axis_hat,
            body_velocity_at_midpoint=v_body,
            body_acceleration_at_midpoint=None,
            fluid_velocity=u_fluid,
            fluid_acceleration=None,
            rho=rho,
            reference_point_inertial=r_ref,
        )
        mid_hist[i] = mid_inertial
    return f6, mid_hist


def recompute_patch_loads(
    element: PlateDragElement,
    *,
    xi: NDArray[np.floating],
    xi_dot: NDArray[np.floating],
    t: NDArray[np.floating],
    fluid_velocity_fn: FluidFieldFn,
    rho: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Re-evaluate one plate element, recovering the **per-patch** normal force.

    ``plate_element_force`` integrates the broadside term over a polar
    quadrature and returns only the 6-vector; the per-patch field ``df_n``
    exists at ``morison.py:583`` and is summed one line later. This recomputes
    it, because a plate-bending model needs the distribution rather than the
    resultant.

    **The tangential term is NOT patch-resolved** -- it is lumped at the disc
    centre (morison.py:589-595), which the G1.0 audit flagged as the one part of
    the plate load that is not distribution-resolved. It is returned separately
    rather than folded in, so the record does not imply a resolution it lacks.

    Returns
    -------
    ``(df_n, f_tangential, patch_pos_inertial)`` -- ``(N, P)`` per-patch normal
    force, ``(N, 3)`` the lumped tangential force, and ``(N, P, 3)`` patch
    positions relative to the body reference point.
    """
    x = np.asarray(xi, dtype=np.float64)
    v = np.asarray(xi_dot, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    b = element.body_index
    sl = slice(6 * b, 6 * b + 6)
    n_patch = element._patch_area.size

    df_n = np.empty((tt.size, n_patch), dtype=np.float64)
    f_tan = np.empty((tt.size, 3), dtype=np.float64)
    pos = np.empty((tt.size, n_patch, 3), dtype=np.float64)
    for i in range(tt.size):
        r_ref, R = _body_pose_from_xi(x[i, sl])
        centre_inertial = r_ref + R @ element.center_body
        u_fluid = np.asarray(
            fluid_velocity_fn(centre_inertial, float(tt[i])), dtype=np.float64
        )
        omega_inertial = R @ v[i, sl][3:6]

        # Mirrors plate_element_force (morison.py:568-595) exactly.
        n_hat = R @ element._n_hat_body
        arms = element._patch_pos_body @ R.T
        u_body = v[i, sl][0:3] + np.cross(omega_inertial, arms)
        u_rel = u_fluid - u_body
        w = u_rel @ n_hat
        df_n[i] = 0.5 * rho * element.Cd_n * np.abs(w) * w * element._patch_area
        pos[i] = arms

        centre_arm = R @ element.center_body
        u_body_c = v[i, sl][0:3] + np.cross(omega_inertial, centre_arm)
        u_rel_c = u_fluid - u_body_c
        u_t = u_rel_c - float(u_rel_c @ n_hat) * n_hat
        f_tan[i] = 0.5 * rho * element.Cd_t * element.edge_area_m2 * float(
            np.linalg.norm(u_t)
        ) * u_t

    return df_n, f_tan, pos


def verify_against_summed(
    per_element: list[NDArray[np.floating]],
    summed: NDArray[np.floating],
) -> float:
    """Max absolute deviation between the re-evaluated parts and the whole.

    The strip export is only meaningful if the parts sum to what the solver
    applied. This is the local half of FloatFEA gate **G1.6** -- the numerical
    check that catches a strip set which is internally consistent but does not
    correspond to the loads the simulator actually used.
    """
    total = np.zeros_like(np.asarray(summed, dtype=np.float64))
    for f in per_element:
        total += np.asarray(f, dtype=np.float64)
    return float(np.abs(total - np.asarray(summed, dtype=np.float64)).max())
