"""Rigid-body 6x6 mass matrix assembly — ARCHITECTURE.md §4.

The mass matrix is expressed at the body reference point, in the body
frame. Velocity ordering is ``[v_x, v_y, v_z, omega_x, omega_y, omega_z]``
(linear first, angular second) — matches ARCHITECTURE.md §3.3.

Derivation (see, e.g., Fossen 2011, §3). Let r denote the vector from the
reference point P to the centre of gravity G, in the body frame, and let
I_P denote the inertia tensor about P. Kinetic energy expressed in [v_P,
omega] gives the symmetric 6×6 mass matrix::

    M = [ m * I_3       -m * r_tilde ]
        [ m * r_tilde    I_P        ]

where r_tilde is the skew-symmetric cross-product matrix of r.

When r = 0 (CoG at the reference point), the cross-coupling blocks vanish
and M reduces to block-diagonal ``diag(m, m, m, Ixx, Iyy, Izz)``. In the
general case, -m * r_tilde captures the coupling between angular velocity
and linear momentum that parallel-axis transforms bake into rigid-body
dynamics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_SYMMETRY_RTOL = 1.0e-8


def _skew(r: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return the 3x3 skew-symmetric cross-product matrix of a 3-vector."""
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    return np.array(
        [
            [0.0, -rz, ry],
            [rz, 0.0, -rx],
            [-ry, rx, 0.0],
        ],
        dtype=np.float64,
    )


def rigid_body_mass_matrix(
    *,
    mass: float,
    inertia_at_reference: NDArray[np.floating],
    cog_offset_body: NDArray[np.floating] | None = None,
) -> NDArray[np.float64]:
    """Assemble the 6x6 rigid-body mass matrix at the body reference point.

    Parameters
    ----------
    mass
        Total body mass in kg. Must be strictly positive.
    inertia_at_reference
        3x3 symmetric inertia tensor about the reference point, expressed in
        the body frame, in kg*m^2.
    cog_offset_body
        3-vector from the reference point to the centre of gravity, in the
        body frame (m). Defaults to zero — CoG at the reference point.

    Returns
    -------
    ndarray of shape (6, 6), float64
        Symmetric positive-definite (for physical inputs) mass matrix in the
        body frame, ordered ``[v_x, v_y, v_z, omega_x, omega_y, omega_z]``.
    """
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"mass must be finite and positive; got {mass}")

    I_P = np.asarray(inertia_at_reference, dtype=np.float64)
    if I_P.shape != (3, 3):
        raise ValueError(
            f"inertia_at_reference must have shape (3, 3); got {I_P.shape}"
        )
    if not np.allclose(I_P, I_P.T, rtol=_SYMMETRY_RTOL, atol=1e-10):
        raise ValueError("inertia_at_reference must be symmetric")
    if not np.all(np.isfinite(I_P)):
        raise ValueError("inertia_at_reference must be all-finite")

    if cog_offset_body is None:
        r = np.zeros(3, dtype=np.float64)
    else:
        r = np.asarray(cog_offset_body, dtype=np.float64)
        if r.shape != (3,):
            raise ValueError(
                f"cog_offset_body must have shape (3,); got {r.shape}"
            )
        if not np.all(np.isfinite(r)):
            raise ValueError("cog_offset_body must be all-finite")

    M = np.zeros((6, 6), dtype=np.float64)
    M[:3, :3] = mass * np.eye(3)
    M[3:, 3:] = I_P
    r_tilde = _skew(r)
    M[:3, 3:] = -mass * r_tilde
    M[3:, :3] = mass * r_tilde
    return M
