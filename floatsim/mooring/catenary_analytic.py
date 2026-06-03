"""Elastic catenary with frictionless seabed contact — ARCHITECTURE.md §4, §7.

Closed-form solution of a linear-elastic mooring line (Irvine 1981, §2.2;
Faltinsen 1990, Ch. 8) hanging between an anchor and a fairlead, with
optional contact on a flat horizontal seabed at ``z = -seabed_depth``.

The line has unstretched length ``L``, submerged weight per unit
unstretched length ``w``, and axial stiffness ``EA``. Each element of
unstretched length ``ds`` under tension ``T`` stretches to
``(1 + T/EA) ds``. Horizontal tension ``H`` is conserved along the line;
vertical tension varies as ``V(s) = V_A + w s`` with ``s`` the
unstretched arc length from the anchor.

Two regimes:

1. **Fully suspended** (``V_A > 0``): the entire line hangs between
   anchor and fairlead.
2. **Touchdown** (``V_A = 0``): part of the line of unstretched length
   ``L_s`` rests on the seabed. The resting portion carries constant
   horizontal tension ``H`` and stretches uniformly to ``L_s (1 + H/EA)``.

Derivation (signs, equations, regime logic) in ``docs/catenary.md`` —
the governing equations appear there as (S1)-(S3) for the suspended case
and (T1)-(T3) for touchdown. This module implements those equations
verbatim.

Coordinate conventions
----------------------
Anchor and fairlead are 2-vectors ``(x, z)`` in the vertical plane of
the line, ``z = 0`` at mean water level, ``z = -h`` at the seabed. The
fairlead's ``x`` coordinate must be **strictly greater** than the
anchor's — callers place the line in a frame aligned with its
horizontal direction and handle sign flips themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

_EARTH: Final[int] = -1

_MinAcceptableHorizontalSpan = 1.0e-9  # [m], below which we treat the line as vertical


@dataclass(frozen=True)
class CatenaryLine:
    """Uniform elastic cable properties.

    Attributes
    ----------
    length
        Unstretched length in m. Must be strictly positive.
    weight_per_length
        Submerged weight per unit unstretched length, in N/m. Must be
        strictly positive (buoyant lines are out of scope).
    EA
        Axial stiffness in N. Must be strictly positive. Pass a very
        large value (e.g. ``1.0e20``) for an effectively inextensible
        line.
    """

    length: float
    weight_per_length: float
    EA: float

    def __post_init__(self) -> None:
        if not (np.isfinite(self.length) and self.length > 0.0):
            raise ValueError(f"length must be finite and positive; got {self.length}")
        if not (np.isfinite(self.weight_per_length) and self.weight_per_length > 0.0):
            raise ValueError(
                f"weight_per_length must be finite and positive; got {self.weight_per_length}"
            )
        if not (np.isfinite(self.EA) and self.EA > 0.0):
            raise ValueError(f"EA must be finite and positive; got {self.EA}")


@dataclass(frozen=True)
class CatenarySolution:
    """Catenary solution at static equilibrium.

    Attributes
    ----------
    regime
        Either ``"suspended"`` (``L_s = 0``) or ``"touchdown"``
        (``0 < L_s < L``).
    H
        Horizontal tension (constant along the line), N.
    V_fairlead
        Vertical tension at the fairlead, N. ``T_fairlead = sqrt(H^2 + V_F^2)``.
    V_anchor
        Vertical tension at the anchor, N. Zero in the touchdown regime.
    touchdown_length
        Unstretched length of line resting on the seabed, m. Zero in the
        suspended regime.
    touchdown_x
        Horizontal coordinate of the touchdown point, m. ``NaN`` for the
        suspended regime.
    top_angle_rad
        Angle between the line and the horizontal at the fairlead,
        ``atan2(V_F, H)``, rad.
    bottom_angle_rad
        Angle at the anchor, ``atan2(V_A, H)``, rad. Zero in the
        touchdown regime (line tangent to seabed).
    """

    regime: Literal["suspended", "touchdown"]
    H: float
    V_fairlead: float
    V_anchor: float
    touchdown_length: float
    touchdown_x: float
    top_angle_rad: float
    bottom_angle_rad: float

    @property
    def T_fairlead(self) -> float:
        """Magnitude of total tension at the fairlead, N."""
        return float(np.hypot(self.H, self.V_fairlead))


# ---------------------------------------------------------------------------
# residuals and Jacobians
# ---------------------------------------------------------------------------


def _suspended_residual(
    unknowns: NDArray[np.float64],
    *,
    L: float,
    w: float,
    EA: float,
    dx: float,
    dz: float,
) -> NDArray[np.float64]:
    """Residual of (S1)-(S2) for the fully-suspended regime.

    ``unknowns = (H, V_A)``; ``V_F = V_A + w L`` (S3) is substituted.
    """
    H, V_A = float(unknowns[0]), float(unknowns[1])
    V_F = V_A + w * L
    r1 = (H / w) * (np.arcsinh(V_F / H) - np.arcsinh(V_A / H)) + H * L / EA - dx
    r2 = (np.hypot(H, V_F) - np.hypot(H, V_A)) / w + (V_A + V_F) * L / (2.0 * EA) - dz
    return np.array([r1, r2], dtype=np.float64)


def _suspended_jacobian(
    unknowns: NDArray[np.float64],
    *,
    L: float,
    w: float,
    EA: float,
    dx: float = 0.0,  # unused; accepted for uniform-call convention
    dz: float = 0.0,
) -> NDArray[np.float64]:
    """Analytical Jacobian of :func:`_suspended_residual` w.r.t. ``(H, V_A)``."""
    H, V_A = float(unknowns[0]), float(unknowns[1])
    V_F = V_A + w * L
    rH_F = np.hypot(H, V_F)
    rH_A = np.hypot(H, V_A)
    # r1 = (H/w) [asinh(V_F/H) - asinh(V_A/H)] + H L/EA - dx
    #    d/dH of (H/w) asinh(V/H) = (1/w) [asinh(V/H) + H * d/dH asinh(V/H)]
    #    d/dH asinh(V/H) = (-V/H^2) / sqrt(1 + V^2/H^2) = -V / (H * hypot(H,V))
    dr1_dH = (
        np.arcsinh(V_F / H) / w
        - V_F / (w * rH_F)
        - np.arcsinh(V_A / H) / w
        + V_A / (w * rH_A)
        + L / EA
    )
    # dV_F/dV_A = 1, dV_A/dV_A = 1
    dr1_dVA = (H / w) * (1.0 / rH_F - 1.0 / rH_A)
    # r2: d/dH of (hypot(H, V_F) - hypot(H, V_A))/w = (H/rH_F - H/rH_A)/w
    dr2_dH = (H / rH_F - H / rH_A) / w
    # d/dV_A: V_F depends on V_A so hypot(H,V_F) derivative is V_F/rH_F; hypot(H,V_A) is V_A/rH_A
    dr2_dVA = (V_F / rH_F - V_A / rH_A) / w + L / EA
    return np.array(
        [
            [dr1_dH, dr1_dVA],
            [dr2_dH, dr2_dVA],
        ],
        dtype=np.float64,
    )


def _touchdown_residual(
    unknowns: NDArray[np.float64],
    *,
    L: float,
    w: float,
    EA: float,
    dx: float,
    dz: float,
) -> NDArray[np.float64]:
    """Residual of (T1)-(T2) for the touchdown regime.

    ``unknowns = (H, L_s)``; ``V_F = w (L - L_s)``, ``V_A = 0``.
    """
    H, L_s = float(unknowns[0]), float(unknowns[1])
    V_F = w * (L - L_s)
    r1 = L_s + H * L / EA + (H / w) * np.arcsinh(V_F / H) - dx
    r2 = (np.hypot(H, V_F) - H) / w + V_F * (L - L_s) / (2.0 * EA) - dz
    return np.array([r1, r2], dtype=np.float64)


def _touchdown_jacobian(
    unknowns: NDArray[np.float64],
    *,
    L: float,
    w: float,
    EA: float,
    dx: float = 0.0,
    dz: float = 0.0,
) -> NDArray[np.float64]:
    """Analytical Jacobian of :func:`_touchdown_residual` w.r.t. ``(H, L_s)``."""
    H, L_s = float(unknowns[0]), float(unknowns[1])
    V_F = w * (L - L_s)
    rH_F = np.hypot(H, V_F)
    # r1 = L_s + H L/EA + (H/w) asinh(V_F/H) - dx
    # d r1 / dH = L/EA + asinh(V_F/H)/w + (H/w) d/dH asinh(V_F/H)
    #           = L/EA + asinh(V_F/H)/w - V_F / (w * rH_F)
    dr1_dH = L / EA + np.arcsinh(V_F / H) / w - V_F / (w * rH_F)
    # d r1 / dL_s = 1 + (H/w) * d/dL_s asinh(V_F/H)
    # V_F = w(L - L_s) -> dV_F/dL_s = -w, d/dL_s asinh(V_F/H) = (-w/H)/sqrt(1 + (V_F/H)^2) = -w/rH_F
    dr1_dLs = 1.0 + (H / w) * (-w / rH_F)  # = 1 - H/rH_F
    # r2 = (rH_F - H)/w + V_F (L - L_s)/(2 EA) - dz
    # d r2 / dH: d(rH_F)/dH = H/rH_F -> (H/rH_F - 1)/w
    dr2_dH = (H / rH_F - 1.0) / w
    # d r2 / dL_s:
    #   d(V_F)/dL_s = -w -> d(rH_F)/dL_s = (V_F/rH_F)*(-w), giving -V_F/rH_F from the (rH_F - H)/w
    #   and d(V_F (L-L_s))/dL_s = -w (L-L_s) + V_F*(-1) = -w(L-L_s) - V_F = -2 V_F.
    dr2_dLs = -V_F / rH_F + (-2.0 * V_F) / (2.0 * EA)
    return np.array(
        [
            [dr1_dH, dr1_dLs],
            [dr2_dH, dr2_dLs],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


_ResidualFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _solve_system(
    residual: _ResidualFn,
    jacobian: _ResidualFn,
    x0: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool]:
    sol = root(residual, x0, jac=jacobian, method="hybr")
    return np.asarray(sol.x, dtype=np.float64), bool(sol.success)


def solve_catenary(
    *,
    line: CatenaryLine,
    anchor_pos: NDArray[np.floating],
    fairlead_pos: NDArray[np.floating],
    seabed_depth: float | None = None,
) -> CatenarySolution:
    """Solve the elastic catenary equilibrium between ``anchor_pos`` and ``fairlead_pos``.

    Parameters
    ----------
    line
        Cable properties (length, weight/length, EA).
    anchor_pos
        Length-2 ``(x, z)`` of the anchor in m. ``z`` should satisfy
        ``z >= -seabed_depth`` when ``seabed_depth`` is supplied.
    fairlead_pos
        Length-2 ``(x, z)`` of the fairlead in m. Must have
        ``fairlead_pos[0] > anchor_pos[0]``.
    seabed_depth
        If not ``None``, seabed contact at ``z = -seabed_depth`` is
        permitted. Must be positive. Without this the suspended regime
        is forced.

    Returns
    -------
    CatenarySolution

    Raises
    ------
    ValueError
        For bad geometry (fairlead not to the right of anchor; anchor
        below seabed) or if the nonlinear solver fails to converge.

    Notes
    -----
    No root-finding initial-condition sensitivity has been observed for
    offshore-typical parameter ranges (`L / span = 1.05 ... 10`,
    `EA / (w L) = 1e3 ... 1e7`). For extreme edge cases the caller can
    bisect the solve by stepping `L` from a large value down to the
    target.
    """
    # Scipy is imported at module level — the caller benefits from its
    # availability implicitly.
    a = np.asarray(anchor_pos, dtype=np.float64)
    f = np.asarray(fairlead_pos, dtype=np.float64)
    if a.shape != (2,) or f.shape != (2,):
        raise ValueError(
            f"anchor_pos and fairlead_pos must have shape (2,); got {a.shape} and {f.shape}"
        )
    dx = float(f[0] - a[0])
    dz = float(f[1] - a[1])
    if dx <= _MinAcceptableHorizontalSpan:
        raise ValueError(
            f"fairlead must be strictly to the right of the anchor; got dx = {dx}. "
            "Rotate into the line's horizontal frame before calling."
        )

    L = line.length
    w = line.weight_per_length
    EA = line.EA

    allow_touchdown = seabed_depth is not None
    if allow_touchdown:
        assert seabed_depth is not None  # for type checker
        if seabed_depth <= 0.0:
            raise ValueError(f"seabed_depth must be positive if supplied; got {seabed_depth}")
        if a[1] < -seabed_depth - 1.0e-9:
            raise ValueError(f"anchor z = {a[1]} is below the seabed at z = {-seabed_depth}")
        # Touchdown is only meaningful when the anchor is on the seabed.
        on_seabed = abs(float(a[1]) - (-seabed_depth)) <= 1.0e-6

        if on_seabed:
            # Attempt touchdown first. Initial guess: inextensible parabolic
            # approximation truncated to (0, L).
            L_s_0 = L - float(np.hypot(dx, dz)) * 0.8
            L_s_0 = float(np.clip(L_s_0, 0.05 * L, 0.95 * L))
            H_0 = max(w * dx / 2.0, 1.0)
            x0 = np.array([H_0, L_s_0], dtype=np.float64)

            def _td_residual(u: NDArray[np.float64]) -> NDArray[np.float64]:
                return _touchdown_residual(u, L=L, w=w, EA=EA, dx=dx, dz=dz)

            def _td_jacobian(u: NDArray[np.float64]) -> NDArray[np.float64]:
                return _touchdown_jacobian(u, L=L, w=w, EA=EA, dx=dx, dz=dz)

            x, ok = _solve_system(_td_residual, _td_jacobian, x0)
            if ok and 0.0 < x[1] < L and x[0] > 0.0:
                H, L_s = float(x[0]), float(x[1])
                V_F = w * (L - L_s)
                return CatenarySolution(
                    regime="touchdown",
                    H=H,
                    V_fairlead=V_F,
                    V_anchor=0.0,
                    touchdown_length=L_s,
                    touchdown_x=float(a[0]) + L_s * (1.0 + H / EA),
                    top_angle_rad=float(np.arctan2(V_F, H)),
                    bottom_angle_rad=0.0,
                )
            # Fall through to suspended attempt.

    # Fully-suspended solve.
    H_0 = max(w * dx / 2.0, 1.0)
    V_A_0 = max(w * L / 4.0, 1.0)
    x0 = np.array([H_0, V_A_0], dtype=np.float64)

    def _sus_residual(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _suspended_residual(u, L=L, w=w, EA=EA, dx=dx, dz=dz)

    def _sus_jacobian(u: NDArray[np.float64]) -> NDArray[np.float64]:
        return _suspended_jacobian(u, L=L, w=w, EA=EA, dx=dx, dz=dz)

    x, ok = _solve_system(_sus_residual, _sus_jacobian, x0)
    if not ok:
        raise RuntimeError(
            f"catenary solver failed to converge (suspended regime): "
            f"initial guess H={H_0:.3e}, V_A={V_A_0:.3e}"
        )
    H, V_A = float(x[0]), float(x[1])
    if H <= 0.0:
        raise RuntimeError(
            f"catenary solver returned non-physical H = {H:.3e}; "
            "geometry may require touchdown but seabed_depth was not supplied"
        )
    V_F = V_A + w * L
    return CatenarySolution(
        regime="suspended",
        H=H,
        V_fairlead=V_F,
        V_anchor=V_A,
        touchdown_length=0.0,
        touchdown_x=float("nan"),
        top_angle_rad=float(np.arctan2(V_F, H)),
        bottom_angle_rad=float(np.arctan2(V_A, H)),
    )


# ---------------------------------------------------------------------------
# 6-DOF state-force composer (M7-Foundation PR3 / F3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatenaryAttachment:
    """One mooring catenary line attached to a body at one end and earth at the other.

    Locked scope at M7-Foundation PR3 (per
    ``docs/m7-foundation-plan.md`` Q4): body-to-earth catenaries
    only. The line hangs in the vertical plane containing the
    inertial-frame fairlead and the inertial-frame anchor; no
    current, no lateral force on the line. Body-to-body catenaries
    are deferred (the inertial-frame anchor side moves with the
    second body, and the geometry / seabed-contact logic gets
    non-trivial).

    Attributes
    ----------
    body_index
        Index of the body whose fairlead the line attaches to.
        Must be ``>= 0`` (no body-to-body, no earth-to-earth).
    fairlead_body
        Length-3 body-frame position of the fairlead, relative to
        the body reference point, in metres.
    anchor_global
        Length-3 inertial-frame position of the anchor in metres,
        with ``z`` typically equal to ``-seabed_depth`` (anchor on
        the seabed).
    line
        Cable physical properties (length, submerged weight per
        unit length, axial stiffness).
    seabed_depth
        Positive depth of the flat horizontal seabed below SWL in
        metres; identical to the ``seabed_depth`` parameter of
        :func:`solve_catenary`.
    """

    body_index: int
    fairlead_body: NDArray[np.float64]
    anchor_global: NDArray[np.float64]
    line: CatenaryLine
    seabed_depth: float

    def __post_init__(self) -> None:
        if self.body_index < 0:
            raise ValueError(
                f"body_index must be >= 0 (body-to-earth only at M7-Foundation PR3); "
                f"got {self.body_index}. Body-to-body catenaries are out of scope -- see "
                f"docs/m7-foundation-plan.md Q4."
            )
        if self.fairlead_body.shape != (3,):
            raise ValueError(
                f"fairlead_body must have shape (3,); got {self.fairlead_body.shape}"
            )
        if self.anchor_global.shape != (3,):
            raise ValueError(
                f"anchor_global must have shape (3,); got {self.anchor_global.shape}"
            )
        if not (np.isfinite(self.seabed_depth) and self.seabed_depth > 0.0):
            raise ValueError(
                f"seabed_depth must be finite and positive; got {self.seabed_depth}"
            )


def _skew_3(r: NDArray[np.floating]) -> NDArray[np.float64]:
    """3x3 skew-symmetric cross-product matrix of ``r``. ``_skew_3(r) @ x == r x x``."""
    r3 = np.asarray(r, dtype=np.float64)
    rx, ry, rz = float(r3[0]), float(r3[1]), float(r3[2])
    return np.array(
        [[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]], dtype=np.float64
    )


def make_catenary_state_force(
    attachments: Sequence[CatenaryAttachment],
    n_dof: int,
) -> Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Build the ``(t, xi, xi_dot) -> F`` closure consumed by
    :func:`floatsim.solver.newmark.integrate_cummins`.

    Mirrors :func:`floatsim.bodies.connector.make_connector_state_force`
    in shape and lag treatment: the integrator evaluates the
    returned closure at the **previous step's state**
    ``(t_{n-1}, xi_{n-1}, xi_dot_{n-1})``, identical to the
    explicit-mu convention of the convolution sum
    (`floatsim/solver/newmark.py` "State-dependent force"
    docstring). The returned force enters the RHS of the
    Cummins step at the same explicit one-step lag as connector
    forces.

    Geometry per call (per attachment):

    1. Read the body's 6-DOF position from ``xi[6 * body_index :
       6 * body_index + 6]``.
    2. Compute the inertial-frame fairlead position via small-
       angle linear rotation:

           r_fairlead_inertial = body_ref_position + r_arm
           r_arm = fairlead_body + theta x fairlead_body
           (where ``theta = xi[6k+3 : 6k+6]``)

       Reduces exactly to ``r_arm = fairlead_body`` at ``theta = 0``,
       matching the M6 PR5 hand-wired path.
    3. Project ``(anchor_global - r_fairlead_inertial)`` onto the
       horizontal plane to get the catenary's local 2D frame.
    4. Call :func:`solve_catenary` in that 2D frame.
    5. Map ``(H, V_fairlead)`` back to a 3D force at the fairlead in
       the inertial frame: ``H`` along the unit horizontal vector
       toward the anchor, ``-V_fairlead`` in z (V_fairlead is
       positive-downward in :class:`CatenarySolution`'s convention).
    6. Translate to a 6-DOF generalised force on the body reference:
       ``F_translation = F_fairlead_inertial``;
       ``F_moment = r_arm x F_translation``.

    Parameters
    ----------
    attachments
        Sequence of :class:`CatenaryAttachment` instances. Each is
        body-to-earth (body_index >= 0); body-to-body raises at
        construction. All ``body_index`` values must satisfy
        ``0 <= body_index < n_dof // 6``.
    n_dof
        Global DOF count ``6 * N`` for the system being integrated.

    Returns
    -------
    Callable
        ``state_force(t, xi, xi_dot)`` returning a length-``n_dof``
        force vector. ``t`` and ``xi_dot`` are accepted for
        signature compatibility with the integrator but unused
        (catenary forces are quasi-static at PR3's locked scope).

    Raises
    ------
    ValueError
        If ``n_dof`` is not a positive multiple of 6, any
        attachment's ``body_index`` is outside ``[0, n_dof // 6)``,
        or ``solve_catenary`` cannot find a solution at runtime
        (degenerate geometry, vertical line, etc.).
    """
    if n_dof <= 0 or n_dof % 6 != 0:
        raise ValueError(f"n_dof must be a positive multiple of 6; got {n_dof}")
    n_bodies = n_dof // 6
    for k, a in enumerate(attachments):
        if not (0 <= a.body_index < n_bodies):
            raise ValueError(
                f"attachment {k}: body_index {a.body_index} outside valid range "
                f"[0, {n_bodies}) for n_dof = {n_dof}"
            )

    att_list = list(attachments)

    def _state_force(
        _t: float,
        xi: NDArray[np.float64],
        _xi_dot: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        F_global = np.zeros(n_dof, dtype=np.float64)
        for a in att_list:
            slc = slice(6 * a.body_index, 6 * a.body_index + 6)
            xi_body = xi[slc]
            theta = xi_body[3:6]
            # Small-angle linear rotated arm: r_arm = fairlead_body + theta x fairlead_body.
            r_arm = a.fairlead_body + _skew_3(theta) @ a.fairlead_body
            r_fairlead_inertial = xi_body[0:3] + r_arm

            # 3D vector from fairlead to anchor; horizontal projection.
            dxy = a.anchor_global[:2] - r_fairlead_inertial[:2]
            horizontal_span = float(np.hypot(dxy[0], dxy[1]))
            if horizontal_span < _MinAcceptableHorizontalSpan:
                raise ValueError(
                    f"catenary attachment with body_index = {a.body_index}: "
                    f"degenerate horizontal span {horizontal_span:.3e} m at this "
                    "body pose; line is effectively vertical. Adjust geometry or "
                    "exclude the line at this pose."
                )
            azimuth_rad = float(np.arctan2(dxy[1], dxy[0]))

            anchor_2d = np.array([0.0, float(a.anchor_global[2])], dtype=np.float64)
            fairlead_2d = np.array(
                [horizontal_span, float(r_fairlead_inertial[2])], dtype=np.float64
            )
            sol = solve_catenary(
                line=a.line,
                anchor_pos=anchor_2d,
                fairlead_pos=fairlead_2d,
                seabed_depth=a.seabed_depth,
            )

            # 3D force at fairlead, inertial frame.
            cos_az = float(np.cos(azimuth_rad))
            sin_az = float(np.sin(azimuth_rad))
            F_fairlead = np.array(
                [sol.H * cos_az, sol.H * sin_az, -sol.V_fairlead], dtype=np.float64
            )

            # Generalised force on body reference (small-angle moment arm).
            F_6 = np.zeros(6, dtype=np.float64)
            F_6[:3] = F_fairlead
            F_6[3:] = np.cross(r_arm, F_fairlead)
            F_global[slc] += F_6

        return F_global

    return _state_force
