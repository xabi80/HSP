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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

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
