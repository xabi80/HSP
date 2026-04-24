"""Milestone 4 gate 2 — elastic catenary closed-form vs shooting method.

Per ``docs/milestone-4-plan.md`` Q2 and ``docs/catenary.md`` §8, the
analytical Irvine catenary solver in ``floatsim.mooring.catenary_analytic``
is cross-validated against an independent shooting method that integrates
the governing ODEs

.. math::

    \\frac{dx}{ds} = \\frac{H}{T(s)} + \\frac{H}{EA},
    \\qquad
    \\frac{dz}{ds} = \\frac{V(s)}{T(s)} + \\frac{V(s)}{EA},

with ``V(s) = V_A + w s`` and ``T(s) = sqrt(H^2 + V(s)^2)`` from ``s = 0``
at the anchor to ``s = L``. The shooting-method root solve finds
``(H, V_A)`` such that ``(x(L), z(L)) = (x_F, z_F)``; it shares no code
with the closed-form solver.

The closed-form ``H`` and ``V_F`` must agree with the shooting-method
values to ``rtol = 1e-4``. We exercise two fixtures:

1. **Suspended** — anchor above the seabed, moderate slack, taut enough
   that ``V_A > 0``.
2. **Touchdown** (Q2 benchmark in the plan) — anchor on the seabed at
   ``z = -200 m``, long slack line so part of it rests. The shooting
   method integrates only the suspended portion ``s \\in [L_s, L]`` with
   ``V = 0`` at ``s = L_s`` (starting at ``(x_TD, -h)``); ``L_s`` itself
   is taken from the closed-form solution and we verify the endpoint
   reaches ``(x_F, z_F)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import root

from floatsim.mooring.catenary_analytic import (
    CatenaryLine,
    CatenarySolution,
    solve_catenary,
)

pytestmark = pytest.mark.slow

_RTOL_H = 1.0e-4
_RTOL_V = 1.0e-4


# ---------------------------------------------------------------------------
# Shooting-method machinery (independent of the closed-form solver)
# ---------------------------------------------------------------------------


def _integrate_catenary_ode(
    H: float, V_A: float, L: float, w: float, EA: float, s_span: tuple[float, float]
) -> tuple[float, float]:
    """Integrate (dx/ds, dz/ds) from ``s_span[0]`` to ``s_span[1]`` starting
    from ``(x, z) = (0, 0)``.

    Returns the endpoint ``(x(s_span[1]), z(s_span[1]))``. ``V(s) = V_A + w (s - s_span[0])``
    is the vertical tension along the line.
    """
    s0 = s_span[0]

    def rhs(s: float, y: np.ndarray) -> np.ndarray:
        V = V_A + w * (s - s0)
        T = float(np.hypot(H, V))
        return np.array([H / T + H / EA, V / T + V / EA])

    out = solve_ivp(
        rhs,
        s_span,
        np.array([0.0, 0.0]),
        method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
        dense_output=False,
    )
    if not out.success:
        raise RuntimeError(f"catenary ODE integration failed: {out.message}")
    return float(out.y[0, -1]), float(out.y[1, -1])


def _shoot_suspended(
    L: float, w: float, EA: float, dx: float, dz: float, x0: np.ndarray
) -> tuple[float, float]:
    """Find (H, V_A) so that integrating from s=0 with (V_A) to s=L
    lands on (dx, dz), using scipy root + ODE integration."""

    def residual(u: np.ndarray) -> np.ndarray:
        H, V_A = float(u[0]), float(u[1])
        x_end, z_end = _integrate_catenary_ode(H, V_A, L, w, EA, (0.0, L))
        return np.array([x_end - dx, z_end - dz])

    sol = root(residual, x0, method="hybr", options={"xtol": 1.0e-12})
    if not sol.success:
        raise RuntimeError(f"shooting method failed: {sol.message}")
    return float(sol.x[0]), float(sol.x[1])


def _shoot_touchdown(
    L: float,
    w: float,
    EA: float,
    dx: float,
    dz: float,
    x0: np.ndarray,
) -> tuple[float, float]:
    """Find ``(H, L_s)`` by shooting the suspended portion ``s in [L_s, L]``
    starting at the touchdown point ``(x_TD, -h)`` with ``V_A = 0``, and
    matching the fairlead endpoint.

    The touchdown point is taken at ``x_TD = L_s (1 + H/EA)`` in a frame
    where the anchor is at ``(0, -h)``. The resting portion is trivial
    (uniform stretch), so only the suspended ODE needs to be integrated.
    Target: ``(x_TD + x_end_rel, z_end_rel) = (dx, dz)`` — matching the
    fairlead horizontal and the vertical rise from the seabed.

    Returns ``(H, L_s)``.
    """

    def residual(u: np.ndarray) -> np.ndarray:
        H, L_s = float(u[0]), float(u[1])
        x_TD = L_s * (1.0 + H / EA)
        x_end_rel, z_end_rel = _integrate_catenary_ode(H, 0.0, L, w, EA, (L_s, L))
        return np.array([x_TD + x_end_rel - dx, z_end_rel - dz])

    sol = root(residual, x0, method="hybr", options={"xtol": 1.0e-12})
    if not sol.success:
        raise RuntimeError(f"touchdown shooting method failed: {sol.message}")
    return float(sol.x[0]), float(sol.x[1])


# ---------------------------------------------------------------------------
# Suspended regime
# ---------------------------------------------------------------------------


class TestSuspendedClosedFormMatchesShooting:
    """Moderately slack line, anchor above seabed, V_A > 0."""

    @pytest.fixture(scope="class")
    def line(self) -> CatenaryLine:
        return CatenaryLine(length=220.0, weight_per_length=100.0, EA=1.0e9)

    @pytest.fixture(scope="class")
    def anchor(self) -> np.ndarray:
        return np.array([0.0, -80.0])

    @pytest.fixture(scope="class")
    def fairlead(self) -> np.ndarray:
        return np.array([180.0, 0.0])

    @pytest.fixture(scope="class")
    def closed_form(
        self, line: CatenaryLine, anchor: np.ndarray, fairlead: np.ndarray
    ) -> CatenarySolution:
        return solve_catenary(line=line, anchor_pos=anchor, fairlead_pos=fairlead)

    @pytest.fixture(scope="class")
    def shooting(
        self,
        line: CatenaryLine,
        anchor: np.ndarray,
        fairlead: np.ndarray,
        closed_form: CatenarySolution,
    ) -> tuple[float, float]:
        dx = float(fairlead[0] - anchor[0])
        dz = float(fairlead[1] - anchor[1])
        # Use the closed-form values as a good initial guess — the test
        # asserts agreement, not convergence from a naive guess.
        x0 = np.array([closed_form.H, closed_form.V_anchor])
        return _shoot_suspended(line.length, line.weight_per_length, line.EA, dx, dz, x0)

    def test_fixture_is_suspended(self, closed_form: CatenarySolution) -> None:
        assert closed_form.regime == "suspended"

    def test_H_matches_shooting(
        self, closed_form: CatenarySolution, shooting: tuple[float, float]
    ) -> None:
        H_shoot = shooting[0]
        assert closed_form.H == pytest.approx(
            H_shoot, rel=_RTOL_H
        ), f"suspended H closed-form={closed_form.H:.6e} vs shooting={H_shoot:.6e}"

    def test_V_fairlead_matches_shooting(
        self,
        closed_form: CatenarySolution,
        shooting: tuple[float, float],
        line: CatenaryLine,
    ) -> None:
        V_F_shoot = shooting[1] + line.weight_per_length * line.length
        assert closed_form.V_fairlead == pytest.approx(V_F_shoot, rel=_RTOL_V), (
            f"V_fairlead closed-form={closed_form.V_fairlead:.6e} vs " f"shooting={V_F_shoot:.6e}"
        )


# ---------------------------------------------------------------------------
# Touchdown regime (Q2 benchmark)
# ---------------------------------------------------------------------------


class TestTouchdownClosedFormMatchesShooting:
    """Q2 benchmark: L=500m, w=1000 N/m, EA=5e8 N, anchor on seabed at -200 m.

    Fairlead at ``(400, 0)`` — deep-water scale, moderately slack line.
    """

    @pytest.fixture(scope="class")
    def line(self) -> CatenaryLine:
        return CatenaryLine(length=500.0, weight_per_length=1000.0, EA=5.0e8)

    @pytest.fixture(scope="class")
    def anchor(self) -> np.ndarray:
        return np.array([0.0, -200.0])

    @pytest.fixture(scope="class")
    def fairlead(self) -> np.ndarray:
        return np.array([400.0, 0.0])

    @pytest.fixture(scope="class")
    def closed_form(
        self, line: CatenaryLine, anchor: np.ndarray, fairlead: np.ndarray
    ) -> CatenarySolution:
        return solve_catenary(
            line=line, anchor_pos=anchor, fairlead_pos=fairlead, seabed_depth=200.0
        )

    def test_fixture_is_touchdown(self, closed_form: CatenarySolution) -> None:
        assert closed_form.regime == "touchdown"
        assert closed_form.V_anchor == 0.0
        assert 0.0 < closed_form.touchdown_length < 500.0

    def test_H_and_L_s_match_shooting(
        self,
        closed_form: CatenarySolution,
        line: CatenaryLine,
        anchor: np.ndarray,
        fairlead: np.ndarray,
    ) -> None:
        dx = float(fairlead[0] - anchor[0])
        dz = float(fairlead[1] - anchor[1])
        x0 = np.array([closed_form.H, closed_form.touchdown_length])
        H_shoot, L_s_shoot = _shoot_touchdown(
            L=line.length,
            w=line.weight_per_length,
            EA=line.EA,
            dx=dx,
            dz=dz,
            x0=x0,
        )
        assert closed_form.H == pytest.approx(
            H_shoot, rel=_RTOL_H
        ), f"touchdown H closed-form={closed_form.H:.6e} vs shooting={H_shoot:.6e}"
        assert closed_form.touchdown_length == pytest.approx(L_s_shoot, rel=_RTOL_H), (
            f"touchdown L_s closed-form={closed_form.touchdown_length:.6e} vs "
            f"shooting={L_s_shoot:.6e}"
        )

    def test_V_fairlead_matches_shooting_via_force_balance(
        self, closed_form: CatenarySolution, line: CatenaryLine
    ) -> None:
        """``V_F = w (L - L_s)`` independently of how ``H`` is found;
        this verifies the closed-form's internal consistency."""
        expected = line.weight_per_length * (line.length - closed_form.touchdown_length)
        assert closed_form.V_fairlead == pytest.approx(expected, rel=1.0e-12)

    def test_shooting_reaches_fairlead_with_closed_form_H(
        self,
        closed_form: CatenarySolution,
        line: CatenaryLine,
        fairlead: np.ndarray,
    ) -> None:
        """Using the closed-form ``(H, L_s)``, integrating the ODE from
        ``(x_TD, -h)`` to ``s = L`` must land on ``(x_F, z_F)`` to within
        ``1e-4 * chord``."""
        x_end_rel, z_end_rel = _integrate_catenary_ode(
            H=closed_form.H,
            V_A=0.0,
            L=line.length,
            w=line.weight_per_length,
            EA=line.EA,
            s_span=(closed_form.touchdown_length, line.length),
        )
        x_reached = closed_form.touchdown_x + x_end_rel
        z_reached = -200.0 + z_end_rel
        chord = float(np.hypot(fairlead[0], fairlead[1] + 200.0))
        assert abs(x_reached - fairlead[0]) < 1.0e-4 * chord, (
            f"x endpoint mismatch: reached {x_reached:.6e} vs " f"fairlead {fairlead[0]:.6e}"
        )
        assert abs(z_reached - fairlead[1]) < 1.0e-4 * chord, (
            f"z endpoint mismatch: reached {z_reached:.6e} vs " f"fairlead {fairlead[1]:.6e}"
        )
