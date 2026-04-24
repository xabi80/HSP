"""Unit tests for the elastic catenary solver.

Covers:

* ``CatenaryLine`` dataclass positivity validation.
* ``CatenarySolution.T_fairlead`` derived property.
* Input validation of ``solve_catenary`` (bad geometry, anchor below seabed,
  non-positive seabed depth).
* Self-consistency: the closed-form solution must zero the governing
  residual (``S1``-``S3`` for suspended, ``T1``-``T3`` for touchdown) to
  ``atol = 1e-6`` m.
* Regime selection: anchor above seabed → suspended; anchor on seabed with
  slack → touchdown; anchor on seabed with a taut line → suspended (solver
  falls back when the touchdown attempt yields ``L_s <= 0``).
* Inextensible limit: with ``EA = 1e20`` the stretch terms in (S1)-(S2)
  vanish and the returned ``(H, V_A)`` matches the classical Irvine
  inextensible form to ``rtol = 1e-6``.
* Relationship invariants: ``V_F = V_A + w L`` (suspended),
  ``V_F = w (L - L_s), V_A = 0`` (touchdown).
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.mooring.catenary_analytic import (
    CatenaryLine,
    CatenarySolution,
    _suspended_residual,
    _touchdown_residual,
    solve_catenary,
)

# ---------------------------------------------------------------------------
# CatenaryLine dataclass
# ---------------------------------------------------------------------------


class TestCatenaryLineValidation:
    def test_accepts_positive_values(self) -> None:
        line = CatenaryLine(length=100.0, weight_per_length=50.0, EA=1.0e8)
        assert line.length == 100.0
        assert line.weight_per_length == 50.0
        assert line.EA == 1.0e8

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_length(self, bad: float) -> None:
        with pytest.raises(ValueError, match="length must be finite and positive"):
            CatenaryLine(length=bad, weight_per_length=50.0, EA=1.0e8)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_weight(self, bad: float) -> None:
        with pytest.raises(ValueError, match="weight_per_length must be finite and positive"):
            CatenaryLine(length=100.0, weight_per_length=bad, EA=1.0e8)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_bad_EA(self, bad: float) -> None:
        with pytest.raises(ValueError, match="EA must be finite and positive"):
            CatenaryLine(length=100.0, weight_per_length=50.0, EA=bad)


# ---------------------------------------------------------------------------
# CatenarySolution derived property
# ---------------------------------------------------------------------------


def test_solution_T_fairlead_is_hypot_of_H_and_VF() -> None:
    sol = CatenarySolution(
        regime="suspended",
        H=3.0,
        V_fairlead=4.0,
        V_anchor=1.0,
        touchdown_length=0.0,
        touchdown_x=float("nan"),
        top_angle_rad=0.9273,
        bottom_angle_rad=0.3217,
    )
    assert sol.T_fairlead == pytest.approx(5.0, rel=1e-12)


# ---------------------------------------------------------------------------
# solve_catenary input validation
# ---------------------------------------------------------------------------


class TestSolveCatenaryInputValidation:
    @pytest.fixture
    def line(self) -> CatenaryLine:
        return CatenaryLine(length=100.0, weight_per_length=50.0, EA=1.0e8)

    def test_rejects_non_2vec_anchor(self, line: CatenaryLine) -> None:
        with pytest.raises(ValueError, match="shape"):
            solve_catenary(
                line=line,
                anchor_pos=np.array([0.0, 0.0, 0.0]),
                fairlead_pos=np.array([10.0, 0.0]),
            )

    def test_rejects_non_2vec_fairlead(self, line: CatenaryLine) -> None:
        with pytest.raises(ValueError, match="shape"):
            solve_catenary(
                line=line,
                anchor_pos=np.array([0.0, 0.0]),
                fairlead_pos=np.array([10.0, 0.0, 0.0]),
            )

    def test_rejects_fairlead_not_right_of_anchor(self, line: CatenaryLine) -> None:
        with pytest.raises(ValueError, match="strictly to the right"):
            solve_catenary(
                line=line,
                anchor_pos=np.array([10.0, 0.0]),
                fairlead_pos=np.array([10.0, 5.0]),
            )

    def test_rejects_negative_seabed_depth(self, line: CatenaryLine) -> None:
        with pytest.raises(ValueError, match="seabed_depth must be positive"):
            solve_catenary(
                line=line,
                anchor_pos=np.array([0.0, 0.0]),
                fairlead_pos=np.array([10.0, 5.0]),
                seabed_depth=-10.0,
            )

    def test_rejects_anchor_below_seabed(self, line: CatenaryLine) -> None:
        with pytest.raises(ValueError, match="anchor z"):
            solve_catenary(
                line=line,
                anchor_pos=np.array([0.0, -150.0]),
                fairlead_pos=np.array([80.0, 0.0]),
                seabed_depth=100.0,
            )


# ---------------------------------------------------------------------------
# Suspended-regime solutions
# ---------------------------------------------------------------------------


class TestSuspendedRegime:
    @pytest.fixture
    def geom(self) -> dict[str, float]:
        return {"L": 250.0, "w": 100.0, "EA": 1.0e9, "dx": 180.0, "dz": 80.0}

    @pytest.fixture
    def line(self, geom: dict[str, float]) -> CatenaryLine:
        return CatenaryLine(length=geom["L"], weight_per_length=geom["w"], EA=geom["EA"])

    def test_returns_suspended_regime(self, line: CatenaryLine) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -80.0]),
            fairlead_pos=np.array([180.0, 0.0]),
        )
        assert sol.regime == "suspended"
        assert sol.touchdown_length == 0.0
        assert np.isnan(sol.touchdown_x)

    def test_V_F_equals_V_A_plus_wL(self, line: CatenaryLine, geom: dict[str, float]) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -80.0]),
            fairlead_pos=np.array([180.0, 0.0]),
        )
        expected_V_F = sol.V_anchor + geom["w"] * geom["L"]
        assert sol.V_fairlead == pytest.approx(expected_V_F, rel=1e-12)

    def test_residual_is_zero_at_solution(self, line: CatenaryLine, geom: dict[str, float]) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -80.0]),
            fairlead_pos=np.array([180.0, 0.0]),
        )
        r = _suspended_residual(
            np.array([sol.H, sol.V_anchor]),
            L=geom["L"],
            w=geom["w"],
            EA=geom["EA"],
            dx=geom["dx"],
            dz=geom["dz"],
        )
        assert np.max(np.abs(r)) < 1.0e-6  # [m]

    def test_top_and_bottom_angles_consistent(self, line: CatenaryLine) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -80.0]),
            fairlead_pos=np.array([180.0, 0.0]),
        )
        assert sol.top_angle_rad == pytest.approx(np.arctan2(sol.V_fairlead, sol.H), rel=1e-12)
        assert sol.bottom_angle_rad == pytest.approx(np.arctan2(sol.V_anchor, sol.H), rel=1e-12)

    def test_H_is_positive(self, line: CatenaryLine) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -80.0]),
            fairlead_pos=np.array([180.0, 0.0]),
        )
        assert sol.H > 0.0

    def test_anchor_above_seabed_does_not_touchdown(self, line: CatenaryLine) -> None:
        """Anchor well above the seabed stays fully suspended even when seabed_depth is given."""
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -80.0]),
            fairlead_pos=np.array([180.0, 0.0]),
            seabed_depth=200.0,
        )
        assert sol.regime == "suspended"


# ---------------------------------------------------------------------------
# Touchdown regime
# ---------------------------------------------------------------------------


class TestTouchdownRegime:
    @pytest.fixture
    def geom(self) -> dict[str, float]:
        return {"L": 500.0, "w": 1000.0, "EA": 5.0e8, "dx": 400.0, "dz": 200.0}

    @pytest.fixture
    def line(self, geom: dict[str, float]) -> CatenaryLine:
        return CatenaryLine(length=geom["L"], weight_per_length=geom["w"], EA=geom["EA"])

    def test_returns_touchdown_regime(self, line: CatenaryLine) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -200.0]),
            fairlead_pos=np.array([400.0, 0.0]),
            seabed_depth=200.0,
        )
        assert sol.regime == "touchdown"
        assert 0.0 < sol.touchdown_length < line.length

    def test_V_anchor_is_zero(self, line: CatenaryLine) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -200.0]),
            fairlead_pos=np.array([400.0, 0.0]),
            seabed_depth=200.0,
        )
        assert sol.V_anchor == 0.0
        assert sol.bottom_angle_rad == 0.0

    def test_V_F_equals_w_times_suspended_length(
        self, line: CatenaryLine, geom: dict[str, float]
    ) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -200.0]),
            fairlead_pos=np.array([400.0, 0.0]),
            seabed_depth=200.0,
        )
        expected_V_F = geom["w"] * (geom["L"] - sol.touchdown_length)
        assert sol.V_fairlead == pytest.approx(expected_V_F, rel=1e-12)

    def test_residual_is_zero_at_solution(self, line: CatenaryLine, geom: dict[str, float]) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -200.0]),
            fairlead_pos=np.array([400.0, 0.0]),
            seabed_depth=200.0,
        )
        r = _touchdown_residual(
            np.array([sol.H, sol.touchdown_length]),
            L=geom["L"],
            w=geom["w"],
            EA=geom["EA"],
            dx=geom["dx"],
            dz=geom["dz"],
        )
        assert np.max(np.abs(r)) < 1.0e-6

    def test_touchdown_x_within_span(self, line: CatenaryLine) -> None:
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -200.0]),
            fairlead_pos=np.array([400.0, 0.0]),
            seabed_depth=200.0,
        )
        # x_TD = x_A + L_s (1 + H/EA); must be between anchor and fairlead.
        assert 0.0 < sol.touchdown_x < 400.0
        expected = 0.0 + sol.touchdown_length * (1.0 + sol.H / line.EA)
        assert sol.touchdown_x == pytest.approx(expected, rel=1e-12)

    def test_taut_line_on_seabed_falls_back_to_suspended(self) -> None:
        """Anchor on seabed but line nearly chord-length → touchdown attempt
        yields L_s <= 0, so the solver must fall back to fully-suspended."""
        # Chord length sqrt(400^2 + 200^2) ~ 447.2 m. With L = 448 there is
        # less than 1 m of slack — no part of the line should rest on the
        # seabed.
        line = CatenaryLine(length=448.0, weight_per_length=1000.0, EA=5.0e8)
        sol = solve_catenary(
            line=line,
            anchor_pos=np.array([0.0, -200.0]),
            fairlead_pos=np.array([400.0, 0.0]),
            seabed_depth=200.0,
        )
        assert sol.regime == "suspended"


# ---------------------------------------------------------------------------
# Inextensible limit
# ---------------------------------------------------------------------------


def test_inextensible_limit_matches_reference_stretch_free() -> None:
    """``EA → infty`` drops the stretch terms; the solution must equal the
    classical Irvine inextensible catenary.

    The equations reduce (suspended) to::

        dx = (H/w) [asinh(V_F/H) - asinh(V_A/H)]
        dz = (1/w) [sqrt(H^2 + V_F^2) - sqrt(H^2 + V_A^2)]
        V_F = V_A + w L

    We compare an ``EA = 1.0e20`` run against an independent solve of these
    inextensible equations.
    """
    from scipy.optimize import root

    L, w = 250.0, 100.0
    dx, dz = 180.0, 80.0

    def inextensible_residual(u: np.ndarray) -> np.ndarray:
        H, V_A = float(u[0]), float(u[1])
        V_F = V_A + w * L
        return np.array(
            [
                (H / w) * (np.arcsinh(V_F / H) - np.arcsinh(V_A / H)) - dx,
                (np.hypot(H, V_F) - np.hypot(H, V_A)) / w - dz,
            ]
        )

    ref = root(inextensible_residual, x0=np.array([w * dx / 2.0, w * L / 4.0]))
    assert ref.success, "reference inextensible solve failed"
    H_ref, V_A_ref = float(ref.x[0]), float(ref.x[1])

    line = CatenaryLine(length=L, weight_per_length=w, EA=1.0e20)
    sol = solve_catenary(
        line=line,
        anchor_pos=np.array([0.0, -80.0]),
        fairlead_pos=np.array([180.0, 0.0]),
    )
    assert sol.H == pytest.approx(H_ref, rel=1.0e-6)
    assert sol.V_anchor == pytest.approx(V_A_ref, rel=1.0e-6)


# ---------------------------------------------------------------------------
# Stretch grows with decreasing EA
# ---------------------------------------------------------------------------


def test_soft_line_pulls_more_horizontal_tension_than_stiff_line() -> None:
    """Monotonicity sanity check: at fixed geometry, softer ``EA`` →
    different ``H`` than stiff. (The sign of the change depends on
    geometry; we just assert they are not equal to within 1%.)"""
    geom_args = {
        "anchor_pos": np.array([0.0, -80.0]),
        "fairlead_pos": np.array([180.0, 0.0]),
    }
    stiff = solve_catenary(
        line=CatenaryLine(length=250.0, weight_per_length=100.0, EA=1.0e12),
        **geom_args,  # type: ignore[arg-type]
    )
    soft = solve_catenary(
        line=CatenaryLine(length=250.0, weight_per_length=100.0, EA=1.0e6),
        **geom_args,  # type: ignore[arg-type]
    )
    assert abs(stiff.H - soft.H) / stiff.H > 1.0e-2
