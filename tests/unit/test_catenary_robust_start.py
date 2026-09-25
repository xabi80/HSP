"""Flume-mooring Phase C1: the catenary solve must not crash mid-run on a solvable geometry.

``solve_catenary`` ran one ``scipy.root(hybr)`` from a fixed cold guess (H = V_A = 1 N). On
very elastic, nearly weightless lines (the flume springs: EA ~ 55 N against ~16 N of
tension) that guess can miss the basin. The moored flume platform at T = 3.5 s stopped at
t = 25.7 s on a line whose mirror image -- geometry equal to 1e-10 m -- solved at 16.1 N.

The cold start stays the FIRST attempt (so every geometry that converged before returns
bit-identical results). Only where it fails does the solve fall back to (1) a caller-supplied
initial guess -- the state force passes the line's previous converged solution (warm start)
-- then (2) a taut-elastic estimate from the chord.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.mooring.catenary_analytic import (
    CatenaryAttachment,
    CatenaryLine,
    make_catenary_state_force,
    solve_catenary,
)

# The failing flume line (buoy7 at t = 25.7 s) and its converging mirror image.
_LINE = CatenaryLine(length=3.3092099787062574, weight_per_length=0.02, EA=55.05312059213587)
_FAIL = (4.277615496851316, 0.028975430906064026)
_OK = (4.2776154967519835, 0.028975430883236175)


def _solve(span_dz, **kw):  # type: ignore[no-untyped-def]
    span, dz = span_dz
    return solve_catenary(
        line=_LINE,
        anchor_pos=np.array([0.0, 0.0]),
        fairlead_pos=np.array([span, -dz]),
        seabed_depth=200.0,
        **kw,
    )


def _residual(sol, span_dz) -> float:  # type: ignore[no-untyped-def]
    """Re-derive the geometry from the returned (H, V_A) with the elastic-catenary closed
    form (Irvine S1-S2) and return the worst position error."""
    span, dz = span_dz
    L, w, EA = _LINE.length, _LINE.weight_per_length, _LINE.EA
    H, VA = sol.H, sol.V_anchor
    VF = VA + w * L
    x = H / w * (np.arcsinh(VF / H) - np.arcsinh(VA / H)) + H * L / EA
    z = H / w * (np.sqrt(1 + (VF / H) ** 2) - np.sqrt(1 + (VA / H) ** 2))
    z += (VA * L + 0.5 * w * L**2) / EA
    return float(max(abs(x - span), abs(z - (-dz))))


def test_previously_failing_geometry_now_solves() -> None:
    sol = _solve(_FAIL)
    assert sol.regime == "suspended"
    assert _residual(sol, _FAIL) < 1e-9
    ref = _solve(_OK)
    assert sol.H == pytest.approx(ref.H, rel=1e-7)
    assert sol.V_fairlead == pytest.approx(ref.V_fairlead, rel=1e-6, abs=1e-9)


def test_warm_start_guess_is_used_and_converges() -> None:
    ref = _solve(_OK)
    sol = _solve(_FAIL, initial_guess=(ref.H, ref.V_anchor))
    assert _residual(sol, _FAIL) < 1e-9
    assert sol.H == pytest.approx(ref.H, rel=1e-7)


def test_cold_start_first_keeps_converging_cases_bit_identical() -> None:
    """Where the cold start converges, a (different) initial guess must not change the answer
    at all: the cold attempt runs first and wins."""
    ref = _solve(_OK)
    guided = _solve(_OK, initial_guess=(3.0 * ref.H, -5.0))
    assert (guided.H, guided.V_fairlead, guided.V_anchor) == (ref.H, ref.V_fairlead, ref.V_anchor)


def test_state_force_runs_through_the_failing_geometry() -> None:
    """Sweep the fairlead across the failing span in 1e-11 m steps: no exception, and the
    force is continuous (the warm start carries each step's solution to the next)."""
    span0, dz = _FAIL
    att = CatenaryAttachment(
        body_index=0,
        fairlead_body=np.zeros(3),
        anchor_global=np.array([0.0, 0.0, 0.0]),
        line=_LINE,
        seabed_depth=200.0,
    )
    f = make_catenary_state_force([att], n_dof=6)
    xi = np.zeros(6)
    xi[2] = -dz
    forces = []
    for d in np.linspace(-5e-10, 5e-10, 101):
        xi[0] = span0 + d
        forces.append(f(0.0, xi, np.zeros(6))[0])
    forces = np.asarray(forces)
    assert np.all(np.isfinite(forces))
    assert np.max(np.abs(np.diff(forces))) < 1e-6


# A very slack flume line (0.3 N/m in air, span about half its length): the cold start
# converged onto the mirror root H = -0.137 N and the solve raised "non-physical H".
_SLACK = CatenaryLine(length=3.831113529965571, weight_per_length=0.3, EA=7.696837288644187)
_SLACK_FAIRLEAD = (1.8884367442035492, 0.7048089507608726)


def test_cold_start_on_the_negative_h_root_falls_back_to_the_physical_one() -> None:
    sol = solve_catenary(
        line=_SLACK,
        anchor_pos=np.array([0.0, 0.717]),
        fairlead_pos=np.array(_SLACK_FAIRLEAD),
        seabed_depth=200.0,
    )
    L, w, EA = _SLACK.length, _SLACK.weight_per_length, _SLACK.EA
    H, VA = sol.H, sol.V_anchor
    assert H > 0.0
    VF = VA + w * L
    x = H / w * (np.arcsinh(VF / H) - np.arcsinh(VA / H)) + H * L / EA
    z = (np.hypot(H, VF) - np.hypot(H, VA)) / w + (VA * L + 0.5 * w * L**2) / EA
    assert x == pytest.approx(_SLACK_FAIRLEAD[0], abs=1e-9)
    assert z == pytest.approx(_SLACK_FAIRLEAD[1] - 0.717, abs=1e-9)
    assert H == pytest.approx(0.12138598, rel=1e-6)


def test_extreme_stretch_geometry_is_a_true_solution() -> None:
    """The elastic catenary reaches any span given enough tension. A 0.5 m line with
    EA = 1e20 spanning 100 m (strain 199) defeated the cold start; the fallback must return
    the genuine elastic solution, verified against the closed form, not an artefact."""
    short = CatenaryLine(length=0.5, weight_per_length=0.02, EA=1.0e20)
    sol = solve_catenary(
        line=short,
        anchor_pos=np.array([0.0, 0.0]),
        fairlead_pos=np.array([100.0, 0.0]),
        seabed_depth=200.0,
    )
    L, w, EA = short.length, short.weight_per_length, short.EA
    H, VA = sol.H, sol.V_anchor
    VF = VA + w * L
    x = H / w * (np.arcsinh(VF / H) - np.arcsinh(VA / H)) + H * L / EA
    assert x == pytest.approx(100.0, rel=1e-9)
    assert H / EA == pytest.approx(199.0, rel=1e-6)


def test_all_starts_failing_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every starting point fails, the solve raises and names each start it tried
    (cold, warm, taut-elastic): the fallbacks rescue the starting point, they never return
    an unconverged answer."""
    import floatsim.mooring.catenary_analytic as ca

    monkeypatch.setattr(ca, "_solve_system", lambda r, j, x0: (np.asarray(x0), False))
    with pytest.raises(RuntimeError, match="from every starting point") as exc:
        _solve(_OK, initial_guess=(16.0, -0.03))
    assert str(exc.value).count("H=") == 3
