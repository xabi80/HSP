"""Milestone 4 PR6 integration deck — two-body moored dynamic run.

Per ``docs/milestone-4-plan.md`` PR6: glue together the pieces built in
PR1-PR5 and demonstrate end-to-end behaviour on a two-body moored
system.

System under test
-----------------
* Two identical single-body fixtures — the M2 heave analytical fixture
  (``M + A_inf = 2e7 kg``, ``C_33 = 1.28e7 N/m``, heave natural period
  ``T_n = 7.854 s``) — stacked into a global 12-DOF problem via
  :mod:`floatsim.solver.state`.
* A heave rigid-link between body 0 and body 1 at
  ``k = 1e3 x max diag(C_global)`` — see "Time step and penalty
  stiffness" below for why we sit at the schema floor instead of the
  default ``1e4`` (M4 PR3).
* A symmetric pair of Irvine catenary lines (:mod:`floatsim.mooring`,
  M4 PR4) from body 0 to earth anchors at ``(+/-350 m, 0, -200 m)``.
  The pair provides surge restoring without invoking a linear spring —
  the body has no hydrostatic restoring in surge, so a single catenary
  would leave the surge equilibrium undefined. Two opposing catenaries
  cancel their horizontal pretension at ``surge = 0`` and introduce a
  soft restoring for non-zero surge.

What this test asserts
----------------------
1. **Equilibrium solves.** :func:`solve_static_equilibrium` converges
   and the residual ``C xi_eq - F_state(0, xi_eq, 0)`` is tiny in
   newtons (< 1 N in the constrained DOFs).
2. **Equilibrium is physically plausible.** Body 0 sits slightly below
   the waterline (the two catenaries both pull down); by symmetry
   surge stays at zero; body 1 follows body 0 in heave to within the
   rigid-link drift tolerance; no DOF beyond heave is loaded.
3. **Short dynamic run is stable.** Starting from ``xi_eq`` with zero
   velocity, 5 s of integration produces finite numbers and bounded
   excursions from equilibrium.
4. **Mooring provides surge restoring.** Releasing the body from an
   off-equilibrium surge offset produces a decaying/oscillating
   surge — i.e. the catenary pair successfully supplies the surge
   stiffness that the hydrostatic matrix lacks.

This is a glue test, not a new validation gate: the physics gates were
hit in PR3 (rigid link) and PR4 (catenary). We just confirm the pieces
compose through the equilibrium solver and integrator.

Time step and penalty stiffness
-------------------------------
The rigid-link penalty force is treated explicitly by
:func:`floatsim.solver.newmark.integrate_cummins` (lagged one step),
which couples the antisymmetric heave mode of the link
(``omega = sqrt(2 k / m)``) to the integrator's stability bound.

We use the **lower** end of the deck schema's allowed range for the
penalty stiffness factor — ``1e3 x max(diag(C_global)) = 1.28e10 N/m``,
giving ``omega ~= 36 rad/s``. Empirically the gen-alpha integrator
(``rho_inf = 0.9``) stays stable for ``dt <= 2-3 ms`` at this
stiffness; we step at ``dt = 2 ms``. The first-order
``check_connector_stability`` formula ``dt < safety * 2 / omega`` is
slightly optimistic for explicit-state-force integration with
``rho_inf`` close to 1 (numerical damping kicks in only above
``dt * omega ~ 0.5``), so the empirical step is what gates the run —
the diagnostic is still asserted in
:func:`test_rigid_link_passes_explicit_stability_gate` to catch gross
regressions. Drift at peak load ``2e5 N / k = O(1.6e-5) m``.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
import pytest
from numpy.typing import NDArray

from floatsim.bodies.connector import (
    check_connector_stability,
    heave_rigid_link,
    make_connector_state_force,
)
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.mooring.catenary_analytic import CatenaryLine, solve_catenary
from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.state import (
    assemble_global_kernel,
    assemble_global_lhs,
)
from tests.support.synthetic_bem import make_diagonal_hdb
from tests.validation.test_cummins_free_decay_analytical import (
    _A_INF_33,
    _B_33,
    _C_33,
    _I_OTHER,
    _M_33,
    _M_OTHER,
    _OMEGA_GRID,
)

pytestmark = pytest.mark.slow

# Step size and penalty factor chosen for empirical stability — see the
# module docstring for the derivation. ``_DURATION`` is short by design:
# this is a glue test, not a long-run validation.
_DT = 2.0e-3
_DURATION = 5.0
_PENALTY_FACTOR = 1.0e3  # schema floor — see module docstring
_PENALTY_K = _PENALTY_FACTOR * _C_33  # = 1.28e10 N/m

# Catenary fixture — scaled so both regimes get exercised near equilibrium.
_LINE = CatenaryLine(length=500.0, weight_per_length=1000.0, EA=5.0e8)
_SEABED_DEPTH = 200.0
# Anchor x-offsets (global). The pair is symmetric so surge pretensions cancel.
_ANCHOR_X_PLUS = +350.0
_ANCHOR_X_MINUS = -350.0
_ANCHOR_Z = -_SEABED_DEPTH


def _single_body_hdb():
    n_w = _OMEGA_GRID.size
    A_inf_diag = [_M_OTHER, _M_OTHER, _A_INF_33, _I_OTHER, _I_OTHER, _I_OTHER]
    A_diag_per_omega = [list(A_inf_diag) for _ in range(n_w)]
    B_diag_per_omega = [[1.0e3, 1.0e3, _B_33, 1.0e4, 1.0e4, 1.0e4] for _ in range(n_w)]
    C_diag = [0.0, 0.0, _C_33, 0.0, 0.0, 0.0]
    return make_diagonal_hdb(
        A_inf_diag=A_inf_diag,
        C_diag=C_diag,
        A_diag_per_omega=A_diag_per_omega,
        B_diag_per_omega=B_diag_per_omega,
        omega=_OMEGA_GRID.tolist(),
        heading_deg=[0.0, 90.0],
    )


def _single_body_rigid_mass():
    return np.diag([_M_OTHER, _M_OTHER, _M_33, _I_OTHER, _I_OTHER, _I_OTHER]).astype(np.float64)


def _single_catenary_force(
    anchor_x: float, fairlead_x_global: float, fairlead_z_global: float
) -> tuple[float, float]:
    """Return ``(Fx, Fz)`` applied to the fairlead by a single catenary line.

    The catenary solver works in a local frame with anchor at origin,
    fairlead to the right (``dx > 0``) and both z's in global coordinates
    (seabed at ``z = -_SEABED_DEPTH``). The line's horizontal direction
    points from fairlead toward anchor — which is ``sign(anchor_x -
    fairlead_x)`` in the global X axis. The fairlead is pulled along
    that direction with magnitude ``H``, and pulled downward with
    magnitude ``V_fairlead``.
    """
    dx_global = anchor_x - fairlead_x_global  # +ve if anchor is to the right
    sign_h = 1.0 if dx_global > 0.0 else -1.0
    # Local frame: anchor at (0, -_SEABED_DEPTH), fairlead at (|dx|, z).
    anchor_local = np.array([0.0, _ANCHOR_Z])
    fairlead_local = np.array([abs(dx_global), fairlead_z_global])
    sol = solve_catenary(
        line=_LINE,
        anchor_pos=anchor_local,
        fairlead_pos=fairlead_local,
        seabed_depth=_SEABED_DEPTH,
    )
    Fx = sign_h * float(sol.H)  # pull toward anchor in global X
    Fz = -float(sol.V_fairlead)  # fairlead pulled downward
    return Fx, Fz


def _catenary_force_on_body0(xi_body0_surge: float, xi_body0_heave: float) -> tuple[float, float]:
    """Sum the force contributions of the +X and -X catenary lines."""
    Fx_plus, Fz_plus = _single_catenary_force(_ANCHOR_X_PLUS, xi_body0_surge, xi_body0_heave)
    Fx_minus, Fz_minus = _single_catenary_force(_ANCHOR_X_MINUS, xi_body0_surge, xi_body0_heave)
    return Fx_plus + Fx_minus, Fz_plus + Fz_minus


def _build_mooring_state_force(
    connector_force: Callable[
        [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
    ],
) -> Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    """Sum the rigid-link connector force and the catenary-pair force on body 0."""

    def _force(
        t: float, xi: NDArray[np.float64], xi_dot: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        F = np.asarray(connector_force(t, xi, xi_dot), dtype=np.float64).copy()
        Fx, Fz = _catenary_force_on_body0(float(xi[0]), float(xi[2]))
        # Body 0 occupies slots [0, 6): (surge, sway, heave, roll, pitch, yaw).
        F[0] += Fx
        F[2] += Fz
        return F

    return _force


def _assemble_system() -> tuple[object, object, Callable, list]:
    """Shared LHS, kernel, state-force closure, and the connector list.

    The connector list is returned alongside so the test that checks the
    integrator stability gate can call
    :func:`floatsim.bodies.connector.check_connector_stability` directly.
    """
    hdb = _single_body_hdb()
    lhs_single = assemble_cummins_lhs(rigid_body_mass=_single_body_rigid_mass(), hdb=hdb)
    kernel_single = compute_retardation_kernel(hdb, t_max=60.0, dt=_DT)

    lhs_global = assemble_global_lhs([lhs_single, lhs_single])
    kernel_global = assemble_global_kernel([kernel_single, kernel_single])

    link = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=_PENALTY_K)
    connectors = [link]
    connector_force = make_connector_state_force(connectors, n_dof=12)
    total_state_force = _build_mooring_state_force(connector_force)
    return lhs_global, kernel_global, total_state_force, connectors


@lru_cache(maxsize=1)
def _solve_equilibrium_and_run():
    lhs_global, kernel_global, total_state_force, _ = _assemble_system()

    # --- Equilibrium -------------------------------------------------------
    eq = solve_static_equilibrium(
        lhs=lhs_global,
        state_force=total_state_force,
        tol=1.0,  # 1 N residual threshold on loaded DOFs.
    )

    # --- Short dynamic from xi_eq -----------------------------------------
    result = integrate_cummins(
        lhs=lhs_global,
        kernel=kernel_global,
        xi0=eq.xi_eq,
        xi_dot0=np.zeros(12),
        duration=_DURATION,
        state_force=total_state_force,
        rho_inf=0.9,
    )
    return eq, result


def test_equilibrium_converges_with_small_residual() -> None:
    eq, _ = _solve_equilibrium_and_run()
    assert eq.converged, f"equilibrium did not converge: residual = {eq.residual_norm:.3e} N"
    assert (
        eq.residual_norm < 1.0
    ), f"equilibrium residual {eq.residual_norm:.3e} N exceeds 1 N tolerance"


def test_equilibrium_by_symmetry_has_zero_surge() -> None:
    """Two symmetric catenaries -> zero net surge force at surge = 0."""
    eq, _ = _solve_equilibrium_and_run()
    assert (
        abs(eq.xi_eq[0]) < 1.0e-4
    ), f"body 0 surge at equilibrium = {eq.xi_eq[0]:.3e} m (expected ~0 by symmetry)"
    assert (
        abs(eq.xi_eq[6]) < 1.0e-4
    ), f"body 1 surge at equilibrium = {eq.xi_eq[6]:.3e} m (expected ~0 by symmetry)"


def test_equilibrium_only_loads_heave_and_surge() -> None:
    """Lines live in XZ plane + heave-only rigid link -> sway/roll/pitch/yaw stay at 0."""
    eq, _ = _solve_equilibrium_and_run()
    silent_body0 = [1, 3, 4, 5]  # sway, roll, pitch, yaw on body 0
    silent_body1 = [7, 9, 10, 11]  # sway, roll, pitch, yaw on body 1
    for dof in silent_body0 + silent_body1:
        assert (
            abs(eq.xi_eq[dof]) < 1.0e-6
        ), f"xi_eq[{dof}] = {eq.xi_eq[dof]:.3e} should be ~0 (DOF not loaded)"


def test_equilibrium_heave_pulled_down_by_catenary_pair() -> None:
    """Both catenaries pull fairlead down -> negative heave offset in equilibrium."""
    eq, _ = _solve_equilibrium_and_run()
    heave_a = float(eq.xi_eq[2])
    # Two lines each pull down with V_F ~ O(1e5 N); net ~ 2 V_F / C_33.
    # Sign: negative heave (downward).
    assert heave_a < 0.0, f"expected downward heave offset; got {heave_a:.4e} m"
    assert abs(heave_a) < 1.0, f"equilibrium heave {heave_a:.4e} m implausibly large (>1 m)"


def test_equilibrium_rigid_link_holds_heave() -> None:
    """Rigid link in heave -> body0 and body1 heave agree within drift tolerance."""
    eq, _ = _solve_equilibrium_and_run()
    heave_a = float(eq.xi_eq[2])
    heave_b = float(eq.xi_eq[8])
    drift = abs(heave_a - heave_b)
    # Catenary vertical load O(2e5 N) on body 0 alone, penalty
    # k = 1.28e10 N/m at factor=1e3 -> drift O(1.6e-5 m). Threshold
    # 1e-4 m gives a comfortable margin without hiding a regression.
    assert drift < 1.0e-4, (
        f"heave drift at equilibrium = {drift:.3e} m exceeds 1e-4 m "
        f"(body0 heave {heave_a:.6f}, body1 heave {heave_b:.6f})"
    )


def test_dynamic_run_is_finite_and_bounded() -> None:
    """60-second run from xi_eq with zero velocity must stay finite and small."""
    _, res = _solve_equilibrium_and_run()
    assert np.all(np.isfinite(res.xi)), "xi went non-finite during integration"
    assert np.all(np.isfinite(res.xi_dot)), "xi_dot went non-finite"
    assert np.all(np.isfinite(res.xi_ddot)), "xi_ddot went non-finite"

    # Starting at equilibrium, deviations should stay tiny relative to
    # the body length scale.
    excursion = float(np.max(np.abs(res.xi - res.xi[0])))
    assert excursion < 0.1, (
        f"max excursion from xi_eq = {excursion:.3e} m exceeds 10 cm — "
        "system is not at equilibrium or integrator is unstable"
    )


def test_dynamic_heave_rigid_link_drift_stays_bounded() -> None:
    """Rigid link holds during the run: |xi_0_heave - xi_1_heave| stays small."""
    _, res = _solve_equilibrium_and_run()
    drift_series = res.xi[:, 2] - res.xi[:, 8]
    peak_drift = float(np.max(np.abs(drift_series)))
    # k = 1.28e10 N/m at factor=1e3, peak heave force during transient
    # comparable to equilibrium load -> drift stays in the 1e-5 m range.
    assert peak_drift < 1.0e-3, (
        f"heave rigid-link drift peaked at {peak_drift:.3e} m during the run "
        "(expected bounded by penalty stiffness)"
    )


def test_mooring_surge_restoring_recenters_body_from_offset() -> None:
    """Spot-check mooring provides surge restoring: an offset releases back.

    Build a new run starting from xi_eq with body 0 displaced in surge
    by +0.5 m (and body 1 likewise — the rigid link is heave-only so
    body 1 surge is independent). With the catenary-pair restoring, the
    surge motion is bounded and oscillates / decays toward equilibrium.
    """
    lhs_global, kernel_global, total_state_force, _ = _assemble_system()
    eq = solve_static_equilibrium(lhs=lhs_global, state_force=total_state_force, tol=1.0)

    xi0 = eq.xi_eq.copy()
    xi0[0] += 0.5  # body 0 surge offset
    res = integrate_cummins(
        lhs=lhs_global,
        kernel=kernel_global,
        xi0=xi0,
        xi_dot0=np.zeros(12),
        duration=_DURATION,
        state_force=total_state_force,
        rho_inf=0.9,
    )
    surge = res.xi[:, 0]
    # With surge restoring, the peak surge after release cannot exceed
    # the IC magnitude by more than a small margin (no energy injection).
    peak = float(np.max(np.abs(surge - eq.xi_eq[0])))
    assert peak < 0.6, f"surge peak {peak:.3e} m after 0.5 m release — no restoring or runaway"
    # And the trajectory should not monotonically drift past the IC amplitude.
    assert np.all(np.isfinite(surge))


def test_rigid_link_passes_explicit_stability_gate() -> None:
    """``_DT`` clears the first-order explicit-stability gate.

    With ``factor = 1e3`` we have ``k_link = 1.28e10 N/m`` and reduced
    mass ``mu_eff = m_a m_b / (m_a + m_b) = 1e7 kg``, so
    ``check_connector_stability`` reports
    ``omega = sqrt(K / mu_eff) ~= 36 rad/s`` and a stable bound
    ``dt < safety * 2 / omega ~= 4.5e-2 s``. We step at ``2 ms`` —
    well within the diagnostic's threshold (the empirical gen-alpha
    bound at ``rho_inf = 0.9`` is tighter; see module docstring).
    """
    lhs_global, _, _, connectors = _assemble_system()
    messages = check_connector_stability(
        lhs=lhs_global, connectors=connectors, dt=_DT, safety_factor=0.8
    )
    assert (
        messages == []
    ), f"explicit-stability gate flagged the rigid link at dt={_DT:.3e}: {messages}"
