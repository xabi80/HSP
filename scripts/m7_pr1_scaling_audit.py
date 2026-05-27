"""Q8 pre-foundation audit diagnostic — M7-Foundation PR1, Step A.

Run static-equilibrium + linear-solve scaling at n_dof = 6, 12, 24
on a block-diagonal stack of the M2 heave-only analytical fixture.
Records nfev, residual_norm, condition number, and wall-clock.

Audit items exercised (per docs/m7-foundation-plan.md Q8):
  1. solve_static_equilibrium scaling -- nfev / iterations / wall.
  2. np.linalg.solve(A_eff, rhs) condition number on A_eff matching
     the integrator's assembly.
  6. make_connector_state_force body-index validation at n_dof=24
     (boundary: bodies [0,4), connector body indices 0..3 should
     pass; index 4 should raise).

Items 3, 4, 5, 7, 8 are docstring-contract checks; documented in
the audit doc directly without a runtime script.
"""
from __future__ import annotations

import time

import numpy as np

from floatsim.bodies.connector import LinearConnector, make_connector_state_force
from floatsim.hydro.radiation import assemble_cummins_lhs, CumminsLHS
from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.state import assemble_global_lhs
from tests.support.synthetic_bem import make_diagonal_hdb, well_behaved_b
from tests.validation.test_cummins_free_decay_analytical import (
    _A_INF_33, _B_33, _C_33, _CUTOFF_OMEGA, _I_OTHER, _M_33, _M_OTHER, _OMEGA_GRID,
)


def _single_body_hdb():
    A_inf_diag = [_M_OTHER, _M_OTHER, _A_INF_33, _I_OTHER, _I_OTHER, _I_OTHER]
    A_diag_per_omega = [list(A_inf_diag) for _ in range(_OMEGA_GRID.size)]
    band_values = [1.0e3, 1.0e3, _B_33, 1.0e4, 1.0e4, 1.0e4]
    rolloff = well_behaved_b(_OMEGA_GRID, band_value=1.0, cutoff_omega=_CUTOFF_OMEGA)
    B_diag_per_omega = [[bv * float(r) for bv in band_values] for r in rolloff]
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


def main() -> None:
    hdb = _single_body_hdb()
    M = _single_body_rigid_mass()
    lhs_single = assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb)

    print("Q8 pre-foundation audit -- scaling diagnostics")
    print("=" * 64)

    # --- Item 1: solve_static_equilibrium scaling ---
    print("\nItem 1: solve_static_equilibrium(F_state=None) scaling")
    print(f"  {'n_dof':>6}  {'N':>3}  {'nfev':>6}  {'res_inf_N':>12}  {'wall_s':>9}  {'conv':>5}")
    for n in (1, 2, 4):
        per_body = [lhs_single] * n
        lhs_global = assemble_global_lhs(per_body)
        t0 = time.perf_counter()
        eq = solve_static_equilibrium(lhs=lhs_global, state_force=None, tol=1.0e-6)
        wall = time.perf_counter() - t0
        print(f"  {lhs_global.n_dof:>6}  {n:>3}  {eq.iterations:>6}  "
              f"{eq.residual_norm:>12.3e}  {wall:>9.4f}  {str(eq.converged):>5}")

    # --- Item 2: A_eff condition number at n_dof = 6, 12, 24 ---
    print("\nItem 2: A_eff condition number (block-diagonal must equal single-body)")
    print(f"  {'n_dof':>6}  {'N':>3}  {'cond(M+Ainf)':>15}  {'cond(C)':>15}  {'cond(A_eff)':>15}")
    rho_inf = 0.9
    alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
    h = 0.01  # representative dt
    for n in (1, 2, 4):
        per_body = [lhs_single] * n
        lhs_global = assemble_global_lhs(per_body)
        A_eff = (1.0 - alpha_m) * lhs_global.M_plus_Ainf + (1.0 - alpha_f) * (h ** 2) * beta * lhs_global.C
        # Add zero-restoring DOFs would make C singular -- regularise A_eff for cond
        try:
            cond_M = np.linalg.cond(lhs_global.M_plus_Ainf)
        except np.linalg.LinAlgError:
            cond_M = float("inf")
        try:
            cond_C = np.linalg.cond(lhs_global.C)
        except np.linalg.LinAlgError:
            cond_C = float("inf")
        try:
            cond_A = np.linalg.cond(A_eff)
        except np.linalg.LinAlgError:
            cond_A = float("inf")
        print(f"  {lhs_global.n_dof:>6}  {n:>3}  {cond_M:>15.3e}  {cond_C:>15.3e}  {cond_A:>15.3e}")

    # --- Item 6: make_connector_state_force body-index validation at n_dof=24 ---
    print("\nItem 6: make_connector_state_force body-index validation at n_dof=24")
    K = np.zeros((6, 6))
    K[2, 2] = 1.0e6
    B_mat = np.zeros((6, 6))
    # Valid: bodies 0..3 in a 4-body system
    for valid_a, valid_b in [(0, 1), (1, 2), (2, 3), (0, 3), (-1, 0), (3, -1)]:
        c = LinearConnector(body_a=valid_a, body_b=valid_b, K=K, B=B_mat)
        try:
            _ = make_connector_state_force([c], n_dof=24)
            print(f"  bodies ({valid_a:>2}, {valid_b:>2}): accepted (n_dof=24)")
        except ValueError as e:
            print(f"  bodies ({valid_a:>2}, {valid_b:>2}): REJECTED unexpectedly -- {e}")
    # Invalid: index 4 (boundary -- only 0..3 are valid for n_dof=24)
    try:
        c = LinearConnector(body_a=0, body_b=4, K=K, B=B_mat)
        _ = make_connector_state_force([c], n_dof=24)
        print(f"  bodies ( 0,  4): accepted ERRONEOUSLY -- audit fail")
    except ValueError as e:
        msg = str(e)
        print(f"  bodies ( 0,  4): correctly rejected -- {msg[:80]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
