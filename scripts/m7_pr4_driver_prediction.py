"""M7-Foundation PR4 Step A -- hand-wired M4 PR6 setup for round-trip identity.

Per docs/m7-foundation-plan.md Q6 PR4 row: predict that
``build_system(M4_PR6_deck)`` produces an identical
``(CumminsLHS.M_plus_Ainf, CumminsLHS.C, kernel.K, state_force(0, xi, 0),
xi0, xi_dot0)`` as the hand-wired M4 PR6 setup using the existing
low-level helpers. Step C identity test compares Step B's
build_system output to Step A's targets at rtol = 1e-12.

The M4 PR6 fixture:
  * 2 bodies, each backed by the M2 synthetic heave-only fixture
    (M = A_inf = 1e7 kg, C_33 = 1.28e7 N/m, narrowband B(omega)
    via well_behaved_b).
  * 1 heave rigid-link between body 0 and body 1 (penalty
    stiffness 1e3 * max(diag(C_global)) per the M4 PR3 schema
    floor).
  * 2 catenaries on body 0 only, fairlead AT the body reference
    point (zero offset), anchors at (+/-350 m, 0, -200 m).

Step A prints:
  - lhs_global.M_plus_Ainf and lhs_global.C (12 x 12 each)
  - kernel_global.K shape + a checksum (sum + a single element)
    for size-stable identity check
  - state_force(0, xi_test, 0) at two poses:
      xi_test_zero = zeros(12)          -- the trivial case
      xi_test_surge = (0.5, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0)
                                          -- body 0 surge + body 1 surge,
                                          identical so the rigid-link
                                          connector force should be 0 in
                                          the heave block but catenaries
                                          asymmetric in surge.
  - xi0 and xi_dot0 (post-equilibrium-solve if requested).
"""

from __future__ import annotations

import numpy as np

from floatsim.bodies.connector import (
    heave_rigid_link,
    make_connector_state_force,
)
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.mooring.catenary_analytic import (
    CatenaryAttachment,
    CatenaryLine,
    make_catenary_state_force,
)
from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.state import assemble_global_kernel, assemble_global_lhs
from tests.support.synthetic_bem import make_diagonal_hdb, well_behaved_b
from tests.validation.test_cummins_free_decay_analytical import (
    _A_INF_33,
    _B_33,
    _C_33,
    _CUTOFF_OMEGA,
    _I_OTHER,
    _M_33,
    _M_OTHER,
    _OMEGA_GRID,
)

# ---------------------------------------------------------------------------
# M4 PR6 fixture (verbatim from test_m4_two_body_moored.py)
# ---------------------------------------------------------------------------

_DT = 2.0e-3
_PENALTY_FACTOR = 1.0e3
_PENALTY_K = _PENALTY_FACTOR * _C_33  # = 1.28e10 N/m

_CATENARY_LINE = CatenaryLine(length=500.0, weight_per_length=1000.0, EA=5.0e8)
_SEABED_DEPTH = 200.0
_ANCHOR_PLUS_GLOBAL = np.array([+350.0, 0.0, -200.0])
_ANCHOR_MINUS_GLOBAL = np.array([-350.0, 0.0, -200.0])


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


def build_m4_pr6_hand_wired():
    """Hand-wire the M4 PR6 setup using existing low-level helpers.

    Returns dict with the round-trip identity targets.
    """
    hdb = _single_body_hdb()
    lhs_single = assemble_cummins_lhs(rigid_body_mass=_single_body_rigid_mass(), hdb=hdb)
    kernel_single = compute_retardation_kernel(hdb, t_max=120.0, dt=_DT)

    lhs_global = assemble_global_lhs([lhs_single, lhs_single])
    kernel_global = assemble_global_kernel([kernel_single, kernel_single])

    # RigidLink: heave between body 0 and body 1.
    link = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=_PENALTY_K)
    connector_force = make_connector_state_force([link], n_dof=12)

    # Catenaries: 2 on body 0, body_b = earth, fairleads at body reference.
    cat_plus = CatenaryAttachment(
        body_index=0,
        fairlead_body=np.zeros(3),
        anchor_global=_ANCHOR_PLUS_GLOBAL.copy(),
        line=_CATENARY_LINE,
        seabed_depth=_SEABED_DEPTH,
    )
    cat_minus = CatenaryAttachment(
        body_index=0,
        fairlead_body=np.zeros(3),
        anchor_global=_ANCHOR_MINUS_GLOBAL.copy(),
        line=_CATENARY_LINE,
        seabed_depth=_SEABED_DEPTH,
    )
    catenary_force = make_catenary_state_force([cat_plus, cat_minus], n_dof=12)

    def total_state_force(t, xi, xi_dot):
        return connector_force(t, xi, xi_dot) + catenary_force(t, xi, xi_dot)

    # Solve equilibrium (default behaviour of build_system).
    eq = solve_static_equilibrium(lhs=lhs_global, state_force=total_state_force, tol=1.0)

    return {
        "lhs_global": lhs_global,
        "kernel_global": kernel_global,
        "state_force": total_state_force,
        "xi0_post_equilibrium": eq.xi_eq,
        "xi_dot0": np.zeros(12),
        "equilibrium_converged": eq.converged,
        "equilibrium_residual_norm": eq.residual_norm,
    }


def main() -> None:
    print("M7-Foundation PR4 Step A -- hand-wired M4 PR6 round-trip targets")
    print("=" * 70)
    targets = build_m4_pr6_hand_wired()

    lhs = targets["lhs_global"]
    kernel = targets["kernel_global"]
    sf = targets["state_force"]

    print(f"\nlhs.M_plus_Ainf shape: {lhs.M_plus_Ainf.shape}, "
          f"sum = {lhs.M_plus_Ainf.sum():.12e}")
    print(f"lhs.C           shape: {lhs.C.shape}, "
          f"sum = {lhs.C.sum():.12e}")
    print(f"kernel.K        shape: {kernel.K.shape}, "
          f"sum = {kernel.K.sum():.12e}, [0,0,0] = {kernel.K[0,0,0]:.12e}")

    xi_test_zero = np.zeros(12)
    F_zero = sf(0.0, xi_test_zero, np.zeros(12))
    print(f"\nstate_force(0, xi=zeros, 0) =")
    for i in range(2):
        print(f"  body {i}: {F_zero[6*i:6*i+6]!r}")

    xi_test_surge = np.zeros(12)
    xi_test_surge[0] = 0.5  # body 0 surge
    xi_test_surge[6] = 0.5  # body 1 surge
    F_surge = sf(0.0, xi_test_surge, np.zeros(12))
    print(f"\nstate_force(0, xi=both-surge-0.5, 0) =")
    for i in range(2):
        print(f"  body {i}: {F_surge[6*i:6*i+6]!r}")

    print(f"\nequilibrium converged: {targets['equilibrium_converged']}, "
          f"residual_norm = {targets['equilibrium_residual_norm']:.3e}")
    print(f"xi0_post_equilibrium = {targets['xi0_post_equilibrium']!r}")

    print("\n--- Numerical fingerprints for tests/unit/test_driver.py ---")
    np.set_printoptions(precision=15, suppress=False, linewidth=200)
    print(f"M_plus_Ainf_sum = {lhs.M_plus_Ainf.sum()!r}")
    print(f"C_sum           = {lhs.C.sum()!r}")
    print(f"K_sum           = {kernel.K.sum()!r}")
    print(f"K_000           = {kernel.K[0,0,0]!r}")
    print(f"F_zero          = np.{repr(F_zero)}")
    print(f"F_surge         = np.{repr(F_surge)}")
    print(f"xi_eq           = np.{repr(targets['xi0_post_equilibrium'])}")


if __name__ == "__main__":
    main()
