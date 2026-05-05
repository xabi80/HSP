"""Milestone 4 gate 1 — two bodies rigidly linked in heave.

Per ARCHITECTURE.md §7 (Validation) + docs/milestone-4-plan.md §PR3:
two identical single-body systems linked by a heave-only penalty rigid
link must reproduce the **single-body** free-decay period, i.e. the
system's symmetric heave mode is unchanged by the linkage.

Derivation
----------
For two identical single-DOF oscillators each of mass ``M`` (= M_body +
A_inf) and hydrostatic restoring ``C``, connected by a heave spring of
stiffness ``k``, the 2x2 mass-stiffness eigenproblem is::

    K = [ C + k   -k    ]        M_mat = [ M   0 ]
        [ -k      C + k ]                [ 0   M ]

    det(K - omega^2 M_mat) = 0
      (C + k - omega^2 M)^2 - k^2 = 0
      C + k - omega^2 M = +- k

Eigenvalues:

* Symmetric mode (``[1, 1]``): ``omega_sym^2 = C / M``
    -> the same period as a free single body — **the penalty does not
    change the symmetric mode**. This is the gate.
* Antisymmetric mode (``[1, -1]``): ``omega_anti^2 = (C + 2k) / M``
    -> ~113 rad/s for the fixture below; stays at zero amplitude if
    both bodies' ICs are identical (only the symmetric mode is excited).

We use the M2 analytical fixture (heave-only, M = A_inf = 1e7 kg,
C_33 = 1.28e7 N/m, B_33 = 1.6e6 N*s/m flat) so the analytical reference
is the same ``T_n = 2*pi*sqrt((M+A_inf)/C) = 7.854 s`` and
``zeta_n = 0.05`` (documented ~0.5 % Kramers-Kronig drift included in
the tolerance).

Penalty stiffness: ``k = 1e4 * max(diag(C_global)) = 1.28e11 N/m``, per
the upper end of the deck-configurable range in Q1.

Stability margin
----------------
With ``k = 1.28e11``, ``M = 2e7 kg`` the antisymmetric mode frequency is
``omega_anti = sqrt(2k/M) ~ 113 rad/s`` (``C << k`` so it's negligible).
Explicit-integrator stability floor: ``dt < 2/omega_anti ~ 0.0177 s``.
We run ``dt = 0.01 s``, giving a ~1.8x margin.
:func:`check_connector_stability` should return an empty list for this
configuration.

Tolerances
----------
* Period:       ``rtol = 1e-2`` (same rationale as the M4 two-body
  assembly test: the M2 fixture's KK-inconsistent ``B(omega)`` drifts
  the period by ~0.5 %).
* Damping:      ``rtol = 5e-2``.
* Drift:        ``max|xi_a_heave - xi_b_heave|`` < 1e-3 m (0.1 % of the
  1.0 m IC — the antisym mode should stay at zero up to round-off).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from floatsim.bodies.connector import (
    check_connector_stability,
    connector_drift,
    heave_rigid_link,
    make_connector_state_force,
)
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.state import (
    assemble_global_kernel,
    assemble_global_lhs,
    pack_state,
)
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
    _ZETA_N,
)
from tests.validation.test_oc4_heave_free_decay import _fit_damping_log_decrement

pytestmark = pytest.mark.slow

_T_N = 2.0 * np.pi * float(np.sqrt((_M_33 + _A_INF_33) / _C_33))
_DT = 0.01
_PENALTY_K = 1.0e4 * _C_33  # 1e4 x max diag(C_global) = 1.28e11 N/m


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


def _fit_period_zero_crossings(t: np.ndarray, x: np.ndarray) -> float:
    """Period from mean spacing of ascending zero crossings."""
    sign = np.sign(x)
    crossings = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if crossings.size < 3:
        raise AssertionError(f"need >= 3 zero crossings; got {crossings.size}")
    t_cross = t[crossings] + (t[crossings + 1] - t[crossings]) * (
        -x[crossings] / (x[crossings + 1] - x[crossings])
    )
    return float(np.mean(np.diff(t_cross)))


@lru_cache(maxsize=1)
def _run_rigid_link_free_decay():
    hdb = _single_body_hdb()
    lhs_single = assemble_cummins_lhs(rigid_body_mass=_single_body_rigid_mass(), hdb=hdb)
    kernel_single = compute_retardation_kernel(hdb, t_max=200.0, dt=_DT)

    lhs_global = assemble_global_lhs([lhs_single, lhs_single])
    kernel_global = assemble_global_kernel([kernel_single, kernel_single])

    link = heave_rigid_link(body_a=0, body_b=1, penalty_stiffness=_PENALTY_K)
    state_force = make_connector_state_force([link], n_dof=12)

    # Both bodies start at heave = 1.0 m (identical -> only symmetric mode excited).
    xi0_body = np.zeros(6)
    xi0_body[2] = 1.0
    xi0 = pack_state([xi0_body, xi0_body.copy()])

    result = integrate_cummins(
        lhs=lhs_global,
        kernel=kernel_global,
        xi0=xi0,
        xi_dot0=np.zeros(12),
        duration=100.0,
        state_force=state_force,
        rho_inf=1.0,
    )
    return result, lhs_global, link


def test_rigid_link_stability_check_is_clean_at_chosen_dt() -> None:
    """With the chosen penalty and dt the explicit-stability check is silent."""
    _res, lhs_global, link = _run_rigid_link_free_decay()
    msgs = check_connector_stability(lhs=lhs_global, connectors=[link], dt=_DT)
    assert msgs == [], f"unexpected stability warnings at dt={_DT}: {msgs}"


def test_rigid_link_both_bodies_heave_at_single_body_period() -> None:
    res, _lhs, _link = _run_rigid_link_free_decay()
    T_a = _fit_period_zero_crossings(res.t, res.xi[:, 2])
    T_b = _fit_period_zero_crossings(res.t, res.xi[:, 8])
    for label, T in (("body 0", T_a), ("body 1", T_b)):
        rel_err = abs(T - _T_N) / _T_N
        assert rel_err < 1.0e-2, (
            f"{label} heave period {T:.5f} s deviates from single-body "
            f"analytical {_T_N:.5f} s by {rel_err:.3%} (limit 1%)"
        )


def test_rigid_link_damping_matches_single_body_analytical() -> None:
    """Symmetric mode zeta = B/(2*omega*M) is invariant under rigid linkage."""
    res, _lhs, _link = _run_rigid_link_free_decay()
    zeta_a = _fit_damping_log_decrement(res.t, res.xi[:, 2])
    zeta_b = _fit_damping_log_decrement(res.t, res.xi[:, 8])
    for label, zeta in (("body 0", zeta_a), ("body 1", zeta_b)):
        rel_err = abs(zeta - _ZETA_N) / _ZETA_N
        assert rel_err < 5.0e-2, (
            f"{label} damping {zeta:.5f} deviates from analytical "
            f"{_ZETA_N:.5f} by {rel_err:.3%} (limit 5%)"
        )


def test_rigid_link_heave_drift_stays_below_0p1_percent_of_ic() -> None:
    """Antisymmetric mode is unexcited (identical ICs) -> heave drift ~ 0."""
    res, _lhs, link = _run_rigid_link_free_decay()
    drift = connector_drift(res.xi, link)
    # 1 m IC on heave -> 0.1 % = 1e-3 m tolerance.
    assert drift[2] < 1.0e-3, (
        f"heave drift peak = {drift[2]:.3e} m exceeds 0.1%% of IC amplitude "
        f"(stiffness {_PENALTY_K:.3e} N/m insufficient or numerical noise)"
    )


def test_rigid_link_non_heave_dofs_stay_silent() -> None:
    """Heave-only penalty + heave-only IC -> all other DOFs at solver noise."""
    res, _lhs, _link = _run_rigid_link_free_decay()
    silent = np.concatenate([[0, 1, 3, 4, 5], [6, 7, 9, 10, 11]])
    peak = float(np.max(np.abs(res.xi[:, silent])))
    assert peak < 1.0e-10, f"non-heave DOFs drifted: max|xi| = {peak:.3e} (expected ~0)"
