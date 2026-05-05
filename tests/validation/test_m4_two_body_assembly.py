"""Milestone 4 gate — N-body assembly on a 2-body uncoupled fixture.

ARCHITECTURE.md §2.2: the multi-body Cummins equation is the single-body
equation with global state ``Xi = [xi_1, xi_2, ..., xi_N]`` and 6N x 6N
block matrices for ``M + A_inf``, ``C``, and ``K(t)``. If the BEM database
for each body is independent (no hydrodynamic interaction), the global
matrices are block-diagonal. This test is the PR1 gate of M4: verify that
stacking two identical single-body systems block-diagonally and running
the integrator in the 6N=12 state space reproduces the M2 analytical
free-decay period and damping on each body independently, with no
spurious cross-body or cross-DOF coupling.

Fixture
-------
Two copies of the :mod:`tests.validation.test_cummins_free_decay_analytical`
heave-only system (M = A_inf = 1.0e7 kg, C_33 = 1.28e7 N/m,
B_33 = 1.6e6 N*s/m on a 60-point grid), stacked block-diagonally. The
analytical references are unchanged::

    omega_n = sqrt(C / (M + A_inf)) = 0.8 rad/s    ->  T_n = 7.854 s
    zeta_n  = B / (2 omega_n (M + A_inf))          =  0.05   (5%)

Initial heave on body 0 is 1.0 m; on body 1 is 0.5 m. Because the global
system is block-diagonal and identical per block, both bodies' heave
responses must coincide after rescaling by their IC amplitudes.

Tolerances
----------
Matches the M2 analytical fixture. The fixture holds ``A(omega) = A_inf``
while setting ``B(omega) = B_0 > 0``; this pair is *not* Kramers-Kronig
consistent, and the true kernel computed by
:func:`floatsim.hydro.retardation.compute_retardation_kernel` therefore
induces a ~0.5 % effective-mass correction at ``omega_n`` — the observed
heave period drifts from ``sqrt(C/(M+A_inf))`` by that amount. The period
bound below (``rtol = 1e-2``) accommodates it; a physically-consistent
BEM reproduces the period to sub-percent accuracy in the M2 OC4 fixture.

* Period:       ``rtol = 1e-2`` (1 %; ~0.5 % KK drift + fit quantization).
* Damping:      ``rtol = 5e-2`` (5 %; explicit-mu O(h) in the convolution).
* Cross-coupling: ``atol = 1e-10`` — block-diagonal fixture, so silent
  DOFs must stay at float-solver noise.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

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
def _run_two_body_free_decay():
    hdb = _single_body_hdb()
    lhs_single = assemble_cummins_lhs(rigid_body_mass=_single_body_rigid_mass(), hdb=hdb)
    dt = 0.01
    kernel_single = compute_retardation_kernel(hdb, t_max=200.0, dt=dt)

    # Stack two copies block-diagonally into the 12-DOF global system.
    lhs_global = assemble_global_lhs([lhs_single, lhs_single])
    kernel_global = assemble_global_kernel([kernel_single, kernel_single])

    xi0_body_a = np.zeros(6)
    xi0_body_a[2] = 1.0  # body 0 heave = 1.0 m
    xi0_body_b = np.zeros(6)
    xi0_body_b[2] = 0.5  # body 1 heave = 0.5 m
    xi0 = pack_state([xi0_body_a, xi0_body_b])

    return integrate_cummins(
        lhs=lhs_global,
        kernel=kernel_global,
        xi0=xi0,
        xi_dot0=np.zeros(12),
        duration=100.0,
        rho_inf=1.0,
    )


def test_two_body_global_state_has_12_components() -> None:
    res = _run_two_body_free_decay()
    assert res.xi.shape[1] == 12
    assert res.xi_dot.shape[1] == 12
    assert res.xi_ddot.shape[1] == 12


def test_body_a_heave_reproduces_single_body_period() -> None:
    res = _run_two_body_free_decay()
    T_fit = _fit_period_zero_crossings(res.t, res.xi[:, 2])
    rel_err = abs(T_fit - _T_N) / _T_N
    assert rel_err < 1.0e-2, (
        f"body 0 heave period {T_fit:.5f} s deviates from analytical "
        f"{_T_N:.5f} s by {rel_err:.3%} (limit 1%)"
    )


def test_body_b_heave_reproduces_single_body_period() -> None:
    res = _run_two_body_free_decay()
    T_fit = _fit_period_zero_crossings(res.t, res.xi[:, 2 + 6])
    rel_err = abs(T_fit - _T_N) / _T_N
    assert rel_err < 1.0e-2, (
        f"body 1 heave period {T_fit:.5f} s deviates from analytical "
        f"{_T_N:.5f} s by {rel_err:.3%} (limit 1%)"
    )


def test_body_a_heave_matches_analytical_damping() -> None:
    res = _run_two_body_free_decay()
    zeta_fit = _fit_damping_log_decrement(res.t, res.xi[:, 2])
    rel_err = abs(zeta_fit - _ZETA_N) / _ZETA_N
    assert rel_err < 5.0e-2, (
        f"body 0 heave damping {zeta_fit:.5f} deviates from analytical "
        f"{_ZETA_N:.5f} by {rel_err:.3%} (limit 5%)"
    )


def test_body_b_heave_matches_analytical_damping() -> None:
    res = _run_two_body_free_decay()
    zeta_fit = _fit_damping_log_decrement(res.t, res.xi[:, 2 + 6])
    rel_err = abs(zeta_fit - _ZETA_N) / _ZETA_N
    assert rel_err < 5.0e-2, (
        f"body 1 heave damping {zeta_fit:.5f} deviates from analytical "
        f"{_ZETA_N:.5f} by {rel_err:.3%} (limit 5%)"
    )


def test_non_heave_dofs_remain_at_rest_on_both_bodies() -> None:
    """Block-diagonal fixture + heave-only IC: no cross-DOF, no cross-body coupling."""
    res = _run_two_body_free_decay()
    # Indices 0,1,3,4,5 on body 0 and 6,7,9,10,11 on body 1 were never excited.
    silent = np.concatenate([[0, 1, 3, 4, 5], [6, 7, 9, 10, 11]])
    peak = float(np.max(np.abs(res.xi[:, silent])))
    assert peak < 1.0e-10, (
        f"uncoupled non-heave DOFs drifted: max|xi| = {peak:.3e} "
        "(expected ~0 under block-diagonal fixture)"
    )


def test_bodies_are_synchronous_up_to_ic_amplitude() -> None:
    """Identical blocks -> body 1 heave should be 0.5 * body 0 heave at every step."""
    res = _run_two_body_free_decay()
    heave_a = res.xi[:, 2]
    heave_b = res.xi[:, 2 + 6]
    # Relative to IC amplitude on body 0: residual should be at float-solver level.
    residual = float(np.max(np.abs(heave_b - 0.5 * heave_a)))
    assert residual < 1.0e-10, (
        f"bodies not synchronous: max|heave_b - 0.5*heave_a| = {residual:.3e} "
        "(expected ~0 under identical blocks)"
    )
