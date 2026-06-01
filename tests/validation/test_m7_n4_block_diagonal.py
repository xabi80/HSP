"""M7-Foundation PR1 -- F4 N = 4 block-diagonal validation (RED GATE).

The first N >= 3 test the repo has ever had. Treated as an
Item-19 code-path exerciser per CLAUDE.md S13: the size-agnostic
solver / integrator / assembly code paths have been validated at
N = 1 (M2-M6) and N = 2 (M4 PR1, PR3, PR6) but never at N >= 3.
This test exercises ``n_dof = 24`` for the first time.

Pre-foundation audit cleared the static / equilibrium / assembly
code paths at ``n_dof = 24`` (see
``docs/diagnostics/m7-pr1-multibody-scaling.md``). What the audit
does NOT exercise — and what this test does — is the integrator's
per-step loop at ``n_dof = 24``: the RadiationConvolution buffer
shifting, the per-step ``np.linalg.solve(A_eff, rhs)``, the
explicit-mu treatment in the convolution sum, and the pack/unpack
indexing through the full integration window.

System under test
-----------------
Four IDENTICAL copies of the M2 heave-only analytical fixture
(``M = A_inf = 1e7 kg``, ``C_33 = 1.28e7 N/m``,
``B_33 = 1.6e6 N*s/m`` on a 60-point grid), stacked into a global
24-DOF block-diagonal system via
:func:`floatsim.solver.state.assemble_global_lhs` and
:func:`floatsim.solver.state.assemble_global_kernel`. No
connectors. No catenaries. No state-dependent forces.

The analytical references are unchanged from M2::

    omega_n = sqrt(C / (M + A_inf)) = 0.8 rad/s -> T_n = 7.854 s
    zeta_n  = B / (2 omega_n (M + A_inf))       =  0.05  (5%)

Initial conditions -- DISTINCT and LOAD-BEARING
-----------------------------------------------
Heave ICs per body: 1.0, 0.8, 0.6, 0.4 m. All other DOFs at zero.
**The distinct ICs are the pack/unpack transposition discriminator
(plan Q5).** With identical bodies AND identical ICs, body k's
signal is indistinguishable from body j's, so a pack/unpack
transposition bug (body 1's state written into body 2's slot in
the global vector, etc.) would PASS every per-body assertion.
Distinct ICs make the transposition immediately observable as a
magnitude mismatch on the affected body via assertion (D).

Rank-deficient C breadcrumb (pre-flight (ii))
---------------------------------------------
The global hydrostatic restoring ``C`` is rank 4 out of 24 by
construction: only heave is restored at each body. The remaining
20 DOFs have zero diagonal entries. Equilibrium relies on the
``lambda_reg * I`` diagonal regularisation in
``floatsim/solver/equilibrium.py`` (verified at ``n_dof = 6`` by
the Q8 pre-foundation audit). **If equilibrium fails to converge
at n_dof = 24, lambda_reg is the FIRST thing to inspect** per the
diagnostic-doc breadcrumb (item ii).

Tolerances (locked at plan Q5)
------------------------------
* (A) Period: ``rtol = 1e-2`` (1 %). Inherits the M2-fixture's
  ~0.5 % Kramers-Kronig drift in ``B(omega)`` that
  ``test_m4_two_body_assembly.py`` already tolerates.
* (B) Damping: ``rtol = 5e-2`` (5 %; explicit-mu O(h) lag).
* (C) Cross-DOF silence per body: ``atol = 1e-10 m`` ABSOLUTE.
  Scale rationale: the maximum excited IC across all bodies is
  1.0 m; leakage from heave to silent DOFs would scale as
  ``(coupling-fraction) x 1.0 m``. ``1e-10 m`` is two decades
  below float64 round-off at the ``M = 1e7 kg`` fixture scale.
* (D) IC-scaling identity: first positive peak amplitude ratio
  ``body_k_peak / body_0_peak == IC_k / IC_0`` to ``rtol = 5e-3``.
* (E) **Pre-flight addendum (Xabier, locked).** Condition-number
  preservation: ``cond(A_eff)`` at ``n_dof = 24`` equals
  ``cond(A_eff)`` at the single-body ``n_dof = 6`` reference to
  ``rtol = 1e-12``. Block-diagonal stacking of identical blocks
  produces a matrix whose condition number equals the per-block
  condition number; any deviation indicates off-diagonal leakage
  in :func:`floatsim.solver.state.assemble_global_lhs`. Free
  structural-correctness check at no extra runtime cost.

Failure-mode response (locked pre-flight item (iv))
---------------------------------------------------
Recorded here for any future investigator: failure routes per
plan Q8 pre-flight item (iv). (a) Q5-A/B fail -> real physics
bug, sub-branch ``fix-m7-f4-<mechanism>``. (b) Q5-C fail ->
pack/unpack or block-stack assembly bug. (c) Q5-D fail ->
indexing transposition (the distinct-IC discriminator did its
job). (d) Equilibrium non-converge -> lambda_reg path first.
(e) Assertion (E) fail -> off-diagonal leakage in
``_block_diagonal``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.equilibrium import solve_static_equilibrium
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
_N_BODIES = 4
_HEAVE_ICS_M = (1.0, 0.8, 0.6, 0.4)
_DT = 0.01
_DURATION_S = 100.0  # ~12 periods; enough for stable log-decrement fit
_T_MAX_KERNEL_S = 200.0
_RHO_INF = 1.0  # trapezoidal limit; matches test_m4_two_body_assembly.py


# ---------------------------------------------------------------------------
# fixture builders -- IDENTICAL to test_m4_two_body_assembly.py and the
# Q8 pre-foundation audit's diagnostic script, so the cross-references stay
# stable.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# peak / period extraction helpers
# ---------------------------------------------------------------------------


def _fit_period_zero_crossings(t: np.ndarray, x: np.ndarray) -> float:
    """Period from mean spacing of ascending zero crossings (M4 PR1 pattern)."""
    sign = np.sign(x)
    crossings = np.where((sign[:-1] < 0) & (sign[1:] >= 0))[0]
    if crossings.size < 3:
        raise AssertionError(f"need >= 3 zero crossings; got {crossings.size}")
    t_cross = t[crossings] + (t[crossings + 1] - t[crossings]) * (
        -x[crossings] / (x[crossings + 1] - x[crossings])
    )
    return float(np.mean(np.diff(t_cross)))


def _first_positive_peak_after_t0(t: np.ndarray, x: np.ndarray) -> float:
    """Return the first positive local maximum after t > 0 (skips the IC peak at t=0).

    Q5 assertion (D) compares first-positive-peak amplitudes across
    bodies. The IC itself is a peak at t = 0; we want the FIRST
    post-IC positive peak (around t = T_n for an underdamped oscillator).
    """
    is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0)
    peak_idx = np.where(is_peak)[0] + 1
    # Skip any peak at t = 0 (the IC itself) by demanding t > T_n / 4.
    skip_t = _T_N / 4.0
    post_t0 = peak_idx[t[peak_idx] > skip_t]
    if post_t0.size == 0:
        raise AssertionError(
            f"no positive peak found at t > {skip_t:.2f} s in the {t[-1]:.1f}-s run"
        )
    return float(x[post_t0[0]])


# ---------------------------------------------------------------------------
# shared 4-body integration (cached so the 5 assertions reuse one run)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _run_n4_block_diagonal_free_decay():
    """Build the N=4 block-diagonal system and integrate; return (eq, res, lhs_*, lhs_single)."""
    hdb = _single_body_hdb()
    lhs_single = assemble_cummins_lhs(rigid_body_mass=_single_body_rigid_mass(), hdb=hdb)
    kernel_single = compute_retardation_kernel(hdb, t_max=_T_MAX_KERNEL_S, dt=_DT)

    # Block-diagonal stack into n_dof = 24.
    lhs_global = assemble_global_lhs([lhs_single] * _N_BODIES)
    kernel_global = assemble_global_kernel([kernel_single] * _N_BODIES)
    assert lhs_global.n_dof == 24
    assert lhs_global.n_bodies == _N_BODIES

    # Per-body 6-vector ICs: heave displacement from _HEAVE_ICS_M; other DOFs zero.
    per_body_ic = []
    for ic_heave in _HEAVE_ICS_M:
        xi_body = np.zeros(6)
        xi_body[2] = ic_heave
        per_body_ic.append(xi_body)
    xi0 = pack_state(per_body_ic)
    assert xi0.shape == (24,)

    # Static equilibrium pre-step -- with no state_force, this should
    # collapse to xi_eq = 0 (or close), regardless of xi0. We're testing
    # that the solve CONVERGES at n_dof = 24 with the rank-deficient C
    # (lambda_reg path; pre-flight item (ii) of the diagnostic doc).
    eq = solve_static_equilibrium(lhs=lhs_global, state_force=None, tol=1.0e-6)

    res = integrate_cummins(
        lhs=lhs_global,
        kernel=kernel_global,
        xi0=xi0,  # NOT eq.xi_eq -- the test integrates from the chosen ICs
        xi_dot0=np.zeros(24),
        duration=_DURATION_S,
        rho_inf=_RHO_INF,
    )
    return eq, res, lhs_global, lhs_single


# ---------------------------------------------------------------------------
# Q5 (A) -- period identity per body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body_idx", range(_N_BODIES))
def test_q5_A_each_body_heave_period_matches_m2_analytical(body_idx: int) -> None:
    """Q5 (A): per-body heave period matches T_n = 7.854 s to rtol = 1e-2."""
    _, res, _, _ = _run_n4_block_diagonal_free_decay()
    heave_slot = 6 * body_idx + 2
    T_fit = _fit_period_zero_crossings(res.t, res.xi[:, heave_slot])
    rel_err = abs(T_fit - _T_N) / _T_N
    assert rel_err < 1.0e-2, (
        f"body {body_idx} heave period {T_fit:.5f} s deviates from analytical "
        f"T_n = {_T_N:.5f} s by {rel_err:.3%} (limit 1%; Q5-A)"
    )


# ---------------------------------------------------------------------------
# Q5 (B) -- damping identity per body
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body_idx", range(_N_BODIES))
def test_q5_B_each_body_heave_damping_matches_m2_analytical(body_idx: int) -> None:
    """Q5 (B): per-body heave log-decrement damping zeta = 0.05 to rtol = 5e-2."""
    _, res, _, _ = _run_n4_block_diagonal_free_decay()
    heave_slot = 6 * body_idx + 2
    zeta_fit = _fit_damping_log_decrement(res.t, res.xi[:, heave_slot])
    rel_err = abs(zeta_fit - _ZETA_N) / _ZETA_N
    assert rel_err < 5.0e-2, (
        f"body {body_idx} heave damping {zeta_fit:.5f} deviates from analytical "
        f"zeta_n = {_ZETA_N:.5f} by {rel_err:.3%} (limit 5%; Q5-B)"
    )


# ---------------------------------------------------------------------------
# Q5 (C) -- cross-DOF silence per body (block-diagonal solver must not leak)
# ---------------------------------------------------------------------------


_SILENT_DOFS_PER_BODY = (0, 1, 3, 4, 5)  # surge, sway, roll, pitch, yaw
_SILENT_ATOL_M = 1.0e-10  # absolute scale per Q5 (C) and diagnostic doc


@pytest.mark.parametrize("body_idx", range(_N_BODIES))
@pytest.mark.parametrize("local_dof", _SILENT_DOFS_PER_BODY)
def test_q5_C_silent_dofs_stay_at_atol(body_idx: int, local_dof: int) -> None:
    """Q5 (C): silent DOFs (surge/sway/roll/pitch/yaw) per body stay at |xi| < 1e-10 m.

    Scale rationale: max excited IC across bodies is 1.0 m; any leakage
    would scale as (coupling-fraction) * 1.0 m. 1e-10 m is two decades
    below float64 round-off at the M = 1e7 kg fixture scale.
    """
    _, res, _, _ = _run_n4_block_diagonal_free_decay()
    global_dof = 6 * body_idx + local_dof
    max_abs = float(np.max(np.abs(res.xi[:, global_dof])))
    assert max_abs < _SILENT_ATOL_M, (
        f"body {body_idx} silent-DOF {local_dof} max |xi| = {max_abs:.3e} "
        f"exceeds atol = {_SILENT_ATOL_M:.0e} m (Q5-C). Off-block leakage in "
        "assembly or pack/unpack indexing bug suspected -- see plan Q8 "
        "pre-flight item (iv), failure mode (b)."
    )


# ---------------------------------------------------------------------------
# Q5 (D) -- IC-scaling identity (pack/unpack transposition discriminator)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("body_idx", range(1, _N_BODIES))
def test_q5_D_first_peak_amplitude_ratio_matches_ic_ratio(body_idx: int) -> None:
    """Q5 (D): body_k first-peak amplitude / body_0 first-peak amplitude == IC_k / IC_0.

    Pack/unpack transposition discriminator. A bug that wrote body_1's
    state to body_2's slot would produce body_2 oscillating at body_1's
    amplitude -> ratio 0.8 instead of expected 0.6 (or any other
    permutation). Tolerance: rtol = 5e-3, one decade tighter than the
    period tolerance.
    """
    _, res, _, _ = _run_n4_block_diagonal_free_decay()
    peak_0 = _first_positive_peak_after_t0(res.t, res.xi[:, 2])  # body 0 heave
    peak_k = _first_positive_peak_after_t0(res.t, res.xi[:, 6 * body_idx + 2])
    expected_ratio = _HEAVE_ICS_M[body_idx] / _HEAVE_ICS_M[0]
    actual_ratio = peak_k / peak_0
    rel_err = abs(actual_ratio - expected_ratio) / expected_ratio
    assert rel_err < 5.0e-3, (
        f"body {body_idx} first-peak amplitude ratio {actual_ratio:.6f} "
        f"deviates from IC ratio {expected_ratio:.6f} by {rel_err:.4%} "
        f"(limit 0.5%; Q5-D). Pack/unpack transposition suspected -- see plan "
        "Q8 pre-flight item (iv), failure mode (c)."
    )


# ---------------------------------------------------------------------------
# Q5 (E) -- condition-number preservation (pre-flight Xabier addendum)
# ---------------------------------------------------------------------------


def _A_eff(M_plus_Ainf: np.ndarray, C: np.ndarray, dt: float, rho_inf: float) -> np.ndarray:
    """Replicate the integrator's per-step LHS matrix.

    A_eff = (1 - alpha_m) (M + A_inf) + (1 - alpha_f) h^2 beta C
    """
    alpha_m = (2.0 * rho_inf - 1.0) / (rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    beta = 0.25 * (1.0 - alpha_m + alpha_f) ** 2
    return (1.0 - alpha_m) * M_plus_Ainf + (1.0 - alpha_f) * (dt ** 2) * beta * C


def test_q5_E_cond_A_eff_preserved_under_block_diagonal_stacking() -> None:
    """Q5 (E): cond(A_eff at n_dof=24) == cond(A_eff at n_dof=6) to rtol = 1e-12.

    Block-diagonal stacking of identical blocks produces a matrix whose
    condition number equals the per-block condition number. Deviation
    indicates off-diagonal leakage in
    :func:`floatsim.solver.state.assemble_global_lhs`. Free
    structural-correctness check; pre-flight addendum locked by Xabier
    before the F4 red test fires.
    """
    _, _, lhs_global, lhs_single = _run_n4_block_diagonal_free_decay()
    A_eff_single = _A_eff(lhs_single.M_plus_Ainf, lhs_single.C, _DT, _RHO_INF)
    A_eff_global = _A_eff(lhs_global.M_plus_Ainf, lhs_global.C, _DT, _RHO_INF)
    cond_single = float(np.linalg.cond(A_eff_single))
    cond_global = float(np.linalg.cond(A_eff_global))
    rel_err = abs(cond_global - cond_single) / cond_single
    assert rel_err < 1.0e-12, (
        f"cond(A_eff) at n_dof = 24 ({cond_global:.6e}) differs from "
        f"cond(A_eff) at n_dof = 6 ({cond_single:.6e}) by {rel_err:.3e} "
        "(limit 1e-12; Q5-E). Off-diagonal leakage in assemble_global_lhs "
        "or _block_diagonal suspected -- see plan Q8 pre-flight item (iv), "
        "failure mode (e)."
    )


# ---------------------------------------------------------------------------
# Equilibrium-converged sanity (pre-flight breadcrumb (ii))
# ---------------------------------------------------------------------------


def test_static_equilibrium_converges_at_n_dof_24() -> None:
    """Sanity: solve_static_equilibrium(state_force=None) converges at n_dof=24.

    The rank-deficient C (rank 4/24) relies on the lambda_reg
    regularisation path; pre-flight item (ii) flags this as the
    first-action diagnosis if this assertion fails.
    """
    eq, _, _, _ = _run_n4_block_diagonal_free_decay()
    assert eq.converged, (
        f"static equilibrium did not converge at n_dof = 24 "
        f"(residual_inf_norm = {eq.residual_norm:.3e} N). "
        "Investigate lambda_reg regularisation path first -- see "
        "docs/diagnostics/m7-pr1-multibody-scaling.md pre-flight item (ii)."
    )


# ---------------------------------------------------------------------------
# Structural shape sanity (cheap; runs first)
# ---------------------------------------------------------------------------


def test_global_state_has_24_components() -> None:
    """Sanity: the 4-body global system has the expected 24 DOFs."""
    _, res, lhs_global, _ = _run_n4_block_diagonal_free_decay()
    assert res.xi.shape[1] == 24
    assert res.xi_dot.shape[1] == 24
    assert res.xi_ddot.shape[1] == 24
    assert lhs_global.n_dof == 24
    assert lhs_global.n_bodies == 4
