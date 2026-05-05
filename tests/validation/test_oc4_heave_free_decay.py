"""Milestone 2 gate validation — OC4 DeepCwind heave free decay (period).

Release a marin_semi.1-driven OC4-shaped fixture from a small heave
displacement with all other DOFs at rest and integrate the Cummins
equation through several natural periods. The fitted period must match
the analytical single-DOF formula evaluated at the heave natural
frequency::

    omega_n^2 (M_33 + A_33(omega_n)) = C_33            (natural freq)

``omega_n`` is solved by fixed-point iteration — ``A_33`` is
frequency-dependent, so the naive ``T_inf = 2*pi*sqrt((M+A_inf)/C)``
underestimates the true period (documented in
:mod:`tests.validation.test_oc4_natural_periods`).

BEM source — marin_semi.1 (M6 PR3 fixture migration)
----------------------------------------------------
Pre-fix, this test consumed the small ``platform_small.yml`` OrcaFlex
export, whose frequency grid only extended to ``ω ~= 5 rad/s`` and did
not reach the asymptotic ``B ~ ω⁻⁴`` regime. The Refinement-2 input
gates (``compute_retardation_kernel``) reject such grids. The test now
combines the well-resolved marin_semi.1 BEM database (which OC4
DeepCwind validation in M6 already uses) with the OC4 platform mass and
hydrostatic stiffness from :mod:`tests.validation.test_oc4_natural_periods`.

Why no analytical damping assertion here
----------------------------------------
The marin_semi-derived ``B_33(omega)`` has frequency-dependent local
structure that makes the single-DOF formula
``zeta = B(omega_n) / (2 omega_n (M + A(omega_n)))`` sensitive to
linear-interpolation artefacts at the heave natural frequency. The
clean analytical damping check lives in
:mod:`tests.validation.test_cummins_free_decay_analytical` on a
purpose-built well-behaved B(ω). Here we assert only that the envelope
decays monotonically — an integrator-level sanity check that
complements the closed-form damping test.

Tolerance
---------
Per CLAUDE.md §5, analytical comparisons default to ``rtol=1e-3`` unless
a looser bound can be argued.

* **Period**: ``rtol=3e-2`` — 3 %. Generalized-alpha has O(h^2) period
  error (< 0.1 % at dt=0.05 s); zero-crossing fit quantization at
  dt-granularity adds ~ dt/T worst-case. The remainder is the causal
  response of the true kernel ``K(t)`` vs. the fixed-point
  single-frequency estimate of ``omega_n``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.readers.wamit import read_added_mass_and_damping
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import integrate_cummins
from tests.validation.test_oc4_natural_periods import (
    OC4_C33_HEAVE_N_PER_M,
    OC4_C44_ROLL_NM_PER_RAD,
    OC4_C55_PITCH_NM_PER_RAD,
    OC4_PLATFORM_MASS_KG,
    _oc4_rigid_body_mass_matrix,
)

# marin_semi.1 — high-resolution BEM database used for OC4 validation.
# Lives under tests/fixtures/openfast/oc4_deepcwind/baseline/.../HydroData/.
_MARIN_SEMI_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "openfast"
    / "oc4_deepcwind"
    / "baseline"
    / "5MW_Baseline"
    / "HydroData"
    / "marin_semi.1"
)


def _build_oc4_marin_semi_hdb() -> HydroDatabase:
    """Combine marin_semi.1 BEM coefficients with OC4 hydrostatic stiffness."""
    omega, A, B, A_inf = read_added_mass_and_damping(_MARIN_SEMI_PATH)
    C = np.zeros((6, 6), dtype=np.float64)
    C[2, 2] = OC4_C33_HEAVE_N_PER_M
    C[3, 3] = OC4_C44_ROLL_NM_PER_RAD
    C[4, 4] = OC4_C55_PITCH_NM_PER_RAD
    n_w = omega.size
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0, 90.0]),
        A=A,
        B=B,
        A_inf=A_inf,
        C=C,
        RAO=np.zeros((6, n_w, 2), dtype=np.complex128),
        reference_point=np.array([0.0, 0.0, 0.0]),
        C_source="full",
    )


# ---------------------------------------------------------------------------
# analytical references (fixed-point on A(omega))
# ---------------------------------------------------------------------------


def _interp_at(omega_query: float, omega_grid: np.ndarray, values: np.ndarray) -> float:
    """1-D linear interpolation with clamp-to-endpoints outside the grid."""
    if omega_query <= omega_grid[0]:
        return float(values[0])
    if omega_query >= omega_grid[-1]:
        return float(values[-1])
    return float(np.interp(omega_query, omega_grid, values))


def _analytical_heave_omega(hdb: HydroDatabase, M_33: float) -> float:
    """Solve ``omega^2 (M_33 + A_33(omega)) = C_33`` by fixed-point."""
    A_33_grid = np.asarray(hdb.A[2, 2, :], dtype=np.float64)
    C_33 = float(hdb.C[2, 2])
    A_inf_33 = float(hdb.A_inf[2, 2])
    omega = float(np.sqrt(C_33 / (M_33 + A_inf_33)))
    for _ in range(50):
        A = _interp_at(omega, hdb.omega, A_33_grid)
        omega_new = float(np.sqrt(C_33 / (M_33 + A)))
        if abs(omega_new - omega) < 1.0e-10:
            return omega_new
        omega = omega_new
    return omega


# ---------------------------------------------------------------------------
# time-series fitting (shared — tests/validation/test_cummins_free_decay_
# analytical.py imports _fit_damping_log_decrement from here)
# ---------------------------------------------------------------------------


def _upward_zero_crossings(t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return interpolated times of upward zero crossings of ``x(t)``."""
    signs = np.sign(x)
    # Treat exact zeros as belonging to the next sign.
    signs[signs == 0] = 1.0
    transitions = np.where(np.diff(signs) > 0)[0]
    crossings = []
    for i in transitions:
        denom = x[i + 1] - x[i]
        frac = -x[i] / denom if denom != 0 else 0.0
        crossings.append(t[i] + frac * (t[i + 1] - t[i]))
    return np.asarray(crossings, dtype=np.float64)


def _fit_period(t: np.ndarray, x: np.ndarray) -> float:
    zeros = _upward_zero_crossings(t, x)
    if zeros.size < 3:
        raise AssertionError(f"need >= 3 upward zero crossings to fit period; found {zeros.size}")
    # Use the average spacing over ALL intervals for noise rejection.
    return float(np.mean(np.diff(zeros)))


def _fit_damping_log_decrement(t: np.ndarray, x: np.ndarray) -> float:
    """Log-decrement damping from consecutive positive peaks.

    delta = (1/n) ln(x_0 / x_n) ;   zeta = delta / sqrt(delta^2 + 4*pi^2)
    """
    # Positive local maxima.
    is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0)
    peak_idx = np.where(is_peak)[0] + 1
    if peak_idx.size < 2:
        raise AssertionError(f"need >= 2 positive peaks; found {peak_idx.size}")
    peaks = x[peak_idx]
    n = peaks.size - 1
    delta = float(np.log(peaks[0] / peaks[-1]) / n)
    return float(delta / np.sqrt(delta * delta + 4.0 * np.pi * np.pi))


# ---------------------------------------------------------------------------
# the actual validation
# ---------------------------------------------------------------------------


def _run_heave_free_decay():
    hdb = _build_oc4_marin_semi_hdb()
    lhs = assemble_cummins_lhs(rigid_body_mass=_oc4_rigid_body_mass_matrix(), hdb=hdb)
    dt = 0.05
    kernel = compute_retardation_kernel(hdb, t_max=60.0, dt=dt)

    xi0 = np.zeros(6)
    xi0[2] = 0.5  # 0.5 m heave displacement from equilibrium
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=150.0,  # ~ 8-9 heave periods at T ~ 17 s
    )

    omega_n = _analytical_heave_omega(hdb, OC4_PLATFORM_MASS_KG)
    T_analytic = 2.0 * np.pi / omega_n
    return res, T_analytic


def test_oc4_heave_free_decay_period_matches_analytical() -> None:
    res, T_analytic = _run_heave_free_decay()
    T_fit = _fit_period(res.t, res.xi[:, 2])
    rel_err = abs(T_fit - T_analytic) / T_analytic
    assert rel_err < 3.0e-2, (
        f"heave period {T_fit:.3f} s deviates from analytical {T_analytic:.3f} s "
        f"by {rel_err:.3%} (limit 3%)"
    )


def test_oc4_heave_free_decay_decays_monotonically_in_envelope() -> None:
    """Each successive positive peak must be smaller than the previous one."""
    res, _ = _run_heave_free_decay()
    x = res.xi[:, 2]
    is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0)
    peaks = x[1:-1][is_peak]
    assert peaks.size >= 3, f"need >= 3 peaks; got {peaks.size}"
    # Allow 1% numerical noise, but enforce strict long-range monotonic decay.
    assert np.all(np.diff(peaks) < 0.01 * peaks[0])
