"""M6 PR3 Mod 4 -- kernel t_max convergence diagnostic.

Locked workflow: don't carry over t_max=200s from the broken-kernel
diagnostic. Verify on the *fixed* kernel that t_max=200s actually
converges the pitch damping, otherwise raise t_max until it does.

Procedure:
  - Load marin_semi.1 BEM via the M5 WAMIT reader.
  - Combine with hand-built OC4 stiffness (Robertson 2014 Table 3-3).
  - Use the OC4 platform mass matrix (Robertson Table 3-1).
  - For t_max in {100, 150, 200, 300} s, compute the Cummins free-decay
    response from a 5-deg pitch IC, fit log-decrement damping ratio.
  - Assert rtol < 1e-3 between t_max=200 and t_max=300.
  - If 200s suffices, lock 200s. Else step up to 400s, etc.

Run from the repo root:
    python scripts/m6_pr3_mod4_tmax_convergence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# scripts/ live outside the floatsim package; add repo root for `tests.*` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.readers.wamit import read_added_mass_and_damping
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import integrate_cummins
from tests.validation.test_oc4_natural_periods import (
    OC4_C33_HEAVE_N_PER_M,
    OC4_C44_ROLL_NM_PER_RAD,
    OC4_C55_PITCH_NM_PER_RAD,
    _oc4_rigid_body_mass_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MARIN_SEMI_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "openfast"
    / "oc4_deepcwind"
    / "baseline"
    / "5MW_Baseline"
    / "HydroData"
    / "marin_semi.1"
)


def _build_oc4_marin_semi_hdb() -> HydroDatabase:
    """marin_semi.1 BEM + OC4 hand-authored hydrostatic stiffness."""
    omega, A, B, A_inf = read_added_mass_and_damping(MARIN_SEMI_PATH)
    C = np.zeros((6, 6), dtype=np.float64)
    C[2, 2] = OC4_C33_HEAVE_N_PER_M
    C[3, 3] = OC4_C44_ROLL_NM_PER_RAD
    C[4, 4] = OC4_C55_PITCH_NM_PER_RAD
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0, 90.0]),
        A=A,
        B=B,
        A_inf=A_inf,
        C=C,
        RAO=np.zeros((6, omega.size, 2), dtype=np.complex128),
        reference_point=np.array([0.0, 0.0, 0.0]),
        C_source="full",
    )


def _fit_pitch_period_and_zeta(t: np.ndarray, pitch: np.ndarray) -> tuple[float, float]:
    """Period (s) from upward zero crossings + log-decrement zeta from peaks."""
    # Subtract any equilibrium offset before fitting.
    pitch_zm = pitch - float(np.mean(pitch[-int(60.0 / (t[1] - t[0])) :]))

    # Upward zero crossings -> period.
    signs = np.sign(pitch_zm)
    signs[signs == 0] = 1
    zero_idx = np.where(np.diff(signs) > 0)[0]
    if zero_idx.size < 6:
        raise AssertionError(f"need >= 6 zero crossings; got {zero_idx.size}")
    t_z = t[zero_idx] + (t[zero_idx + 1] - t[zero_idx]) * (
        -pitch_zm[zero_idx] / (pitch_zm[zero_idx + 1] - pitch_zm[zero_idx])
    )
    period = float(np.mean(np.diff(t_z[:6])))

    # Positive peaks -> log-decrement.
    is_peak = (
        (pitch_zm[1:-1] > pitch_zm[:-2]) & (pitch_zm[1:-1] > pitch_zm[2:]) & (pitch_zm[1:-1] > 0)
    )
    peak_idx = np.where(is_peak)[0] + 1
    peaks = pitch_zm[peak_idx]
    if peaks.size < 6:
        raise AssertionError(f"need >= 6 positive peaks; got {peaks.size}")
    n = 5
    delta = float(np.log(peaks[0] / peaks[n]) / n)
    zeta = float(delta / np.sqrt(delta * delta + 4.0 * np.pi * np.pi))
    return period, zeta


def _run_floatsim_pitch_decay(t_max_kernel: float) -> tuple[float, float]:
    """One Cummins free-decay run with the given kernel truncation."""
    hdb = _build_oc4_marin_semi_hdb()
    M_rigid = _oc4_rigid_body_mass_matrix()
    lhs = assemble_cummins_lhs(rigid_body_mass=M_rigid, hdb=hdb)

    dt = 0.05  # match OpenFAST sample rate
    kernel = compute_retardation_kernel(hdb, t_max=t_max_kernel, dt=dt)

    xi0 = np.zeros(6)
    xi0[4] = np.deg2rad(5.0)  # pitch IC
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=600.0,  # match OpenFAST reference (12 cycles at T~26.8s)
        rho_inf=1.0,
    )

    period, zeta = _fit_pitch_period_and_zeta(res.t, res.xi[:, 4])
    return period, zeta


def main() -> None:
    print("M6 PR3 Mod 4 -- kernel t_max convergence (post-fix kernel)")
    print("=" * 64)
    print()
    print(f"{'t_max [s]':>10}  {'period [s]':>11}  {'zeta':>10}")
    print("-" * 64)

    results: list[tuple[float, float, float]] = []
    for t_max in (100.0, 150.0, 200.0, 300.0):
        period, zeta = _run_floatsim_pitch_decay(t_max)
        print(f"{t_max:>10.0f}  {period:>11.4f}  {zeta:>10.5f}")
        results.append((t_max, period, zeta))

    print()
    # Pin against t_max=300 (longest tested).
    period_ref, zeta_ref = results[-1][1], results[-1][2]
    print(f"Reference (t_max=300s): period={period_ref:.4f} s, zeta={zeta_ref:.5f}")
    print()
    print(f"{'t_max':>10}  {'rel-err period':>16}  {'rel-err zeta':>14}")
    print("-" * 64)
    for t_max, period, zeta in results:
        rerr_T = abs(period - period_ref) / period_ref
        rerr_z = abs(zeta - zeta_ref) / abs(zeta_ref) if abs(zeta_ref) > 1e-9 else 0.0
        print(f"{t_max:>10.0f}  {rerr_T:>16.4e}  {rerr_z:>14.4e}")

    # Locking decision: smallest t_max where rel-err vs t_max=300 is < 1e-3
    # on BOTH period and zeta.
    print()
    rtol = 1.0e-3
    for t_max, period, zeta in results[:-1]:
        rerr_T = abs(period - period_ref) / period_ref
        rerr_z = abs(zeta - zeta_ref) / abs(zeta_ref) if abs(zeta_ref) > 1e-9 else 0.0
        if rerr_T < rtol and rerr_z < rtol:
            print(f"LOCKED: t_max = {t_max:.0f} s converges within rtol={rtol:.0e}")
            return
    print(f"WARNING: no t_max < 300 s converges within rtol={rtol:.0e}; lock 300s.")


if __name__ == "__main__":
    main()
