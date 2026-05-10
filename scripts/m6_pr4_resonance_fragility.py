"""M6 PR4 — heave-resonance-peak fragility quantification.

Per Group B-(a) of the post-PR4 disposition: compute heave RAO peak
amplitude using FloatSim's impedance pipeline with B(omega) interpolated
three ways and report whether peak amplitudes vary substantially
across schemes. If yes → F-RESONANCE-PEAK-FRAGILITY is empirically
confirmed; if no → revisit bug-vs-fragility classification.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from floatsim.hydro.readers.wamit import (  # noqa: E402
    read_added_mass_and_damping,
    read_excitation_force,
)
from tests.validation.test_m6_openfast_free_decay import (  # noqa: E402
    _MARIN_SEMI as MARIN_PATH,
)
from tests.validation.test_m6_openfast_free_decay import _build_setup_b  # noqa: E402

S2_DECK = REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/inputs/s2_pitch_decay"
HEAVE_DOF = 2


def _impedance_amp_at_omega(
    omega_query: float,
    omega_grid: np.ndarray,
    M: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    F_exc: np.ndarray,
    interp: str,
) -> float:
    """Compute |xi[heave]| at omega_query via Z^-1 F_exc with three
    different interpolation schemes for B(omega).

    A and F_exc are interpolated linearly in all schemes (the question
    is specifically about B at the resonance peak). M and C are
    constants (no omega dependence).
    """
    if interp == "linear":
        B_q = float(np.interp(omega_query, omega_grid, B[HEAVE_DOF, HEAVE_DOF, :]))
    elif interp == "cubic":
        cs = CubicSpline(omega_grid, B[HEAVE_DOF, HEAVE_DOF, :])
        B_q = float(cs(omega_query))
    elif interp == "nearest":
        idx = int(np.argmin(np.abs(omega_grid - omega_query)))
        B_q = float(B[HEAVE_DOF, HEAVE_DOF, idx])
    else:
        raise ValueError(f"unknown interp: {interp}")
    A_q = float(np.interp(omega_query, omega_grid, A[HEAVE_DOF, HEAVE_DOF, :]))
    F_re = float(np.interp(omega_query, omega_grid, F_exc[HEAVE_DOF, :, 0].real))
    F_im = float(np.interp(omega_query, omega_grid, F_exc[HEAVE_DOF, :, 0].imag))
    F_q = complex(F_re, F_im)
    Z = (
        -(omega_query**2) * (M[HEAVE_DOF, HEAVE_DOF] + A_q)
        + 1j * omega_query * B_q
        + C[HEAVE_DOF, HEAVE_DOF]
    )
    return float(np.abs(F_q / Z))


def main() -> None:
    print("M6 PR4 -- heave-resonance-peak fragility quantification")
    print("=" * 72)
    print()
    setup = _build_setup_b(S2_DECK)
    M = setup.M
    C = setup.C
    omega, A, B, _A_inf = read_added_mass_and_damping(MARIN_PATH)
    _headings, F_exc = read_excitation_force(MARIN_PATH.parent / "marin_semi.3", omega=omega)

    # Find heave natural frequency by fixed point.
    omega_n = float(
        np.sqrt(C[HEAVE_DOF, HEAVE_DOF] / (M[HEAVE_DOF, HEAVE_DOF] + A[HEAVE_DOF, HEAVE_DOF, 0]))
    )
    for _ in range(50):
        A_n = float(np.interp(omega_n, omega, A[HEAVE_DOF, HEAVE_DOF, :]))
        omega_new = float(np.sqrt(C[HEAVE_DOF, HEAVE_DOF] / (M[HEAVE_DOF, HEAVE_DOF] + A_n)))
        if abs(omega_new - omega_n) < 1e-12:
            break
        omega_n = omega_new
    T_n = 2 * np.pi / omega_n
    B_at_n_linear = float(np.interp(omega_n, omega, B[HEAVE_DOF, HEAVE_DOF, :]))
    A_at_n = float(np.interp(omega_n, omega, A[HEAVE_DOF, HEAVE_DOF, :]))
    M_plus_A = M[HEAVE_DOF, HEAVE_DOF] + A_at_n
    crit = 2.0 * np.sqrt(M_plus_A * C[HEAVE_DOF, HEAVE_DOF])
    zeta = B_at_n_linear / crit

    print(f"Heave natural period T_n = {T_n:.4f} s, omega_n = {omega_n:.4f} rad/s")
    print(f"M+A(omega_n) = {M_plus_A:.4e} kg, B(omega_n)_linear = {B_at_n_linear:.4e} N*s/m")
    print(f"Radiation-only zeta = {zeta * 100:.4f} %")
    print()

    # Sweep across the +/- 15 % band around omega_n at 5 sample omegas + omega_n itself.
    relative_offsets = (-0.15, -0.10, -0.05, 0.0, +0.05, +0.10, +0.15)
    print(
        f"{'rel. offset':>11}  {'omega':>9}  {'T':>7}  "
        f"{'amp linear':>11}  {'amp cubic':>11}  {'amp nearest':>11}  "
        f"{'span %':>8}"
    )
    print("-" * 80)
    max_span = 0.0
    for rel in relative_offsets:
        omega_q = omega_n * (1.0 + rel)
        T_q = 2 * np.pi / omega_q
        amp_lin = _impedance_amp_at_omega(omega_q, omega, M, A, B, C, F_exc, "linear")
        amp_cub = _impedance_amp_at_omega(omega_q, omega, M, A, B, C, F_exc, "cubic")
        amp_near = _impedance_amp_at_omega(omega_q, omega, M, A, B, C, F_exc, "nearest")
        amps = [amp_lin, amp_cub, amp_near]
        span_pct = (max(amps) - min(amps)) / max(amps) * 100.0
        max_span = max(max_span, span_pct)
        print(
            f"{rel*100:>+10.1f}%  {omega_q:>9.4f}  {T_q:>7.3f}  "
            f"{amp_lin:>11.4e}  {amp_cub:>11.4e}  {amp_near:>11.4e}  "
            f"{span_pct:>7.2f}%"
        )

    print()
    print(f"Maximum span across the +/- 15 % band: {max_span:.2f} %")
    print()
    if max_span > 10.0:
        print("VERDICT: F-RESONANCE-PEAK-FRAGILITY confirmed empirically. Peak")
        print("amplitude varies by >10% across linear/cubic/nearest interpolation")
        print("schemes. Lightly-damped resonance peaks are not bug-suitable for")
        print("tight cross-checks across tools.")
    elif max_span > 1.0:
        print("VERDICT: marginal fragility. Span 1-10%; some sensitivity but not")
        print("dominant. Bug-vs-fragility classification ambiguous; default to")
        print("fragility but flag for follow-up if cross-tool gap exceeds span.")
    else:
        print("VERDICT: NOT fragile (span < 1%). Revisit bug-vs-fragility")
        print("classification -- the heave-resonance gap may be a real bug.")


if __name__ == "__main__":
    main()
