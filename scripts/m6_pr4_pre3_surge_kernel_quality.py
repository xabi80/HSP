"""M6 PR4 Pre-3 Option E diagnostic -- per-DOF kernel quality on marin_semi.

The post-fix-wamit-dimensionalisation BEM data has surge B(omega_max)
at 1.7% of peak (vs 0.23% pre-fix when the data was incorrectly
non-dim). The kernel's Refinement-2 Check 1 input-proxy gate fires
on this; the question is whether the gate is catching a real
problem (kernel does NOT decay cleanly) or a proxy problem (gate
is conservative, kernel decays cleanly post-tail-extension).

This script bypasses the Check 1 gate by calling the integration
internals directly, computes K(t) per DOF, and plots / reports
the decay ratios:

    |K_ii(t = 200 s)| / max_t |K_ii(t)|
    |K_ii(t = 400 s)| / max_t |K_ii(t)|

If the surge kernel decays cleanly (ratios < ~1% at t = 200 s),
Option E is justified: the input gate's proxy is too strict, the
post-extension kernel is the right thing to gate on.

If the surge kernel shows artifacts (sustained oscillation, or
ratios >> 1% at long lag), the input gate is correctly conservative
and the marin_semi grid genuinely fails to support clean kernel
extension for surge.

Run from the repo root:
    python scripts/m6_pr4_pre3_surge_kernel_quality.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from floatsim.hydro._filon import (  # noqa: E402
    compute_tail_contribution,
    filon_trap_cosine,
    fit_per_entry_tail_constants,
)
from floatsim.hydro.readers.wamit import read_added_mass_and_damping  # noqa: E402

MARIN_SEMI = (
    REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/baseline/5MW_Baseline/HydroData/marin_semi.1"
)
DIAG_PNG = REPO_ROOT / "docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png"

DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
RESTORED = (False, False, True, True, True, False)  # OC4 unmoored: only heave/roll/pitch

# Mirror the production kernel-config defaults from retardation.py.
TAIL_FIT_POINTS: int = 10
TAIL_UPPER_BOUND_FACTOR: float = 5.0


def _build_kernel_internal(omega: np.ndarray, B: np.ndarray, t_arr: np.ndarray) -> np.ndarray:
    """Bypass Check 1 / Check 2 gates: compute K(t) for the full 6x6 B stack.

    Returns shape ``(6, 6, n_t)``. No skip_tail_mask -- every entry
    gets its full per-entry tail contribution, so the diagnostic
    measures the kernel as it would be IF the gates didn't fire.
    """
    # Prepend B(omega=0)=0 if needed (matches production behaviour).
    if omega[0] > 1.0e-12:
        omega = np.concatenate([[0.0], omega])
        B = np.concatenate([np.zeros((6, 6, 1), dtype=np.float64), B], axis=2)

    K_in = filon_trap_cosine(omega, B, t_arr)
    C_tail = fit_per_entry_tail_constants(omega, B, n_tail_points=TAIL_FIT_POINTS)
    K_tail = compute_tail_contribution(
        C_tail,
        float(omega[-1]),
        t_arr,
        upper_bound_factor=TAIL_UPPER_BOUND_FACTOR,
    )
    return (2.0 / np.pi) * (K_in + K_tail)


def _b_max_at_omega_max_ratio(B: np.ndarray, dof: int) -> float:
    """Pre-extension Check 1 ratio: |B[i,i](omega_max)| / max|B[i,i]|."""
    diag = np.abs(B[dof, dof, :])
    peak = float(np.max(diag))
    if peak < 1.0e-30:
        return float("nan")
    return float(diag[-1] / peak)


def _kernel_decay_ratios(K_ii: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """Post-extension diagnostics: |K(t_query)| / max|K| at t = 200, 400 s."""
    peak = float(np.max(np.abs(K_ii)))
    if peak < 1.0e-30:
        return {"peak": peak, "r200": float("nan"), "r400": float("nan")}
    idx_200 = int(np.argmin(np.abs(t - 200.0)))
    idx_400 = int(np.argmin(np.abs(t - 400.0)))
    return {
        "peak": peak,
        "r200": float(np.abs(K_ii[idx_200]) / peak),
        "r400": float(np.abs(K_ii[idx_400]) / peak),
    }


def main() -> None:
    print("M6 PR4 Pre-3 Option E -- per-DOF kernel quality on marin_semi")
    print("=" * 78)
    print(f"BEM: {MARIN_SEMI.relative_to(REPO_ROOT)}")
    print()

    omega, _A, B, _A_inf = read_added_mass_and_damping(MARIN_SEMI)
    print(f"omega range: [{omega[0]:.3f}, {omega[-1]:.3f}] rad/s, n = {omega.size}")
    print("WAMIT dimensionalisation applied (default assume_dimensional=False).")
    print()

    # Compute kernel out to t = 500 s with dt = 0.05 s.
    dt = 0.05
    t_max = 500.0
    n_t = round(t_max / dt) + 1
    t = dt * np.arange(n_t, dtype=np.float64)
    K = _build_kernel_internal(omega, B, t)

    # Per-DOF report.
    rows = []
    print(
        f"{'i':>2} {'DOF':>6} {'restored':>9} {'B(om)/peak':>11}  "
        f"{'K_peak':>14}  {'K(200)/peak':>13}  {'K(400)/peak':>13}  {'verdict':>8}"
    )
    print("-" * 90)
    for i in range(6):
        b_ratio = _b_max_at_omega_max_ratio(B, i)
        k_diag = _kernel_decay_ratios(K[i, i, :], t)
        # Verdict: kernel decays cleanly if r200 < 1% AND r400 < 0.5%.
        clean = k_diag["r200"] < 1.0e-2 and k_diag["r400"] < 5.0e-3
        verdict = "CLEAN" if clean else "ARTIFACT"
        peak_disp = f"{k_diag['peak']:.3e}"
        r200_disp = f"{k_diag['r200']:.4e}" if not np.isnan(k_diag["r200"]) else "n/a"
        r400_disp = f"{k_diag['r400']:.4e}" if not np.isnan(k_diag["r400"]) else "n/a"
        b_ratio_disp = f"{b_ratio:.4e}" if not np.isnan(b_ratio) else "n/a"
        print(
            f"{i:>2} {DOF_NAMES[i]:>6} {RESTORED[i]!s:>9} {b_ratio_disp:>11}  "
            f"{peak_disp:>14}  {r200_disp:>13}  {r400_disp:>13}  {verdict:>8}"
        )
        rows.append((i, DOF_NAMES[i], RESTORED[i], b_ratio, k_diag, clean))

    # Plot K_ii(t) for all 6 DOFs with peak-normalised amplitude.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for i, ax in enumerate(axes.flat):
        k_ii = K[i, i, :]
        peak = float(np.max(np.abs(k_ii))) or 1.0
        ax.plot(t, k_ii / peak, "k", lw=0.8)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axhline(0.01, color="red", ls=":", lw=0.5, label="±1 % of peak")
        ax.axhline(-0.01, color="red", ls=":", lw=0.5)
        ax.set_xlim(0.0, t[-1])
        ax.set_ylim(-0.2, 1.0)
        ax.set_title(
            f"{i}: {DOF_NAMES[i]}  (restored={RESTORED[i]})\n"
            f"K_peak={peak:.3e}, K(200)/peak={rows[i][4]['r200']:.2e}"
        )
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        if i // 3 == 1:
            ax.set_xlabel("t [s]")
        if i % 3 == 0:
            ax.set_ylabel("K_ii(t) / peak")
    fig.suptitle(
        "Per-DOF retardation kernel on marin_semi (post-WAMIT-dim-fix, " "Filon + 1/omega^4 tail)"
    )
    fig.tight_layout()
    fig.savefig(DIAG_PNG, dpi=110)
    plt.close(fig)
    print()
    print(f"-> {DIAG_PNG.relative_to(REPO_ROOT)}")
    print()

    # Disposition summary.
    n_blocked = sum(1 for _, _, _, b_r, _, _ in rows if b_r > 1.0e-2)
    n_clean_post = sum(1 for _, _, _, _, _, c in rows if c)
    n_blocked_artifact = sum(1 for _, _, _, b_r, _, c in rows if b_r > 1.0e-2 and not c)
    n_blocked_clean = sum(1 for _, _, _, b_r, _, c in rows if b_r > 1.0e-2 and c)

    print(f"Pre-extension Check 1 gate (1% B/peak): {n_blocked} of 6 DOFs blocked.")
    print(
        f"Post-extension kernel decay (CLEAN: r200 < 1%, r400 < 0.5%): "
        f"{n_clean_post} of 6 DOFs clean."
    )
    print()
    if n_blocked_artifact == 0 and n_blocked_clean > 0:
        print("DECISION TREE -- Case 1: kernels of blocked DOFs decay cleanly post-")
        print("extension. The input-proxy gate is too strict; Option E is justified.")
        print()
        print("Suggested implementation: replace Check 1 with a post-extension")
        print("check on K(t_max)/max(K) < 1% (matching the existing decay-")
        print("diagnostic warning at retardation.py:_emit_decay_diagnostic).")
    elif n_blocked_artifact > 0 and n_blocked_clean == 0:
        print("DECISION TREE -- Case 2: kernels of blocked DOFs show artifacts.")
        print("The input gate is correctly catching a real problem. Loosening")
        print("would propagate artifacts into off-diagonal K(t) and contaminate")
        print("mode coupling. Option B (skip for unrestored DOFs) is UNSAFE; the")
        print("right fix is to extend the BEM grid (Option C, expensive).")
    else:
        print(f"DECISION TREE -- Case 3 (mixed): {n_blocked_clean} blocked DOFs decay")
        print(f"cleanly, {n_blocked_artifact} show artifacts. Per-DOF treatment needed.")


if __name__ == "__main__":
    main()
