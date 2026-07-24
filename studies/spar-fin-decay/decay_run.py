"""Steps D + E: static equilibrium + both free-decay integrations.

Step D: solve_static_equilibrium on the BEM-only LHS. The eqdraft mesh
        translation placed the waterline so equilibrium is ~0 by
        construction; the gate |z_eq| < 0.01 m verifies it.
Step E: integrate both decays (BEM-only, BEM+Morison) from
        xi_eq + 0.10 m heave. Writes results/decay_bem_only.csv and
        results/decay_bem_morison.csv (t + 6 DOF + heave velocity).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import study_common as sc

from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.newmark import integrate_cummins

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_DOF_NAMES = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def _save_csv(path: Path, res) -> None:
    header = "t," + ",".join(_DOF_NAMES) + ",heave_vel"
    data = np.column_stack([res.t, res.xi, res.xi_dot[:, 2]])
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main() -> None:
    _RESULTS.mkdir(exist_ok=True)
    print("=" * 70)
    print("Steps D + E: equilibrium + free decay (eqdraft BEM)")
    print("=" * 70)

    hdb = sc.load_hdb()
    lhs = sc.build_lhs(hdb)
    kernel = sc.build_kernel(hdb)
    print(f"  (M+A_inf)[heave] = {lhs.M_plus_Ainf[2,2]:.4f} kg")
    print(f"  C[heave]         = {lhs.C[2,2]:.4f} N/m")

    # --- Step D: static equilibrium (BEM-only, no state force) ---
    print("\n[Step D] static equilibrium (no moorings, no state force) ...")
    eq = solve_static_equilibrium(lhs=lhs, state_force=None)
    z_eq = float(eq.xi_eq[2])
    print(f"  converged: {eq.converged}, residual_norm: {eq.residual_norm:.3e} N")
    print(f"  xi_eq = {np.array2string(eq.xi_eq, precision=4)}")
    print(f"  z_eq (heave) = {z_eq:+.5e} m   GATE |z_eq| < 0.01 m")
    gate_ok = abs(z_eq) < 0.01
    (_RESULTS / "equilibrium.json").write_text(
        json.dumps(
            {
                "converged": bool(eq.converged),
                "residual_norm_N": eq.residual_norm,
                "xi_eq": eq.xi_eq.tolist(),
                "z_eq_m": z_eq,
                "gate_abs_z_lt_0p01": bool(gate_ok),
                "note": (
                    "Equilibrium ~0 by construction: the eqdraft mesh was "
                    "translated so the free-floating waterline is at z=0. "
                    "Gate verifies the translation."
                ),
            },
            indent=2,
        )
    )
    if not gate_ok:
        raise SystemExit(
            f"STOP (Step D gate): |z_eq| = {abs(z_eq):.3e} m >= 0.01 m. "
            f"The mesh translation did not place equilibrium at 0."
        )
    print("  GATE PASSED.")

    # --- Step E: both decays from xi_eq + 0.10 m heave ---
    xi0 = eq.xi_eq.copy()
    xi0[2] += sc.IC_HEAVE
    xd0 = np.zeros(6, dtype=np.float64)
    print(f"\n[Step E] decay IC: heave = {xi0[2]:+.4f} m (xi_eq + {sc.IC_HEAVE})")

    print("  (1) BEM-only decay ...")
    res_bem = integrate_cummins(
        lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=xd0,
        duration=sc.DURATION, dt=sc.DT,
    )
    _save_csv(_RESULTS / "decay_bem_only.csv", res_bem)
    print(f"      steps: {res_bem.t.size}, "
          f"heave[0]={res_bem.xi[0,2]:+.4f}, heave[-1]={res_bem.xi[-1,2]:+.4e}")

    print("  (2) BEM + Morison heave-plate decay ...")
    morison = sc.make_morison_force()
    res_mor = integrate_cummins(
        lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=xd0,
        duration=sc.DURATION, dt=sc.DT, state_force=morison,
    )
    _save_csv(_RESULTS / "decay_bem_morison.csv", res_mor)
    print(f"      steps: {res_mor.t.size}, "
          f"heave[0]={res_mor.xi[0,2]:+.4f}, heave[-1]={res_mor.xi[-1,2]:+.4e}")

    # NaN / instability guard.
    for label, res in (("bem_only", res_bem), ("bem_morison", res_mor)):
        if not np.all(np.isfinite(res.xi)):
            raise SystemExit(f"STOP: non-finite values in {label} decay.")
        if np.max(np.abs(res.xi[:, 2])) > 10.0 * sc.IC_HEAVE:
            raise SystemExit(f"STOP: {label} heave grew > 10x IC (instability).")
    print("\n  Both decays finite and bounded. CSVs written to results/.")


if __name__ == "__main__":
    main()
