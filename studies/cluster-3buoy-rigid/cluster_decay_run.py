"""Step 4 + 5(run): static equilibrium gate + both cluster free decays."""

from __future__ import annotations

import json
from pathlib import Path

import cluster_study_common as sc
import numpy as np

from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.newmark import integrate_cummins

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_DOF = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def _save_csv(path: Path, res) -> None:
    header = "t," + ",".join(_DOF) + ",heave_vel"
    np.savetxt(path, np.column_stack([res.t, res.xi, res.xi_dot[:, 2]]),
               delimiter=",", header=header, comments="")


def main() -> None:
    _RESULTS.mkdir(exist_ok=True)
    print("=" * 70)
    print("Cluster Step 4 + 5: equilibrium gate + free decay")
    print("=" * 70)
    hdb = sc.load_hdb()
    lhs = sc.build_lhs(hdb)
    kernel = sc.build_kernel(hdb)
    print(f"  (M+A_inf)[heave] = {lhs.M_plus_Ainf[2,2]:.4f} kg "
          f"(98.01 + A33_comp)")
    print(f"  C[heave]         = {lhs.C[2,2]:.4f} N/m")

    # Step 4: static equilibrium.
    print("\n[Step 4] static equilibrium (BEM-only) ...")
    eq = solve_static_equilibrium(lhs=lhs, state_force=None)
    z_eq = float(eq.xi_eq[2])
    print(f"  converged: {eq.converged}; z_eq = {z_eq:+.5e} m  GATE |z_eq|<0.01")
    (_RESULTS / "equilibrium.json").write_text(json.dumps(
        {"converged": bool(eq.converged), "z_eq_m": z_eq,
         "xi_eq": eq.xi_eq.tolist(),
         "gate_pass": bool(abs(z_eq) < 0.01)}, indent=2))
    if abs(z_eq) >= 0.01:
        raise SystemExit(f"STOP (Step 4 gate): |z_eq| = {abs(z_eq):.3e} >= 0.01")
    print("  GATE PASSED.")

    # Step 5: both decays.
    xi0 = eq.xi_eq.copy()
    xi0[2] += sc.IC_HEAVE
    xd0 = np.zeros(6)
    print(f"\n[Step 5] decays from heave = {xi0[2]:+.4f} m")

    print("  (1) BEM-only ...")
    r1 = integrate_cummins(lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=xd0,
                           duration=sc.DURATION, dt=sc.DT)
    _save_csv(_RESULTS / "decay_bem_only.csv", r1)
    print(f"      heave[-1] = {r1.xi[-1,2]:+.4e}")

    print("  (2) BEM + Morison ...")
    r2 = integrate_cummins(lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=xd0,
                           duration=sc.DURATION, dt=sc.DT,
                           state_force=sc.make_morison_force())
    _save_csv(_RESULTS / "decay_bem_morison.csv", r2)
    print(f"      heave[-1] = {r2.xi[-1,2]:+.4e}")

    for label, r in (("bem_only", r1), ("bem_morison", r2)):
        if not np.all(np.isfinite(r.xi)):
            raise SystemExit(f"STOP: non-finite in {label}")
        if np.max(np.abs(r.xi[:, 2])) > 10 * sc.IC_HEAVE:
            raise SystemExit(f"STOP: {label} heave grew >10x IC")
    print("\n  Both decays finite and bounded.")


if __name__ == "__main__":
    main()
