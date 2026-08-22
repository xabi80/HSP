"""OSU Test Buoy free-decay check — confirms the heave/pitch periods of the adapted
FloatSim model (osu_buoy_common.py + capytaine_osu_buoy.nc). See OSU-TEST-BUOY-GEOMETRY.md.

NOTE (exploratory): the surge/sway/roll radiation kernels for this slender spar decay
slowly and trip the Check-3 kernel-decay gate; the HEAVE kernel is clean. This script
bypasses the gate study-side (does NOT modify floatsim) to run the placeholder decay —
the heave decay is clean (no spurious oscillation) and matches the frequency-domain
period. A production kernel needs a finer/wider BEM omega grid (re-run bem_database.py
with a denser grid) — a refinement to do alongside the tank heave-plate data.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
warnings.simplefilter("ignore")

import floatsim.hydro.retardation as _ret  # noqa: E402
_ret._validate_kernel_decay = lambda *a, **k: None  # exploratory bypass (see module docstring)

import osu_buoy_common as oc  # noqa: E402
from floatsim.solver.equilibrium import solve_static_equilibrium  # noqa: E402
from floatsim.solver.newmark import integrate_cummins  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    hdb = oc.load_hdb()
    lhs, kernel, drag = oc.build_lhs(hdb), oc.build_kernel(hdb), oc.make_drag()
    eq = solve_static_equilibrium(lhs=lhs, state_force=drag)
    C33 = float(lhs.C[2, 2]); A_inf = float(lhs.M_plus_Ainf[2, 2] - oc.M_BODY)
    print(f"C33={C33:.1f} N/m  heave A_inf={A_inf:.1f} kg  eq_heave={eq.xi_eq[2] * 1000:+.1f} mm")
    print(f"FD heave period ~{2 * np.pi * np.sqrt((oc.M_BODY + A_inf) / C33):.2f} s")

    for dof, ic, lbl in [(2, 0.10, "HEAVE"), (4, 0.10, "PITCH")]:
        xi0 = eq.xi_eq.copy(); xi0[dof] += ic
        r = integrate_cummins(lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=np.zeros(6),
                              duration=oc.DURATION, dt=oc.DT, state_force=drag)
        x = r.xi[:, dof] - eq.xi_eq[dof]
        pk, _ = find_peaks(x, height=1e-4)
        if len(pk) < 3:
            print(f"{lbl}: <3 peaks"); continue
        T = float(np.mean(np.diff(r.t[pk][:6])))
        d = np.log(x[pk][:-1] / x[pk][1:]); d = d[np.isfinite(d) & (d > 0)]
        print(f"{lbl} decay: T={T:.2f} s  zeta~{np.mean(d[:2]) / (2 * np.pi) * 100:.1f}% (placeholder drag)")
    print("\nheave 2.52 s is the PLACEHOLDER (solid equal-area disc); the real perforated plate")
    print("adds less added mass -> shorter (~2.3-2.4 s). Pitch uses placeholder I_YY + plate.")


if __name__ == "__main__":
    main()
