"""PR2 sanity: confirm BOTH the articulated and whole-chain-rigid 16-buoy decks
assemble and solve, on a single short regular-wave case. Prints the constraint
count, platform + buoy7 heave RAO, and settle status for each. Not a converged
RAO -- just an end-to-end smoke test before the full sweep.

Run from the local mirror for speed (see STUDY-PLAN.md / the OneDrive note).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import platform16_rao as prp16  # noqa: E402
import pvr_common as pvr  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore")
    hdb = pvr.load_hdb("0215")
    hydro_dof = prp16._hydro_dof(pvr.build_deck(False))
    print(f"16-buoy: {prp16._N_DOF} DOF, {len(hydro_dof)} hydro DOF; "
          f"single case H=0.10 m, T=2.5 s\n")

    for rigid, label in [(False, "ARTICULATED (yaw_locked)"), (True, "RIGID (whole chain weld)")]:
        setup = pvr.build_setup(rigid, hdb)
        n_con = setup.constraints.n_constraints
        c = prp16.run_case(
            setup, hdb, hydro_dof, height_m=0.10, period_s=2.5,
            ramp_s=8.0, cap_settle_s=12.0, window_periods=3.0, dt=0.01,
        )
        rao_p = c["rao"]["platform_heave"]
        rao_b = c["rao"]["buoy7_heave"]
        finite = np.isfinite([rao_p, rao_b]).all()
        print(f"{label}: {n_con} constraints ({n_con // 20}/joint x 20 joints)")
        print(f"   platform heave RAO = {rao_p:.4f} | buoy7 heave RAO = {rao_b:.4f} | "
              f"finite={finite} settled={c['settled']} dur={c['duration_s']:.1f}s "
              f"steps={c['n_steps']}\n")
    print("sanity OK -- both decks assemble and integrate")


if __name__ == "__main__":
    main()
