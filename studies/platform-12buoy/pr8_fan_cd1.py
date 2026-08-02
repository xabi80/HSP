"""M11b PR8 follow-up: the SAME in-band amplitude fan (7 periods x 5 heights)
at plate Cd_n = 1.0, for side-by-side comparison with the committed operational
Cd_n = 5.0 fan (pr8_fan_out/). Reuses platform_rao_pilot verbatim -- only the
plate drag coefficient and the output directory change -- so the per-case CSVs
(heave + vertical/Nz acceleration channels) are byte-format-identical to the
Cd=5 fan and the comparison plotter can read both the same way.

Cd_n=1.0 is the light-drag end already characterised in the PR8 Cd sweep
(cd_check.csv / cd_peak_pin.csv); this extends it to the full (H, T) grid.
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))

import platform_rao_pilot as prp  # noqa: E402
from floatsim.driver import build_system  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402

prp._PLATE_CD_N = 1.0  # the only physics change vs the committed fan
_OUT = _HERE / "pr8_fan_cd1_out"


def main() -> None:
    _OUT.mkdir(exist_ok=True)
    deck = prp._deck_with_drag()  # reads _PLATE_CD_N = 1.0
    hydro_dof = prp._hydro_dof(deck)
    shared = read_capytaine(prp._PLAT_NC)
    hdb_force = read_capytaine(prp._PLAT_NC)
    dt = 0.01
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setup = build_system(
            deck, bem_databases={}, dt=dt, t_max_kernel=30.0, solve_equilibrium=False,
            shared_hydro_database=shared, asymptote_check_override=prp._ASYMPTOTE_OVR,
            kernel_decay_floor_override=prp._KERNEL_EXEMPT,
        )

    fan_heights = [0.05, 0.15, 0.30, 0.60, 1.00]
    fan_periods = [2.0, 2.5, 2.8, 3.0, 3.141, 3.257, 3.3]
    matrix = [(h, p, 20.0, 450.0, 8.0 if p < 2.6 else 6.0)
              for p in fan_periods for h in fan_heights]

    summary_rows: list[dict] = []
    for i, (h, p, r, s, wp) in enumerate(matrix):
        print(f"[{i + 1}/{len(matrix)}] Cd_n=1.0  H={h} m  T={p} s  (cap {s}s)...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            case = prp.run_case(setup, hdb_force, hydro_dof, height_m=h, period_s=p,
                                ramp_s=r, cap_settle_s=s, window_periods=wp, dt=dt)
        tag = f"H{h:g}_T{p:g}".replace(".", "p")
        prp._write_case_csv(_OUT / f"case_{tag}.csv", case)
        row = {
            "height_m": h, "period_s": p, "omega": case["omega"], "amp_m": case["amp_m"],
            "settle_ratio": case["settle_ratio"], "settled": case["settled"],
            "converged_early": case["converged_early"], "duration_used_s": case["duration_s"],
            "duration_cap_s": case["duration_cap_s"], "n_steps": case["n_steps"],
        }
        row.update({f"rao_{k}": v for k, v in case["rao"].items()})
        summary_rows.append(row)
        print(f"    plat RAO={case['rao']['platform_heave']:.4f}  "
              f"buoy7 RAO={case['rao']['buoy7_heave']:.4f}  settled={case['settled']}  "
              f"dur={case['duration_s']:.0f}s  converged_early={case['converged_early']}",
              flush=True)

    with (_OUT / "rao_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with (_OUT / "manifest.json").open("w") as fh:
        json.dump({
            "study": "M11b PR8 -- 12-buoy fan at plate Cd_n=1.0 (light-drag comparison to Cd_n=5.0)",
            "plate_Cd_n": 1.0, "plate_Cd_t": prp._PLATE_CD_T,
            "grid_periods_s": fan_periods, "grid_heights_m": fan_heights,
            "rao_normalization": "per wave amplitude A = H/2",
            "acceleration": "vertical (Nz) heave acceleration, m/s^2, in per-case CSVs",
            "sibling_fan": "pr8_fan_out/ (operational Cd_n=5.0)",
            "cases": summary_rows,
        }, fh, indent=2)
    print(f"\nDone. Outputs in {_OUT}")
    n_un = sum(1 for r in summary_rows if not r["settled"])
    if n_un:
        print(f"WARNING: {n_un}/{len(summary_rows)} cases did not settle within the cap.")


if __name__ == "__main__":
    main()
