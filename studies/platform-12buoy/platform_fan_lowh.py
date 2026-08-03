"""12-buoy platform heave RAO + Nz-accel wave sweep at the corrected model-scale
heights (0.04..0.12 m = 2..6 m full-scale at 1:50), both plate Cd_n in {5, 1}.

Reuses platform_rao_pilot verbatim (build / run_case / drag); only the height
grid and the output layout change, and the summary is written in the uniform
cross-model schema (center = platform ref point, buoy = buoy7 cluster C) shared
with sparfin_rao.py and cluster_rao.py so one plotter reads all three models."""

from __future__ import annotations

import csv
import json
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))

import cluster_common as cc  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402

from floatsim.driver import _build_drag_state_force, build_system  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402

_OUT = _HERE / "platform_rao_lowh_out"
_HEIGHTS = [0.04, 0.06, 0.08, 0.10, 0.12]
_PERIODS = [2.0, 2.5, 2.8, 3.0, 3.141, 3.257, 3.3]
_CENTER = 6 * prp._buoy_body_index_platform() + 2  # platform ref heave DOF = 98
_BUOY = 6 * prp._buoy_body_index(6) + 2  # buoy7 (cluster C) heave DOF = 50


def _acc_amp(acc: np.ndarray, dof: int) -> tuple[float, float]:
    v = acc[:, dof]
    return 0.5 * (v.max() - v.min()), float(np.max(np.abs(v)))


def main() -> None:
    _OUT.mkdir(exist_ok=True)
    deck = prp._deck_with_drag()
    hydro_dof = prp._hydro_dof(deck)
    shared = read_capytaine(prp._PLAT_NC)
    hdb = read_capytaine(prp._PLAT_NC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setup = build_system(
            deck,
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=shared,
            asymptote_check_override=prp._ASYMPTOTE_OVR,
            kernel_decay_floor_override=prp._KERNEL_EXEMPT,
        )
    print(
        f"platform: center heave DOF {_CENTER}, buoy7 heave DOF {_BUOY}; "
        f"assembled T_n 3.143 s (STEP A)"
    )

    for cd in (5.0, 1.0):
        prp._PLATE_CD_N = cd
        drag = _build_drag_state_force(prp._deck_with_drag(), 102, rho=cc.RHO)
        s_cd = replace(setup, state_force=drag)
        rows: list[dict] = []
        for p in _PERIODS:
            for h in _HEIGHTS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c = prp.run_case(
                        s_cd,
                        hdb,
                        hydro_dof,
                        height_m=h,
                        period_s=p,
                        ramp_s=20.0,
                        cap_settle_s=450.0,
                        window_periods=6.0,
                        dt=0.01,
                    )
                ca, cp = _acc_amp(c["acc"], _CENTER)
                ba, bp = _acc_amp(c["acc"], _BUOY)
                tag = f"Cd{cd:g}_H{h:g}_T{p:g}".replace(".", "p")
                with (_OUT / f"case_{tag}.csv").open("w", newline="") as fh:
                    wc = csv.writer(fh)
                    wc.writerow(
                        [
                            "t_s",
                            "center_heave_m",
                            "center_heave_acc_mps2",
                            "buoy_heave_m",
                            "buoy_heave_acc_mps2",
                        ]
                    )
                    wc.writerows(
                        np.column_stack(
                            [
                                c["t"],
                                c["xi"][:, _CENTER],
                                c["acc"][:, _CENTER],
                                c["xi"][:, _BUOY],
                                c["acc"][:, _BUOY],
                            ]
                        ).tolist()
                    )
                rows.append(
                    dict(
                        height_m=h,
                        period_s=p,
                        omega=c["omega"],
                        amp_m=c["amp_m"],
                        rao_center=c["rao"]["platform_heave"],
                        acc_center_amp=ca,
                        acc_center_peak=cp,
                        rao_buoy=c["rao"]["buoy7_heave"],
                        acc_buoy_amp=ba,
                        acc_buoy_peak=bp,
                        settled=c["settled"],
                        converged_early=c["converged_early"],
                        duration_used_s=c["duration_s"],
                    )
                )
                print(
                    f"  Cd={cd} H={h:.2f} T={p:.3f}: RAO_ctr={c['rao']['platform_heave']:.4f} "
                    f"RAO_b7={c['rao']['buoy7_heave']:.4f} settled={c['settled']} "
                    f"dur={c['duration_s']:.0f}s",
                    flush=True,
                )
        with (_OUT / f"rao_summary_Cd{cd:g}.csv").open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    prp._PLATE_CD_N = 5.0
    with (_OUT / "manifest.json").open("w") as fh:
        json.dump(
            {
                "model": "12-buoy articulated platform (102-DOF, 64 constraint rows)",
                "T_n_s": 3.143,
                "grid_heights_m": _HEIGHTS,
                "grid_periods_s": _PERIODS,
                "plate_Cd_n": [5.0, 1.0],
                "scale": "1:50 model; H 0.04-0.12 m = 2-6 m full-scale; accel Froude-invariant",
                "outputs": "platform ref centre + buoy7 (cluster C) heave RAO & Nz-accel",
            },
            fh,
            indent=2,
        )
    print(f"\nDone. Outputs in {_OUT}")


if __name__ == "__main__":
    main()
