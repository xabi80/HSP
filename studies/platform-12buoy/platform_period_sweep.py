"""M11b PR8 STEP 2: platform heave RAO period sweep at one height, to locate
the actual resonance peaks (adaptive settle).

Sweep T = 1.2 -> 3.6 s at H = 0.30 m: coarse 0.2 s below 2.8 s, fine 0.05 s
from 2.8 to 3.6 s. Extends past 3.3 s deliberately (the campaign shoulder went
to 3.6 s; the pilot cannot rule out a peak above 3.3). H = 0.30 is clear of the
steepness flag over the near-resonance band and in the amplitude-sensitive
regime.

Reports platform-heave and buoy1-heave RAO vs T, the peak locations + widths,
and the per-case adaptive duration. Reference marks: PR7's 3.141 s (single-
cluster heave scaling) and M10's 3.257 s (rotational).
"""

from __future__ import annotations

import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))
import platform_rao_pilot as prp  # noqa: E402

from floatsim.driver import build_system  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402

_OUT = _HERE / "pr8_pilot_out" / "period_sweep_H0p30.csv"
_H = 0.30


def _periods() -> np.ndarray:
    coarse = np.arange(1.2, 2.8, 0.2)  # 1.2 .. 2.6
    fine = np.arange(2.8, 3.6 + 1e-9, 0.05)  # 2.8 .. 3.60
    return np.round(np.concatenate([coarse, fine]), 3)


def main() -> None:
    deck = prp._deck_with_drag()
    hydro_dof = prp._hydro_dof(deck)
    shared = read_capytaine(prp._PLAT_NC)
    hdb_force = read_capytaine(prp._PLAT_NC)
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

    rows: list[dict] = []
    t_start = time.perf_counter()
    periods = _periods()
    print(f"sweeping {periods.size} periods at H={_H} m ...", flush=True)
    for i, T in enumerate(periods):
        wp = 8.0 if T < 2.6 else 6.0  # more windows for the low-response short periods
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = prp.run_case(
                setup,
                hdb_force,
                hydro_dof,
                height_m=_H,
                period_s=float(T),
                ramp_s=20.0,
                cap_settle_s=450.0,
                window_periods=wp,
                dt=0.01,
            )
        lam = prp.cc.G * T**2 / (2.0 * np.pi)
        rows.append(
            dict(
                period_s=float(T),
                omega=c["omega"],
                rao_platform_heave=c["rao"]["platform_heave"],
                rao_buoy1_heave=c["rao"]["buoy1_heave"],
                settled=c["settled"],
                converged_early=c["converged_early"],
                duration_used_s=c["duration_s"],
                steepness=_H / lam,
                steepness_flag=bool(_H / lam > 0.04),
            )
        )
        print(
            f"  [{i+1}/{periods.size}] T={T:.2f}s  RAO_heave={c['rao']['platform_heave']:.4f}"
            f"  dur={c['duration_s']:.0f}s  settled={c['settled']}",
            flush=True,
        )

    _OUT.parent.mkdir(exist_ok=True)
    with _OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Peak location (parabolic refine around the discrete max of the fine band).
    rao = np.array([r["rao_platform_heave"] for r in rows])
    Ts = np.array([r["period_s"] for r in rows])
    kmax = int(np.argmax(rao))
    print(f"\ndiscrete peak: T={Ts[kmax]:.3f}s  RAO={rao[kmax]:.4f}")
    if 0 < kmax < len(rao) - 1:
        y0, y1, y2 = rao[kmax - 1], rao[kmax], rao[kmax + 1]
        denom = y0 - 2 * y1 + y2
        dk = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        Tpk = Ts[kmax] + dk * (Ts[kmax + 1] - Ts[kmax])
        print(f"parabolic peak: T={Tpk:.3f}s")
    total = time.perf_counter() - t_start
    print(
        f"\nsweep wall time: {total/60:.1f} min for {periods.size} cases "
        f"({total/periods.size:.0f}s/case avg). Output: {_OUT.name}"
    )
    n_unsettled = sum(1 for r in rows if not r["settled"])
    if n_unsettled:
        print(f"WARNING: {n_unsettled} case(s) did not settle within the cap.")


if __name__ == "__main__":
    main()
