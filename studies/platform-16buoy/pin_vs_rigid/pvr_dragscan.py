"""Drag sensitivity of the pin-vs-rigid comparison: re-run the drag-limited KKT model (as in
pvr_td.py) with every Morison Cd (spar, plate normal, plate tangential) scaled by a factor.

Question: the radiation-only FD gives identical pin/rigid heave (< 1 %), the drag-limited TD a
few-% gap near resonance -- does the TD converge to the FD as drag is removed? Quadratic drag
relative to the linear forces scales with Cd * H, so a small wave height stands in for a very
small Cd while keeping the motions small (inside the small-angle joint range).

Usage:
    python pvr_dragscan.py <cd_scale> [T1,T2,...] [--height H] [--cap S] [--tol X]
        defaults: T = 3.0,3.2,3.5 s; H = 0.10 m; cap_settle = 150 s; settle tol = 0.02
    The near-linear runs used:  --height 0.01 --cap 400 --tol 0.002 (cd_scale 0.01) and
    --height 0.1 --cap 400 --tol 0.002 (cd_scale 0.001) at T = 3.2 s. A lightly damped
    resonance approaches steady state slowly, so the default 2 % window-to-window settle check
    would stop too early there.
Writes resumable per-case JSON to pvr_drag_rows/; pvr_dragscan_plot.py tabulates and plots.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import platform16_common as pc16  # noqa: E402
import platform16_rao as prp16  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402  (_fit_amplitude, _SETTLE_TOL)
import pvr_common as pvr  # noqa: E402

_OUT = _HERE / "pvr_drag_rows"
_PLAT = pc16.platform_body_index()
_HEAVE, _PITCH = 6 * _PLAT + 2, 6 * _PLAT + 4
_BASE_CD = (prp16._SPAR_CD, prp16._PLATE_CD_N, prp16._PLATE_CD_T)   # read at deck build


def row_path(scale: float, height: float, key: str, period: float) -> Path:
    name = f"drag_s{scale:g}_H{height:g}_{key}_T{period:g}".replace(".", "p")
    return _OUT / f"{name}.json"


def set_cd_scale(scale: float) -> None:
    """Scale all Morison Cd; platform16_rao reads these module constants when it builds the
    drag elements, so this must run before build_setup."""
    prp16._SPAR_CD, prp16._PLATE_CD_N, prp16._PLATE_CD_T = (c * scale for c in _BASE_CD)


def run(scale: float, periods: list[float], height: float, cap_s: float, tol: float) -> None:
    _OUT.mkdir(exist_ok=True)
    set_cd_scale(scale)
    prp._SETTLE_TOL = tol
    hydro_dof = prp16._hydro_dof(pvr.build_deck(False))
    hdb = pvr.load_hdb("0215")
    for rigid, key in [(False, "artic"), (True, "rigid")]:
        setup = None
        for T in periods:
            jp = row_path(scale, height, key, T)
            if jp.exists():
                continue
            if setup is None:
                setup = pvr.build_setup(rigid, hdb)
            t0 = time.perf_counter()
            c = prp16.run_case(setup, hdb, hydro_dof, height_m=height, period_s=T, ramp_s=20.0,
                               cap_settle_s=cap_s, window_periods=6.0, dt=0.01)
            amp, om, t, xi, acc = c["amp_m"], c["omega"], c["t"], c["xi"], c["acc"]
            row = dict(config=key, T=float(T), scale=float(scale), height=float(height),
                       heave=prp._fit_amplitude(t, xi[:, _HEAVE], om) / amp,
                       pitch=prp._fit_amplitude(t, xi[:, _PITCH], om) / amp,
                       acc_heave=float(np.max(np.abs(acc[:, _HEAVE]))),
                       duration_s=c["duration_s"], settle_ratio=c["settle_ratio"],
                       settle_tol=tol, settled=c["settled"],
                       wall_min=(time.perf_counter() - t0) / 60)
            jp.write_text(json.dumps(row))
            print(f"  Cd x{scale:g} H {height:g} [{key}] T={T:.2f}: heave {row['heave']:.3f}, "
                  f"pitch {row['pitch'] * 1e3:.1f} mrad/m, settle {row['settle_ratio']:.4f}",
                  flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("scale", type=float)
    ap.add_argument("periods", nargs="?", default="3.0,3.2,3.5")
    ap.add_argument("--height", type=float, default=0.10)
    ap.add_argument("--cap", type=float, default=150.0)
    ap.add_argument("--tol", type=float, default=0.02)
    a = ap.parse_args()
    run(a.scale, [float(x) for x in a.periods.split(",")], a.height, a.cap, a.tol)


if __name__ == "__main__":
    main()
