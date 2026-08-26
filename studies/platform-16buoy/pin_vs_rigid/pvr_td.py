"""Drag-limited time-domain confirmation: articulated vs whole-chain-rigid.

The FD sweep (pvr_fd.py) gives resonance placement + loads but radiation-only
magnitudes (upper bounds). Here we run the full nonlinear KKT model WITH Morison
drag at a handful of key periods (heading 0°) and extract the REAL platform
deck-stillness metrics: heave, tilt (pitch), and peak deck accelerations, for both
configurations. Per-case JSON so the run is resumable (126-DOF KKT is heavy).

Usage: python pvr_td.py [T1,T2,...]   (default periods below)
Writes pvr_td_rows/*.json + pvr_td_summary.csv next to this script.
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
sys.path.insert(0, str(_HERE.parent))

import platform16_common as pc16  # noqa: E402
import platform16_rao as prp16  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402  (_fit_amplitude)
import pvr_common as pvr  # noqa: E402

warnings.simplefilter("ignore")

_PLAT = pc16.platform_body_index()
_HEAVE, _ROLL, _PITCH = 6 * _PLAT + 2, 6 * _PLAT + 3, 6 * _PLAT + 4
_ROWDIR = _HERE / "pvr_td_rows"
_DEFAULT_T = [2.5, 3.0, 3.2, 3.5, 4.0]
_HEIGHT = 0.10  # m wave height (modest -> buoy tilt stays near the joint small-angle bound)


def _row_path(key: str, T: float) -> Path:
    return _ROWDIR / (f"pvr_td_{key}_T{T:g}".replace(".", "p") + ".json")


def run_one(key: str, rigid: bool, setup, hydro_dof, T: float) -> dict:
    c = prp16.run_case(
        setup, _HDB, hydro_dof,
        height_m=_HEIGHT, period_s=T, ramp_s=20.0, cap_settle_s=150.0,
        window_periods=6.0, dt=0.01,
    )
    amp, omega, t, xi, acc = c["amp_m"], c["omega"], c["t"], c["xi"], c["acc"]
    heave = prp._fit_amplitude(t, xi[:, _HEAVE], omega) / amp
    roll = prp._fit_amplitude(t, xi[:, _ROLL], omega) / amp
    pitch = prp._fit_amplitude(t, xi[:, _PITCH], omega) / amp
    row = dict(
        config=key, T=float(T), omega=float(omega), amp_m=float(amp),
        heave_RAO=float(heave), roll_RAO=float(roll), pitch_RAO=float(pitch),
        acc_heave_peak=float(np.max(np.abs(acc[:, _HEAVE]))),
        acc_pitch_peak=float(np.max(np.abs(acc[:, _PITCH]))),
        settled=bool(c["settled"]), duration_s=float(c["duration_s"]),
    )
    return row


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    global _HDB
    periods = ([float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else _DEFAULT_T)
    _ROWDIR.mkdir(exist_ok=True)
    _HDB = pvr.load_hdb("0215")
    hydro_dof = prp16._hydro_dof(pvr.build_deck(False))

    for rigid, key in [(False, "artic"), (True, "rigid")]:
        setup = None
        for T in periods:
            rp = _row_path(key, T)
            if rp.exists():
                continue
            if setup is None:
                setup = pvr.build_setup(rigid, _HDB)
            row = run_one(key, rigid, setup, hydro_dof, T)
            rp.write_text(json.dumps(row))
            print(f"[{key}] T={T:.2f}s  heave={row['heave_RAO']:.3f}  "
                  f"pitch={row['pitch_RAO']*1e3:.2f} mrad/m  "
                  f"acc_z={row['acc_heave_peak']:.3f} m/s²  settled={row['settled']} "
                  f"dur={row['duration_s']:.0f}s", flush=True)

    # combined summary
    rows = []
    for key in ("artic", "rigid"):
        for T in periods:
            rp = _row_path(key, T)
            if rp.exists():
                rows.append(json.loads(rp.read_text()))
    if rows:
        with (_HERE / "pvr_td_summary.csv").open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nwrote pvr_td_summary.csv ({len(rows)} cases)")


if __name__ == "__main__":
    main()
