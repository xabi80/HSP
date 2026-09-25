"""The chosen single-buoy design (attachment_sweep.py: collar r = 0.2 m at z = -0.50 m, T0 2.40 N,
k x1; yaw stiffness ~2.0 N m/rad, period ~1.12 s) run in FloatSim with the conservative drift
sum (line_hardware.extreme_run: 210 s at dt = 0.0025 s, unseeded heading 0). Round 2 (DESIGN-BASIS
C5) found the r = 0.12 m collar at T0 = 4 N -- the same yaw stiffness -- parametrically unstable
at H = 0.5 m, T = 2.65 s; this checks where the chosen design stands, operational band included.
(buoy_stability_check_k0p5.json: the same cases for the k x0.5 variant, an intermediate choice.)

The buoy is mirror-symmetric to round-off (DESIGN-BASIS C6), so heading-0 yaw is never forced:
any yaw above round-off is growth of the model itself (parametric). The operational runs sit at
1e-13 - 2e-11 rad; the classes below put "stable" at <= 1e-8 rad.

Writes buoy_stability_check.json.  Run: python buoy_stability_check.py [report [file.json]]
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import attachment_sweep as asw

CASES = ((0.04, 2.75), (0.12, 2.45), (0.12, 2.55), (0.12, 2.65), (0.12, 2.75), (0.2, 2.65),
         (0.35, 2.65), (0.5, 2.65),
         # the rest of the extreme band (H 0.5 m from 2.35 s; H/lambda <= 0.08)
         (0.2, 2.35), (0.2, 2.75), (0.2, 3.5), (0.35, 2.0), (0.35, 2.35), (0.35, 3.0),
         (0.35, 3.5), (0.5, 2.35), (0.5, 3.0), (0.5, 3.5))


OUT = HERE / "buoy_stability_check.json"


def classify(r: dict) -> str:
    """stable: yaw at round-off (<= 1e-8 rad); 'yaw grows': model growth, still below 1 deg at
    the end of the 210 s run (the loads are valid over the run, a longer test grows further);
    diverges: yaw > 1 deg or tilt > 60 deg (no prediction)."""
    if r["yaw_max_deg"] > 1.0 or r["tilt_max_deg"] > 60.0:
        return "diverges"
    return "yaw grows" if r["yaw_max_deg"] > np.degrees(1e-8) else "stable"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1:2] == ["report"]:
        rows = json.loads((HERE / sys.argv[2]).read_text() if sys.argv[2:] else OUT.read_text())
    else:
        with ProcessPoolExecutor(max_workers=len(CASES)) as ex:
            rows = list(ex.map(asw.extreme, [("buoy", H, T) for H, T in CASES]))
    for r in rows:
        r.pop("stable", None)
        r["yaw_max_deg"] = float(np.degrees(r["antisymmetric_max"]["yaw"]))
        r["roll_max_deg"] = float(np.degrees(r["antisymmetric_max"]["roll"]))
        r["state"] = classify(r)
        print(f"buoy H {r['H']} T {r['T']}: {r['state']}; yaw max "
              f"{r['yaw_max_deg']:.1e} deg, roll {r['roll_max_deg']:.1e} deg, tilt "
              f"{r['tilt_max_deg']:.1f} deg, T_min/T0 {r['T_min_ratio']:.2f}, T max "
              f"{max(r['T_max_N']):.2f} N, surge {r['surge_min_m']:.2f}..{r['surge_max_m']:.2f} m",
              flush=True)
    (HERE / sys.argv[2] if sys.argv[2:] else OUT).write_text(json.dumps(rows, indent=1,
                                                                   default=float))


if __name__ == "__main__":
    main()
