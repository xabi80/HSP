"""FloatSim cross-check of cluster_statics.py at small hub tilt.

FloatSim's own model of the cluster (studies/flume-mooring mooring_verify.article_cluster: the
four pinned buoys on the 12-sided flume BEM, the hub, the gimbal joints) is solved statically
with its linear KKT system [[K, G^T], [G, 0]]. The hub weight and a payload at offset e in the
pin plane are applied as a vertical force and its moments on the hub. The large-angle statics are
evaluated on the same 12-sided hydrostatic table. At small tilt the two must agree:
- hub heave (sinkage) per kg;
- hub tilt per kg m of payload moment.
Tiny regularising springs hold FloatSim's neutral surge / sway / yaw, as reference_modes does.

Run: python floatsim_crosscheck.py   (prints the comparison and writes floatsim_crosscheck.json)
"""

# ruff: noqa: E402  -- sys.path bootstrap precedes the study imports
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "flume-mooring"))
warnings.simplefilter("ignore")

import cluster_statics as cs
import mooring_verify as mv

CASES = [(2.0, 0.0, 0.0), (10.0, 0.0, 0.0), (1.0, 0.10, 0.0), (1.0, 0.10, 45.0),
         (2.0, 0.20, 0.0), (2.0, 0.20, 30.0)]     # (payload kg, offset m, direction deg)


def floatsim_static(art: dict, m: float, e: np.ndarray) -> dict:
    n, h = art["n"], art["ref"]
    K = art["C"] + mv.roll_split(n) + mv.point_k(n, h, (0.0, 0.0, 0.0), (1e-6, 1e-6, 0.0), 1e-6)
    W = cs.G_ACC * (cs.M_HUB + m)
    F = np.zeros(n)
    F[6 * h + 2] = -W
    F[6 * h + 3] = -cs.G_ACC * m * e[1]           # moment of the payload weight about the hub
    F[6 * h + 4] = cs.G_ACC * m * e[0]
    xi = mv.static(art, K, F)
    roll, pitch = xi[6 * h + 3], xi[6 * h + 4]
    return {"heave_m": float(xi[6 * h + 2]), "tilt_deg": float(np.degrees(np.hypot(roll, pitch)))}


def statics(m: float, e: np.ndarray, b: cs.Buoy) -> dict:
    q, _ = cs.solve(m, e, b, np.zeros(3))
    if q is None:
        raise RuntimeError(f"no small-angle equilibrium for m = {m}, e = {e}")
    return {"heave_m": float(q[0]), "tilt_deg": cs.tilt_deg(q)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    art = mv.article_cluster()
    b = cs.Buoy("12")
    rows = []
    print(" payload  offset  dir | FloatSim heave / tilt     | statics heave / tilt")
    for m, r, d in CASES:
        e = np.array([r * np.cos(np.deg2rad(d)), r * np.sin(np.deg2rad(d)), 0.0])
        f, s = floatsim_static(art, m, e), statics(m, e, b)
        rows.append({"m_kg": m, "offset_m": r, "dir_deg": d, "floatsim": f, "statics": s})
        print(f" {m:5.1f} kg {r:.2f} m {d:4.0f} | {1e3 * f['heave_m']:7.2f} mm "
              f"{f['tilt_deg']:6.3f} deg | {1e3 * s['heave_m']:7.2f} mm {s['tilt_deg']:6.3f} deg")
    (HERE / "floatsim_crosscheck.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("wrote floatsim_crosscheck.json")


if __name__ == "__main__":
    main()
