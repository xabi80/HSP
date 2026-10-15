"""FloatSim cross-check of platform_statics.py at small deck tilt.

FloatSim's own model of the platform is solved statically with its linear KKT system
[[K, G^T], [G, 0]]. The model is studies/flume-mooring mooring_verify.article_platform: the 16
pinned buoys on the coupled flume BEM, the four hubs, the deck, and the gimbal joints. The loads
are:
- each hub's weight on its hub;
- the deck weight, the payload weight and the payload's moments on the deck.
The large-angle statics run on the hydrostatic table with FloatSim's panel count, in two forms:
- 'like for like': every weight is put in the pin plane. FloatSim's linear model has no weight
  term, so this is the comparison that must agree.
- 'real heights': the deck at +0.183 m and the payload on the deck frame. The difference is the
  m g h destabilisation that FloatSim's linear model leaves out.
Tiny regularising springs hold FloatSim's neutral surge / sway / yaw, as reference_modes does.

Run: python platform_crosscheck.py   (prints the comparison and writes platform_crosscheck.json)
"""

# ruff: noqa: E402, E702, I001  -- platform_statics is imported first (it sets the platform
# environment the record modules read at import); compact setup lines
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import platform_statics as ps   # sets PLAT_SQUARE_M / PLAT_ROT_DEG before the record imports

sys.path.insert(0, str(HERE.parent / "flume-mooring"))
warnings.simplefilter("ignore")

import cluster_statics as cs
import mooring_verify as mv

CASES = [(10.0, 0.0, 0.0), (50.0, 0.0, 0.0), (5.0, 0.5, 0.0), (5.0, 0.5, 45.0),
         (10.0, 1.0, 0.0), (10.0, 1.0, 30.0)]     # (payload kg, offset m, direction deg)
HUBS = [5 * c + 4 for c in range(4)]              # articulated_wall body order: 4 buoys + hub


def floatsim_static(art: dict, m: float, offset_m: float, dir_deg: float) -> dict:
    n, p = art["n"], art["ref"]
    K = art["C"] + mv.roll_split(n) + mv.point_k(n, p, (0.0, 0.0, 0.0), (1e-6, 1e-6, 0.0), 1e-6)
    a = np.deg2rad(dir_deg); ex, ey = offset_m * np.cos(a), offset_m * np.sin(a)
    F = np.zeros(n)
    for h in HUBS:
        F[6 * h + 2] = -cs.G_ACC * ps.M_HUB
    F[6 * p + 2] = -cs.G_ACC * (ps.M_DECK + m)
    F[6 * p + 3] = -cs.G_ACC * m * ey               # moment of the payload weight about the deck
    F[6 * p + 4] = cs.G_ACC * m * ex
    xi = mv.static(art, K, F)
    roll, pitch = xi[6 * p + 3], xi[6 * p + 4]
    return {"heave_m": float(xi[6 * p + 2]), "tilt_deg": float(np.degrees(np.hypot(roll, pitch)))}


def statics(m: float, offset_m: float, dir_deg: float, b: cs.Buoy, hd: float) -> dict:
    ld = ps.payload(m, offset_m, dir_deg, 0.0, hd)
    q, _ = ps.solve(ld, b, np.zeros(3))
    if q is None:
        raise RuntimeError(f"no small-angle equilibrium for {m} kg at {offset_m} m")
    return {"heave_m": float(q[0]), "tilt_deg": ps.tilt_deg(ps.full(q))}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    art = mv.article_platform()
    c33 = art["C"][6 * 0 + 2, 6 * 0 + 2]
    nt = min(cs.TABLE["rows"], key=lambda k: abs(cs.TABLE["rows"][k][0]["C33"] - c33))
    print(f"FloatSim C33 per buoy {c33:.2f} N/m -> statics on the NT {nt} table "
          f"(C33 {cs.TABLE['rows'][nt][0]['C33']:.2f})")
    b = cs.Buoy(nt)
    rows = []
    print(" payload  offset  dir | FloatSim heave / tilt  | like for like         | real heights")
    for m, r, d in CASES:
        f = floatsim_static(art, m, r, d)
        s0, s1 = statics(m, r, d, b, 0.0), statics(m, r, d, b, ps.H_DECK)
        rows.append({"m_kg": m, "offset_m": r, "dir_deg": d, "floatsim": f,
                     "statics_like_for_like": s0, "statics_real_heights": s1})
        print(f" {m:5.1f} kg {r:.2f} m {d:4.0f} | {1e3 * f['heave_m']:7.2f} mm "
              f"{f['tilt_deg']:6.3f} deg | {1e3 * s0['heave_m']:7.2f} mm {s0['tilt_deg']:6.3f} deg"
              f" | {1e3 * s1['heave_m']:7.2f} mm {s1['tilt_deg']:6.3f} deg")
    out = {"nt": nt, "floatsim_c33_N_per_m": float(c33), "rows": rows}
    (HERE / "platform_crosscheck.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote platform_crosscheck.json")


if __name__ == "__main__":
    main()
