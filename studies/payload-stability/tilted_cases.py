"""Static equilibria of the payload cases for the dynamics in waves (tilted_dynamics.py).

Xabier's cases (2026-10-16):
- the platform with 50 kg on the deck and the cluster with 12.5 kg (50 / 4) on the hub;
- the payload shifted along +x (the wave direction) until the deck / hub tilts 5 deg and 10 deg;
- for comparison, the same payload centred (level) and the unloaded article.
+x is between hubs (platform) and between buoys (cluster), in both records' test orientations
(articulated_wall PLAT_ROT_DEG = 45; mooring_verify CLUSTER_ROT_DEG = 45).

The equilibria come from the large-angle statics (platform_statics / cluster_statics) on the
12-panel hydrostatic table. That is the resolution of the coupled BEM FloatSim runs on, so each
geometry is in equilibrium for the model's own hydrostatics. For every body the output gives:
- the position (buoy CoGs, pins, hubs, deck);
- each buoy's sinkage and pin load;
- the pin load's tipping stiffness P L against the buoy's C55(s). FloatSim's linear model leaves
  P L out, so tilted_dynamics.py reports it alongside the results.

Writes tilted_cases.json. Run: python tilted_cases.py
"""

# ruff: noqa: E702  -- compact lines, as the statics scripts
from __future__ import annotations

import json
import sys
from pathlib import Path

import cluster_statics as cs
import numpy as np
import platform_statics as ps  # sets the platform environment before the record imports

HERE = Path(__file__).resolve().parent
NT = "12"                                     # the coupled BEM's resolution
M_PLATFORM, M_CLUSTER = 50.0, 12.5            # kg
TILTS_DEG = (5.0, 10.0)


def cluster_eq(m: float, e: float, b: cs.Buoy) -> np.ndarray:
    """Hub equilibrium (z, phi, theta) for m kg at offset e along +x, by continuation in mass."""
    q = np.zeros(3); ev = np.array([e, 0.0, 0.0])
    for mm in np.linspace(0.0, m, max(2, int(np.ceil(m / 0.5)) + 1)):
        q, why = cs.solve(float(mm), ev, b, q)
        if q is None:
            raise RuntimeError(f"cluster: no equilibrium at {mm:.2f} kg, e {e} m ({why})")
    return q


def platform_eq(m: float, e: float, b: cs.Buoy) -> np.ndarray:
    q = np.zeros(3)
    for mm in np.linspace(0.0, m, max(2, int(np.ceil(m / 2.0)) + 1)):
        q, why = ps.solve(ps.payload(float(mm), e, 0.0, 0.0), b, q)
        if q is None:
            raise RuntimeError(f"platform: no equilibrium at {mm:.2f} kg, e {e} m ({why})")
    return q


def offset_for(tilt_deg: float, eq, tilt_fn, e_hi: float) -> float:  # type: ignore[no-untyped-def]
    """Payload offset (m) giving the target tilt, by bisection to 1e-6 m."""
    lo, hi = 0.0, e_hi
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tilt_fn(eq(mid)) < tilt_deg else (lo, mid)
    return 0.5 * (lo + hi)


def buoy_rows(pins: np.ndarray, b: cs.Buoy) -> list[dict]:
    s = cs.S_TOP - pins[:, 2]
    P = b.pin_load(s); c55 = b.c55(s)
    return [{"x": float(p[0]), "y": float(p[1]), "sink_m": float(si),
             "z_cog": float(p[2] - cs.L_PIN),
             "pin": [float(v) for v in p], "pin_load_N": float(Pi), "C55_Nm_per_rad": float(ci),
             "PL_Nm_per_rad": float(Pi * cs.L_PIN)}
            for p, si, Pi, ci in zip(pins, s, P, c55, strict=True)]


def inertia_about(points: list[tuple[float, np.ndarray, np.ndarray]], cg: np.ndarray) -> np.ndarray:
    """3x3 inertia tensor about cg (world axes) of parts (mass, position, own inertia tensor)."""
    J = np.zeros((3, 3))
    for m, r, Jo in points:
        d = r - cg
        J += Jo + m * (d @ d * np.eye(3) - np.outer(d, d))
    return J


def cluster_case(name: str, m: float, e: float, b: cs.Buoy) -> dict:
    q = cluster_eq(m, e, b)
    R = cs.rot(q[1], q[2])[0]
    hub = np.array([0.0, 0.0, cs.S_TOP + q[0]])
    pins = hub + cs.Q_PIN @ R.T
    pay = hub + R @ np.array([e, 0.0, 0.0])
    mt = cs.M_HUB + m
    cg = (cs.M_HUB * hub + m * pay) / mt
    J_hub = R @ np.diag([0.3, 0.3, 0.6]) @ R.T                 # mooring_verify._cluster_deck hub
    J = inertia_about([(cs.M_HUB, hub, J_hub), (m, pay, np.zeros((3, 3)))], cg)
    return {"article": "cluster", "case": name, "m_kg": m, "offset_m": e,
            "tilt_deg": cs.tilt_deg(q), "q": q.tolist(), "buoys": buoy_rows(pins, b),
            "hubs": [{"centre": hub.tolist(), "pins": pins.tolist()}],
            "top": {"name": "hub", "mass_kg": mt, "cg": cg.tolist(), "inertia": J.tolist()},
            "payload": pay.tolist()}


def platform_case(name: str, m: float, e: float, b: cs.Buoy) -> dict:
    q = platform_eq(m, e, b)
    R = cs.rot(q[1], q[2])[0]
    o = np.array([0.0, 0.0, cs.S_TOP + q[0]])
    hubs = o + ps.HUB @ R.T
    pins = np.concatenate([h + ps.PIN for h in hubs])          # hubs hang level
    deck = o + R @ np.array([0.0, 0.0, ps.H_DECK])
    pay = o + R @ np.array([e, 0.0, ps.H_DECK])
    mt = ps.M_DECK + m
    cg = (ps.M_DECK * deck + m * pay) / mt
    J_deck = R @ np.diag([6.0, 6.0, 12.0]) @ R.T                # articulated_wall platform body
    J = inertia_about([(ps.M_DECK, deck, J_deck), (m, pay, np.zeros((3, 3)))], cg)
    return {"article": "platform", "case": name, "m_kg": m, "offset_m": e,
            "tilt_deg": ps.tilt_deg(ps.full(q)), "q": q.tolist(), "buoys": buoy_rows(pins, b),
            "hubs": [{"centre": h.tolist(), "pins": (h + ps.PIN).tolist()} for h in hubs],
            "top": {"name": "platform", "mass_kg": mt, "cg": cg.tolist(), "inertia": J.tolist()},
            "payload": pay.tolist()}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    b = cs.Buoy(NT)
    cases = []
    for m, eq, case_fn, e_hi in (
            (M_CLUSTER, lambda e: cluster_eq(M_CLUSTER, e, b), cluster_case, 0.4),
            (M_PLATFORM, lambda e: platform_eq(M_PLATFORM, e, b), platform_case, 1.2)):
        cases.append(case_fn("unloaded", 0.0, 0.0, b))
        cases.append(case_fn("centred", m, 0.0, b))
        for t in TILTS_DEG:
            e = offset_for(t, eq, lambda q: cs.tilt_deg(q[:3]), e_hi)
            cases.append(case_fn(f"tilt{t:g}", m, e, b))
    for c in cases:
        s = [r["sink_m"] for r in c["buoys"]]
        fb = cs.S_TOP - max(s)
        worst = max(c["buoys"], key=lambda r: r["PL_Nm_per_rad"] / r["C55_Nm_per_rad"])
        print(f"{c['article']:8s} {c['case']:8s} m {c['m_kg']:5.1f} kg e {c['offset_m']:.4f} m "
              f"tilt {c['tilt_deg']:5.2f} deg | sink {min(s):+.3f} .. {max(s):+.3f} m, "
              f"pin freeboard {1e3 * fb:4.0f} mm | worst P L / C55 "
              f"{worst['PL_Nm_per_rad']:.0f} / {worst['C55_Nm_per_rad']:.0f}")
    (HERE / "tilted_cases.json").write_text(json.dumps({"nt": NT, "cases": cases}, indent=1),
                                            encoding="utf-8")
    print("wrote tilted_cases.json")


if __name__ == "__main__":
    main()
