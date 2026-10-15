"""Large-angle hydrostatic statics of the 4x4 platform with a payload on the deck.

The platform is the rev D test platform (MOORING-SPEC). It is the model that
mooring_verify.article_platform runs: studies/platform-12buoy/flume-wall-effect
articulated_wall.deck with PLAT_SQUARE_M = 2.4 and PLAT_ROT_DEG = 45.
- 16 OSU buoys on a 0.8 m square grid (outer buoy-centre span 2.4 m), in four 2 x 2 clusters.
- Each buoy hangs from its cluster hub (2 kg) by a gimbal at its spar top, in the pin plane
  (+0.717 m).
- Each hub hangs from the deck frame (6 kg, +0.90 m) by a gimbal at the hub centre, also in the
  pin plane.
- Every gimbal has roll and pitch free and yaw locked.
- The frame is the 45 deg test orientation: the grid is square to x and y, so the hubs sit on the
  diagonals, 1.131 m from the centre.

Payload: a point mass m on the deck, at an offset e parallel to the waterline. Its CG sits h_pay
above the deck frame. h_pay = 0 (on the deck frame) carries over the cluster study's choice of a
mass on the structure; raised CGs are run as a sensitivity.

Model (exact at any deck tilt, in calm water), extending cluster_statics:
- Every load is vertical, so each buoy hangs vertically with pin load dW(s_i), the Capytaine
  table (NT 36). Each hub carries the deck's load at its own pivot, which lies in the plane of its
  four pins, so the load has no lever arm to tilt the hub: the hub hangs level and its four buoys
  sink together. The hub tilts stay in the Hessian, so their stability is checked, not assumed.
- Coordinates: q = (z, phi, theta, alpha_1, beta_1, ..., alpha_4, beta_4).
  - z: heave of the deck's pin-plane centre.
  - phi, theta: deck roll and pitch, R = Ry(theta) Rx(phi).
  - alpha_c, beta_c: each hub's absolute tilt. The yaw that the gimbal sets does not change any
    height, so it does not enter.
- Potential energy:
  U(q) = sum_i int_0^{s_i} dW + g (m_hub sum_c z_hub,c + m_deck z_deck + m z_payload).
  Equilibrium is solved on (z, phi, theta) with level hubs, an exact equilibrium of the hub
  coordinates by symmetry (the residual is recorded).
- Deck and hub stability: the Hessian of U over all 11 coordinates is positive definite.
  Each buoy on its gimbal: C55(s_i) - P_i L > 0, as for the cluster.
- Limit: the lower of the loss of restoring and the first pin awash (Xabier's decisions for the
  cluster, 2026-10-14: free gimbals, pins dry, calm water, free-floating). Other limits:
  - a buoy lifted beyond the draft table (spar bottom at the waterline);
  - the deck on its side (85 deg, the model's range).
  Buoy contact is recorded for information.

Writes platform_envelope_h<h>.json (one per payload CG height) and platform_vs_height.json.
Run: python platform_statics.py envelope <h_pay_m>   |   python platform_statics.py heights
"""

# ruff: noqa: E402, E702  -- sys.path / environment set before the record imports; compact lines
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
os.environ["PLAT_SQUARE_M"] = "2.4"     # MOORING-SPEC rev D platform
os.environ["PLAT_ROT_DEG"] = "45"       # grid square to x and y; hubs on the diagonals
sys.path.insert(0, str(HERE.parent / "platform-12buoy" / "flume-wall-effect"))

import articulated_wall as aw
import cluster_statics as cs

G_ACC, L_PIN, S_TOP = cs.G_ACC, cs.L_PIN, cs.S_TOP
M_BUOY, M_HUB, M_DECK = aw.M_BUOY, 2.0, 6.0      # kg: articulated_wall.deck hub / platform bodies
H_DECK = aw.ZP - aw.ZH                            # deck frame above the pin plane (0.183 m)
R_HUB = aw.S_ARM                                  # hub centres from the deck centre (1.131 m)
R_PIN = 0.5 * aw.S_ARM                            # buoy pins from their hub centre (0.566 m)
HUB = np.array([[R_HUB * np.cos(a), R_HUB * np.sin(a), 0.0]
                for a in np.deg2rad(aw.CLUSTER_DEG)])
PIN = np.array([[R_PIN * np.cos(a), R_PIN * np.sin(a), 0.0] for a in np.deg2rad(aw.BUOY_DEG)])
N_Q = 3 + 2 * len(HUB)
SCALE = np.diag([1.0, 1 / R_HUB, 1 / R_HUB] + [1 / R_PIN] * (N_Q - 3))   # N/m footing
TILT_LINES_DEG = (1.0, 2.0, 5.0, 10.0, 20.0, 30.0)
DIRS_DEG = (0.0, 15.0, 30.0, 45.0)               # payload direction; 45 = towards hub 1 (a corner)
OFFSETS = np.round(np.arange(0.0, 1.2001, 0.05), 3)   # to the outer buoy-centre line (1.2 m)
H_PAYS = np.round(np.arange(0.0, 2.001, 0.1), 2)      # payload CG heights, centred sweep
M_MAX, DM = 260.0, 1.0
TILT_MODEL_LIMIT_DEG = cs.TILT_MODEL_LIMIT_DEG
FAIL_LABEL = {**cs.FAIL_LABEL,
              "on_side": f"deck on its side (>= {TILT_MODEL_LIMIT_DEG:g} deg tilt, model limit)",
              "lifted": "a buoy lifted out (beyond the draft table)"}


class Load(NamedTuple):
    """Payload mass (kg), its position in the deck frame relative to the pin-plane centre (m),
    and the deck mass height above the pin plane (m)."""
    m: float
    r: np.ndarray
    hd: float = H_DECK


def payload(m: float, offset_m: float, dir_deg: float, h_pay_m: float, hd: float = H_DECK) -> Load:
    a = np.deg2rad(dir_deg)
    return Load(m, np.array([offset_m * np.cos(a), offset_m * np.sin(a), hd + h_pay_m]), hd)


def full(q3: np.ndarray) -> np.ndarray:
    return np.concatenate([q3, np.zeros(N_Q - 3)])


def heights(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Hub-centre heights (4) and pin heights (4 x 4) relative to the design pin plane."""
    Rp = cs.rot(q[1], q[2])
    zh = q[0] + (HUB @ Rp[0].T)[:, 2]
    Rh = [cs.rot(q[3 + 2 * c], q[4 + 2 * c]) for c in range(len(HUB))]
    zp = zh[:, None] + np.array([(PIN @ R[0].T)[:, 2] for R in Rh])
    return zh, zp, [Rp, *Rh]


def energy(q: np.ndarray, ld: Load, b: cs.Buoy) -> float:
    zh, zp, rots = heights(q)
    Rp = rots[0][0]
    return float(np.sum(b.work(-zp)) + G_ACC * (
        M_HUB * np.sum(zh) + M_DECK * (q[0] + Rp[2, 2] * ld.hd) + ld.m * (q[0] + (Rp @ ld.r)[2])))


def grad(q: np.ndarray, ld: Load, b: cs.Buoy) -> np.ndarray:
    """dU/dq, analytic: out-of-balance vertical force, deck moments, hub moments."""
    _, zp, rots = heights(q)
    P = b.pin_load(-zp)                                  # (hub, buoy)
    Pc = P.sum(axis=1)
    g = np.zeros(N_Q)
    g[0] = -np.sum(P) + G_ACC * (len(HUB) * M_HUB + M_DECK + ld.m)
    for k, dR in ((1, rots[0][1]), (2, rots[0][2])):
        dzh = (HUB @ dR.T)[:, 2]
        g[k] = (-np.sum(Pc * dzh) + G_ACC * (M_HUB * np.sum(dzh) + M_DECK * dR[2, 2] * ld.hd
                                            + ld.m * (dR @ ld.r)[2]))
    for c, (_, dRa, dRb) in enumerate(rots[1:]):
        g[3 + 2 * c] = -np.sum(P[c] * (PIN @ dRa.T)[:, 2])
        g[4 + 2 * c] = -np.sum(P[c] * (PIN @ dRb.T)[:, 2])
    return g


def hessian(q: np.ndarray, ld: Load, b: cs.Buoy, n: int = N_Q, h: float = 1e-7) -> np.ndarray:
    H = np.zeros((n, n))
    for k in range(n):
        dq = np.zeros(N_Q); dq[k] = h
        H[:, k] = (grad(q + dq, ld, b)[:n] - grad(q - dq, ld, b)[:n]) / (2 * h)
    return 0.5 * (H + H.T)


def tilt_deg(q: np.ndarray) -> float:
    return cs.tilt_deg(q[:3])


def state(q3: np.ndarray, ld: Load, b: cs.Buoy) -> dict:
    q = full(q3)
    zh, zp, rots = heights(q); s = -zp; P = b.pin_load(s)
    H = hessian(q, ld, b)
    lam = float(np.linalg.eigvalsh(SCALE @ H @ SCALE).min())
    gimbal = b.c55(s) - P * L_PIN
    xy = np.concatenate([(HUB @ rots[0][0].T)[c, :2] + (PIN @ R[0].T)[:, :2]
                         for c, R in enumerate(rots[1:])])     # buoy axes (they hang vertically)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    gap = float(d[np.triu_indices(len(xy), 1)].min())
    return {"m_kg": ld.m, "q": q3.tolist(), "tilt_deg": tilt_deg(q), "hub_sink_m": (-zh).tolist(),
            "min_sink_m": float(s.min()), "min_axis_gap_m": gap,
            "cluster_load_N": P.sum(axis=1).tolist(), "min_stiffness_N_per_m": lam,
            "hub_moment_residual_Nm": float(np.max(np.abs(grad(q, ld, b)[3:]))),
            "min_gimbal_net_Nm_per_rad": float(gimbal.min()),
            "min_pin_freeboard_m": float(S_TOP - s.max())}


def stable(st: dict) -> bool:
    return st["min_stiffness_N_per_m"] > 0 and st["min_gimbal_net_Nm_per_rad"] > 0


def solve(ld: Load, b: cs.Buoy, q0: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Deck equilibrium near q0 with level hubs, accepted on the residual (N and N m)."""
    sol = root(lambda q: grad(full(q), ld, b)[:3], q0,
               jac=lambda q: hessian(full(q), ld, b, n=3), method="hybr")
    w = G_ACC * (len(HUB) * M_HUB + M_DECK + ld.m + 1.0)
    if np.max(np.abs(grad(full(sol.x), ld, b)[:3])) > 1e-6 * w:
        return None, "no_equilibrium"
    if tilt_deg(sol.x) > TILT_MODEL_LIMIT_DEG:
        return None, "on_side"
    if np.max(np.abs(sol.x - q0)[1:]) > 0.35:                  # jumped to another branch
        return None, "jump"
    if -heights(full(sol.x))[1].max() < b.s[0]:
        return None, "lifted"
    return np.asarray(sol.x), "ok"


def branch(load_at, b: cs.Buoy, m_max: float = M_MAX) -> dict:
    """Follow the equilibrium from m = 0 until the restoring is lost; refine to 0.02 kg.
    load_at(m) gives the Load for payload mass m."""
    q = np.zeros(3)
    q, _ = solve(load_at(0.0), b, q)                      # deck and hub weights alone
    if q is None:
        raise RuntimeError("no equilibrium without payload")
    path = [state(q, load_at(0.0), b)]
    m = 0.0
    while m < m_max:
        m_try = min(m + DM, m_max)
        q_try, why = solve(load_at(m_try), b, q)
        if q_try is not None and stable(st := state(q_try, load_at(m_try), b)):
            q, m = q_try, m_try; path.append(st)
            continue
        lo, hi, qlo = m, m_try, q
        while hi - lo > 0.02:
            mid = 0.5 * (lo + hi); qm, _ = solve(load_at(mid), b, qlo)
            if qm is not None and stable(state(qm, load_at(mid), b)):
                lo, qlo = mid, qm
            else:
                hi = mid
        crit = state(qlo, load_at(lo), b)
        reason = (FAIL_LABEL[why] if q_try is None else
                  "deck/hub restoring" if st["min_stiffness_N_per_m"] <= 0 else "buoy gimbal")
        return {"path": path, "critical": crit, "m_crit_kg": lo, "reason": reason}
    return {"path": path, "critical": path[-1], "m_crit_kg": None, "reason": "none up to M_MAX"}


def platform_cg(m: float, offset_m: float, h_pay_m: float) -> dict:
    """Combined CG of the level, design-draft platform (buoys + hubs + deck + payload)."""
    nb, nh = len(HUB) * len(PIN), len(HUB)
    mt = nb * M_BUOY + nh * M_HUB + M_DECK + m
    z = (nb * M_BUOY * aw.ZB + nh * M_HUB * aw.ZH + M_DECK * aw.ZP + m * (aw.ZP + h_pay_m)) / mt
    return {"total_mass_kg": mt, "cg_offset_m": float(m * offset_m / mt), "cg_z_m": float(z)}


def summarise(br: dict, extra: dict, offset_m: float, h_pay_m: float) -> dict:
    path, crit = br["path"], br["critical"]
    row = {**extra, "m_crit_kg": br["m_crit_kg"], "reason": br["reason"],
           "tilt_at_crit_deg": crit["tilt_deg"], "freeboard_at_crit_m": crit["min_pin_freeboard_m"],
           "m_pin_awash_kg": cs.awash(path, br),
           "m_buoys_touch_kg": cs.crossing(path, "min_axis_gap_m", cs.PLATE_D, rising=False),
           "m_tilt_kg": {f"{t:g}": cs.crossing(path, "tilt_deg", t) for t in TILT_LINES_DEG},
           "m_floatsim_valid_kg": cs.crossing(path, "tilt_deg", float(np.degrees(0.1))),
           "max_hub_moment_residual_Nm": max(p["hub_moment_residual_Nm"] for p in path),
           "critical_state": crit}
    lim = [v for v in (row["m_crit_kg"], row["m_pin_awash_kg"]) if v is not None]
    row["m_limit_kg"] = min(lim) if lim else None
    row["limit_reason"] = ("pin awash" if row["m_pin_awash_kg"] is not None
                           and row["m_limit_kg"] == row["m_pin_awash_kg"] else row["reason"])
    m_lim = row["m_limit_kg"] or M_MAX
    row["cg_at_limit"] = platform_cg(m_lim, offset_m, h_pay_m)
    row["tilt_at_limit_deg"] = cs.at_mass(path, m_lim, "tilt_deg")
    row["path"] = [{k: p[k] for k in ("m_kg", "tilt_deg", "min_pin_freeboard_m",
                                      "min_stiffness_N_per_m", "min_gimbal_net_Nm_per_rad")}
                   for p in path]
    return row


def header(h_pay_m: float | None) -> dict:
    return {"model": __doc__.split("\n\n")[0], "m_hub_kg": M_HUB, "m_deck_kg": M_DECK,
            "m_buoy_kg": M_BUOY, "n_buoys": len(HUB) * len(PIN), "r_hub_m": R_HUB,
            "r_pin_m": R_PIN, "deck_above_pins_m": H_DECK, "pin_above_cog_m": L_PIN,
            "spar_top_sink_m": S_TOP, "h_pay_m": h_pay_m, "tilt_lines_deg": list(TILT_LINES_DEG)}


def envelope(h_pay_m: float) -> None:
    b = cs.Buoy("36")
    out = {**header(h_pay_m), "rows": []}
    for d in DIRS_DEG:
        print(f"h_pay {h_pay_m:.2f} m, direction {d:.0f} deg (45 = towards a hub)", flush=True)
        for r in OFFSETS:
            br = branch(lambda m, r=r, d=d: payload(m, r, d, h_pay_m), b)
            row = summarise(br, {"dir_deg": d, "offset_m": float(r)}, float(r), h_pay_m)
            out["rows"].append(row)
            print(f"  e {r:.2f} m: limit {cs.fm(row['m_limit_kg'])} kg ({row['limit_reason']}), "
                  f"tilt {row['tilt_at_limit_deg']:5.1f} deg | 5 / 10 deg at "
                  f"{cs.fm(row['m_tilt_kg']['5'])} / {cs.fm(row['m_tilt_kg']['10'])} kg | touch "
                  f"{cs.fm(row['m_buoys_touch_kg'])} kg", flush=True)
    tag = f"{h_pay_m:.1f}".replace(".", "p")
    (HERE / f"platform_envelope_h{tag}.json").write_text(json.dumps(out, indent=1),
                                                         encoding="utf-8")
    print(f"wrote platform_envelope_h{tag}.json")


def vs_height() -> None:
    """Centred payload: the limit against the payload CG height above the deck."""
    b = cs.Buoy("36")
    out = {**header(None), "rows": []}
    for h in H_PAYS:
        # a tiny offset picks the tilt direction once a raised payload makes level unstable
        br = branch(lambda m, h=h: payload(m, 1e-4, 45.0, float(h)), b)
        row = summarise(br, {"h_pay_m": float(h)}, 0.0, float(h))
        out["rows"].append(row)
        print(f"  h_pay {h:.1f} m: limit {cs.fm(row['m_limit_kg'])} kg ({row['limit_reason']}), "
              f"tilt {row['tilt_at_limit_deg']:5.1f} deg", flush=True)
    (HERE / "platform_vs_height.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote platform_vs_height.json")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    mode = sys.argv[1] if len(sys.argv) > 1 else "envelope"
    if mode == "heights":
        vs_height()
    else:
        envelope(float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)


if __name__ == "__main__":
    main()
