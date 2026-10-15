"""Large-angle hydrostatic statics of the hinged 4-buoy cluster with a payload on the hub.

The cluster (studies/flume-mooring mooring_verify._cluster_deck): four OSU buoys at r = 0.4167 m,
45 / 135 / 225 / 315 deg, each hung from the hub by a gimbal at its spar top (the pin plane,
+0.717 m), with roll and pitch free and yaw locked. The hub (2 kg) carries a payload of mass m at
an offset e in the pin plane.

Why not FloatSim: an off-centre payload tilts the hub far beyond FloatSim's small-angle validity
(0.1 rad) before the cluster loses its restoring, and FloatSim's restoring is fixed at the design
draft. Xabier approved this static calculation as an exception (2026-10-14). FloatSim's own linear
static solve cross-checks it at small angles (floatsim_crosscheck.py).

Model (exact at any hub tilt, in calm water):
- Every load is vertical, so each buoy carries a vertical pin load P_i and hangs vertically.
  Its sinkage s_i below the design draft is the depth of its pin below the design pin plane.
  P_i = dW(s_i) = rho g (V(s_i) - V(0)) from the Capytaine table (buoy_hydrostatics_vs_draft.json,
  NT 36). P_i stops growing once the spar top (+0.717 m) is under water, and stays at the lifted
  value once the buoy has risen out.
- Hub coordinates q = (z_c, roll phi, pitch theta), R = Ry(theta) Rx(phi). The potential energy is
  U(q) = sum_i int_0^{s_i} dW + g m_hub z_c + g m (z_c + (R e)_z).
  Equilibrium: grad U = 0. Hub stability: the Hessian of U is positive definite.
- Each buoy's own tilt on its gimbal: net stiffness C55(s_i) - P_i * L > 0, with C55 about the
  buoy CoG from the table and L = 1.624 m from the CoG up to the pin.
- Loss of restoring: the first payload mass, along the equilibrium branch from m = 0, at which the
  branch ends (a fold), the Hessian stops being positive definite, or a buoy's net gimbal
  stiffness reaches zero.
- Lines also recorded on each branch:
  - hub tilt 2 / 5 / 10 / 20 / 30 deg;
  - the first pin reaching the water (pin freeboard 0, the hub corner awash);
  - FloatSim's 0.1 rad validity limit.

Decisions (Xabier, 2026-10-14): the hub tilts freely on the gimbals (no stop), and the pins must
stay dry. Each position's limit is the lower of the loss of restoring and the first pin awash.
Buoy contact (axes closer than a heave-plate diameter) is recorded for information.

Writes cluster_envelope.json. Run: python cluster_statics.py
"""

# ruff: noqa: E702  -- compact multi-statement setup lines, as the flume study scripts
from __future__ import annotations

import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.optimize import root

HERE = Path(__file__).resolve().parent
TABLE = json.loads((HERE / "buoy_hydrostatics_vs_draft.json").read_text(encoding="utf-8"))
G_ACC = TABLE["g"]
L_PIN = TABLE["pin_above_cog_m"]                 # buoy CoG up to the pin (1.624 m)
S_TOP = TABLE["z_pin_design_m"]                  # sinkage that puts the spar top / pin awash
M_BUOY, M_HUB = 21.52, 2.0                       # kg (articulated_wall.M_BUOY, the cluster hub)
R_BUOY = 0.5 * 1.25 / 1.5                        # buoy centres from the hub centre
ANG_BUOY = np.deg2rad([45.0, 135.0, 225.0, 315.0])
Q_PIN = np.array([[R_BUOY * np.cos(a), R_BUOY * np.sin(a), 0.0] for a in ANG_BUOY])
TILT_LINES_DEG = (2.0, 5.0, 10.0, 20.0, 30.0)
PLATE_D = 2 * 0.1437                             # heave-plate diameter (coupled_bem_osu)
AWASH_TOL = 1e-3                                 # m: a pin within 1 mm of the waterline
DIRS_DEG = (0.0, 15.0, 30.0, 45.0)               # payload direction; 45 = towards buoy 1
OFFSETS = np.round(np.arange(0.0, 0.501, 0.025), 3)
M_MAX, DM = 70.0, 0.25


class Buoy:
    """Pin load P(s), its integral U(s) and C55(s) from the Capytaine table (NT 36)."""

    def __init__(self, nt: str = "36") -> None:
        rows = TABLE["rows"][nt]
        s = np.array([r["sink_m"] for r in rows]); p = np.array([r["dW_N"] for r in rows])
        c55 = np.array([r["C55"] for r in rows]); c33 = rows[-1]["C33"]
        # the last centimetre to the spar top: constant section, P grows at C33
        s = np.append(s, S_TOP); p = np.append(p, p[-1] + c33 * (S_TOP - s[-2]))
        c55 = np.append(c55, c55[-1] + (c55[-1] - c55[-2]) / (s[-2] - s[-3]) * (S_TOP - s[-2]))
        self.s, self.p, self.c55_tab = s, p, c55
        seg = 0.5 * (p[1:] + p[:-1]) * np.diff(s)            # exact integral of the linear P(s)
        u = np.concatenate([[0.0], np.cumsum(seg)])
        self.u = u - np.interp(0.0, s, u)                     # U(0) = 0

    def pin_load(self, s: np.ndarray) -> np.ndarray:
        return np.interp(s, self.s, self.p)                   # flat outside the table

    def work(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, float)
        sc = np.clip(s, self.s[0], self.s[-1])
        i = np.clip(np.searchsorted(self.s, sc) - 1, 0, len(self.s) - 2)
        ds = sc - self.s[i]
        slope = (self.p[i + 1] - self.p[i]) / (self.s[i + 1] - self.s[i])
        u = self.u[i] + self.p[i] * ds + 0.5 * slope * ds**2
        return u + self.pin_load(sc) * (s - sc)                      # linear beyond the table

    def c55(self, s: np.ndarray) -> np.ndarray:
        return np.interp(s, self.s, self.c55_tab)


def rot(phi: float, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R = Ry(theta) Rx(phi) and its derivatives dR/dphi, dR/dtheta."""
    cx, sx, cy, sy = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    drx = np.array([[0, 0, 0], [0, -sx, -cx], [0, cx, -sx]])
    dry = np.array([[-sy, 0, cy], [0, 0, 0], [-cy, 0, -sy]])
    return ry @ rx, ry @ drx, dry @ rx


def sinkage(q: np.ndarray) -> np.ndarray:
    R = rot(q[1], q[2])[0]
    return -(q[0] + (Q_PIN @ R.T)[:, 2])


def energy(q: np.ndarray, m: float, e: np.ndarray, b: Buoy) -> float:
    R = rot(q[1], q[2])[0]
    return float(np.sum(b.work(sinkage(q))) + G_ACC * M_HUB * q[0]
                 + G_ACC * m * (q[0] + (R @ e)[2]))


def grad(q: np.ndarray, m: float, e: np.ndarray, b: Buoy) -> np.ndarray:
    """dU/dq, analytic: the out-of-balance vertical force and the two tilting moments."""
    _, dRp, dRt = rot(q[1], q[2])
    P = b.pin_load(sinkage(q))
    return np.array([
        -np.sum(P) + G_ACC * (M_HUB + m),
        -np.sum(P * (Q_PIN @ dRp.T)[:, 2]) + G_ACC * m * (dRp @ e)[2],
        -np.sum(P * (Q_PIN @ dRt.T)[:, 2]) + G_ACC * m * (dRt @ e)[2],
    ])


def hessian(q: np.ndarray, m: float, e: np.ndarray, b: Buoy, h: float = 1e-7) -> np.ndarray:
    H = np.zeros((3, 3))
    for k in range(3):
        dq = np.zeros(3); dq[k] = h
        H[:, k] = (grad(q + dq, m, e, b) - grad(q - dq, m, e, b)) / (2 * h)
    return 0.5 * (H + H.T)


def tilt_deg(q: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(rot(q[1], q[2])[0][2, 2], -1.0, 1.0))))


def state(q: np.ndarray, m: float, e: np.ndarray, b: Buoy) -> dict:
    s = sinkage(q); P = b.pin_load(s)
    H = hessian(q, m, e, b)
    scale = np.diag([1.0, 1.0 / R_BUOY, 1.0 / R_BUOY])          # N/m on a common footing
    lam = float(np.linalg.eigvalsh(scale @ H @ scale).min())
    gimbal = b.c55(s) - P * L_PIN
    xy = (Q_PIN @ rot(q[1], q[2])[0].T)[:, :2]                # buoy axes (they hang vertically)
    gap = min(float(np.linalg.norm(xy[i] - xy[j])) for i in range(4) for j in range(i + 1, 4))
    return {"m_kg": m, "q": q.tolist(), "tilt_deg": tilt_deg(q), "sink_m": s.tolist(),
            "min_axis_gap_m": gap,
            "pin_load_N": P.tolist(), "hub_min_stiffness_N_per_m": lam,
            "gimbal_net_Nm_per_rad": gimbal.tolist(), "min_pin_freeboard_m": float(S_TOP - s.max())}


def stable(st: dict) -> bool:
    return st["hub_min_stiffness_N_per_m"] > 0 and min(st["gimbal_net_Nm_per_rad"]) > 0


TILT_MODEL_LIMIT_DEG = 85.0   # beyond this the hub is on its side (and the roll-pitch angles near
                              # their 90 deg singularity): the model's range, not a stability limit
FAIL_LABEL = {"no_equilibrium": "fold (no equilibrium beyond)",
              "jump": "fold (the branch jumps)",
              "on_side": f"hub on its side (>= {TILT_MODEL_LIMIT_DEG:g} deg tilt, model limit)"}


def solve(m: float, e: np.ndarray, b: Buoy, q0: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Equilibrium near q0, accepted on the residual (N and N m), not on the solver's flag."""
    sol = root(lambda q: grad(q, m, e, b), q0, jac=lambda q: hessian(q, m, e, b), method="hybr")
    if np.max(np.abs(grad(sol.x, m, e, b))) > 1e-6 * G_ACC * (M_HUB + m + 1.0):
        return None, "no_equilibrium"
    if tilt_deg(sol.x) > TILT_MODEL_LIMIT_DEG:
        return None, "on_side"
    if np.max(np.abs(sol.x - q0)[1:]) > 0.35:                  # jumped to another branch
        return None, "jump"
    return np.asarray(sol.x), "ok"


def branch(e: np.ndarray, b: Buoy) -> dict:
    """Follow the equilibrium from m = 0 until the restoring is lost; refine to 0.01 kg."""
    q = np.zeros(3); path = [state(q, 0.0, e, b)]
    m = 0.0
    while m < M_MAX:
        m_try = m + DM
        q_try, why = solve(m_try, e, b, q)
        if q_try is not None and stable(st := state(q_try, m_try, e, b)):
            q, m = q_try, m_try; path.append(st)
            continue
        lo, hi, qlo = m, m_try, q                          # bisection between stable and lost
        while hi - lo > 0.01:
            mid = 0.5 * (lo + hi); qm, _ = solve(mid, e, b, qlo)
            if qm is not None and stable(state(qm, mid, e, b)):
                lo, qlo = mid, qm
            else:
                hi = mid
        crit = state(qlo, lo, e, b)
        reason = (FAIL_LABEL[why] if q_try is None else
                  "hub restoring" if st["hub_min_stiffness_N_per_m"] <= 0 else "buoy gimbal")
        return {"path": path, "critical": crit, "m_crit_kg": lo, "reason": reason}
    return {"path": path, "critical": path[-1], "m_crit_kg": None, "reason": "none up to M_MAX"}


def crossing(path: list[dict], key: str, level: float, rising: bool = True) -> float | None:
    """Payload mass where path[key] first crosses level (linear interpolation)."""
    for a, c in pairwise(path):
        va, vc = a[key], c[key]
        if (va - level) * (vc - level) <= 0 and va != vc and (vc > va) == rising:
            return float(a["m_kg"] + (level - va) / (vc - va) * (c["m_kg"] - a["m_kg"]))
    return None


def awash(path: list[dict], br: dict) -> float | None:
    """First payload mass putting a pin at the waterline (within AWASH_TOL); a branch that ends
    exactly there (all loaded pins saturating together) is awash at its critical mass."""
    m = crossing(path, "min_pin_freeboard_m", AWASH_TOL, rising=False)
    if m is None and br["critical"]["min_pin_freeboard_m"] <= AWASH_TOL:
        m = br["m_crit_kg"]
    return m


def at_mass(path: list[dict], m: float, key: str) -> float:
    ms = [p["m_kg"] for p in path]
    return float(np.interp(m, ms, [p[key] for p in path]))


def cluster_cg(m: float, e: np.ndarray) -> dict:
    """Combined CG of the unloaded-geometry cluster (buoys + hub + payload), design frame:
    horizontal offset and height above the design waterline."""
    mt = 4 * M_BUOY + M_HUB + m
    z = (4 * M_BUOY * (-0.907) + (M_HUB + m) * S_TOP) / mt
    return {"total_mass_kg": mt, "cg_offset_m": float(m * np.linalg.norm(e) / mt), "cg_z_m": z}


def fm(v: float | None) -> str:
    return "  —  " if v is None else f"{v:5.2f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    b = Buoy("36")
    out: dict = {"model": __doc__.split("\n\n")[0], "m_hub_kg": M_HUB, "m_buoy_kg": M_BUOY,
                 "r_buoy_m": R_BUOY, "pin_above_cog_m": L_PIN, "spar_top_sink_m": S_TOP,
                 "tilt_lines_deg": list(TILT_LINES_DEG), "rows": []}
    for d in DIRS_DEG:
        print(f"direction {d:.0f} deg (45 = towards a buoy)")
        for r in OFFSETS:
            e = np.array([r * np.cos(np.deg2rad(d)), r * np.sin(np.deg2rad(d)), 0.0])
            br = branch(e, b)
            path, crit = br["path"], br["critical"]
            row = {"dir_deg": d, "offset_m": float(r), "m_crit_kg": br["m_crit_kg"],
                   "reason": br["reason"], "tilt_at_crit_deg": crit["tilt_deg"],
                   "freeboard_at_crit_m": crit["min_pin_freeboard_m"],
                   "m_pin_awash_kg": awash(path, br),
                   "m_buoys_touch_kg": crossing(path, "min_axis_gap_m", PLATE_D, rising=False),
                   "m_tilt_kg": {f"{t:g}": crossing(path, "tilt_deg", t) for t in TILT_LINES_DEG},
                   "m_floatsim_valid_kg": crossing(path, "tilt_deg", float(np.degrees(0.1))),
                   "critical_state": crit}
            lim = [v for v in (row["m_crit_kg"], row["m_pin_awash_kg"]) if v is not None]
            row["m_limit_kg"] = min(lim) if lim else None
            row["limit_reason"] = ("pin awash" if row["m_pin_awash_kg"] is not None
                                   and row["m_limit_kg"] == row["m_pin_awash_kg"]
                                   else row["reason"])
            row["cg_at_limit"] = cluster_cg(row["m_limit_kg"] or M_MAX, e)
            row["tilt_at_limit_deg"] = at_mass(path, row["m_limit_kg"] or M_MAX, "tilt_deg")
            out["rows"].append(row)
            print(f"  e {r:.3f} m: limit {fm(row['m_limit_kg'])} kg ({row['limit_reason']}), "
                  f"tilt {row['tilt_at_limit_deg']:5.1f} deg | 5 / 10 deg at "
                  f"{fm(row['m_tilt_kg']['5'])} / {fm(row['m_tilt_kg']['10'])} kg | touch "
                  f"{fm(row['m_buoys_touch_kg'])} kg", flush=True)
    (HERE / "cluster_envelope.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote cluster_envelope.json")


if __name__ == "__main__":
    main()
