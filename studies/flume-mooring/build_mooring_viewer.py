"""Moored motion viewer for the three flume articles (1 buoy, 1 cluster, 4x4 platform).

Reuses the FloatSim motion-viewer renderer (studies/platform-12buoy/fin_study/
platform_motion.html), extended in mooring_motion_template.html with the flume (walls, floor),
the wall anchors, the mooring lines coloured by live tension, the true spar / heave-plate
geometry, a free single buoy and a hub-only cluster.

Motion: drift_td.simulate() -- the drag-limited FloatSim time-domain model with Morison drag
relative to the wave velocity and the design mooring (T_surge = 15 s, lines at the SWL of the
recommended spars) -- in a regular wave. The settled window (last 10 periods) is fitted with
harmonics 1-3 of the wave frequency (plus a constant and a linear trend that absorb the slow
surge transient and are dropped), so each case loops seamlessly over one wave period.
Line tension per frame: T = T0 + k_line * (unit anchor->attachment) . displacement, with the
per-line stiffness / pretension of the design (mooring_sizing.xspread). The mean splash-zone
drift offset is not part of the time-domain model; the readout quotes its upper bound.

Usage:
    python build_mooring_viewer.py run <buoy|cluster|platform> <T1,T2,...> [--H 0.3] [--free]
    python build_mooring_viewer.py html
Writes viewer_rows/*.json (per case) and mooring_motion.html.
"""
# ruff: noqa: E402, E702  -- sys.path bootstrap first; compact lines
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PLAT_ROT_DEG", "45")

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent.parent, HERE.parent / "platform-12buoy" / "flume-wall-effect",
           HERE.parent / "osu-test-buoy", HERE):
    sys.path.insert(0, str(_p))

import articulated_wall as aw
import drift_td as dtd
import mooring_sizing as ms
import mooring_verify as mv

ROWS = HERE / "viewer_rows"
TEMPLATE = HERE / "mooring_motion_template.html"
OUT_HTML = HERE / "mooring_motion.html"
NF = 90                     # frames per (looped) wave period
N_HARM = 3
N_WIN = 10
Z_TOP, Z_WL, Z_BOT, Z_PLATE = 0.717, 0.0, -0.967, -1.383
ANCHOR_X, W2 = mv.L_ANCHOR, ms.W_FLUME / 2
NAMES = {"buoy": "1 buoy", "cluster": "1 cluster (4 buoys)", "platform": "4x4 platform (45°)"}


# ------------------------------------------------------------------ article geometry
def bodies(article: str) -> list[dict]:
    """Viewer bodies: type buoy / hub / platform, parent index, equilibrium reference point."""
    if article == "buoy":
        return [{"name": "buoy", "type": "buoy", "parent": -1, "x0": 0.0, "y0": 0.0, "z0": 0.0}]
    dk = mv._cluster_deck() if article == "cluster" else aw.deck()
    names = [b.name for b in dk.bodies]
    parent = {j.body_a: j.body_b for j in dk.joints}
    out = []
    for b in dk.bodies:
        typ = "buoy" if b.hydro_body_label else ("platform" if b.name == "platform" else "hub")
        par = names.index(parent[b.name]) if b.name in parent else -1
        out.append({"name": b.name, "type": typ, "parent": par,
                    **{k: round(float(v), 5) for k, v in zip(("x0", "y0", "z0"),
                                                              b.reference_point, strict=True)}})
    return out


def swl_offset(article: str) -> np.ndarray:
    """Body-frame offset from a buoy's reference point to its spar at the SWL."""
    return np.zeros(3) if article == "buoy" else np.array([0.0, 0.0, -aw.ZB])


def lines(article: str, B: list[dict]) -> list[dict]:
    """The four X-spread lines: anchor, attachment legs (buoy index, SWL), optional bridle ring."""
    buoys = [i for i, b in enumerate(B) if b["type"] == "buoy"]
    out = []
    for sx in (-1, 1):
        for sy in (1, -1):
            anchor = [sx * ANCHOR_X, sy * W2, 0.0]
            if article == "buoy":
                legs, ring = [0], None
            elif article == "cluster":
                legs = [next(i for i in buoys if np.sign(B[i]["x0"]) == sx
                             and np.sign(B[i]["y0"]) == sy)]
                ring = None
            else:
                xr = max(abs(B[i]["x0"]) for i in buoys)
                legs = [i for i in buoys if np.isclose(B[i]["x0"], sx * xr)
                        and np.sign(B[i]["y0"]) == sy]
                ring = [sx * (xr + 0.75), sy * 0.59, 0.0]
            out.append({"anchor": anchor, "legs": legs, "ring": ring})
    return out


def line_design(article: str, n_spar: int) -> dict:
    a = ms.ARTICLES[NAMES[article]]
    f05 = max(sum(ms.drift_per_spar(0.5, T)[:2]) for T in ms.T_WAVE)
    Kx = (a["M"] + a["A"]) * (2 * np.pi / mv.T_SURGE) ** 2
    xs = ms.xspread(Kx, n_spar * f05 / Kx, 0.25, ANCHOR_X)
    return {"Kx": float(Kx), "k_line": float(xs["kl"]), "T0": float(xs["T0"])}


# ------------------------------------------------------------------ one case
def harmonic_fit(t, X, w):  # type: ignore[no-untyped-def]
    """Least-squares [1, t, cos(n w t), sin(n w t)] fit; returns the harmonic coefficients."""
    cols = [np.ones_like(t), t - t.mean()]
    for n in range(1, N_HARM + 1):
        cols += [np.cos(n * w * t), np.sin(n * w * t)]
    D = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(D, X, rcond=None)
    return c[2:]                                  # (2*N_HARM, n_dof)


def synth(coef, w, tf, deriv=0):  # type: ignore[no-untyped-def]
    out = np.zeros((tf.size, coef.shape[1]))
    for n in range(1, N_HARM + 1):
        a, b = coef[2 * (n - 1)], coef[2 * (n - 1) + 1]
        nw = n * w
        if deriv == 0:
            out += np.outer(np.cos(nw * tf), a) + np.outer(np.sin(nw * tf), b)
        elif deriv == 1:
            out += nw * (np.outer(-np.sin(nw * tf), a) + np.outer(np.cos(nw * tf), b))
        else:
            out += -nw * nw * (np.outer(np.cos(nw * tf), a) + np.outer(np.sin(nw * tf), b))
    return out


def point(Q, b, r):  # type: ignore[no-untyped-def]
    """Small-angle point kinematics of body b at body-frame offset r: Q[:, 6b:6b+6] -> (n, 3)."""
    return Q[:, 6 * b:6 * b + 3] + np.cross(Q[:, 6 * b + 3:6 * b + 6], r)


def run(article: str, periods: list[float], H: float, free: bool = False) -> None:
    """free=True: the same case without the mooring (free-floating article)."""
    ROWS.mkdir(exist_ok=True)
    m = dtd.build(article)
    B = bodies(article)
    L = lines(article, B)
    buoys = [i for i, b in enumerate(B) if b["type"] == "buoy"]
    n_spar = len(buoys)
    des = line_design(article, n_spar)
    r_wl = swl_offset(article)
    ref = next((i for i, b in enumerate(B) if b["type"] == "platform"),
               next((i for i, b in enumerate(B) if b["type"] == "hub"), 0))
    up = min(buoys, key=lambda i: (B[i]["x0"], B[i]["y0"]))
    dn = max(buoys, key=lambda i: (B[i]["x0"], B[i]["y0"]))
    for T in periods:
        jp = ROWS / (f"{article}_H{H:g}_T{T:g}".replace(".", "p") + ("_free" if free else "")
                     + ".json")
        if jp.exists():
            continue
        t0 = time.perf_counter()
        r, wv = dtd.simulate(m, T, H, n_win=N_WIN, moored=not free)
        w, A = wv["omega"], wv["A"]
        msk = r.t >= r.t[-1] - N_WIN * T + 0.5 * dtd.DT
        coef = harmonic_fit(r.t[msk], r.xi[msk] - m["xi0"], w)
        tf = np.arange(NF) * T / NF
        Q, Qd, Qdd = (synth(coef, w, tf, k) for k in (0, 1, 2))
        frames = [[[round(float(v), 6) for v in Q[f, 6 * b:6 * b + 6]] for b in range(len(B))]
                  for f in range(NF)]
        # scope signals: reference centre + upwave / downwave spar at the SWL (single buoy:
        # its SWL point, spar top and heave plate)
        if article == "buoy":
            pts = {"ref": (0, np.zeros(3)), "up": (0, np.array([0, 0, Z_TOP + 0.001])),
                   "dn": (0, np.array([0, 0, Z_PLATE]))}
        else:
            pts = {"ref": (ref, np.zeros(3)), "up": (up, r_wl), "dn": (dn, r_wl)}
        sig = {k: {q: np.round(point(M, b, rr), 6).tolist() for q, M in
                   (("disp", Q), ("vel", Qd), ("acc", Qdd))} for k, (b, rr) in pts.items()}
        # line tensions (true displacement of the SWL attachments; bridle ring = leg mean)
        tension = np.zeros((NF, len(L)))
        for j, ln in enumerate(L):
            d = np.mean([point(Q, i, r_wl) for i in ln["legs"]], axis=0)
            att0 = np.array(ln["ring"] if ln["ring"] else
                            [B[ln["legs"][0]]["x0"], B[ln["legs"][0]]["y0"], 0.0])
            e = att0 - np.array(ln["anchor"]); e /= np.linalg.norm(e)
            tension[:, j] = des["T0"] + des["k_line"] * (d @ e) if not free else np.nan
        tilt = max(float(np.degrees(np.max(np.hypot(Q[:, 6 * b + 3], Q[:, 6 * b + 4]))))
                   for b in buoys)
        zref = point(Q, ref, np.zeros(3))[:, 2]
        F_bound = n_spar * sum(ms.drift_per_spar(H, T)[:2])
        row = {"T": T, "H": round(wv["H_used"], 4), "omega": round(float(w), 5),
               "amp_m": round(float(A), 5), "n_frames": NF, "dt_frame_s": round(T / NF, 6),
               "frames": frames, "sig": sig,
               "tension": None if free else np.round(tension, 3).tolist(),
               "rao_ref": round(float(0.5 * (zref.max() - zref.min()) / A), 4),
               "max_tilt_deg": round(tilt, 2), "moored": not free,
               "max_tension_N": None if free else round(float(tension.max()), 2),
               "min_tension_N": None if free else round(float(tension.min()), 2),
               "offset_bound_m": round(float(F_bound / des["Kx"]), 3),
               "wall_min": round((time.perf_counter() - t0) / 60, 2)}
        jp.write_text(json.dumps(row, separators=(",", ":")))
        print(f"{article}{' (free)' if free else ''} T {T} H {row['H']}: heave RAO "
              f"{row['rao_ref']:.3f}, tilt {row['max_tilt_deg']:.1f}°, {row['wall_min']:.1f} min",
              flush=True)
    meta = {"key": article, "name": NAMES[article], "bodies": B, "lines": L, "ref": ref,
            "up": up, "dn": dn, "k_line": round(des["k_line"], 3), "T0": round(des["T0"], 3),
            "Kx": round(des["Kx"], 3)}
    (ROWS / f"{article}_meta.json").write_text(json.dumps(meta))


# ------------------------------------------------------------------ assemble the viewer
SIGLAB = {"buoy": ["spar at SWL", "spar top", "heave plate"],
          "cluster": ["hub", "upwave spar (SWL)", "downwave spar (SWL)"],
          "platform": ["deck centre", "upwave spar (SWL)", "downwave spar (SWL)"]}


def html() -> None:
    arts = []
    for key in ("buoy", "cluster", "platform"):
        mp = ROWS / f"{key}_meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        rows = {f.stem: json.loads(f.read_text()) for f in ROWS.glob(f"{key}_H*_T*.json")}
        cases = []
        for stem, c in rows.items():
            if stem.endswith("_free"):
                continue
            fr = rows.get(stem + "_free")
            if fr:
                c["free"] = {k: fr[k] for k in ("frames", "sig", "rao_ref", "max_tilt_deg")}
            cases.append(c)
        cases.sort(key=lambda c: (c["H"], c["T"]))
        arts.append({**meta, "siglabels": SIGLAB[key], "cases": cases})
    data = {"geom": {"z_top": Z_TOP, "z_wl": Z_WL, "z_bot": Z_BOT, "z_plate": Z_PLATE,
                     "plate_r": ms.PLATE_R, "spar_d": ms.SPAR_D, "flume_w": ms.W_FLUME,
                     "flume_h": ms.H_FLUME, "flume_x": 6.2, "t_surge": mv.T_SURGE},
            "articles": arts}
    h = TEMPLATE.read_text(encoding="utf-8")
    img = base64.b64encode((HERE / "mooring_layout.png").read_bytes()).decode()
    h = h.replace("__LAYOUT_PNG__", "data:image/png;base64," + img)
    h = h.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    OUT_HTML.write_text(h, encoding="utf-8")
    print(f"wrote {OUT_HTML.name} ({len(h) / 1e6:.2f} MB, {sum(len(a['cases']) for a in arts)} "
          f"cases)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "html"])
    ap.add_argument("article", nargs="?", choices=["buoy", "cluster", "platform"])
    ap.add_argument("periods", nargs="?", default="2.2,2.8,3.5")
    ap.add_argument("--H", type=float, default=0.3)
    ap.add_argument("--free", action="store_true", help="run without the mooring")
    a = ap.parse_args()
    if a.step == "run":
        run(a.article, [float(x) for x in a.periods.split(",")], a.H, a.free)
    else:
        html()


if __name__ == "__main__":
    main()
