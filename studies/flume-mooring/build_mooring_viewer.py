"""Moored motion viewer for the three flume articles (1 buoy, 1 cluster, 4x4 platform) -- all
motion from FloatSim.

Reuses the FloatSim motion-viewer renderer (studies/platform-12buoy/fin_study/
platform_motion.html), extended in mooring_motion_template.html with the flume (walls, floor),
the wall anchors, the mooring lines coloured by live tension, the true spar / heave-plate
geometry, a free single buoy and a hub-only cluster, and a mooring on / off control.

Motion: FloatSim decks from ``floatsim_decks`` (deck bodies + Capytaine BEM through the driver,
the deck's Morison drag elements, FloatSim ``Catenary`` mooring lines), integrated by FloatSim
(``integrate_cummins`` under ``make_regular_wave_force``) in a regular wave. The viewer frames
are FloatSim's own output over the last LOOP_PERIODS wave periods of the settled run (decimated,
not refitted), as displacements from FloatSim's UNMOORED equilibrium -- so a moored case shows the
lines' static effect too (the pretension tilt of the moored pinned buoys, "static_tilt_deg"). The
moored cluster / platform start from FloatSim's settled moored equilibrium
(``floatsim_decks.moored_equilibrium``). Line tension per frame = the magnitude of FloatSim's
catenary line force at that frame's pose (``make_catenary_state_force`` of each line). The mean
splash-zone drift offset is not part of the model; the readout quotes its upper bound.

Usage:
    python build_mooring_viewer.py run <buoy|cluster|platform> [T1,T2,...] [--H 0.1] [--free]
    python build_mooring_viewer.py html
Writes viewer_rows/*.json (per case) and mooring_motion.html.
"""
# ruff: noqa: E402, E702  -- sys.path bootstrap first; compact lines
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import floatsim_decks as fd
import mooring_sizing as ms
import mooring_verify as mv

from floatsim.mooring.catenary_analytic import (
    CatenaryAttachment,
    CatenaryLine,
    make_catenary_state_force,
)

ROWS = HERE / "viewer_rows"
TEMPLATE = HERE / "mooring_motion_template.html"
OUT_HTML = HERE / "mooring_motion.html"
FPP = 60                    # frames per wave period
LOOP_PERIODS = 2
N_SETTLE = 45.0             # s of settling after the 15 s ramp
Z_TOP, Z_WL, Z_BOT, Z_PLATE = 0.717, 0.0, -0.967, -1.383


def bodies(dk) -> list[dict]:  # type: ignore[no-untyped-def]
    """Viewer bodies: type buoy / hub / platform, parent index, equilibrium reference point."""
    names = [b.name for b in dk.bodies]
    parent = {j.body_a: j.body_b for j in dk.joints}
    out = []
    for b in dk.bodies:
        typ = ("buoy" if (b.hydro_body_label or b.hydro_database) else
               "platform" if b.name == "platform" else "hub")
        out.append({"name": b.name, "type": typ,
                    "parent": names.index(parent[b.name]) if b.name in parent else -1,
                    **{k: round(float(v), 5) for k, v in zip(("x0", "y0", "z0"),
                                                              b.reference_point, strict=True)}})
    return out


def point(Q, b, r):  # type: ignore[no-untyped-def]
    """Small-angle point kinematics of body b at body-frame offset r: Q[:, 6b:6b+6] -> (n, 3)."""
    return Q[:, 6 * b:6 * b + 3] + np.cross(Q[:, 6 * b + 3:6 * b + 6], r)


def run(article: str, periods: list[float], H: float, free: bool = False) -> None:
    ROWS.mkdir(exist_ok=True)
    dk0 = fd.deck(article)
    dk_m, lines = fd.moored(dk0, article)
    B = bodies(dk_m)
    dk = dk0 if free else dk_m
    # frames are FloatSim displacements from its UNMOORED equilibrium, for both variants, so the
    # moored runs show what the lines do to the article (incl. any static pretension tilt)
    setup0, hd = fd.build(dk0, article)
    setup = setup0 if free else fd.build(dk, article, hd=hd)[0]
    xi_ref = setup0.xi0
    n = setup.lhs.n_dof
    buoys = [i for i, b in enumerate(B) if b["type"] == "buoy"]
    ref = next((i for i, b in enumerate(B) if b["type"] == "platform"),
               next((i for i, b in enumerate(B) if b["type"] == "hub"), buoys[0]))
    up = min(buoys, key=lambda i: (B[i]["x0"], B[i]["y0"]))
    dn = max(buoys, key=lambda i: (B[i]["x0"], B[i]["y0"]))
    # one FloatSim catenary force per line, to read its tension at each frame
    line_forces = []
    refs = np.array([b.reference_point for b in dk_m.bodies], dtype=float)
    for ln, c in zip(lines, dk_m.connections, strict=True):
        att = CatenaryAttachment(body_index=ln["body"], fairlead_body=np.asarray(c.attach_a_body),
                                 anchor_global=np.asarray(c.attach_b_body),
                                 line=CatenaryLine(length=c.line.length,
                                                   weight_per_length=c.line.weight_per_length,
                                                   EA=c.line.EA), seabed_depth=200.0)
        line_forces.append((ln["body"], make_catenary_state_force(
            [att], n_dof=n, body_reference_points=refs)))
    wl = fd.WL_B
    for T in periods:
        jp = ROWS / (f"{article}_H{H:g}_T{T:g}".replace(".", "p") + ("_free" if free else "")
                     + ".json")
        if jp.exists():
            continue
        t0 = time.perf_counter()
        try:
            r = fd.run_wave(setup, hd, dk, T, H, N_SETTLE, LOOP_PERIODS)
        except RuntimeError as exc:
            # a catenary solve that fails from every start (cold, warm, taut-elastic; Phase C1)
            # is recorded as a failed case instead of inventing the motion
            if free or "catenary" not in str(exc):
                raise
            jp.write_text(json.dumps({"T": T, "H": H, "moored": True, "failed": str(exc)}))
            print(f"{article} T {T} H {H}: FloatSim run failed ({exc})", flush=True)
            continue
        tf = r.t[-1] - LOOP_PERIODS * T + np.arange(LOOP_PERIODS * FPP) * T / FPP
        idx = np.clip(np.searchsorted(r.t, tf), 0, r.t.size - 1)
        Q, Qd, Qdd = (r.xi[idx] - xi_ref), r.xi_dot[idx], r.xi_ddot[idx]
        frames = [[[round(float(v), 6) for v in Q[f, 6 * b:6 * b + 6]] for b in range(len(B))]
                  for f in range(Q.shape[0])]
        pts = ({"ref": (0, wl), "up": (0, np.array([0.0, 0.0, Z_TOP - fd.aw.ZB])),
                "dn": (0, np.array([0.0, 0.0, Z_PLATE - fd.aw.ZB]))} if article == "buoy" else
               {"ref": (ref, np.zeros(3)), "up": (up, wl), "dn": (dn, wl)})
        sig = {k: {q: np.round(point(M, b, rr), 6).tolist() for q, M in
                   (("disp", Q), ("vel", Qd), ("acc", Qdd))} for k, (b, rr) in pts.items()}
        tension = None
        if not free:
            X = r.xi[idx]; zero = np.zeros(n)
            tension = [[round(float(np.linalg.norm(f(0.0, X[k], zero)[6 * b:6 * b + 3])), 3)
                        for (b, f) in line_forces] for k in range(X.shape[0])]
        tilt = max(float(np.degrees(np.max(np.hypot(Q[:, 6 * b + 3], Q[:, 6 * b + 4]))))
                   for b in buoys)
        d0 = setup.xi0 - xi_ref
        tilt0 = max(float(np.degrees(np.hypot(d0[6 * b + 3], d0[6 * b + 4]))) for b in buoys)
        zref = point(Q, ref, wl if article == "buoy" else np.zeros(3))[:, 2]
        F_bound = len(buoys) * sum(ms.drift_per_spar(H, T)[:2])
        Kx = fd.line_design(article, len(buoys))["Kx"]
        tt = np.asarray(tension) if tension is not None else None
        row = {"T": T, "H": H, "omega": round(2 * np.pi / T, 5), "amp_m": 0.5 * H,
               "n_frames": len(frames), "dt_frame_s": round(T / FPP, 6), "frames": frames,
               "sig": sig, "tension": tension, "moored": not free,
               "rao_ref": round(float(0.5 * (zref.max() - zref.min()) / (0.5 * H)), 4),
               "max_tilt_deg": round(tilt, 2), "static_tilt_deg": round(tilt0, 2),
               "max_tension_N": None if tt is None else round(float(tt.max()), 2),
               "min_tension_N": None if tt is None else round(float(tt.min()), 2),
               "offset_bound_m": round(float(F_bound / Kx), 3),
               "wall_min": round((time.perf_counter() - t0) / 60, 2)}
        jp.write_text(json.dumps(row, separators=(",", ":")))
        print(f"{article}{' (free)' if free else ''} T {T} H {H}: heave RAO {row['rao_ref']:.3f}, "
              f"peak tilt {row['max_tilt_deg']:.1f}°, {row['wall_min']:.1f} min", flush=True)
    scale = 2 if article == "platform" else 1      # platform: each bridle = 2 FloatSim lines
    meta = {"key": article, "name": fd.NAMES[article], "bodies": B, "ref": ref, "up": up,
            "dn": dn, "lines": [{"anchor": ln["anchor"].tolist(), "legs": [ln["body"]],
                                 "ring": None, "T0": round(ln["T0"], 3), "k": round(ln["k"], 3)}
                                for ln in lines],
            "k_line": round(max(ln["k"] for ln in lines) * scale, 3),
            "T0": round(max(ln["T0"] for ln in lines) * scale, 3),
            "Kx": round(fd.line_design(article, len(buoys))["Kx"], 3)}
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
            if c.get("failed"):
                if fr:     # moored FloatSim run failed: show the free run only, and say why
                    cases.append({**fr, "no_moored": "the moored FloatSim run stopped: its "
                                  "catenary solver did not converge (see README)"})
                continue
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
    ap.add_argument("periods", nargs="?", default="2.2,2.9,3.5")
    ap.add_argument("--H", type=float, default=0.1)
    ap.add_argument("--free", action="store_true", help="run without the mooring")
    a = ap.parse_args()
    if a.step == "run":
        run(a.article, [float(x) for x in a.periods.split(",")], a.H, a.free)
    else:
        html()


if __name__ == "__main__":
    main()
