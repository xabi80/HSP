"""Single-buoy yaw collar: r = 0.12 m vs the decided r = 0.2 m (flume-mooring, decision 1).

1. STATICS (FloatSim, as single_buoy_redesign.py): yaw stiffness and period, surge/sway periods,
   and every design state's line clearance to the 0.6 H crest, slack ratio, tension and spring
   stretch, for both radii. Drift: the conservative envelope (H/lambda <= 0.08 over 1.4-3.5 s)
   AND the confirmed matrix (H = 0.5 m from T = 2.35 s).

2. YAW RESPONSE near the moored pitch resonance (T = 2.45 / 2.55 / 2.65 s; H = 0.04 and 0.12 m),
   collared buoy, wave-relative drag.
   - Unseeded, heading 0: the system is mirror-symmetric (symmetry_check.py), so yaw stays at
     exactly zero -- it is never forced.
   - Seeded (yaw 1e-8 rad, roll 1e-8 rad at t = 0): measures whether pitch pumps yaw. FloatSim's
     one-step-lagged state force (the catenary supplies all yaw restoring) grows ANY yaw at
     sigma_num = K_yaw dt / (2 I_zz) (tracker STATE-FORCE-LAG-NEGATIVE-DAMPING), and the spar's
     yaw damping is ~0, so the growth rate is measured at dt = 0.01 and 0.005 s over 15-60 s
     (amplitudes stay ~1e-4 rad or less: linear) and the physical part is extracted two ways:
       sigma_phys = 2 sigma(dt/2) - sigma(dt)          (Richardson, first order in dt)
       sigma_phys = sigma(dt) - K_yaw dt / (2 I_zz)     (the known numerical rate)
     A pitch-yaw parametric coupling would show as sigma_phys > 0, peaking where the yaw period
     sits in a parametric zone of the pitch motion.

Writes buoy_yaw_collar.json.  Run: python buoy_yaw_collar.py [statics|dynamics|report]
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import dataclasses
import itertools
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import floatsim_decks as fd
import mooring_sizing as ms
import resonance_bandwidth as rb
import single_buoy_redesign as sb

OUT = HERE / "buoy_yaw_collar.json"
RADII = (0.12, 0.2)
T_LIST, H_LIST, DT_LIST = (2.45, 2.55, 2.65), (0.04, 0.12), (0.01, 0.005)
SEED = 1.0e-8
T_END, WIN = 60.0, (15.0, 60.0)
H_MATRIX = {0.12: None, 0.2: None, 0.35: 1.85, 0.5: 2.35}   # confirmed-matrix worst period


def _deck(r: float):  # type: ignore[no-untyped-def]
    opts = {**rb.mooring_opts("buoy", collar=False), "collar": "radial", "collar_r": r}
    return fd.moored(fd.deck("buoy"), "buoy", **opts)[0]


def statics() -> dict:
    out = {}
    T0 = rb.mooring_opts("buoy")["T0"]
    for r in RADII:
        s, dk, _l, hdb = sb.setup(T0, r, "radial")
        xc = sb.mean_offset(s, 0.0)
        s = dataclasses.replace(s, xi0=xc)
        per = sb.periods(s, dk, hdb)
        rows = [sb.design_state(s, dk, H, xc) for H in sb.H_CASES]
        matrix = []
        for H, T in H_MATRIX.items():
            Ts = np.arange(1.4, 3.5001, 0.01) if T is None else np.array([T])
            F = max(sum(ms.drift_per_spar(H, t, steep=ms.STEEP_MATRIX)[:2]) for t in Ts)
            xm = sb.mean_offset(s, F)
            xd = xm.copy()
            xd[0] += 0.5 * H
            ls = sb.line_states(dk, xd)
            T0a = float(np.mean([ln["T_fair"] for ln in sb.line_states(dk, xc)]))
            matrix.append({"H": H, "drift_N": F, "mean_offset_m": float(xm[0] - xc[0]),
                           "clearance_m": min(ln["z_low"] for ln in ls) - sb.CREST * H,
                           "T_min_ratio": min(ln["T_fair"] for ln in ls) / T0a,
                           "T_max_N": max(ln["T_fair"] for ln in ls),
                           "stretch_max_m": max(ln["T_fair"] / ln["k"] for ln in ls)})
        out[f"r{r:g}"] = {"periods": per, "envelope_states": rows, "matrix_states": matrix,
                          "spar_clearance_at_collar_m": r - fd.aw.SPAR_D / 2}
        p = per
        print(f"r {r}: yaw {p['yaw']['T_s']:.3f} s (K {p['yaw']['K']:.3f}, I {p['yaw']['M']:.4f}),"
              f" surge {p['surge']['T_s']:.1f}, sway {p['sway']['T_s']:.1f}, heave "
              f"{p['heave']['T_s']:.3f}", flush=True)
        for st in rows:
            print(f"   envelope H {st['H']}: clr {st['clearance_m']:+.3f} Tmin/T0 "
                  f"{st['T_min_ratio']:.2f} Tmax {st['T_max_N']:.2f} stretch "
                  f"{st['stretch_max_m']:.2f}", flush=True)
        for st in matrix:
            print(f"   matrix   H {st['H']}: drift {st['drift_N']:.3f} off "
                  f"{st['mean_offset_m']:.3f} clr {st['clearance_m']:+.3f} Tmin/T0 "
                  f"{st['T_min_ratio']:.2f} Tmax {st['T_max_N']:.2f} stretch "
                  f"{st['stretch_max_m']:.2f}", flush=True)
    return out


def _growth(t: np.ndarray, y: np.ndarray, T_y: float) -> float:
    """Exponential growth rate (1/s) of |y|'s envelope over WIN (max per yaw period)."""
    m = (t >= WIN[0]) & (t <= WIN[1])
    tt, yy = t[m], np.abs(y[m])
    edges = np.arange(tt[0], tt[-1], T_y)
    pk_t, pk = [], []
    for a, b in itertools.pairwise(edges):
        w = (tt >= a) & (tt < b)
        if w.any() and yy[w].max() > 0:
            pk_t.append(tt[w][np.argmax(yy[w])])
            pk.append(yy[w].max())
    if len(pk) < 4:
        return float("nan")
    return float(np.polyfit(pk_t, np.log(pk), 1)[0])


def one(args: tuple) -> dict:
    r, T, H, dt, seeded = args
    t0 = time.perf_counter()
    dk = _deck(r)
    s, hd, wave, ramp = fd.wave_setup(dk, "buoy", T, H, dt=dt)
    xi0 = np.asarray(s.xi0).copy()
    if seeded:
        xi0[5] += SEED
        xi0[3] += SEED
    s = dataclasses.replace(s, xi0=xi0)
    rr = fd.run_case(s, hd, dk, wave, ramp, T_END - fd.RAMP_S, 0, dt=dt)
    K_yaw = sb.periods(dataclasses.replace(s, xi0=np.asarray(s.xi0)), dk,
                       hd["bem_databases"]["buoy"])["yaw"]
    T_y = K_yaw["T_s"]
    yaw, roll, pitch = rr.xi[:, 5], rr.xi[:, 3], rr.xi[:, 4] - xi0[4]
    last = rr.t >= rr.t[-1] - 4 * T
    f = np.fft.rfftfreq(rr.t.size, dt)
    spec = np.abs(np.fft.rfft(np.where(rr.t >= WIN[0], yaw, 0.0)))
    row = {"r": r, "T": T, "H": H, "dt": dt, "seeded": seeded,
           "sigma_yaw": _growth(rr.t, yaw, T_y), "sigma_roll": _growth(rr.t, roll, T_y),
           "sigma_num": K_yaw["K"] * dt / (2 * K_yaw["M"]),
           "T_yaw_s": T_y, "K_yaw": K_yaw["K"], "I_yaw": K_yaw["M"],
           "yaw_max_rad": float(np.abs(yaw).max()), "roll_max_rad": float(np.abs(roll).max()),
           "yaw_spectral_peak_s": float(1.0 / f[1:][np.argmax(spec[1:])]),
           "pitch_amp_deg": float(np.degrees(0.5 * (pitch[last].max() - pitch[last].min()))),
           "wall_min": (time.perf_counter() - t0) / 60}
    print(f"r {r} T {T} H {H} dt {dt} seeded {seeded}: sigma yaw {row['sigma_yaw']:+.4f} "
          f"(num {row['sigma_num']:.4f}) roll {row['sigma_roll']:+.4f} /s, max yaw "
          f"{row['yaw_max_rad']:.2e}, yaw peak {row['yaw_spectral_peak_s']:.3f} s, pitch "
          f"{row['pitch_amp_deg']:.2f} deg [{row['wall_min']:.1f} min]", flush=True)
    return row


def dynamics() -> list[dict]:
    cases = [(r, T, H, dt, True) for r in RADII for T in T_LIST for H in H_LIST for dt in DT_LIST]
    cases += [(r, T, H, 0.01, False) for r in RADII for T in T_LIST for H in H_LIST]
    with ProcessPoolExecutor(max_workers=12) as ex:
        return list(ex.map(one, cases))


def report(rows: list[dict]) -> list[dict]:
    summary = []
    for r in RADII:
        for T in T_LIST:
            for H in H_LIST:
                g = {row["dt"]: row for row in rows if row["seeded"] and
                     (row["r"], row["T"], row["H"]) == (r, T, H)}
                u = next(row for row in rows if not row["seeded"] and
                         (row["r"], row["T"], row["H"]) == (r, T, H))
                a, b = g[0.01], g[0.005]
                rich = 2 * b["sigma_yaw"] - a["sigma_yaw"]
                summary.append({"r": r, "T": T, "H": H, "T_yaw_s": a["T_yaw_s"],
                                "sigma_dt01": a["sigma_yaw"], "sigma_dt005": b["sigma_yaw"],
                                "sigma_num_dt01": a["sigma_num"], "sigma_phys_richardson": rich,
                                "sigma_phys_subtract_dt005": b["sigma_yaw"] - b["sigma_num"],
                                "zeta_phys_equiv": -rich / (2 * np.pi / a["T_yaw_s"]),
                                "unseeded_yaw_max_rad": u["yaw_max_rad"],
                                "pitch_amp_deg": u["pitch_amp_deg"]})
                s_ = summary[-1]
                print(f"r {r} T {T} H {H}: sigma(dt .01/.005) {a['sigma_yaw']:+.4f}/"
                      f"{b['sigma_yaw']:+.4f}, num {a['sigma_num']:.4f}/{b['sigma_num']:.4f}"
                      f" -> phys {rich:+.4f} (subtract {s_['sigma_phys_subtract_dt005']:+.4f}) "
                      f"/s; unseeded max yaw {u['yaw_max_rad']:.1e}; pitch "
                      f"{u['pitch_amp_deg']:.2f} deg", flush=True)
    return summary


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if step in ("statics", "all"):
        res["statics"] = statics()
        OUT.write_text(json.dumps(res, indent=1, default=float))
    if step in ("dynamics", "all"):
        res["runs"] = dynamics()
        OUT.write_text(json.dumps(res, indent=1, default=float))
    if step in ("dynamics", "report", "all"):
        res["summary"] = report(res["runs"])
        OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
