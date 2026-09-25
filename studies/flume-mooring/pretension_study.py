"""Does the OPERATIONAL mooring need pretension? (flume-mooring, 2026-09-25; an evaluation for
Xabier -- MOORING-SPEC rev C is NOT changed.)

Variants, cluster and platform (FloatSim throughout; the rev C operational cords, same k, same
attachments and anchors):
  V0  rev C operational set (at-rest 1.44 / 1.42 N per line/leg).
  V1  "taut minimum": T0 = k (tolerance + creep). Tolerance +-10 mm (Xabier). Creep: natural
      rubber creeps ~2.4-4 % of its deflection per decade of time in tension, ~4 % wet (Gent,
      Engineering with Rubber ch. 7; PMC6728486, "Long-time creep in a pure-gum rubber
      vulcanizate"); a test day is 1 min -> 1 day = 3.2 decades, so ~13 % of the loaded stretch.
      The loaded stretch is taken conservatively as the operational peak (V0's moored runs).
  T0=0 exactly taut at zero tension (Xabier's physics case; statics only).
  V2  installed 10 mm slack (L0 = chord + 10 mm); statics only: the dead band.

Steps:
  weight   -- the line weight the record uses, and rev C's sag / tension with the heavier
              submerged weights of a 1.1 / 1.2 g/cm3 cord (dry 0.3 N/m).
  settle   -- FloatSim's calm equilibrium of V1 (cluster, platform).
  statics  -- per variant: calm tilt; tilt / heave period shift vs free (bem_waterline modal,
              attachment_sweep.evaluate); pull stiffness in surge / sway / yaw, small amplitude
              and secant at operating amplitude, BOTH directions; periods (decay-scaled); the
              dead band (lines kinematically slack).
  runs V0|V1 -- H = 0.04 and 0.12 m at the tilt resonance and at 1.4 s, the drift sum in-run
              (line_hardware.extreme_run): mean offset, surge amplitude, heave RAO, max tilt,
              line tensions, the fraction of the window each line is slack (chord <= L0), and
              the fouling geometry at each line's slackest instant (the catenary profile against
              every spar and heave plate; dynamic cord motion is NOT modelled).
  extreme  -- the extreme set at V1's at-rest tension: statics estimate of its H = 0.5 m max
              surge; FloatSim runs only if it moves by > 5 %.
  buoy     -- statics: yaw period and tilt shift vs T0, 2.40 N down to V1.

Writes pretension_study.json.  Run: python pretension_study.py <step>
"""
# ruff: noqa: E402, E501, RUF001  -- sys.path bootstrap first; report rows are long Markdown tables with typographic signs
from __future__ import annotations

import json
import math
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import attachment_sweep as asw
import build_spec as bs
import extreme_set as es
import floatsim_decks as fd
import line_hardware as lh
import mooring_figures as mf
import tank_predictions as tp

OUT = HERE / "pretension_study.json"
ARTS = ("cluster", "platform")
TOL_M = 0.010                                  # installation length tolerance (Xabier)
CREEP_PER_DECADE = 0.04                        # wet natural rubber, of the deflection
DECADES_DAY = math.log10(24 * 60)              # 1 min -> 1 day
CREEP_DAY = CREEP_PER_DECADE * DECADES_DAY     # ~13 % of the loaded stretch
RHO_W = 998.0
W_DRY = 0.3                                    # N/m, the record's dry weight (resonance_bandwidth)
H_RUN = (0.04, 0.12)
T_OFF = 1.4


def _load() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def _save(res: dict) -> None:
    OUT.write_text(json.dumps(res, indent=1, default=float))


def s_peak_op(art: str) -> float:
    """The operational peak stretch of the V0 cords (its moored runs; the buoy: every predicted
    run of its one set)."""
    k = bs.ST[art]["lines"][0]["k_N_per_m"]
    return max(max(r["T_max_N"]) for r in bs.op_runs(art)) / k


def t0_v1(art: str, creep_day: float = CREEP_DAY) -> float:
    k = bs.ST[art]["lines"][0]["k_N_per_m"]
    return k * (TOL_M + creep_day * s_peak_op(art))


def variant_opts(art: str, name: str) -> dict:
    """Operational options with the per-line/leg nominal T0 of the variant."""
    o = dict(asw.chosen(art)["opts"])
    ln = bs.ST[art]["lines"][0]
    t_leg = {"V0": ln["T0_nominal_N"], "V1": t0_v1(art), "T0=0": 0.0,
             "V2": -ln["k_N_per_m"] * TOL_M}[name]
    o["T0"] = o["T0"] * t_leg / ln["T0_nominal_N"]
    return o


def v1_tag(art: str) -> str:
    return f"{art}:pretension_V1"


def calm_xi(art: str, name: str) -> np.ndarray:
    """The calm state: V0 / V1 FloatSim settles; T0=0 and V2 the deck (unmoored) equilibrium --
    no tension, so no pretension moment (line weight only)."""
    n = 6 * len(fd.deck(art).bodies)
    cache = json.loads(fd.EQ_CACHE.read_text())
    if name == "V0":
        return np.asarray(cache[asw.chosen(art)["tag"]]["xi"])
    if name == "V1" and v1_tag(art) in cache:
        return np.asarray(cache[v1_tag(art)]["xi"])
    return np.zeros(n)


# ----------------------------------------------------------------------------- weight
def weight() -> dict:
    out = {"record_w_N_per_m": fd.W_LINE, "dry_w_N_per_m": W_DRY,
           "implied_density_g_cm3": RHO_W / (1 - fd.W_LINE / W_DRY) / 1000}
    ws = {"record": fd.W_LINE}
    for rho in (1.1, 1.2):
        ws[f"rho {rho}"] = W_DRY * (1 - RHO_W / 1000 / rho)
    out["w_N_per_m"] = ws
    for art in ("buoy", *ARTS):
        des = asw.chosen(art)
        xi = (np.asarray(lh._setup(art, opts=des["opts"], tag=des["tag"])[1]) if art == "buoy"
              else calm_xi(art, "V0"))
        rows = {}
        for lab, w in ws.items():
            dk, _ = fd.moored(fd.deck(art), art, **{**des["opts"], "w_line": w})
            st = lh.line_states(dk, xi, n_pts=41)[0]
            p = st["pts"]
            s_ = np.linalg.norm(p - p[0], axis=1)
            chord = p[0] + np.outer(s_ / s_[-1], p[-1] - p[0])
            rows[lab] = {"w": w, "sag_mm": float(1000 * (chord[:, 2] - p[:, 2]).max()),
                         "T_N": st["T"]}
        out[art] = rows
        print(f"{art:8s}: " + ", ".join(f"{k} w {v['w']:.3f} N/m -> sag {v['sag_mm']:.0f} mm, "
                                        f"T {v['T_N']:.3f} N" for k, v in rows.items()), flush=True)
    return out


# ----------------------------------------------------------------------------- statics
def _pulls(art: str, opts: dict, xi: np.ndarray):  # type: ignore[no-untyped-def]
    dk, lines = fd.moored(fd.deck(art), art, **opts)
    lf = tp._line_forces(dk)
    z = np.zeros(xi.size)

    def res(x: np.ndarray) -> np.ndarray:
        return tp._resultant(dk, x, np.sum([f(0.0, x, z) for f in lf], axis=0))

    R0 = res(xi)

    def pull(mode: str, j: int, a: float) -> float:
        return float(R0[j] - res(tp.rigid(dk, xi, mode, a))[j])

    return dk, lines, pull


def _dead_band(art: str, opts: dict, xi: np.ndarray, mode: str) -> float:
    """Width of the offset interval around the calm state in which every line is kinematically
    slack (chord <= L0), rigid article (m)."""
    dk, lines = fd.moored(fd.deck(art), art, **opts)
    atts, ref, _n, _F = asw._lines(dk)
    anch = np.array([a.anchor_global for a in atts])
    L0 = np.array([ln["L0"] for ln in lines])
    grid = np.linspace(-0.10, 0.10, 4001)

    def all_slack(a: float) -> bool:
        p = asw._attach_paths(atts, ref, tp.rigid(dk, xi, mode, a)[None, :])[0]
        return bool(np.all(np.linalg.norm(anch - p, axis=1) <= L0))

    ok = np.array([all_slack(a) for a in grid])
    if not ok[len(grid) // 2]:
        return 0.0
    i0 = len(grid) // 2
    lo, hi = i0, i0
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(grid) - 1 and ok[hi + 1]:
        hi += 1
    return float(grid[hi] - grid[lo])


def statics_one(args: tuple) -> dict:
    art, name = args
    opts = variant_opts(art, name)
    xi = calm_xi(art, name)
    dk, _lines, pull = _pulls(art, opts, xi)
    K = {}
    for mode, j, small, big in (("surge", 0, 2e-3, 0.05), ("sway", 1, 2e-3, 0.05),
                                ("yaw", 3, math.radians(0.1), math.radians(5.0))):
        K[mode] = {"small_+": pull(mode, j, small) / small,
                   "small_-": pull(mode, j, -small) / -small,
                   "secant_+": pull(mode, j, big) / big, "secant_-": pull(mode, j, -big) / -big,
                   "amp_small": small, "amp_secant": big}
    per = {}
    for mode in ("surge", "sway", "yaw"):
        dec = json.loads((HERE / "tank_rows" / f"decay_{art}_{mode}.json").read_text())
        K_dec = json.loads((HERE / "tank_rows" / f"pull_{art}.json").read_text())[mode]["K0"]
        for kind in ("small", "secant"):
            kk = 0.5 * (K[mode][f"{kind}_+"] + K[mode][f"{kind}_-"])
            per[f"{mode}_{kind}"] = (float(dec["T_s"] * math.sqrt(K_dec / kk)) if kk > 1e-9
                                     else math.inf)
    d = asw.chosen(art)["row"]
    ln0 = bs.ST[art]["lines"][0]
    t_leg = {"V0": ln0["T0_nominal_N"], "V1": t0_v1(art), "T0=0": 0.0,
             "V2": -ln0["k_N_per_m"] * TOL_M}[name]
    modal = asw._shifts(asw.evaluate(art, asw._free(art), asw._kin(art), d["height"],
                                     d["t0_scale"] * t_leg / ln0["T0_nominal_N"], d["k_scale"]),
                        json.loads(asw.OUT.read_text())[art][0]["free"])
    cache = json.loads(fd.EQ_CACHE.read_text())
    tag = asw.chosen(art)["tag"] if name == "V0" else v1_tag(art) if name == "V1" else None
    tilt = cache[tag]["max_buoy_tilt_deg"] if tag and tag in cache else 0.0
    band = {m: _dead_band(art, opts, xi, m) for m in ("surge", "sway")} if name in ("V2", "T0=0") \
        else {}
    T_rest = [s["T"] for s in lh.line_states(dk, xi)]
    out = {"article": art, "variant": name, "T0_leg_nominal_N": t_leg,
           "T_rest_N": [min(T_rest), max(T_rest)], "calm_tilt_deg": tilt,
           "calm_tilt_source": "FloatSim settle" if tag and tag in cache else
           "no tension: no pretension moment (not settled)",
           "tilt_T_s": modal["tilt_T_s"], "heave_T_s": modal["heave_T_s"],
           "tilt_shift_pct": modal["tilt_shift_pct"], "heave_shift_pct": modal["heave_shift_pct"],
           "K": K, "periods_s": per, "dead_band_m": band}
    print(f"{art:8s} {name:5s}: T0 {t_leg:+.3f} N, calm tilt {tilt:.2f} deg, tilt "
          f"{modal['tilt_shift_pct']:+.2f} %, surge K small {K['surge']['small_+']:.1f}/"
          f"{K['surge']['small_-']:.1f}, secant {K['surge']['secant_+']:.1f}/"
          f"{K['surge']['secant_-']:.1f} N/m, sway small {K['sway']['small_+']:.2f}, secant "
          f"{K['sway']['secant_+']:.2f}, band {band}", flush=True)
    return out


# ----------------------------------------------------------------------------- runs
def _dist_seg(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    ab = B - A
    t = np.clip(((P - A) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (A + np.outer(t, ab)), axis=1)


def _fouling(art: str, dk, xi: np.ndarray, line_idx: int) -> dict:  # type: ignore[no-untyped-def]
    """Minimum distance from line ``line_idx``'s catenary profile (beyond 0.25 m from its
    attachment) to every OTHER spar's surface and to every heave plate (its own included: the
    droop risk), at state ``xi`` (small-angle kinematics). Its own spar is excluded: the line
    starts on it."""
    g = mf.geometry(art)
    ls = lh.line_states(dk, xi, n_pts=81)[line_idx]
    p = ls["pts"]
    s_ = np.linalg.norm(p - p[-1], axis=1)             # from the fairlead end
    p = p[s_ > 0.25]
    names = fd.fsd._validate_body_names(dk)
    own = fd.fsd._materialise_catenary(dk.connections[line_idx], names).body_index
    buoys = [b for b, bd in enumerate(dk.bodies) if bd.hydro_body_label or bd.hydro_database]
    d_spar, d_plate = np.inf, np.inf
    for b in buoys:
        ref = np.asarray(dk.bodies[b].reference_point, dtype=float)
        u, th = xi[6 * b:6 * b + 3], xi[6 * b + 3:6 * b + 6]

        def at(zb: float, ref=ref, u=u, th=th) -> np.ndarray:
            r = np.array([0.0, 0.0, zb - g["cog"]])
            return ref + u + r + np.cross(th, r)

        A, B = at(g["spar_bot"]), at(g["top"])
        if b != own:
            d_spar = min(d_spar, float((_dist_seg(p, A, B) - g["spar_r"]).min()))
        C = at(g["plate_z"])
        n = np.cross(th, [0.0, 0.0, 1.0]) + np.array([0.0, 0.0, 1.0])
        n /= np.linalg.norm(n)
        v = p - C
        h = v @ n
        rho = np.linalg.norm(v - np.outer(h, n), axis=1)
        d = np.where(rho <= g["plate_r"], np.abs(h), np.hypot(rho - g["plate_r"], h))
        d_plate = min(d_plate, float((d - g["plate_t"] / 2).min()))
    q = ls["pts"]
    s2 = np.linalg.norm(q - q[0], axis=1)
    chord = q[0] + np.outer(s2 / s2[-1], q[-1] - q[0])
    return {"min_to_spar_m": d_spar, "min_to_plate_m": d_plate,
            "sag_below_chord_mm": float(1000 * max(0.0, (chord[:, 2] - q[:, 2]).max()))}


def run_one(args: tuple) -> dict:
    art, name, H, T = args
    opts = variant_opts(art, name)
    tag = asw.chosen(art)["tag"] if name == "V0" else v1_tag(art)
    dk, _ln = fd.moored(fd.deck(art), art, **opts)
    K = _pulls(art, opts, calm_xi(art, name))[2]
    r = lh.extreme_run((art, H, T, True, 1.0, {"opts": opts, "tag": tag,
                                              "K_surge": K("surge", 0, 1e-3) / 1e-3,
                                              "history": True}))
    t, X = r.pop("_t"), r.pop("_xi")
    atts, ref, _n, _F = asw._lines(dk)
    anch = np.array([a.anchor_global for a in atts])
    lines = fd.moored(fd.deck(art), art, **opts)[1]
    L0 = np.array([ln["L0"] for ln in lines])
    paths = asw._attach_paths(atts, ref, X)                         # (n_t, n_lines, 3)
    chord = np.linalg.norm(anch[None, :, :] - paths, axis=2)
    slack = chord <= L0[None, :]
    names = [b.name for b in dk.bodies]
    rb = names.index("platform") if "platform" in names else names.index("hub")
    xi_eq = calm_xi(art, name)
    heave = X[:, 6 * rb + 2]
    surge = X[:, 6 * rb] - xi_eq[6 * rb]
    Ts = np.array([[s["T"] for s in lh.line_states(dk, x)] for x in X])
    foul = []
    for i in range(len(atts)):
        k = int(np.argmin(Ts[:, i]))
        foul.append({"line": i, "t_s": float(t[k]), "T_N": float(Ts[k, i]),
                     "slack": bool(slack[k, i]), **_fouling(art, dk, X[k], i)})
    up = anch[:, 0] < 0
    r.update({"variant": name, "heave_rao": float(0.5 * (heave.max() - heave.min()) / (0.5 * H)),
              "surge_amp_m": 0.5 * (r["surge_max_m"] - r["surge_min_m"]),   # article-mean surge
              "ref_body_surge_amp_m": float(0.5 * (surge.max() - surge.min())),
              "slack_fraction": slack.mean(axis=0).tolist(),
              "slack_fraction_upflume_max": float(slack[:, up].mean(axis=0).max()),
              "slack_fraction_downflume_max": float(slack[:, ~up].mean(axis=0).max()),
              "fouling": foul,
              "min_to_spar_m": min(f["min_to_spar_m"] for f in foul),
              "min_to_plate_m": min(f["min_to_plate_m"] for f in foul)})
    print(f"{art:8s} {name} H {H} T {T}: mean {r['mean_offset_m']:.3f} m, surge amp "
          f"{r['surge_amp_m']:.3f} m, heave RAO {r['heave_rao']:.3f}, "
          f"tilt {r['tilt_max_deg']:.1f}, "
          f"T {min(r['T_min_N']):.2f}-{max(r['T_max_N']):.2f} N, slack up/down "
          f"{r['slack_fraction_upflume_max']:.2f}/{r['slack_fraction_downflume_max']:.2f}, "
          f"clear spar {r['min_to_spar_m']:.2f} plate {r['min_to_plate_m']:.2f} m", flush=True)
    return r


# ----------------------------------------------------------------------------- extreme
def _ext_t0(art: str, o0: dict, tr1: float, m: float, t01: float) -> float:
    """The options' T0 (per anchor line; the platform's bridle shares it over 2 legs) giving the
    extreme set V1's at-rest tension: per-leg nominal T_rest + m (T0 - T_rest)."""
    share = 0.5 if art == "platform" else 1.0
    return (tr1 + m * (t01 - tr1)) / share


def extreme_estimate() -> dict:
    out = {}
    fin = {(r["article"], r["T"]): r for r in es.json.loads(es.OUT.read_text())["final"]}
    for art in ARTS:
        m = es.json.loads(es.OUT.read_text())["choice"][art]["m_criterion4"]
        xi0, xi1 = calm_xi(art, "V0"), calm_xi(art, "V1")
        o0 = es.opts(art, m, rest=True)
        # V1's extreme set: the same at-rest tension as V1 (nominal T0 = T_rest + m (T0 - T_rest))
        dk1, _ = fd.moored(fd.deck(art), art, **variant_opts(art, "V1"))
        tr1 = lh.line_states(dk1, xi1)[0]["T"]
        t01 = t0_v1(art)
        o1 = dict(o0)
        o1["T0"] = _ext_t0(art, o0, tr1, m, t01)
        p0 = _pulls(art, o0, xi0)[2]
        p1 = _pulls(art, o1, xi1)[2]
        rows = {}
        for T in es.T_EXT:
            r = fin[(art, T)]
            F = p0("surge", 0, r["mean_offset_m"])
            lo, hi = 0.0, 3.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if p1("surge", 0, mid) < F else (lo, mid)
            d = hi - r["mean_offset_m"]
            rows[T] = {"mean_V0_m": r["mean_offset_m"], "mean_V1_est_m": hi,
                       "max_V0_m": r["surge_max_m"], "max_V1_est_m": r["surge_max_m"] + d,
                       "change_pct": 100 * d / r["surge_max_m"]}
        out[art] = {"m": m, "T_rest_V1_N": tr1, "rows": rows,
                    "run": any(abs(v["change_pct"]) > 5.0 for v in rows.values())}
        print(f"{art:8s} extreme set at V1 tension: " + ", ".join(
            f"T {T}: max {v['max_V0_m']:.3f} -> {v['max_V1_est_m']:.3f} m "
            f"({v['change_pct']:+.1f} %)" for T, v in rows.items()), flush=True)
    return out


def extreme_run_v1(args: tuple) -> dict:
    art, T = args
    res = _load()["extreme"][art]
    m = res["m"]
    o0 = es.opts(art, m, rest=True)
    tr1, t01 = res["T_rest_V1_N"], t0_v1(art)
    o1 = dict(o0)
    o1["T0"] = _ext_t0(art, o0, tr1, m, t01)
    K = _pulls(art, o1, calm_xi(art, "V1"))[2]("surge", 0, 1e-3) / 1e-3
    return lh.extreme_run((art, es.H_EXT, T, True, 1.0, {"opts": o1, "tag": v1_tag(art),
                                                         "K_surge": K, "reuse_eq": True}))


# ----------------------------------------------------------------------------- buoy
def buoy_scan() -> list[dict]:
    art = "buoy"
    free, kin = asw._free(art), asw._kin(art)
    d = asw.chosen(art)["row"]
    free_p = json.loads(asw.OUT.read_text())[art][0]["free"]
    t_v1 = t0_v1(art)
    out = []
    for T0 in sorted({2.40, 2.0, 1.84, 1.6, 1.2, 0.8, 0.46, round(t_v1, 3), 0.25, 0.15},
                     reverse=True):
        r = asw._shifts(asw.evaluate(art, free, kin, d["height"], T0 / 4.0, d["k_scale"]), free_p)
        ty, tp_ = r["periods_s"]["yaw"], r["tilt_T_s"]
        zones = {"T_p": tp_, "2T_p": 2 * tp_, "T_p/2": tp_ / 2}
        near = [z for z, v in zones.items() if abs(ty / v - 1) <= 0.15]
        out.append({"T0_N": T0, "is_V1": abs(T0 - round(t_v1, 3)) < 1e-9, "yaw_T_s": ty,
                    "K_yaw": r["K"]["yaw"], "tilt_T_s": tp_, "tilt_shift_pct": r["tilt_shift_pct"],
                    "heave_shift_pct": r["heave_shift_pct"], "surge_T_s": r["periods_s"]["surge"],
                    "in_wave_band": 1.40 <= ty <= 3.50, "near_parametric": near,
                    "zones_s": zones})
        print(f"buoy T0 {T0:.3f} N: yaw {ty:.2f} s (K {r['K']['yaw']:.3f}), tilt "
              f"{r['tilt_shift_pct']:+.2f} %, band {1.40 <= ty <= 3.50}, near {near}", flush=True)
    return out


def main() -> None:
    """Each step merges only its own keys into a freshly loaded JSON (steps may run in parallel)."""
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1]
    new: dict = {}
    if step == "weight":
        new["weight"] = weight()
    elif step == "settle":
        with ProcessPoolExecutor(max_workers=2) as ex:
            list(ex.map(_settle, ARTS))
        new["v1_T0_leg_N"] = {a: t0_v1(a) for a in (*ARTS, "buoy")}
        new["creep"] = {"per_decade": CREEP_PER_DECADE, "decades_day": DECADES_DAY,
                        "day": CREEP_DAY, "tol_m": TOL_M,
                        "s_peak_op_m": {a: s_peak_op(a) for a in (*ARTS, "buoy")}}
    elif step == "statics":
        todo = [(a, v) for a in ARTS for v in ("V0", "V1", "T0=0", "V2")]
        with ProcessPoolExecutor(max_workers=len(todo)) as ex:
            new["statics"] = list(ex.map(statics_one, todo))
    elif step == "runs":
        names = sys.argv[2:] or ["V0", "V1"]
        todo = [(a, v, H, T) for v in names for a in ARTS for H in H_RUN
                for T in (round(asw.chosen(a)["row"]["tilt_T_s"], 2), T_OFF)]
        with ProcessPoolExecutor(max_workers=len(todo)) as ex:
            runs = list(ex.map(run_one, todo))
        keep = [r for r in _load().get("runs", []) if r["variant"] not in names]
        new["runs"] = keep + runs
    elif step == "extreme":
        est = extreme_estimate()
        new["extreme"] = est
        res = _load()
        res.update(new)
        _save(res)
        todo = [(a, T) for a in ARTS if est[a]["run"] for T in es.T_EXT]
        if todo:
            with ProcessPoolExecutor(max_workers=len(todo)) as ex:
                new["extreme_runs"] = list(ex.map(extreme_run_v1, todo))
    elif step == "buoy":
        new["buoy"] = buoy_scan()
    elif step == "fouling_calm":
        out = {}
        for a in ARTS:
            for v in ("V0", "V1"):
                dk, _ = fd.moored(fd.deck(a), a, **variant_opts(a, v))
                xi = calm_xi(a, v)
                ff = [_fouling(a, dk, xi, i) for i in range(len(dk.connections))]
                out[f"{a}:{v}"] = {"min_to_spar_m": min(f["min_to_spar_m"] for f in ff),
                                   "min_to_plate_m": min(f["min_to_plate_m"] for f in ff),
                                   "sag_mm_max": max(f["sag_below_chord_mm"] for f in ff)}
                print(a, v, out[f"{a}:{v}"], flush=True)
        new["fouling_calm"] = out
    elif step == "report":
        report()
        return
    res = _load()
    res.update(new)
    _save(res)


def _f(x: float, n: int = 2) -> str:
    return "∞" if not math.isfinite(x) else f"{x:.{n}f}"


def report_tables(res: dict) -> dict:
    """Markdown tables for PRETENSION-EVAL.md (every number from pretension_study.json)."""
    w = res["weight"]
    wt = ["| article | " + " | ".join(f"{k}: w {v:.3f} N/m" for k, v in w["w_N_per_m"].items())
          + " |", "|---|" + "---|" * len(w["w_N_per_m"])]
    for art in ("buoy", *ARTS):
        wt.append(f"| {art} | " + " | ".join(f"sag {v['sag_mm']:.0f} mm, T {v['T_N']:.3f} N"
                                             for v in w[art].values()) + " |")
    st = {(r["article"], r["variant"]): r for r in res["statics"]}
    names = ("V0", "V1", "T0=0", "V2")
    stat = {}
    for art in ARTS:
        rows = [f"| {art} | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]

        def row(lab: str, fn, art: str = art) -> str:  # type: ignore[no-untyped-def]
            return f"| {lab} | " + " | ".join(fn(st[(art, v)]) for v in names) + " |"

        def kk(mode: str, kind: str, unit: str):  # type: ignore[no-untyped-def]
            return lambda r: (f"{_f(r['K'][mode][kind + '_+'])} / {_f(r['K'][mode][kind + '_-'])}"
                              f" {unit}")

        rows += [
            row("nominal T0 per line/leg (at rest)", lambda r: f"{r['T0_leg_nominal_N']:+.3f} N "
                f"({r['T_rest_N'][0]:.3f})"),
            row("calm static tilt", lambda r: f"{r['calm_tilt_deg']:.2f}°"
                + ("" if r["calm_tilt_source"] == "FloatSim settle" else " (no tension)")),
            row("tilt period shift vs free (modal)", lambda r: f"{r['tilt_shift_pct']:+.2f} %"),
            row("heave period shift vs free", lambda r: f"{r['heave_shift_pct']:+.2f} %"),
            row("surge pull K, ±2 mm: + / −", kk("surge", "small", "N/m")),
            row("surge pull K, ±50 mm secant: + / −", kk("surge", "secant", "N/m")),
            row("sway pull K, ±2 mm: + / −", kk("sway", "small", "N/m")),
            row("sway pull K, ±50 mm secant: + / −", kk("sway", "secant", "N/m")),
            row("yaw pull K, ±0.1°: + / −", kk("yaw", "small", "N·m/rad")),
            row("yaw pull K, ±5° secant: + / −", kk("yaw", "secant", "N·m/rad")),
            row("surge period: small / secant", lambda r: f"{_f(r['periods_s']['surge_small'], 1)}"
                f" / {_f(r['periods_s']['surge_secant'], 1)} s"),
            row("sway period: small / secant", lambda r: f"{_f(r['periods_s']['sway_small'], 1)}"
                f" / {_f(r['periods_s']['sway_secant'], 1)} s"),
            row("yaw period: small / secant", lambda r: f"{_f(r['periods_s']['yaw_small'], 1)}"
                f" / {_f(r['periods_s']['yaw_secant'], 1)} s"),
            row("dead band (all lines slack): surge / sway", lambda r: (
                f"{1000 * r['dead_band_m']['surge']:.0f} / {1000 * r['dead_band_m']['sway']:.0f} mm"
                if r["dead_band_m"] else "—")),
        ]
        stat[art] = "\n".join(rows)
    rt = ["| article | H | T | set | mean offset | surge amp. | heave RAO | max tilt | line T min / max"
          " | slack fraction: up-flume / down-flume lines | max sag at the slackest instant | slackest instant:"
          " min clearance to another spar / a heave plate |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(res.get("runs", []), key=lambda x: (x["article"], x["H"], -x["T"], x["variant"])):
        slack_any = max(r["slack_fraction_upflume_max"], r["slack_fraction_downflume_max"]) > 0
        if slack_any:
            foul = f"{r['min_to_spar_m']:.2f} / {r['min_to_plate_m']:.2f} m"
        else:                            # never slack: the calm (taut) geometry
            fc = res["fouling_calm"][f"{r['article']}:{r['variant']}"]
            foul = f"never slack (calm: {fc['min_to_spar_m']:.2f} / {fc['min_to_plate_m']:.2f} m)"
        rt.append(f"| {r['article']} | {r['H']} m | {r['T']} s | {r['variant']} | "
                  f"{r['mean_offset_m']:.3f} m | {0.5 * (r['surge_max_m'] - r['surge_min_m']):.3f} m | "
                  f"{r['heave_rao']:.3f} | "
                  f"{r['tilt_max_deg']:.1f}° | {min(r['T_min_N']):.2f} / {max(r['T_max_N']):.2f} N | "
                  f"{r['slack_fraction_upflume_max']:.2f} / {r['slack_fraction_downflume_max']:.2f} | "
                  f"{max(f['sag_below_chord_mm'] for f in r['fouling']):.0f} mm | {foul} |")
    bt = ["| buoy T0 per line | yaw K | yaw period | in the wave band (1.40–3.50 s) | near a parametric"
          " zone (±15 %: T_p, 2T_p, T_p/2) | tilt shift vs free | heave shift | surge period |",
          "|---|---|---|---|---|---|---|---|"]
    for b in res["buoy"]:
        z = ", ".join(f"{n} = {b['zones_s'][n]:.2f} s" for n in b["near_parametric"]) or "—"
        bt.append(f"| {b['T0_N']:.3f} N{' (V1)' if b['is_V1'] else ' (rev C)' if b['T0_N'] == 2.4 else ''}"
                  f" | {b['K_yaw']:.3f} N·m/rad | {b['yaw_T_s']:.2f} s | "
                  f"{'**yes**' if b['in_wave_band'] else 'no'} | {z} | {b['tilt_shift_pct']:+.2f} % | "
                  f"{b['heave_shift_pct']:+.2f} % | {b['surge_T_s']:.1f} s |")
    et = ["| article | T | max surge, rev C extreme set | estimated at V1's at-rest tension | change |",
          "|---|---|---|---|---|"]
    for art in ARTS:
        e = res["extreme"][art]
        for T, v in e["rows"].items():
            et.append(f"| {art} (k ×{e['m']:g}; at rest {e['T_rest_V1_N']:.3f} N) | {T} s | "
                      f"{v['max_V0_m']:.3f} m | {v['max_V1_est_m']:.3f} m | {v['change_pct']:+.1f} % |")
    for r in res.get("extreme_runs", []):
        et.append(f"| {r['article']} FloatSim run at V1 tension | {r['T']} s | — | "
                  f"**{r['surge_max_m']:.3f} m** (mean {r['mean_offset_m']:.3f}) | "
                  f"{'**> +1.0 m**' if r['surge_max_m'] > 1.0 else 'within +1.0 m'} |")
    return {"WEIGHT": "\n".join(wt), "STAT_CLUSTER": stat["cluster"],
            "STAT_PLATFORM": stat["platform"], "RUNS": "\n".join(rt), "BUOY": "\n".join(bt),
            "EXTREME": "\n".join(et)}


def report() -> None:
    res = _load()
    md = (HERE / "PRETENSION-EVAL.template.md").read_text(encoding="utf-8")
    for k, v in report_tables(res).items():
        md = md.replace("{{" + k + "}}", v)
    assert "{{" not in md, md[md.index("{{"):md.index("{{") + 40]
    (HERE / "PRETENSION-EVAL.md").write_text(md, encoding="utf-8")
    print("wrote PRETENSION-EVAL.md")


def _settle(art: str) -> None:
    fd.moored_equilibrium(art, tag=v1_tag(art), **variant_opts(art, "V1"))


if __name__ == "__main__":
    main()
