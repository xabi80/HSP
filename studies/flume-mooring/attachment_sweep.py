"""Attachment height x pretension x stiffness: pick the mooring that disturbs the articles least
(flume-mooring, 2026-09-25; decision rule by Xabier).

Why: the pin-plane lines shift the buoys' tilt resonance ~9 % (k x lever^2 about the tilt mode's
rotation centre near the CoG), several times the resonance's tolerance. The static pretension
moment acts about the PIN; the dynamic coupling acts about the tilt mode's rotation centre (near
the CoG). No single height satisfies both unless the pretension is low.

Steps (FloatSim throughout):
  kin    -- the operational excursion: each article FREE (no lines), held only by the criterion-6
            numerical restraint (60 s surge/sway springs at each body's reference point = the
            buoys' CoGs, so no tilt coupling; the single buoy also gets its collar's yaw
            stiffness), H = 0.12 m at its free tilt resonance and at 1.4 s, wave-relative drag,
            no applied drift. The steady window of the motion is saved (kin_rows/*.npz).
  sweep  -- per article, height, pretension and stiffness:
            * tilt / heave modal periods vs the free article (bem_waterline.periods: K = C + the
              linearised FloatSim catenary force at the free equilibrium; M + A(w) iterated);
            * calm static tilt: articulated articles, pretension moment about the pin over the
              record's per-buoy tilt stiffness (75.36 / 81.45 N m/rad, from FloatSim settles);
              the single buoy's collar balances its pretension (0);
            * surge / sway / yaw periods: the round-1 FloatSim decays scaled by
              sqrt(K_decay / K) (same effective mass), K from the rigid pull (FloatSim lines);
            * operational slack: every line's attachment-point path from the kin runs (small-angle
              kinematics at that height) plus the static mean offset under the drift SUM (the
              recorded bound + FloatSim's own mean force, read from the restraint), T_min =
              T0 - k * max shortening; T0_min = the pretension that keeps T_min >= 0.15 T0;
            * anchor load at rest and in the operational band, stretch at rest.
  choose -- the decision rule: HARD tilt shift <= zeta/3 (zeta at H = 0.04 m), heave shift
            <= 1 %, calm static tilt <= 1 deg, surge >= 14 s, T_min >= 0.15 T0 in the
            operational band; among passing, the MOST pretension, then the attachment closest to
            the waterline, then (added) the stiffest surge. If nothing passes: the least tilt
            shift with static tilt <= 1 deg and no operational slack.

  settle -- the chosen designs' calm equilibrium by FloatSim (articulated: the moored settle;
            buoy: the checked static solve with the collar) -> the calm static tilt.
  trim   -- the FloatSim settle wins over the K_TILT estimate: where the settled tilt of the
            chosen articulated design exceeds 1 deg, its row takes the settled value and a row
            with T0 trimmed by the settled ratio (linear in T0) is added; choose again, then
            settle again to verify.
  extremes -- cluster and platform on the chosen design, H = 0.5 m at T = 2.35 / 2.65 s, the
            conservative drift sum (line_hardware.extreme_run); plus the operational check, the
            moored design at H = 0.12 m (its tilt resonance and 1.4 s) with the drift sum.

Writes attachment_sweep.json (kin/sweep/choose) and attachment_design.json (settle/extremes).
Run: python attachment_sweep.py kin|sweep|choose|settle|trim|extremes
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import dataclasses
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

import bem_waterline as bw
import floatsim_decks as fd
import line_hardware as lh
import mooring_sizing as ms
import tank_predictions as tp

import floatsim.driver as fsd
from floatsim.driver import build_system
from floatsim.mooring.catenary_analytic import make_catenary_state_force

OUT = HERE / "attachment_sweep.json"
OUT_DESIGN = HERE / "attachment_design.json"
KIN = HERE / "kin_rows"
ARTICLES = ("buoy", "cluster", "platform")
N_SPAR = {"buoy": 1, "cluster": 4, "platform": 16}
Z_REF = fd.aw.ZB                                   # buoy reference = CoG, -0.907 m
HEIGHTS = {"pin": 0.717, "swl": 0.0, "z-0.15": -0.15, "z-0.30": -0.30, "z-0.50": -0.50,
           "cog": Z_REF}
T0_SCALES = (1.0, 0.75, 0.6, 0.5, 0.35, 0.25)
K_SCALES = (1.0, 0.5)
H_OP = 0.12
T_TILT_FREE = {"buoy": 2.75, "cluster": 2.85, "platform": 2.90}   # free tilt resonance (modal)
T_KIN = (1.4,)                                                    # + the tilt resonance
ZETA = {a: json.loads((HERE / "resonance_bandwidth.json").read_text())["cases"][f"{a}@H0.04"]
        ["pitch"]["zeta"] for a in ARTICLES}                      # tilt zeta at H = 0.04 m
K_TILT = {"cluster": 75.36, "platform": 81.45}                    # N m/rad per buoy (record)
T_REST = 60.0
COLLAR = 0.2
YAW_K_BUOY = 3.337                                               # collar yaw stiffness, T0 4 N


def mooring_opts(art: str, z: float, t0_scale: float, k_scale: float) -> dict:
    des = fd.line_design(art, N_SPAR[art])
    t0 = 4.0 if art == "buoy" else des["T0"]
    opts = {"fairlead": np.array([0.0, 0.0, z - Z_REF]), "anchor_z": z,
            "w_line": 0.3 if z > 0.0 else fd.W_LINE, "T0": t0_scale * t0, "k_scale": k_scale}
    if art == "buoy":
        opts.update(collar="radial", collar_r=COLLAR)
    return opts


# ----------------------------------------------------------------------------- kinematics
def restraint(dk, xi0: np.ndarray, yaw_k: float = 0.0):  # type: ignore[no-untyped-def]
    """The criterion-6 numerical restraint: 60 s surge/sway springs at each body's reference."""
    n = 6 * len(dk.bodies)
    ks = np.array([b.mass for b in dk.bodies]) * (2 * np.pi / T_REST) ** 2

    def f(_t: float, xi: np.ndarray, _xd: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        out[0::6] = -ks * (xi[0::6] - xi0[0::6])
        out[1::6] = -ks * (xi[1::6] - xi0[1::6])
        if yaw_k:
            out[5] -= yaw_k * (xi[5] - xi0[5])
        return out

    return f, float(ks.sum())


def kin_run(args: tuple) -> str:
    art, T = args
    t0 = time.perf_counter()
    dk = fd.deck(art)
    dt = 0.005 if art == "buoy" else None
    s, hd, wave, ramp = fd.wave_setup(dk, art, T, H_OP, dt=dt)
    f, K_tot = restraint(dk, np.asarray(s.xi0), YAW_K_BUOY if art == "buoy" else 0.0)
    r = fd.run_case(s, hd, dk, wave, ramp, 120.0, 4, extra_force=f, dt=dt)
    keep = r.t >= r.t[-1] - 4 * T
    KIN.mkdir(exist_ok=True)
    path = KIN / f"{art}_T{T:g}".replace(".", "p")
    np.savez_compressed(path, t=r.t[keep], xi=r.xi[keep], xi0=np.asarray(s.xi0), K_rest=K_tot,
                        T=T, H=H_OP)
    tilt = max(float(np.degrees(np.abs(r.xi[keep, 6 * b + 4] - s.xi0[6 * b + 4]).max()))
               for b in range(len(dk.bodies)) if dk.bodies[b].hydro_body_label
               or dk.bodies[b].hydro_database)
    return (f"{art} T {T}: max tilt {tilt:.1f} deg, mean surge "
            f"{float(np.mean(r.xi[keep, 0] - s.xi0[0])):.4f} m "
            f"[{(time.perf_counter() - t0) / 60:.1f} min]")


# ----------------------------------------------------------------------------- sweep
def _free(art: str):  # type: ignore[no-untyped-def]
    hd = fd.hdbs(art)
    hdb = hd.get("shared_hydro_database") or hd["bem_databases"]["buoy"]
    dk = fd.deck(art, drag=False)
    s = (fd.build_single(dk, hdb, True) if art == "buoy" else
         build_system(dk, dt=fd.DT, t_max_kernel=fd.T_KERNEL, solve_equilibrium=True, **hd))
    return s, dk, hdb


def _lines(dk_m):  # type: ignore[no-untyped-def]
    ref = np.array([b.reference_point for b in dk_m.bodies], dtype=float)
    names = fsd._validate_body_names(dk_m)
    n = 6 * len(dk_m.bodies)
    atts = [fsd._materialise_catenary(c, names) for c in dk_m.connections]
    return atts, ref, n, make_catenary_state_force(atts, n_dof=n, body_reference_points=ref)


def _attach_paths(atts, ref, xi: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Attachment points (n_t, n_lines, 3), small-angle kinematics (as FloatSim's catenary)."""
    out = []
    for a in atts:
        b = a.body_index
        th = xi[:, 6 * b + 3:6 * b + 6]
        arm = a.fairlead_body + np.cross(th, a.fairlead_body)
        out.append(ref[b] + xi[:, 6 * b:6 * b + 3] + arm)
    return np.stack(out, axis=1)


def evaluate(art: str, free, kin: list[dict], z: float, t0s: float, ks: float) -> dict:
    s0, dk0, hdb = free
    opts = mooring_opts(art, z, t0s, ks)
    dk_m, lines = fd.moored(fd.deck(art, drag=False), art, **opts)
    atts, ref, n, F = _lines(dk_m)
    xi = np.asarray(s0.xi0)
    zero = np.zeros(n)
    s = dataclasses.replace(s0, state_force=F)          # hydro of the free article + these lines
    p = bw.periods(s, dk0, hdb)
    # rigid pull (lines only) and periods scaled from the round-1 decays
    lf = [make_catenary_state_force([a], n_dof=n, body_reference_points=ref) for a in atts]

    def res(x: np.ndarray) -> np.ndarray:
        return tp._resultant(dk0, x, np.sum([f(0.0, x, zero) for f in lf], axis=0))

    per = {}
    K = {}
    for mode, j, hh in (("surge", 0, 1e-3), ("sway", 1, 1e-3), ("yaw", 3, np.radians(0.1))):
        K[mode] = float(-(res(tp.rigid(dk0, xi, mode, hh))[j]
                          - res(tp.rigid(dk0, xi, mode, -hh))[j]) / (2 * hh))
        if art == "buoy" and mode == "yaw":
            per[mode] = float(2 * np.pi * np.sqrt(0.063 / K[mode]))
            continue
        dec = json.loads((HERE / "tank_rows" / f"decay_{art}_{mode}.json").read_text())
        K_dec = json.loads((HERE / "tank_rows" / f"pull_{art}.json").read_text())[mode]["K0"]
        per[mode] = float(dec["T_s"] * np.sqrt(K_dec / K[mode]))
    T0 = np.array([ln["T0"] for ln in lines])
    k = np.array([ln["k"] for ln in lines])
    # calm static tilt: pretension moment about the pin (articulated); the collar balances (buoy)
    if art == "buoy":
        tilt_static = 0.0
    else:
        per_spar: dict[int, float] = {}
        for ln in lines:
            per_spar[ln["body"]] = per_spar.get(ln["body"], 0.0) + ln["T0"]
        lever = 0.717 - z                                   # attachment depth below the pin
        tilt_static = float(np.degrees(max(per_spar.values()) * lever / K_TILT[art]))
    # operational slack: attachment paths from the kin runs + the drift-sum mean offset
    anch = np.array([a.anchor_global for a in atts])
    worst_short, worst_long = 0.0, 0.0
    for kr in kin:
        path = _attach_paths(atts, ref, kr["xi"])
        mean_surge = float(np.mean(kr["xi"][:, 0::6] - kr["xi0"][0::6]))
        path[:, :, 0] -= mean_surge                                  # remove the restraint's mean
        F_own = kr["K_rest"] * mean_surge                            # FloatSim's own mean force
        bound = sum(ms.drift_per_spar(H_OP, kr["T"], steep=ms.STEEP_MATRIX)[:2])
        F_sum = N_SPAR[art] * bound + F_own                          # the conservative drift SUM
        path[:, :, 0] += F_sum / K["surge"]
        p_calm = _attach_paths(atts, ref, xi[None, :])[0]
        l0 = np.linalg.norm(anch - p_calm, axis=1)
        d = l0[None, :] - np.linalg.norm(anch[None, :, :] - path, axis=2)   # >0 = shortening
        worst_short = max(worst_short, float((d * k[None, :] / T0[None, :]).max()))
        worst_long = max(worst_long, float((-d * k[None, :]).max()))
    T_min_ratio = 1.0 - worst_short
    t0_min_scale = t0s * worst_short / 0.85          # T0 with T_min = 0.15 T0 (k kept)
    anchors: dict[tuple, float] = {}
    for ln in lines:
        key = tuple(np.round(ln["anchor"], 3))
        anchors[key] = anchors.get(key, 0.0) + ln["T0"]
    return {"article": art, "height": z, "t0_scale": t0s, "k_scale": ks,
            "T0_line_N": float(T0.max()), "k_line": float(k.max()),
            "tilt_T_s": p["tilt"], "heave_T_s": p["heave"], "static_tilt_deg": tilt_static,
            "periods_s": per, "K": K, "T_min_ratio_op": T_min_ratio,
            "t0_scale_min_op": t0_min_scale, "T_max_op_N": float(T0.max() + worst_long),
            "anchor_rest_N": max(anchors.values()),
            "anchor_op_N": max(anchors.values()) + worst_long * (2 if art == "platform" else 1),
            "stretch_rest_m": float((T0 / k).max())}


def _shifts(r: dict, p_free: dict) -> dict:
    r["name"] = next(k for k, v in HEIGHTS.items() if v == r["height"])
    r["tilt_shift_pct"] = 100 * (r["tilt_T_s"] / p_free["tilt"] - 1)
    r["heave_shift_pct"] = 100 * (r["heave_T_s"] / p_free["heave"] - 1)
    return r


def _kin(art: str) -> list[dict]:
    kin = [dict(np.load(p)) for p in sorted(KIN.glob(f"{art}_T*.npz"))]
    for kr in kin:
        kr["T"] = float(kr["T"])
        kr["K_rest"] = float(kr["K_rest"])
    return kin


def trim(res: dict, settled: dict) -> None:
    for art in ("cluster", "platform"):
        d = res["choice"][art]["design"]
        tilt = settled[art]["max_buoy_tilt_deg"]
        same = [r for r in res[art] if "article" in r and r["height"] == d["height"]
                and r["k_scale"] == d["k_scale"] and abs(r["t0_scale"] - d["t0_scale"]) < 1e-12]
        for r in same:
            r["static_tilt_estimate_deg"] = r["static_tilt_deg"]
            r["static_tilt_deg"] = tilt
            r["static_tilt_source"] = "FloatSim settle"
        print(f"{art:8s}: settled static tilt {tilt:.4f} deg (estimate "
              f"{same[0]['static_tilt_estimate_deg']:.4f})", flush=True)
        if tilt <= 1.0:
            continue
        t0n = d["t0_scale"] / tilt * (1.0 - 1e-3)
        row = _shifts(evaluate(art, _free(art), _kin(art), d["height"], t0n, d["k_scale"]),
                      res[art][0]["free"])
        row.update(t0_is_settle_trim=True, static_tilt_estimate_deg=row["static_tilt_deg"],
                   static_tilt_deg=tilt * t0n / d["t0_scale"],
                   static_tilt_source="FloatSim settle, scaled linearly in T0 (verify: settle)")
        res[art].append(row)
        print(f"{art:8s}: T0 trimmed x{t0n / d['t0_scale']:.4f} -> {row['T0_line_N']:.3f} N, "
              f"tilt shift {row['tilt_shift_pct']:+.2f} %, op T_min/T0 "
              f"{row['T_min_ratio_op']:.2f}", flush=True)


def sweep(art: str) -> list[dict]:
    free = _free(art)
    kin = _kin(art)
    p_free = bw.periods(free[0], free[1], free[2])
    rows = []
    for z in HEIGHTS.values():
        for ks in K_SCALES:
            for t0s in T0_SCALES:
                r = evaluate(art, free, kin, z, t0s, ks)
                rows.append(r)
            # the operational minimum for this height and k (from the design-T0 row's shortening)
            base = next(x for x in rows if x["height"] == z and x["k_scale"] == ks
                        and x["t0_scale"] == 1.0)
            t0m = max(base["t0_scale_min_op"], 1e-3)
            rows.append({**evaluate(art, free, kin, z, t0m, ks), "t0_is_min_op": True})
            # articulated: the pretension at exactly 1 deg of calm static tilt (the most the
            # static-tilt rule allows), if it lies above the operational minimum
            if art != "buoy" and base["static_tilt_deg"] > 1.0:
                t0c = 1.0 / base["static_tilt_deg"]
                if t0c > t0m:
                    rows.append({**evaluate(art, free, kin, z, t0c, ks), "t0_is_static_cap": True})
    for r in rows:
        _shifts(r, p_free)
        print(f"{art:8s} {r['name']:7s} k x{r['k_scale']:<3} T0 x{r['t0_scale']:.3f} "
              f"({r['T0_line_N']:.2f} N): tilt {r['tilt_shift_pct']:+.2f} % heave "
              f"{r['heave_shift_pct']:+.2f} % static {r['static_tilt_deg']:.2f} deg surge "
              f"{r['periods_s']['surge']:.1f} s op T_min/T0 {r['T_min_ratio_op']:.2f}", flush=True)
    return [{"free": p_free}, *rows]


def choose(res: dict) -> dict:
    out = {}
    for art in ARTICLES:
        rows = [r for r in res[art] if "article" in r]
        lim = ZETA[art] / 3 * 100

        def hard(r: dict, lim: float = lim) -> bool:
            return (abs(r["tilt_shift_pct"]) <= lim and abs(r["heave_shift_pct"]) <= 1.0
                    and r["static_tilt_deg"] <= 1.0 + 1e-6 and r["periods_s"]["surge"] >= 14.0
                    and r["T_min_ratio_op"] >= 0.15 - 1e-6)
        ok = [r for r in rows if hard(r)]
        if ok:
            # most pretension; then closest to the waterline; then (added tie-break, from the
            # confirmed Phase B criterion 4, excursion) the stiffest surge = the least offset
            best = max(ok, key=lambda r: (round(r["T0_line_N"], 6), r["height"],
                                          r["K"]["surge"]))
            verdict = "pass"
        else:
            cand = [r for r in rows
                    if r["static_tilt_deg"] <= 1.0 + 1e-6 and r["T_min_ratio_op"] >= 0.15 - 1e-6]
            best = min(cand, key=lambda r: abs(r["tilt_shift_pct"])) if cand else None
            verdict = "nothing passes: least tilt shift"
        out[art] = {"verdict": verdict, "tilt_limit_pct": lim, "n_pass": len(ok), "design": best}
        if best:
            print(f"{art:8s}: {verdict} ({len(ok)} pass) -> {best['name']} k x{best['k_scale']} "
                  f"T0 {best['T0_line_N']:.2f} N: tilt {best['tilt_shift_pct']:+.2f} % (limit "
                  f"{lim:.2f}), heave {best['heave_shift_pct']:+.2f} %, static "
                  f"{best['static_tilt_deg']:.2f} deg, surge {best['periods_s']['surge']:.1f} s, "
                  f"op T_min/T0 {best['T_min_ratio_op']:.2f}", flush=True)
    return out


# ----------------------------------------------------------------------------- chosen design
def chosen(art: str) -> dict:
    """The chosen design's mooring options, settle tag and surge stiffness."""
    d = json.loads(OUT.read_text())["choice"][art]["design"]
    return {"opts": mooring_opts(art, d["height"], d["t0_scale"], d["k_scale"]),
            "tag": f"{art}:attach_{d['name']}_k{d['k_scale']:g}", "K_surge": d["K"]["surge"],
            "row": d}


def settle(art: str) -> dict:
    des = chosen(art)
    if art == "buoy":
        dk, _ = fd.moored(fd.deck(art), art, **des["opts"])
        xi = np.asarray(fd.build_single(dk, fd.hdbs(art)["bem_databases"]["buoy"], True).xi0)
        rec = {"max_buoy_tilt_deg": float(np.degrees(np.hypot(xi[3], xi[4]))),
               "surge_m": float(xi[0]), "note": "checked static solve (residual <= 1e-4 N)"}
    else:
        fd.moored_equilibrium(art, tag=des["tag"], **des["opts"])
        rec = json.loads(fd.EQ_CACHE.read_text())[des["tag"]]
        rec = {k: v for k, v in rec.items() if k != "xi"}
    rec["tag"] = des["tag"]
    print(f"{art:8s} {des['tag']}: calm static tilt {rec['max_buoy_tilt_deg']:.2f} deg "
          f"(estimate {des['row']['static_tilt_deg']:.2f})", flush=True)
    return rec


def extreme(args: tuple) -> dict:
    art, H, T = args
    des = chosen(art)
    return lh.extreme_run((art, H, T, True, 1.0,
                           {k: des[k] for k in ("opts", "tag", "K_surge")}))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if step == "kin":
        cases = [(a, T) for a in ARTICLES for T in (T_TILT_FREE[a], *T_KIN)]
        with ProcessPoolExecutor(max_workers=len(cases)) as ex:
            for line in ex.map(kin_run, cases):
                print(line, flush=True)
    elif step == "sweep":
        for a in (sys.argv[2:] or ARTICLES):
            res[a] = sweep(a)
            OUT.write_text(json.dumps(res, indent=1, default=float))
    elif step == "choose":
        res["choice"] = choose(res)
        OUT.write_text(json.dumps(res, indent=1, default=float))
    elif step == "trim":
        dres = json.loads(OUT_DESIGN.read_text())
        trim(res, dres["settle"])
        res["choice"] = choose(res)
        OUT.write_text(json.dumps(res, indent=1, default=float))
        dres["settle_before_trim"] = dres.pop("settle")
        OUT_DESIGN.write_text(json.dumps(dres, indent=1, default=float))
    elif step in ("settle", "extremes"):
        dres = json.loads(OUT_DESIGN.read_text()) if OUT_DESIGN.exists() else {}
        if step == "settle":
            arts = tuple(sys.argv[2:]) or ARTICLES
            with ProcessPoolExecutor(max_workers=len(arts)) as ex:
                dres["settle"] = {**dres.get("settle", {}),
                                  **dict(zip(arts, ex.map(settle, arts), strict=True))}
        else:
            todo = [(a, 0.5, T) for a in ("cluster", "platform") for T in (2.35, 2.65)]
            todo += [(a, H_OP, T) for a in ("cluster", "platform")
                     for T in (round(chosen(a)["row"]["tilt_T_s"], 2), *T_KIN)]
            with ProcessPoolExecutor(max_workers=len(todo)) as ex:
                dres["extremes"] = list(ex.map(extreme, todo))
        OUT_DESIGN.write_text(json.dumps(dres, indent=1, default=float))


if __name__ == "__main__":
    main()
