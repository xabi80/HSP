"""John's bungee cords on the single buoy and the cluster's operational set (flume-mooring,
2026-10-29; Xabier: "try these cords for buoy and cluster to see the effect").

John's test (one loading run, Bungee_cord_01.xlsx): pull in lbf (scale in pounds mode) at 0-100 %
elongation in 10 % steps, 1/8", 3/16" and 1/4" shock cord. These cords are too stiff for the
spec's softest lines (MOORING-SPEC rev D: buoy 2.01 N/m, cluster operational 4.15 N/m): even one
1/8" strand over the whole anchor-to-spar chord is ~2-4x stiffer. This checks what that costs.

The cord in FloatSim: its catenary is a linear line, EA = k L0. The lines sit at 2-9 % stretch
(at-rest tensions 1.44-2.40 N, swings of ~1 N), inside the cord's first 10 %, where the data
has one point; that part is taken as straight: EA0 = F(10 %) / 0.1 per strand. (If the cord's
stiff start ends before 10 %, it is stiffer still.) One strand spans the chord, so
k = (EA0 + T0) / chord (then L0 = chord - T0 / k and k L0 = EA0). The pretension is the spec's
at-rest tension (extreme_set.rest_ratio); everything else is the spec's operational design.

Steps (FloatSim throughout, through the spec's own functions):
  fit      -- EA0 per cord; k and the multiple m of the operational k, per article.
  statics  -- the spec's selection criteria (attachment_sweep.evaluate: tilt / heave shift vs
              the unmoored article, calm tilt, surge period, operational slack), each cord, at
              the spec's attachment; the spec's own row for comparison.
  sweep    -- 1/8": attachment height x pretension (attachment_sweep.sweep) at the cord's k, and
              the spec's decision rule (attachment_sweep.choose): does any attachment pass?
  settle   -- FloatSim's calm settle of the cluster design the sweep picks (its own tag).
  runs     -- 1/8": the buoy's 18 spec cases at the spec's attachment (buoy_stability_check.CASES,
              210 s, dt 0.0025 s, drift sum), and the cluster's operational band (H 0.12 m at
              1.4 s and its moored tilt resonance) at the spec's attachment and at the sweep's
              pick, as line_hardware.extreme_run.
  lowt0    -- the buoy with the 1/8" cord at the spec's attachment and T0 = 1.0 N (the sweep's
              pretension that brings the tilt shift inside zeta/3; k recomputed for that T0):
              the criteria and the 18 spec cases. Lower pretension softens the collar's yaw
              stiffness, so its yaw stability is checked, not assumed.

Writes cord_check.json.  Run: python cord_check.py fit|statics|sweep|settle|runs|lowt0|report
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import attachment_sweep as asw
import buoy_stability_check as bsc
import extreme_set as es
import line_hardware as lh

OUT = HERE / "cord_check.json"
LBF = 4.448222
ELONG = np.arange(0.0, 1.01, 0.1)
JOHN_LBF = {"1/8": (0, 0.91, 1.1, 1.4, 1.51, 1.66, 1.87, 2.2, 2.45, 2.73, 3.31),
            "3/16": (0, 1.1, 1.55, 1.79, 1.95, 2.2, 2.51, 2.87, 3.31, 4.8, 7.2),
            "1/4": (0, 1.58, 2.87, 3.3, 3.84, 4.41, 4.9, 5.51, 5.99, 6.61, 7.23)}
ARTS = ("buoy", "cluster")
RUN_CORD = "1/8"
LOW_T0_N = 1.0                       # buoy pretension for the 1/8" cord (sweep: tilt -1.34 %)


def ea0(cord: str) -> float:
    """Per-strand axial stiffness over the first 10 % (N per unit strain)."""
    return JOHN_LBF[cord][1] * LBF / 0.1


def line_ref(art: str) -> dict:
    return json.loads((HERE / "spec_statics.json").read_text())[art]["lines"][0]


def cord_m(art: str, cord: str, strands: int = 1) -> dict:
    """k of n strands spanning the chord at the at-rest-matched pretension, and m = k / k_spec."""
    ln = line_ref(art)
    k_op, chord = ln["k_N_per_m"], ln["chord_m"]
    m = 1.0
    for _ in range(50):                    # T0 nominal depends on m (at-rest matching)
        t0n = ln["T0_nominal_N"] * es.rest_ratio(art, m)
        m_new = (strands * ea0(cord) + t0n) / chord / k_op
        if abs(m_new - m) < 1e-12:
            break
        m = m_new
    k = m * k_op
    return {"cord": cord, "strands": strands, "EA0_N": strands * ea0(cord), "k_N_per_m": k,
            "m": m, "T0_nominal_N": t0n, "L0_m": chord - t0n / k, "chord_m": chord,
            "strain_rest": t0n / (k * (chord - t0n / k))}


def fit() -> dict:
    out = {a: [cord_m(a, c) for c in JOHN_LBF] for a in ARTS}
    for a, rows in out.items():
        print(f"{a:8s} spec k {line_ref(a)['k_N_per_m']:.2f} N/m: " + "; ".join(
            f"{r['cord']}\" k {r['k_N_per_m']:.2f} N/m (x{r['m']:.2f}), rest strain "
            f"{100 * r['strain_rest']:.1f} %" for r in rows), flush=True)
    return {"EA0_N": {c: ea0(c) for c in JOHN_LBF}, "articles": out}


def _row(art: str, m: float, free, kin, p_free: dict) -> dict:
    d = asw.chosen(art)["row"]
    r = asw._shifts(asw.evaluate(art, free, kin, d["height"], d["t0_scale"] * es.rest_ratio(art, m),
                                 d["k_scale"] * m), p_free)
    r["m"] = m
    return _checked(art, r)


def _checked(art: str, r: dict) -> dict:
    """The spec's hard criteria (attachment_sweep.choose) on one evaluated row."""
    lim = asw.ZETA[art] / 3 * 100
    r["checks"] = {"tilt shift <= zeta/3": abs(r["tilt_shift_pct"]) <= lim,
                   "heave shift <= 1 %": abs(r["heave_shift_pct"]) <= 1.0,
                   "calm tilt <= 1 deg": r["static_tilt_deg"] <= 1.0 + 1e-6,
                   "surge >= 14 s": r["periods_s"]["surge"] >= 14.0,
                   "op T_min >= 0.15 T0": r["T_min_ratio_op"] >= 0.15 - 1e-6}
    r["tilt_limit_pct"] = lim
    return r


def statics(res: dict) -> dict:
    out = {}
    for art in ARTS:
        free = asw._free(art)
        kin = asw._kin(art)
        p_free = json.loads(asw.OUT.read_text())[art][0]["free"]
        rows = [{"cord": "spec", **_row(art, 1.0, free, kin, p_free)}]
        rows += [{"cord": f["cord"], **_row(art, f["m"], free, kin, p_free)}
                 for f in res["fit"]["articles"][art]]
        for r in rows:
            fails = [k for k, ok in r["checks"].items() if not ok]
            print(f"{art:8s} {r['cord']:5s} k {r['k_line']:.2f} N/m: tilt "
                  f"{r['tilt_shift_pct']:+.2f} % (limit {r['tilt_limit_pct']:.2f}), heave "
                  f"{r['heave_shift_pct']:+.2f} %, surge "
                  f"{r['periods_s']['surge']:.1f} s, sway {r['periods_s']['sway']:.1f} s, yaw "
                  f"{r['periods_s']['yaw']:.2f} s, op T_min/T0 {r['T_min_ratio_op']:.2f} -> "
                  f"{'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}", flush=True)
        out[art] = rows
    return out


def sweep(res: dict) -> dict:
    """attachment_sweep's height x pretension sweep with k fixed at the 1/8" cord, then its rule."""
    k_des = {a: asw.fd.line_design(a, asw.N_SPAR[a])["k_line"] for a in ARTS}
    fit_m = {a: next(f for f in res["fit"]["articles"][a] if f["cord"] == RUN_CORD) for a in ARTS}
    k_scale = {a: fit_m[a]["k_N_per_m"] / k_des[a] for a in ARTS}
    sw = {}
    for art in ARTS:
        asw.K_SCALES = (k_scale[art],)
        sw[art] = asw.sweep(art)
    asw.ARTICLES = ARTS
    return {"k_scale": k_scale, "rows": sw, "choice": asw.choose(sw)}


def _design(art: str, m: float, row: dict) -> dict:
    return {"opts": es.opts(art, m, rest=True), "tag": asw.chosen(art)["tag"],
            "K_surge": row["K"]["surge"], "reuse_eq": True}


def new_design(res: dict) -> dict:
    """The cluster design the sweep's rule picks with the 1/8" cord."""
    d = res["sweep"]["choice"]["cluster"]["design"]
    return {"opts": asw.mooring_opts("cluster", d["height"], d["t0_scale"], d["k_scale"]),
            "tag": f"cluster:cord{RUN_CORD.replace('/', '')}_attach_{d['name']}", "row": d}


def settle(res: dict) -> dict:
    nd = new_design(res)
    asw.fd.moored_equilibrium("cluster", tag=nd["tag"], **nd["opts"])
    rec = json.loads(asw.fd.EQ_CACHE.read_text())[nd["tag"]]
    print(f"cluster {nd['tag']}: calm tilt {rec['max_buoy_tilt_deg']:.3f} deg, joint residual "
          f"{rec['joint_residual_N']:.4f} N", flush=True)
    return {**{k: v for k, v in rec.items() if k != "xi"}, "tag": nd["tag"]}


def run_one(args: tuple) -> dict:
    """One case; a run whose motion blows up (the catenary solver then fails) is recorded as
    diverged -- no prediction, as buoy_stability_check's 'diverges'."""
    art, H, T, design, label = args
    try:
        r = lh.extreme_run((art, H, T, True, 1.0, design))
    except RuntimeError as e:
        return {"article": art, "H": H, "T": T, "cord": RUN_CORD, "design": label,
                "error": str(e)[:200]}
    return {**r, "cord": RUN_CORD, "design": label}


def runs(res: dict) -> list[dict]:
    rows = {a: next(r for r in res["statics"][a] if r["cord"] == RUN_CORD) for a in ARTS}
    spec = {a: _design(a, rows[a]["m"], rows[a]) for a in ARTS}
    nd = new_design(res)
    new = {"opts": nd["opts"], "tag": nd["tag"], "K_surge": nd["row"]["K"]["surge"],
           "reuse_eq": True}
    todo = [("buoy", H, T, spec["buoy"], "spec attachment") for H, T in bsc.CASES]
    todo += [("cluster", asw.H_OP, T, spec["cluster"], "spec attachment")
             for T in (round(rows["cluster"]["tilt_T_s"], 2), *asw.T_KIN)]
    todo += [("cluster", asw.H_OP, T, new, nd["row"]["name"])
             for T in (round(nd["row"]["tilt_T_s"], 2), *asw.T_KIN)]
    with ProcessPoolExecutor(max_workers=len(todo)) as ex:
        out = list(ex.map(run_one, todo))
    for r in out:
        if r["article"] == "buoy" and "error" not in r:
            r["yaw_max_deg"] = float(np.degrees(r["antisymmetric_max"]["yaw"]))
            r["state"] = bsc.classify(r)
    return out


def buoy_low_t0() -> dict:
    """The buoy, 1/8" cord, spec attachment, T0 = LOW_T0_N: criteria and the 18 spec cases."""
    ln = line_ref("buoy")
    k = (ea0(RUN_CORD) + LOW_T0_N) / ln["chord_m"]          # one strand: k L0 = EA0
    k_des = asw.fd.line_design("buoy", 1)["k_line"]
    d = asw.chosen("buoy")["row"]
    t0s, ks = LOW_T0_N / 4.0, k / k_des                     # mooring_opts: buoy T0 = 4 N x t0s
    p_free = json.loads(asw.OUT.read_text())["buoy"][0]["free"]
    r = _checked("buoy", asw._shifts(asw.evaluate("buoy", asw._free("buoy"), asw._kin("buoy"),
                                                  d["height"], t0s, ks), p_free))
    fails = [c for c, ok in r["checks"].items() if not ok]
    print(f"buoy 1/8\" T0 {LOW_T0_N} N: k {k:.2f} N/m, tilt {r['tilt_shift_pct']:+.2f} % (limit "
          f"{r['tilt_limit_pct']:.2f}), heave {r['heave_shift_pct']:+.2f} %, surge "
          f"{r['periods_s']['surge']:.1f} s, yaw {r['periods_s']['yaw']:.2f} s, op T_min/T0 "
          f"{r['T_min_ratio_op']:.2f} -> {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}",
          flush=True)
    design = {"opts": asw.mooring_opts("buoy", d["height"], t0s, ks),
              "tag": f"buoy:cord18_t0_{LOW_T0_N:g}", "K_surge": r["K"]["surge"]}
    label = f"spec attachment, T0 {LOW_T0_N:g} N"
    todo = [("buoy", H, T, design, label) for H, T in bsc.CASES]
    with ProcessPoolExecutor(max_workers=len(todo)) as ex:
        out = list(ex.map(run_one, todo))
    for x in out:
        if "error" not in x:
            x["yaw_max_deg"] = float(np.degrees(x["antisymmetric_max"]["yaw"]))
            x["state"] = bsc.classify(x)
    return {"k_N_per_m": k, "T0_N": LOW_T0_N, "statics": r, "runs": out}


def report_low_t0(res: dict) -> None:
    base = {(r["H"], r["T"]): r for r in json.loads(bsc.OUT.read_text())}
    hi = {(r["H"], r["T"]): r for r in res["runs"] if r["article"] == "buoy"}
    print(f"buoy, 1/8\" at T0 {LOW_T0_N:g} N | the 1/8\" at 2.40 N | the spec cord:")

    def cell(r: dict | None) -> str:
        if r is None:
            return "-"
        if "error" in r:
            return "diverges"
        st = r.get("state") or bsc.classify(
            {**r, "yaw_max_deg": float(np.degrees(r["antisymmetric_max"]["yaw"]))})
        return f"{st}, tilt {r['tilt_max_deg']:.1f}"

    for r in res["buoy_low_t0"]["runs"]:
        key = (r["H"], r["T"])
        extra = ("" if "error" in r else f", T_min/T0 {r['T_min_ratio']:.2f}, T max "
                 f"{max(r['T_max_N']):.2f} N, surge {r['surge_min_m']:.2f}.."
                 f"{r['surge_max_m']:.2f} m")
        print(f"  H {r['H']:<4} T {r['T']:<4}: {cell(r)}{extra} | {cell(hi.get(key))} | "
              f"{cell(base.get(key))}", flush=True)


def report(res: dict) -> None:
    base_b = {(r["H"], r["T"]): r for r in json.loads(bsc.OUT.read_text())}
    print("buoy (1/8\" cord vs the spec cord): H T: state, max tilt, T_min/T0, T max, surge range")
    for r in (x for x in res["runs"] if x["article"] == "buoy"):
        b = base_b.get((r["H"], r["T"]))
        if "error" in r:
            print(f"  H {r['H']:<4} T {r['T']:<4}: diverges (solver failure: no prediction)")
            continue
        bs = bsc.classify({**b, "yaw_max_deg": float(np.degrees(b["antisymmetric_max"]["yaw"]))}) \
            if b else "-"
        bt = b["tilt_max_deg"] if b else float("nan")
        print(f"  H {r['H']:<4} T {r['T']:<4}: {r['state']:10s} (spec {bs:10s}) tilt "
              f"{r['tilt_max_deg']:5.1f} deg (spec {bt:5.1f}), "
              f"T_min/T0 {r['T_min_ratio']:.2f}, T max {max(r['T_max_N']):.2f} N, surge "
              f"{r['surge_min_m']:.2f}..{r['surge_max_m']:.2f} m", flush=True)
    ad = json.loads(asw.OUT_DESIGN.read_text())["extremes"]
    def line(r: dict) -> str:
        return (f"T {r['T']}: T min/max {min(r['T_min_N']):.2f} / {max(r['T_max_N']):.2f} N, "
                f"T_min/T0 {r['T_min_ratio']:.2f}, mean offset {r['mean_offset_m']:.3f} m, surge "
                f"{r['surge_min_m']:.3f}..{r['surge_max_m']:.3f} m, max tilt "
                f"{r['tilt_max_deg']:.1f} deg")

    print("cluster operational band (1/8\" cord; spec rows below each):")
    for r in (x for x in res["runs"] if x["article"] == "cluster"):
        if "error" in r:
            print(f"  {r['design']}, H {r['H']} T {r['T']}: diverges (solver failure)")
            continue
        print(f"  {r['design']}, H {r['H']} " + line(r))
        for b in (x for x in ad if x["article"] == "cluster" and x["H"] == r["H"]
                  and abs(x["T"] - r["T"]) < 0.06):
            print("     spec " + line(b))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if step == "fit":
        res["fit"] = fit()
    elif step == "statics":
        res["statics"] = statics(res)
    elif step == "sweep":
        res["sweep"] = sweep(res)
    elif step == "settle":
        res["settle"] = settle(res)
    elif step == "runs":
        res["runs"] = runs(res)
        report(res)
    elif step == "lowt0":
        res["buoy_low_t0"] = buoy_low_t0()
        report_low_t0(res)
    elif step == "report":
        report(res)
        if "buoy_low_t0" in res:
            report_low_t0(res)
        return
    OUT.write_text(json.dumps(res, indent=1, default=es._jsonable))


if __name__ == "__main__":
    main()
