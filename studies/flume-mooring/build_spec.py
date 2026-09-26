"""Build MOORING-SPEC.md (and MOORING-SPEC.pdf via headless Edge, if present) from the FloatSim
results. Rev C (2026-09-25): two cord sets on the same anchors and attachment points -- the
operational set (rev B's attachment-height design, response tests H <= 0.12 m) and a stiffer
extreme set (load / survival tests H = 0.2-0.5 m) -- with the figures of mooring_figures.py.

Sources (every number in the tables comes from them):
- spec_statics.json / spec_statics_extreme.json (mooring_spec.py [extreme]): the lines, pull
  stiffness and periods of each set;
- attachment_sweep.json (attachment_sweep.py): the sweep and the operational selection;
- attachment_design.json (attachment_sweep.py settle/extremes): the operational set's settles,
  its moored operational checks and its H = 0.5 m runs (rev B);
- extreme_set.json (extreme_set.py): the extreme set's k scan, choice, settle, declared periods;
- buoy_stability_check.json: the single buoy's moored runs (one cord set).
- figs/*.png (mooring_figures.py), embedded in the PDF.

Run: python build_spec.py
"""
# ruff: noqa: E501, RUF001  -- Markdown table rows are long; tables print the multiplication sign
from __future__ import annotations

import base64
import html
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def _load(name: str) -> dict:
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else {}


ST = _load("spec_statics.json")
STX = _load("spec_statics_extreme.json")
AS = _load("attachment_sweep.json")
AD = _load("attachment_design.json")
EX = _load("extreme_set.json")
BS = json.loads((HERE / "buoy_stability_check.json").read_text())
ARTS = ("buoy", "cluster", "platform")
ART2 = ("cluster", "platform")
SAFETY, STROKE, SLACK = 3.0, 1.25, 0.15
NAMES = {"buoy": "Single buoy", "cluster": "Cluster (4 buoys, 45°)",
         "platform": "4×4 platform (16 buoys, 45°)"}
FOV_MAX = 1.0
H_OP = 0.12
RHO = 998.0                     # articulated_wall.RHO, the decks' water density
COLLAR_R, COLLAR_ARMS = 0.2, 4  # attachment_sweep.COLLAR, one arm per line
ADDED_MASS_BUDGET_KG = 0.4      # Xabier (rev C decisions): the slender collar's budget
HEAVE_PLATE_AM_KG = 7.9         # record (DESIGN-BASIS Phase F, F4)
ANCHOR_RATING_N = 300.0         # Xabier (rev C record): every wall anchor rated >= 300 N
# rev C.1: cord creep / relaxation, an ASSUMPTION to confirm against the purchased cord's datasheet:
# wet natural rubber, ~4 % of the stretch per decade of time (Gent, Engineering with Rubber
# ch. 7; "Long-time creep in a pure-gum rubber vulcanizate", PMC6728486); pretension_study.py
CREEP_PER_DECADE = 0.04
T_REF_MIN = 1.0                 # the installation reading: 1 min after tensioning
CREEP_TIMES_MIN = (1, 10, 60, 240, 480, 1440)
MARGIN = 0.15                   # the operational no-slack criterion, T_min >= 0.15 T_rest
YAW_ZONE_CLEAR = 0.15           # buoy yaw kept 15 % clear of the T_p/2 parametric zone
PS = _load("pretension_study.json")


def f(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def xyz(v: list[float]) -> str:
    return "(" + ", ".join(f"{c:+.3f}" for c in v) + ")"


def lim(art: str) -> float:
    return AS["choice"][art]["tilt_limit_pct"]


def design(art: str) -> dict:
    return AS["choice"][art]["design"]


def m_ext(art: str) -> float:
    return EX["choice"][art]["m_criterion4"]


def op_runs(art: str) -> list[dict]:
    """The operational set's predicted runs in its band (H <= 0.12 m); the buoy's single cord
    set: every predicted run (its diverging runs are not predictions)."""
    if art == "buoy":
        return [r for r in BS if r["state"] != "diverges"]
    return [r for r in AD["extremes"] if r["article"] == art and r["H"] <= H_OP + 1e-9]


def ext_runs(art: str) -> list[dict]:
    """The extreme set's final runs: the chosen k, at-rest-matched pretension (extreme_set.py)."""
    return [r for r in EX["final"] if r["article"] == art]


def fails(art: str, r: dict) -> list[str]:
    out = []
    if abs(r["tilt_shift_pct"]) > lim(art):
        out.append(f"tilt {r['tilt_shift_pct']:+.2f} %")
    if abs(r["heave_shift_pct"]) > 1.0:
        out.append(f"heave {r['heave_shift_pct']:+.2f} %")
    if r["static_tilt_deg"] > 1.0 + 1e-6:
        out.append(f"static {r['static_tilt_deg']:.2f}°")
    if r["periods_s"]["surge"] < 14.0:
        out.append(f"surge {r['periods_s']['surge']:.1f} s")
    if r["T_min_ratio_op"] < SLACK - 1e-6:
        out.append(f"op slack {r['T_min_ratio_op']:.4f}")
    return out


def sweep_selection() -> list[dict]:
    """Per article, height and k: the passing row with the most pretension, or else the row with
    the most pretension that keeps the static tilt and the operational band (it fails on the
    tilt shift), or else the operational-minimum row (no pretension keeps both). The sweep table
    and the design-rationale chart (mooring_figures.py) both use this selection."""
    out = []
    for art in ARTS:
        rows = [r for r in AS[art] if "article" in r]
        ch = design(art)
        for z in sorted({r["height"] for r in rows}, reverse=True):
            for ks in (1.0, 0.5):
                grp = [r for r in rows if r["height"] == z and r["k_scale"] == ks]
                ok = [r for r in grp if not fails(art, r)]
                keep = [r for r in grp if r["static_tilt_deg"] <= 1.0 + 1e-6
                        and r["T_min_ratio_op"] >= SLACK - 1e-6]
                r = (max(ok, key=lambda x: x["T0_line_N"]) if ok else
                     max(keep, key=lambda x: x["T0_line_N"]) if keep else
                     next(x for x in grp if x.get("t0_is_min_op")))
                chosen = r is ch or (r["height"] == ch["height"] and r["k_scale"] == ch["k_scale"]
                                     and abs(r["T0_line_N"] - ch["T0_line_N"]) < 1e-9)
                verdict = ("**CHOSEN**" if chosen else "pass") if ok else \
                    "fails: " + ", ".join(fails(art, r)) + \
                    ("" if keep else " (even the least T0 that keeps the operational band taut)")
                out.append({"article": art, "height": z, "k_scale": ks, "row": r,
                            "passes": bool(ok), "chosen": chosen, "verdict": verdict})
    return out


def sweep_rows() -> str:
    out = ["| article | attachment z | k | T0 per line/leg | tilt shift (limit) | heave shift | calm static tilt | surge | op. T_min/T0 | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for sel in sweep_selection():
        art, z, ks, r = sel["article"], sel["height"], sel["k_scale"], sel["row"]
        zl = "pin +0.717" if r["name"] == "pin" else f"{z:+.2f}" + (" (CoG)" if r["name"] == "cog" else "")
        out.append(
            f"| {art} | {zl} m | ×{ks:g} | {f(r['T0_line_N'])} N | "
            f"{r['tilt_shift_pct']:+.2f} % ({f(lim(art))}) | {r['heave_shift_pct']:+.2f} % | "
            f"{f(r['static_tilt_deg'])}° | {f(r['periods_s']['surge'], 1)} s | "
            f"{f(r['T_min_ratio_op'])} | {sel['verdict']} |")
    return "\n".join(out)


def retension(r: dict) -> float:
    """Peak tension of the lines that went slack (T_min < 0.15 T0) in the run; 0 if none."""
    out = [tx for tx, tn, t0 in zip(r["T_max_N"], r["T_min_N"], r["T0_N"], strict=True)
           if tn < SLACK * t0]
    return max(out) if out else 0.0


def retension_cell(r: dict) -> str:
    """The slack lines' peak tension; "stay slack" when they never come taut in the window."""
    rt = retension(r)
    if not rt:
        return "none"
    if rt < SLACK * max(r["T0_N"]):
        return f"stay slack (≤ {f(rt, 2)} N, their hanging weight): no re-tension in the wave train"
    return f"**{f(rt, 2)} N** (slack → taut within the wave train)"


def line_table(st: dict, rr: list[dict], band: tuple[float, float]) -> tuple[str, dict]:
    """Per-line table of one cord set: peaks over the runs ``rr``; k band (fractions)."""
    tmax = [max(r["T_max_N"][i] for r in rr) for i in range(st["n_lines"])]
    rows = [f"| line | anchor (x, y, z) m | attachment (x, y, z) m | L0 unstretched | k target (band {100 * (band[0] - 1):+.0f} / {100 * (band[1] - 1):+.0f} %) | pretension T0 (at rest) | at-rest stretch | peak stretch | elongation capacity ×1.25 | max tension | working load ×3 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    strain = []
    for i, ln in enumerate(st["lines"]):
        k = ln["k_N_per_m"]
        ps = tmax[i] / k
        strain.append(STROKE * ps / ln["L0_m"])
        rows.append(
            f"| {i + 1} ({ln['body']}) | {xyz(ln['anchor_m'])} | {xyz(ln['attachment_m'])} | "
            f"{f(ln['L0_m'], 3)} m | {f(k, 2)} N/m ({f(band[0] * k, 2)}–{f(band[1] * k, 2)}) | "
            f"{f(ln['T0_nominal_N'], 2)} N ({f(ln['T_at_rest_N'], 2)}) | "
            f"{f(1000 * ln['stretch_at_rest_m'], 0)} mm | {f(ps, 2)} m | {f(STROKE * ps, 2)} m "
            f"({f(100 * strain[-1], 0)} % of L0) | {f(tmax[i], 2)} N | {f(SAFETY * tmax[i], 1)} N |")
    anchors: dict[tuple, list[int]] = {}
    for i, ln in enumerate(st["lines"]):
        anchors.setdefault(tuple(round(c, 3) for c in ln["anchor_m"]), []).append(i)
    loads = {a: sum(tmax[i] for i in idx) for a, idx in anchors.items()}
    return "\n".join(rows), {
        "tmax": max(tmax), "peak_stretch": max(t / ln["k_N_per_m"] for t, ln in zip(tmax, st["lines"], strict=True)),
        "strain": max(strain), "anchor_loads": loads, "anchor_lines": anchors}


def anchor_table(sets: list[tuple[str, dict]]) -> tuple[str, float]:
    """Anchor design load per set and the working-load limit = 3 x the max over the sets."""
    head = "| anchor (x, y, z) m | lines | " + " | ".join(f"design load, {n}" for n, _ in sets) + " | anchor working-load limit ×3 (max of the sets) |"
    out = [head, "|" + "---|" * (3 + len(sets))]
    wll = 0.0
    first = sets[0][1]
    for a, idx in first["anchor_lines"].items():
        loads = [s["anchor_loads"][a] for _, s in sets]
        w = SAFETY * max(loads)
        wll = max(wll, w)
        out.append(f"| {xyz(list(a))} | {', '.join(str(i + 1) for i in idx)} | "
                   + " | ".join(f"{f(x, 2)} N" for x in loads) + f" | **{f(w, 1)} N** |")
    return "\n".join(out), wll


def envelope_op() -> str:
    out = ["| article | case (drift sum) | min / max line tension | T_min / T0 (≥ 0.15) | mean offset (range) | max tilt | max yaw | validity |",
           "|---|---|---|---|---|---|---|---|"]
    for art in ART2:
        for r in sorted(op_runs(art), key=lambda x: x["T"]):
            ok = "PASS" if r["T_min_ratio"] >= SLACK else "**FAIL**"
            out.append(
                f"| {art} | H {r['H']} m, T {r['T']} s | {f(min(r['T_min_N']))} / {f(max(r['T_max_N']))} N | "
                f"{f(r['T_min_ratio'])} {ok} | {f(r['mean_offset_m'])} m ({f(r['surge_min_m'])}–{f(r['surge_max_m'])}) | "
                f"{f(r['tilt_max_deg'], 1)}° | {r['antisymmetric_max']['yaw'] * 57.29578:.1e}° | "
                f"{'indicative (tilt > 5.7°)' if r['tilt_max_deg'] > 5.7 else 'LEVEL1 valid'} |")
    return "\n".join(out)


def envelope_ext(rows: list[dict], label: str) -> str:
    out = ["| article | set | case (drift sum) | min / max line tension | slack lines: peak re-tension | mean offset (range) | max tilt | max yaw | line above the local surface | validity |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        fov = f" **> +{FOV_MAX:g} m**" if r["surge_max_m"] > FOV_MAX else ""
        em = r.get("line_above_local_surface_max_m", float("nan"))
        out.append(
            f"| {r['article']} | {label if 'm' not in r else 'extreme, k ×' + format(r['m'], 'g')} | H {r['H']} m, T {r['T']} s | "
            f"{f(min(r['T_min_N']))} / {f(max(r['T_max_N']))} N | {retension_cell(r)} | "
            f"{f(r['mean_offset_m'])} m ({f(r['surge_min_m'])}–{f(r['surge_max_m'])}){fov} | "
            f"{f(r['tilt_max_deg'], 1)}° | {r['antisymmetric_max']['yaw'] * 57.29578:.2f}° | "
            f"{'breaks the surface by ' + f(em) + ' m' if em > 0 else 'submerged (' + f(em) + ' m)'} | "
            f"indicative (tilt > 5.7°) |")
    return "\n".join(out)


def kscan_rows() -> str:
    out = ["| article | k (× operational) | T = 2.35 s: mean / max surge | T = 2.65 s: mean / max surge | peak line tension | slack lines: peak re-tension | verdict (max surge ≤ 1.0 m, both periods) |",
           "|---|---|---|---|---|---|---|"]
    for art in ART2:
        scan = [r for r in EX["runs"] if r["article"] == art and r["t0x"] == 1.0]
        for m in sorted({r["m"] for r in scan}):
            rr = {r["T"]: r for r in scan if r["m"] == m}
            cells = [f"{f(rr[T]['mean_offset_m'])} / {f(rr[T]['surge_max_m'])} m" if T in rr else "—"
                     for T in (2.35, 2.65)]
            ok = all(T in rr and rr[T]["surge_max_m"] <= FOV_MAX for T in (2.35, 2.65))
            mean_ok = all(T in rr and rr[T]["mean_offset_m"] <= FOV_MAX for T in (2.35, 2.65))
            ch = m == m_ext(art)
            verdict = ("**CHOSEN**" if ch else "pass") if ok else ("fails (mean alone passes)" if mean_ok else "fails")
            out.append(f"| {art} | ×{m:g} ({f(m * design(art)['k_line'], 2)} N/m) | {cells[0]} | {cells[1]} | "
                       f"{f(max(max(r['T_max_N']) for r in rr.values()))} N | "
                       f"{f(max(retension(r) for r in rr.values()), 2)} N | {verdict} |")
        fin = {r["T"]: r for r in ext_runs(art)}
        cells = [f"{f(fin[T]['mean_offset_m'])} / {f(fin[T]['surge_max_m'])} m" for T in (2.35, 2.65)]
        ok = all(r["surge_max_m"] <= FOV_MAX for r in fin.values())
        out.append(f"| {art} | **×{m_ext(art):g} FINAL** (at-rest-matched T0) | {cells[0]} | {cells[1]} | "
                   f"{f(max(max(r['T_max_N']) for r in fin.values()))} N | "
                   f"{f(max(retension(r) for r in fin.values()), 2)} N | {'**PASS: the spec**' if ok else '**FAIL**'} |")
    return "\n".join(out)


def buoy_rows() -> str:
    out = ["| H | T | FloatSim | max tilt | max yaw | T_min / T0 | max tension | slack lines: peak re-tension | surge range |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(BS, key=lambda x: (x["H"], x["T"])):
        pred = r["state"] != "diverges"
        st = {"stable": "stable", "yaw grows": "**yaw grows**", "diverges": "**diverges: no prediction**"}[r["state"]]
        fov = f" **> +{FOV_MAX:g} m**" if pred and r["surge_max_m"] > FOV_MAX else ""
        out.append(
            f"| {r['H']} m | {r['T']} s | {st} | "
            + (f"{f(r['tilt_max_deg'], 1)}° | {r['yaw_max_deg']:.1e}° | {f(r['T_min_ratio'])} | "
               f"{f(max(r['T_max_N']))} N | {retension_cell(r)} | "
               f"{f(r['surge_min_m'])}–{f(r['surge_max_m'])} m{fov} |" if pred else "— | — | — | — | — | — |"))
    return "\n".join(out)


def tank_rows() -> str:
    out = ["| article | cord set | surge period | sway period | yaw period | surge pull K / F at 0.25, 0.5 m | sway pull K / F at 0.1 m | yaw pull K / M at 5° | calm static tilt (FloatSim settle) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for art in ARTS:
        for name, st in (("operational" if art != "buoy" else "one set", ST[art]),
                         *((("extreme", STX[art]),) if art in STX else ())):
            p = st["pull"]
            out.append(
                f"| {art} | {name} | {f(st['periods_s']['surge'], 1)} s | {f(st['periods_s']['sway'], 1)} s | "
                f"{f(st['periods_s']['yaw'], 2)} s | {f(p['surge']['K0'], 2)} N/m / "
                f"{f(p['surge']['points'][0]['F'], 2)}, {f(p['surge']['points'][1]['F'], 2)} N | "
                f"{f(p['sway']['K0'], 2)} N/m / {f(p['sway']['points'][0]['F'], 3)} N | "
                f"{f(p['yaw']['K0'], 2)} N·m/rad / {f(p['yaw']['points'][0]['F'], 3)} N·m | "
                f"{f(st['static_tilt_settle_deg'])}° |")
    return "\n".join(out)


def id_threshold(art: str) -> float:
    """Pull-stiffness threshold between the two cord sets: their geometric mean (N/m)."""
    return float((ST[art]["pull"]["surge"]["K0"] * STX[art]["pull"]["surge"]["K0"]) ** 0.5)


def id_rows() -> str:
    out = ["| article | cord set | surge pull stiffness K | pull force at 0.25 / 0.5 m | reads as the extreme set if K ≥ |",
           "|---|---|---|---|---|"]
    for art in ART2:
        for name, st in (("operational", ST[art]), ("extreme", STX[art])):
            p = st["pull"]["surge"]
            out.append(f"| {art} | {name} | {f(p['K0'], 1)} N/m | {f(p['points'][0]['F'], 1)} / "
                       f"{f(p['points'][1]['F'], 1)} N | **{f(id_threshold(art), 0)} N/m** |")
    return "\n".join(out)


def wrong_set() -> tuple[str, str]:
    """The operational cords at H = 0.5 m (the rev B runs): drift, and load against their spec."""
    drift, load = [], []
    for art in ART2:
        rr = [r for r in AD["extremes"] if r["article"] == art and r["H"] == 0.5]
        tmax_op = max(max(r["T_max_N"]) for r in op_runs(art))
        k = ST[art]["lines"][0]["k_N_per_m"]
        drift.append(f"{f(max(r['surge_max_m'] for r in rr))} m (mean {f(max(r['mean_offset_m'] for r in rr))} m)")
        load.append(f"{art} peak tension {f(max(max(r['T_max_N']) for r in rr), 1)} N against their "
                    f"working load {f(SAFETY * tmax_op, 1)} N, stretch {f(max(r['stretch_max_m'] for r in rr))} m "
                    f"against their elongation capacity {f(STROKE * tmax_op / k)} m")
    return " / ".join(drift), "; ".join(load)


def relax(t_min: float) -> float:
    """Fraction of the at-rest tension (and of the pull stiffness) left t_min after tensioning."""
    import math
    return 1.0 - CREEP_PER_DECADE * math.log10(t_min / T_REF_MIN)


def _t_reach(frac: float) -> float:
    """Minutes after tensioning at which the tension has relaxed to ``frac`` of its value."""
    return T_REF_MIN * 10 ** ((1.0 - frac) / CREEP_PER_DECADE)


def _fmt_t(t_min: float) -> str:
    if t_min > 60 * 24 * 365:
        return "never (> 1 year)"
    if t_min > 60 * 24 * 2:
        return f"~{t_min / 1440:.0f} days"
    return f"~{t_min:.0f} min" if t_min < 90 else f"~{t_min / 60:.1f} h"


def creep_rows() -> str:
    out = ["| time since tensioning | tension left | calm tilt, cluster and platform (both sets) | at-rest tension: buoy / cluster / platform (per leg) | surge pull K, operational: cluster / platform | surge pull K, extreme: cluster / platform |",
           "|---|---|---|---|---|---|"]
    for t in CREEP_TIMES_MIN:
        r = relax(t)
        lab = f"{t} min" if t < 60 else f"{t / 60:g} h"
        out.append(f"| {lab} | {100 * r:.1f} % | {ST['cluster']['static_tilt_settle_deg'] * r:.2f}° | "
                   f"{f(ST['buoy']['lines'][0]['T_at_rest_N'] * r)} / {f(ST['cluster']['lines'][0]['T_at_rest_N'] * r)} / "
                   f"{f(ST['platform']['lines'][0]['T_at_rest_N'] * r)} N | "
                   f"{f(ST['cluster']['pull']['surge']['K0'] * r, 1)} / {f(ST['platform']['pull']['surge']['K0'] * r, 1)} N/m | "
                   f"{f(STX['cluster']['pull']['surge']['K0'] * r, 1)} / {f(STX['platform']['pull']['surge']['K0'] * r, 1)} N/m |")
    return "\n".join(out)


def thresholds() -> list[dict]:
    """Re-tension thresholds from the record: the operational no-slack margin (op_runs), the
    tracking window of the extreme set (the pretension study's FloatSim pair: rev C tension and
    V1 tension), the buoy's yaw clear of the T_p/2 parametric zone."""
    out = []
    for art in ART2:
        tr = ST[art]["lines"][0]["T_at_rest_N"]
        tilt0 = ST[art]["static_tilt_settle_deg"]
        dT = tr - min(min(r["T_min_N"]) for r in op_runs(art))
        req = dT / (1 - MARGIN)
        out.append({"article": art, "set": "operational", "T_req": req, "tilt": tilt0 * req / tr,
                    "t_reach": _t_reach(req / tr),
                    "basis": f"operational no-slack margin (T_min ≥ {MARGIN:g} T_rest at H = 0.12 m near the tilt resonance)",
                    "taut_tilt": tilt0 * dT / tr, "taut_t": _t_reach(dT / tr)})
        e = PS["extreme"][art]
        runs = {x["T"]: x for x in PS.get("extreme_runs", []) if x["article"] == art}
        dTr = tr - e["T_rest_V1_N"]
        slope = max(((runs[float(T)]["surge_max_m"] if float(T) in runs else v["max_V1_est_m"])
                     - v["max_V0_m"]) / dTr for T, v in e["rows"].items())
        margin = 1.0 - max(x["surge_max_m"] for x in ext_runs(art))
        req_x = tr - margin / slope
        out.append({"article": art, "set": "extreme", "T_req": max(req_x, 0.0),
                    "tilt": tilt0 * max(req_x, 0.0) / tr,
                    "t_reach": _t_reach(req_x / tr) if req_x > 0 else float("inf"),
                    "basis": f"tracking window at H = 0.5 m (max surge {f(max(x['surge_max_m'] for x in ext_runs(art)), 3)} m "
                             f"+ {slope * 1000:.0f} mm per N of lost tension, FloatSim)"})
    tb = ST["buoy"]["lines"][0]["T_at_rest_N"]
    ty = ST["buoy"]["periods_s"]["yaw"]
    edge = (1 - YAW_ZONE_CLEAR) * design("buoy")["tilt_T_s"] / 2
    req_b = tb * (ty / edge) ** 2                  # K_yaw is proportional to T0 (pretension_study)
    out.append({"article": "buoy", "set": "one set", "T_req": req_b, "tilt": None,
                "t_reach": _t_reach(req_b / tb),
                "basis": f"yaw period ≤ {edge:.3f} s, {100 * YAW_ZONE_CLEAR:.0f} % clear of the T_p/2 parametric zone "
                         f"({design('buoy')['tilt_T_s'] / 2:.2f} s); yaw {ty:.2f} s at {tb:.2f} N, ∝ 1/√T0"})
    return out


def threshold_rows() -> str:
    out = ["| article | cord set | re-tension below: calm tilt / at-rest tension (per line/leg) | basis | reached, at the assumed creep, after |",
           "|---|---|---|---|---|"]
    for t in thresholds():
        tl = "—" if t["tilt"] is None else ("none" if t["T_req"] <= 0 else f"**{t['tilt']:.2f}°**")
        tn = "none (the window margin exceeds the whole tension)" if t["T_req"] <= 0 else f"**{f(t['T_req'])} N**"
        out.append(f"| {t['article']} | {t['set']} | {tl} / {tn} | {t['basis']} | {_fmt_t(t['t_reach'])} |")
    return "\n".join(out)


def pullband_rows() -> str:
    out = ["| article | cord set | surge pull K target | acceptance at 1 min / 1 h / 1 day after tensioning (low edge × tension left; high edge +30 %) |",
           "|---|---|---|---|"]
    for art in ART2:
        for name, st, low in (("operational", ST[art], 0.87), ("extreme", STX[art], 1.0)):
            K = st["pull"]["surge"]["K0"]
            cells = " / ".join(f"{f(low * K * relax(t), 1)}–{f(1.30 * K, 1)}" for t in (1, 60, 1440))
            out.append(f"| {art} | {name} | {f(K, 1)} N/m | {cells} N/m |")
    return "\n".join(out)


def fig(name: str, caption: str, width: int = 100) -> str:
    return f'![{caption}](figs/{name}.png "{width}")\n\n*{caption}*'


def main() -> None:
    tmpl = (HERE / "MOORING-SPEC.template.md").read_text(encoding="utf-8")
    ops, exts, sections, wll = {}, {}, {}, {}
    for art in ARTS:
        band_op = (0.87, 1.30)
        md_op, ops[art] = line_table(ST[art], op_runs(art), band_op)
        parts = []
        if art == "buoy":
            parts.append("**One cord set** (response and extreme tests; its predicted H = 0.5 m offsets stay "
                         f"within +{FOV_MAX:g} m). Peaks over every predicted run (§5):\n\n" + md_op)
            atab, wll[art] = anchor_table([("the cord set", ops[art])])
        else:
            md_x, exts[art] = line_table(STX[art], ext_runs(art), (1.0, 1.30))
            parts.append(f"**Operational cord set** (response tests, H ≤ {H_OP} m). Peaks over its moored "
                         "operational runs (§5):\n\n" + md_op)
            parts.append(f"**Extreme cord set, k ×{m_ext(art):g}** (load / survival tests, H = 0.2–0.5 m). "
                         "Same anchors, attachments and nominal T0. Its stiffness is a MINIMUM: a softer cord "
                         f"puts H = 0.5 m past +{FOV_MAX:g} m, so the band is −0 / +30 %. Peaks over its H = 0.5 m "
                         "runs (§5):\n\n" + md_x)
            atab, wll[art] = anchor_table([("operational", ops[art]), ("extreme", exts[art])])
        sections[art] = "\n\n".join([
            fig(f"plan_{art}", f"{NAMES[art]}: plan, to scale (mooring_figures.py, from spec_statics.json)", 92),
            fig(f"elevation_{art}", f"{NAMES[art]}: elevation looking across the flume, to scale", 92),
            *parts, "Anchors (working-load limit = 3 × the larger design load of the cord sets):\n\n" + atab])

    def row(label: str, fn) -> str:  # type: ignore[no-untyped-def]
        return f"| {label} | " + " | ".join(fn(a) for a in ARTS) + " |"

    def attach(a: str) -> str:
        z = design(a)["height"]
        return {"buoy": f"slender radial collar r = {COLLAR_R:g} m on the spar, 4 points (4 lines) at z = {z:+.2f} m",
                "cluster": f"each of the 4 spars (4 lines) at z = {z:+.2f} m",
                "platform": f"the 8 up-/down-stream row spars at z = {z:+.2f} m (4 two-leg bridles, 8 legs)"}[a]

    def cord(a: str, st: dict, s: dict, band: str) -> str:
        ln = st["lines"][0]
        L0 = " / ".join(sorted({f(x['L0_m'], 3) for x in st['lines']}))
        return (f"k {f(ln['k_N_per_m'], 2)} N/m ({band}), L0 {L0} m, pre-stretch at rest "
                f"{f(1000 * ln['stretch_at_rest_m'], 0)} mm, elongation ≥ {f(STROKE * s['peak_stretch'], 2)} m "
                f"({f(100 * s['strain'], 0)} % of L0)")

    summary = ["| | single buoy | cluster | 4×4 platform |", "|---|---|---|---|",
               row("anchors (under water, on the side walls)",
                   lambda a: f"x ±{abs(ST[a]['lines'][0]['anchor_m'][0]):.1f}, y ±{abs(ST[a]['lines'][0]['anchor_m'][1]):.2f}, "
                             f"z {ST[a]['lines'][0]['anchor_m'][2]:+.2f} m"),
               row("attachment points", attach),
               row("pretension at rest per line/leg, both sets (set by tension or calm tilt)",
                   lambda a: f"**{f(ST[a]['lines'][0]['T_at_rest_N'], 2)} N**"),
               row(f"operational cord (response tests, H ≤ {H_OP} m)",
                   lambda a: cord(a, ST[a], ops[a], "−13 / +30 %")),
               row("extreme cord (H = 0.2–0.5 m)",
                   lambda a: "the same cord (one set)" if a == "buoy" else
                   f"**k ×{m_ext(a):g}**: " + cord(a, STX[a], exts[a], "minimum; −0 / +30 %")),
               row("anchor working load ×3 (max of both sets)", lambda a: f"{f(wll[a], 1)} N"),
               row("anchor rating (specified, every anchor)", lambda a: f"**≥ {ANCHOR_RATING_N:.0f} N**"),
               row("set identification: surge pull stiffness, operational / extreme (threshold)",
                   lambda a: "one set" if a == "buoy" else
                   f"{f(ST[a]['pull']['surge']['K0'], 1)} / {f(STX[a]['pull']['surge']['K0'], 1)} N/m "
                   f"(extreme if ≥ {f(id_threshold(a), 0)} N/m)"),
               row("calm static tilt, FloatSim: operational / extreme",
                   lambda a: (f"{f(ST[a]['static_tilt_settle_deg'])}° (collar)" if a == "buoy" else
                              f"{f(ST[a]['static_tilt_settle_deg'])}° / {f(STX[a]['static_tilt_settle_deg'])}°")),
               row("H = 0.5 m max surge (limit +1.0 m)",
                   lambda a: (f"{f(max(r['surge_max_m'] for r in op_runs(a) if r['H'] == 0.5))} m (predicted cases)"
                              if a == "buoy" else
                              f"{f(max(r['surge_max_m'] for r in ext_runs(a)))} m (extreme set; operational set "
                              f"{f(max(r['surge_max_m'] for r in AD['extremes'] if r['article'] == a and r['H'] == 0.5))} m)")),
                   ]
    arms = COLLAR_ARMS * COLLAR_R
    d_max = (4 * ADDED_MASS_BUDGET_KG / (RHO * 3.141592653589793 * arms)) ** 0.5
    disk = 8 / 3 * RHO * COLLAR_R ** 3
    tmax_buoy = ops["buoy"]["tmax"]
    m_root = tmax_buoy * COLLAR_R
    d_rod = 0.010
    sigma = m_root / (3.141592653589793 * d_rod ** 3 / 32) / 1e6
    fill = {
        "SUMMARY": "\n".join(summary),
        "BUOY": sections["buoy"], "CLUSTER": sections["cluster"], "PLATFORM": sections["platform"],
        "SWEEP": sweep_rows(), "KSCAN": kscan_rows(), "TANK": tank_rows(),
        "ENV_OP": envelope_op(),
        "ENV_EXT": envelope_ext([r for a in ART2 for r in sorted(ext_runs(a), key=lambda x: x["T"])], "extreme"),
        "ENV_REVB": envelope_ext([r for r in AD["extremes"] if r["H"] == 0.5], "operational (rev B)"),
        "BUOYRUNS": buoy_rows(),
        "FIG_RATIONALE": fig("rationale", "Design rationale: tilt-period shift and pretension against attachment height (the sweep table's rows)", 100),
        "FIG_OFFSETS": fig("offsets", "H = 0.5 m mean positions of both cord sets against the tracking window (FloatSim, drift sum)", 78),
        "PINSHIFT": ", ".join(f"{a} {next(s['row'] for s in sweep_selection() if s['article'] == a and s['row']['name'] == 'pin' and s['k_scale'] == 1.0)['tilt_shift_pct']:+.1f} % against ±{f(lim(a))} %" for a in ARTS),
        "ZB": f"{design('buoy')['height']:+.2f}", "ZC": f"{design('cluster')['height']:+.2f}",
        "ZP": f"{design('platform')['height']:+.2f}",
        "WLL": ", ".join(f"{a} {f(wll[a], 1)} N" for a in ARTS),
        "MC": f"{m_ext('cluster'):g}", "MP": f"{m_ext('platform'):g}",
        "MC_MEAN": f"{EX['choice']['cluster']['m_mean_only']:g}", "MP_MEAN": f"{EX['choice']['platform']['m_mean_only']:g}",
        "DMAX": f"{1000 * d_max:.0f}", "ARMS": f"{arms:g}", "DISK": f"{disk:.0f}", "BUDGET": f"{ADDED_MASS_BUDGET_KG:g}",
        "ROD": f"{1000 * d_rod:.0f}", "ROD_AM": f"{RHO * 3.141592653589793 * d_rod ** 2 / 4 * arms:.2f}",
        "SIGMA": f"{sigma:.0f}", "TMAXB": f(tmax_buoy, 2), "PLATE_AM": f"{HEAVE_PLATE_AM_KG:g}",
        "TILT_OP": " / ".join(f"{f(ST[a]['static_tilt_settle_deg'])}°" for a in ART2),
        "TILT_EXT": " / ".join(f"{f(STX[a]['static_tilt_settle_deg'])}°" for a in ART2),
        "SENS": " / ".join(f"{ST[a]['static_tilt_settle_deg'] / ST[a]['lines'][0]['T_at_rest_N']:.2f}°/N" for a in ART2),
        "PRE_EXT": " / ".join(f"{1000 * STX[a]['lines'][0]['stretch_at_rest_m']:.0f} mm" for a in ART2),
        "DECL": ", ".join(f"{a} tilt {EX['declare'][a]['tilt_shift_pct']:+.2f} %, heave {EX['declare'][a]['heave_shift_pct']:+.2f} %, surge {f(EX['declare'][a]['periods_s']['surge'], 1)} s" for a in ART2),
        "T0RAISE": t0_raise_text(),
        "SURGE_AMP": " / ".join(f"{a} +{100 * (1 / (1 - (3.5 / STX[a]['periods_s']['surge']) ** 2) - 1):.0f} % "
                                f"(T_surge {f(STX[a]['periods_s']['surge'], 1)} s)" for a in ART2),
        "RESID": resid_text(),
        "ANCHOR_RATING": f"{ANCHOR_RATING_N:.0f}", "WLL_P": f(wll["platform"], 1),
        "IDTABLE": id_rows(),
        "IDK": ", ".join(f"{a} {f(ST[a]['pull']['surge']['K0'], 1)} → {f(STX[a]['pull']['surge']['K0'], 1)} N/m "
                         f"(×{STX[a]['pull']['surge']['K0'] / ST[a]['pull']['surge']['K0']:.1f})" for a in ART2),
        "WRONG_DRIFT": wrong_set()[0], "WRONG_LOAD": wrong_set()[1],
        "SURGE_T": " / ".join(f"{a} {f(STX[a]['periods_s']['surge'], 1)} s" for a in ART2),
        "CREEP_TABLE": creep_rows(), "THRESH_TABLE": threshold_rows(), "PULLBAND_TABLE": pullband_rows(),
        "CREEP_PCT": f"{100 * CREEP_PER_DECADE:g}", "DAY_TILT": f"{ST['cluster']['static_tilt_settle_deg'] * relax(1440):.2f}",
        "DAY_LEFT": f"{100 * relax(1440):.0f}",
        "PLAT_MARGIN_T": _fmt_t(next(t["t_reach"] for t in thresholds() if t["article"] == "platform" and t["set"] == "operational")),
        "PLAT_TAUT_TILT": f"{next(t['taut_tilt'] for t in thresholds() if t['article'] == 'platform' and t['set'] == 'operational'):.2f}",
        "PLAT_TAUT_T": _fmt_t(next(t["taut_t"] for t in thresholds() if t["article"] == "platform" and t["set"] == "operational")),
        "BUOY_T": _fmt_t(next(t["t_reach"] for t in thresholds() if t["article"] == "buoy")),
    }
    assert max(wll.values()) <= ANCHOR_RATING_N, "an anchor working load exceeds the specified rating"
    md = tmpl
    for k, v in fill.items():
        md = md.replace("{{" + k + "}}", v)
    assert "{{" not in md, re.findall(r"\{\{\w+\}\}", md)
    (HERE / "MOORING-SPEC.md").write_text(md, encoding="utf-8")
    print("wrote MOORING-SPEC.md")
    if EDGE.exists():
        htm = HERE / "MOORING-SPEC.html"
        htm.write_text(to_html(md), encoding="utf-8")
        pdf = HERE / "MOORING-SPEC.pdf"
        pdf.unlink(missing_ok=True)
        subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", htm.as_uri()], check=False, timeout=180,
                       capture_output=True)
        last = -1                    # Edge may finish printing after it returns: wait for a
        for _ in range(180):         # stable file before removing the HTML it reads
            size = pdf.stat().st_size if pdf.exists() else -1
            if size > 0 and size == last:
                break
            last = size
            time.sleep(1.0)
        htm.unlink(missing_ok=True)
        print("wrote MOORING-SPEC.pdf" if pdf.exists() else "PDF not produced")


def resid_text() -> str:
    r = _load("extreme_set_residuals.json")
    if not r:
        return ""
    return ("FloatSim's joint-projected static residual at the operational settle: "
            + "; ".join(f"{a} {r[a]['operational']:.4f} N with the operational lines, "
                        f"{r[a]['extreme at-rest matched']:.4f} N with the extreme lines "
                        f"(nominal T0 instead: {r[a]['extreme nominal T0']:.2f} N)" for a in ART2)
            + " (extreme_set_residuals.json).")


def t0_raise_text() -> str:
    t = EX.get("t0_raise")
    if not t:
        none = all(retension(r) < SLACK * max(r["T0_N"]) for a in ART2 for r in ext_runs(a))
        if not none:
            return "Not run: the slack lines re-tension, but the check was not run (see *Follow-ups*)."
        cap = []
        for a in ART2:
            ln = STX[a]["lines"][0]
            pre3 = ln["T_at_rest_N"] * 3.0 / ST[a]["static_tilt_settle_deg"] / ln["k_N_per_m"]
            off = min(r["mean_offset_m"] for r in ext_runs(a))
            cap.append(f"{a} {1000 * pre3:.0f} mm against a mean offset ≥ {f(off)} m")
        return ("Not run, because there is nothing to cut. In the final runs the down-flume lines "
                "go slack and **stay slack through the wave train** (their peak is their hanging "
                "weight). Their pre-stretch is far below the mean offset, and it stays so even at "
                "the 3° calm-tilt cap (T0 ≈ 3× the at-rest tension: pre-stretch "
                + "; ".join(cap) + "). Raising T0 would only add to the up-flume peak.")
    base = {r["T"]: r for r in ext_runs(t["article"])}
    cells = []
    for r in t["runs"]:
        b = base[r["T"]]
        cells.append(f"T = {r['T']} s: peak re-tension {f(retension(b), 2)} → {f(retension(r), 2)} N, "
                     f"max tension {f(max(b['T_max_N']))} → {f(max(r['T_max_N']))} N, max surge "
                     f"{f(b['surge_max_m'])} → {f(r['surge_max_m'])} m")
    return (f"Checked on the cluster (cheap): T0 ×{t['t0x']:.2f} on the extreme set (calm tilt "
            f"{f(t['calm_tilt_deg'])}°, FloatSim settle). " + "; ".join(cells) + ".")


def _inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


def _img(m: re.Match) -> str:
    alt, path, width = m.group(1), HERE / m.group(2), m.group(3) or "100"
    data = base64.b64encode(path.read_bytes()).decode()
    return (f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{data}" '
            f'style="width:{width}%;display:block;margin:4px auto">')


def to_html(md: str) -> str:
    """Minimal Markdown (headings, tables, lists, paragraphs, bold, code, images) for the PDF."""
    out, lines, i = [], md.splitlines(), 0
    img = re.compile(r'!\[(.*?)\]\((\S+?)(?: "(\d+)")?\)')
    while i < len(lines):
        ln = lines[i]
        if img.fullmatch(ln.strip()):
            out.append(img.sub(_img, ln.strip()))
        elif ln.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append("<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in tbl[0]) + "</tr>")
            for r in tbl[2:]:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</table>")
            continue
        elif m := re.match(r"(#+) (.*)", ln):
            n = len(m.group(1))
            # page one = title + the summary; each later section on a new page
            brk = (' style="page-break-before:always"'
                   if ln.startswith("## ") and any(o.startswith("<h2") for o in out) else "")
            out.append(f"<h{n}{brk}>{_inline(m.group(2))}</h{n}>")
        elif re.match(r" *- ", ln):
            out.append("<ul>")
            while i < len(lines) and re.match(r" *- ", lines[i]):
                item = lines[i].lstrip()[2:]
                while i + 1 < len(lines) and lines[i + 1].startswith("  ") \
                        and not re.match(r" *- ", lines[i + 1]):
                    i += 1
                    item += " " + lines[i].strip()
                out.append(f"<li>{_inline(item)}</li>")
                i += 1
            out.append("</ul>")
            continue
        elif re.match(r"\d+\. ", ln):
            out.append("<ol>")
            while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                item = re.sub(r"^\d+\. ", "", lines[i])
                while i + 1 < len(lines) and lines[i + 1].startswith("   ") \
                        and not re.match(r" *- ", lines[i + 1]):
                    i += 1
                    item += " " + lines[i].strip()
                out.append(f"<li>{_inline(item)}</li>")
                i += 1
            out.append("</ol>")
            continue
        elif ln.strip() == "---":
            out.append("<hr>")
        elif ln.strip():
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    css = ("body{font-family:Segoe UI,Arial,sans-serif;font-size:9pt;margin:0;color:#1b2430}"
           "h1{font-size:15pt;margin:0 0 4px}h2{font-size:12pt;border-bottom:1px solid #9aa5b1}"
           "h3{font-size:10pt}table{border-collapse:collapse;margin:5px 0;font-size:7.5pt}"
           "td,th{border:1px solid #9aa5b1;padding:2px 4px;vertical-align:top}th{background:#e8edf2}"
           "code{font-size:7.5pt}p{margin:4px 0}img{page-break-inside:avoid}"
           "@page{size:A4 landscape;margin:9mm}")
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>" + \
        "\n".join(out) + "</body></html>"


if __name__ == "__main__":
    main()
