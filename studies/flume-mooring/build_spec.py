"""Build MOORING-SPEC.md (and MOORING-SPEC.pdf via headless Edge, if present) from the FloatSim
results. Rev B (2026-09-25): the attachment-height design.

Sources (every number in the tables comes from them):
- spec_statics.json (mooring_spec.py): the lines, pull stiffness and periods;
- attachment_sweep.json (attachment_sweep.py kin/sweep/choose/trim): the sweep and selection;
- attachment_design.json (attachment_sweep.py settle/extremes): settles, worst cases and the
  moored operational checks of the cluster and platform;
- buoy_stability_check.json: the single buoy's moored runs.

Run: python build_spec.py
"""
# ruff: noqa: E501, RUF001  -- Markdown table rows are long; tables print the multiplication sign
from __future__ import annotations

import html
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
ST = json.loads((HERE / "spec_statics.json").read_text())
AS = json.loads((HERE / "attachment_sweep.json").read_text())
AD = json.loads((HERE / "attachment_design.json").read_text())
BS = json.loads((HERE / "buoy_stability_check.json").read_text())
ARTS = ("buoy", "cluster", "platform")
SAFETY, STROKE, SLACK = 3.0, 1.25, 0.15
NAMES = {"buoy": "Single buoy", "cluster": "Cluster (4 buoys, 45°)",
         "platform": "4×4 platform (16 buoys, 45°)"}
FOV_MAX = 1.0


def f(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def xyz(v: list[float]) -> str:
    return "(" + ", ".join(f"{c:+.3f}" for c in v) + ")"


def lim(art: str) -> float:
    return AS["choice"][art]["tilt_limit_pct"]


def design(art: str) -> dict:
    return AS["choice"][art]["design"]


def runs(art: str) -> list[dict]:
    """Every FloatSim moored run of the design that is a prediction (the buoy's diverging runs are
    not)."""
    if art == "buoy":
        return [r for r in BS if r["state"] != "diverges"]
    return [r for r in AD["extremes"] if r["article"] == art]


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


def sweep_rows() -> str:
    """Per article, height and k: the passing row with the most pretension, or else the row with
    the most pretension that keeps the static tilt and the operational band (it fails on the
    tilt shift), or else the operational-minimum row (no pretension keeps both)."""
    out = ["| article | attachment z | k | T0 per line/leg | tilt shift (limit) | heave shift | calm static tilt | surge | op. T_min/T0 | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|"]
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
                zl = "pin +0.717" if r["name"] == "pin" else f"{z:+.2f}" + (" (CoG)" if r["name"] == "cog" else "")
                out.append(
                    f"| {art} | {zl} m | ×{ks:g} | {f(r['T0_line_N'])} N | "
                    f"{r['tilt_shift_pct']:+.2f} % ({f(lim(art))}) | {r['heave_shift_pct']:+.2f} % | "
                    f"{f(r['static_tilt_deg'])}° | {f(r['periods_s']['surge'], 1)} s | "
                    f"{f(r['T_min_ratio_op'])} | {verdict} |")
    return "\n".join(out)


def line_table(art: str) -> tuple[str, dict]:
    st = ST[art]
    rr = runs(art)
    tmax = [max(r["T_max_N"][i] for r in rr) for i in range(st["n_lines"])]
    rows = ["| line | anchor (x, y, z) m | attachment (x, y, z) m | L0 unstretched | k target (band −13 / +30 %) | pretension T0 | at-rest stretch | peak stretch | elongation capacity ×1.25 | max tension | working load ×3 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    strain = []
    for i, ln in enumerate(st["lines"]):
        k = ln["k_N_per_m"]
        ps = tmax[i] / k
        strain.append(STROKE * ps / ln["L0_m"])
        rows.append(
            f"| {i + 1} ({ln['body']}) | {xyz(ln['anchor_m'])} | {xyz(ln['attachment_m'])} | "
            f"{f(ln['L0_m'], 3)} m | {f(k, 2)} N/m ({f(ln['k_band_N_per_m'][0], 2)}–"
            f"{f(ln['k_band_N_per_m'][1], 2)}) | {f(ln['T0_nominal_N'], 2)} N | "
            f"{f(ln['stretch_at_rest_m'], 2)} m | {f(ps, 2)} m | {f(STROKE * ps, 2)} m "
            f"({f(100 * strain[-1], 0)} % of L0) | {f(tmax[i], 2)} N | "
            f"{f(SAFETY * tmax[i], 1)} N |")
    anchors: dict[tuple, list[int]] = {}
    for i, ln in enumerate(st["lines"]):
        anchors.setdefault(tuple(round(c, 3) for c in ln["anchor_m"]), []).append(i)
    arows = ["| anchor (x, y, z) m | lines | anchor design load (sum of line maxima) | anchor working-load limit ×3 |",
             "|---|---|---|---|"]
    wll = 0.0
    for a, idx in anchors.items():
        load = sum(tmax[i] for i in idx)
        wll = max(wll, SAFETY * load)
        arows.append(f"| {xyz(list(a))} | {', '.join(str(i + 1) for i in idx)} | {f(load, 2)} N | "
                     f"**{f(SAFETY * load, 1)} N** |")
    return "\n".join(rows) + "\n\n" + "\n".join(arows), {
        "tmax": max(tmax), "peak_stretch": max(t / ln["k_N_per_m"] for t, ln in zip(tmax, st["lines"], strict=True)),
        "wll_anchor": wll, "strain": max(strain)}


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
    return f"{f(rt, 2)} N"


def envelope_rows() -> str:
    out = ["| article | case (drift sum) | min / max line tension | T_min / T0 | slack lines: peak re-tension | mean offset (range) | max tilt | max yaw | line above the local surface | validity |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for art in ("cluster", "platform"):
        for r in sorted(runs(art), key=lambda x: (-x["H"], x["T"])):
            band = "operational" if r["H"] <= 0.12 + 1e-9 else "extreme"
            rt = retension(r)
            slack = retension_cell(r)
            yaw = f"{r['antisymmetric_max']['yaw'] * 57.29578:.1e}°"
            ok = ("PASS" if r["T_min_ratio"] >= SLACK else "**FAIL**") if band == "operational" \
                else ("slack (allowed)" if rt else "taut")
            fov = f" **> +{FOV_MAX:g} m**" if r["surge_max_m"] > FOV_MAX else ""
            emerge = r.get("line_above_local_surface_max_m", float("nan"))
            out.append(
                f"| {art} | {band}: H {r['H']} m, T {r['T']} s | "
                f"{f(min(r['T_min_N']))} / {f(max(r['T_max_N']))} N | "
                f"{f(r['T_min_ratio'])} {ok} | {slack} | {f(r['mean_offset_m'])} m "
                f"({f(r['surge_min_m'])}–{f(r['surge_max_m'])}){fov} | "
                f"{f(r['tilt_max_deg'], 1)}° | {yaw} | "
                f"{'breaks the surface by ' + f(emerge) + ' m' if emerge > 0 else 'submerged (' + f(emerge) + ' m)'} | "
                f"{'indicative (tilt > 5.7°)' if r['tilt_max_deg'] > 5.7 else 'LEVEL1 valid'} |")
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


def main() -> None:
    tmpl = (HERE / "MOORING-SPEC.template.md").read_text(encoding="utf-8")
    parts, summ = {}, {}
    for art in ARTS:
        parts[art], summ[art] = line_table(art)
    st = ST
    free = {a: AS[a][0]["free"] for a in ARTS}

    def row(label: str, fn) -> str:  # type: ignore[no-untyped-def]
        return f"| {label} | " + " | ".join(fn(a) for a in ARTS) + " |"

    def attach(a: str) -> str:
        z = design(a)["height"]
        return {"buoy": f"radial collar r = 0.2 m on the spar at **z = {z:+.2f} m** (submerged)",
                "cluster": f"each spar at **z = {z:+.2f} m**",
                "platform": f"the 8 up-/down-stream row spars at **z = {z:+.2f} m** (bridles)"}[a]

    def op(a: str) -> str:
        d = design(a)
        direct = ([r["T_min_ratio"] for r in BS if r["H"] <= 0.12 + 1e-9] if a == "buoy" else
                  [r["T_min_ratio"] for r in runs(a) if r["H"] <= 0.12 + 1e-9])
        return (f"{f(d['T_min_ratio_op'])} (kinematic) / {f(min(direct))} (moored FloatSim)"
                if direct else f"{f(d['T_min_ratio_op'])} (kinematic)")

    head = ["| | single buoy | cluster | 4×4 platform |", "|---|---|---|---|",
            row("attachment", attach),
            row("anchors (walls, submerged)", lambda a: f"x ±5.0, y ±1.83, z {design(a)['height']:+.2f} m"),
            row("lines (legs)", lambda a: str(st[a]["n_lines"])),
            row("cord stiffness k (band)", lambda a: f"{f(st[a]['lines'][0]['k_N_per_m'], 2)} N/m "
                f"({f(st[a]['lines'][0]['k_band_N_per_m'][0], 2)}–"
                f"{f(st[a]['lines'][0]['k_band_N_per_m'][1], 2)})"),
            row("pretension T0 per line/leg", lambda a: f"**{f(st[a]['lines'][0]['T0_nominal_N'], 2)} N**"),
            row("unstretched length L0", lambda a: " / ".join(sorted({f(ln['L0_m'], 3) for ln in st[a]['lines']})) + " m"),
            row("at-rest stretch", lambda a: f"{f(st[a]['lines'][0]['stretch_at_rest_m'], 2)} m"),
            row("peak stretch (worst predicted case)", lambda a: f"{f(summ[a]['peak_stretch'], 2)} m"),
            row("elongation capacity ×1.25", lambda a: f"{f(1.25 * summ[a]['peak_stretch'], 2)} m "
                f"({f(100 * summ[a]['strain'], 0)} % of L0)"),
            row("max line tension / working load ×3", lambda a: f"{f(summ[a]['tmax'], 2)} / "
                f"{f(3 * summ[a]['tmax'], 1)} N"),
            row("anchor working-load limit ×3", lambda a: f"**{f(summ[a]['wll_anchor'], 1)} N**"),
            row("calm static tilt (FloatSim; ≤ 1°, proposed)", lambda a: f"{f(st[a]['static_tilt_settle_deg'])}°"),
            row("tilt period vs unmoored (limit ζ/3)", lambda a: f"{design(a)['tilt_shift_pct']:+.2f} % "
                f"({f(free[a]['tilt'])} → {f(design(a)['tilt_T_s'])} s; ≤ {f(lim(a))} %)"),
            row("heave period vs unmoored (≤ 1 %)", lambda a: f"{design(a)['heave_shift_pct']:+.2f} %"),
            row("operational band T_min/T0 (≥ 0.15)", op),
            row("surge / sway / yaw period", lambda a: f"{f(st[a]['periods_s']['surge'], 1)} / "
                f"{f(st[a]['periods_s']['sway'], 1)} / {f(st[a]['periods_s']['yaw'], 2)} s"),
            row("pull stiffness surge / sway", lambda a: f"{f(st[a]['pull']['surge']['K0'], 2)} / "
                f"{f(st[a]['pull']['sway']['K0'], 2)} N/m"),
            row("pull stiffness yaw", lambda a: f"{f(st[a]['pull']['yaw']['K0'], 2)} N·m/rad")]
    tank = ["| | surge period | sway period | yaw period | surge pull K / F at 0.25, 0.5 m | sway pull K / F at 0.1 m | yaw pull K / M at 5° |",
            "|---|---|---|---|---|---|---|"]
    for a in ARTS:
        p = st[a]["pull"]
        tank.append(
            f"| {a} | {f(st[a]['periods_s']['surge'], 1)} s | {f(st[a]['periods_s']['sway'], 1)} s | "
            f"{f(st[a]['periods_s']['yaw'], 2)} s | {f(p['surge']['K0'], 2)} N/m / "
            f"{f(p['surge']['points'][0]['F'], 2)}, {f(p['surge']['points'][1]['F'], 2)} N | "
            f"{f(p['sway']['K0'], 2)} N/m / {f(p['sway']['points'][0]['F'], 3)} N | "
            f"{f(p['yaw']['K0'], 2)} N·m/rad / {f(p['yaw']['points'][0]['F'], 3)} N·m |")
    pin = {a: next(r for r in AS[a] if "article" in r and r["name"] == "pin"
                   and r["k_scale"] == 1.0 and r["t0_scale"] == 1.0) for a in ARTS}
    pins = ", ".join(f"{a} {pin[a]['tilt_shift_pct']:+.1f} % against ±{f(lim(a))} % "
                     f"({abs(pin[a]['tilt_shift_pct']) / lim(a):.1f} ×)" for a in ARTS)
    fill = {"HEADLINE": "\n".join(head), "BUOY": parts["buoy"], "CLUSTER": parts["cluster"],
            "PLATFORM": parts["platform"], "ENVELOPE": envelope_rows(), "BUOYRUNS": buoy_rows(),
            "TANK": "\n".join(tank), "SWEEP": sweep_rows(), "PINSHIFT": pins,
            "WLL": ", ".join(f"{a} {f(summ[a]['wll_anchor'], 1)} N" for a in ARTS),
            "ZB": f"{design('buoy')['height']:+.2f}", "ZC": f"{design('cluster')['height']:+.2f}",
            "ZP": f"{design('platform')['height']:+.2f}",
            "ZETA": ", ".join(f"{a} {3 * lim(a):.2f} %" for a in ARTS),
            "LIMS": ", ".join(f"{a} {f(lim(a))} %" for a in ARTS)}
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
                        f"--print-to-pdf={pdf}", htm.as_uri()], check=False, timeout=120,
                       capture_output=True)
        last = -1                    # Edge may finish printing after it returns: wait for a
        for _ in range(120):         # stable file before removing the HTML it reads
            size = pdf.stat().st_size if pdf.exists() else -1
            if size > 0 and size == last:
                break
            last = size
            time.sleep(1.0)
        htm.unlink(missing_ok=True)
        print("wrote MOORING-SPEC.pdf" if pdf.exists() else "PDF not produced")


def _inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t


def to_html(md: str) -> str:
    """Minimal Markdown (headings, tables, lists, paragraphs, bold, code) for the PDF print."""
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("|"):
            tbl = []
            while i < len(lines) and lines[i].startswith("|"):
                tbl.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            out.append("<table><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in tbl[0]) + "</tr>")
            for r in tbl[2:]:
                out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
            out.append("</table>")
            continue
        m = re.match(r"(#+) (.*)", ln)
        if m:
            n = len(m.group(1))
            # page one = title + the facility assumptions; each later section on a new page
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
    css = ("body{font-family:Segoe UI,Arial,sans-serif;font-size:9.5pt;margin:14mm;color:#1b2430}"
           "h1{font-size:16pt}h2{font-size:12.5pt;border-bottom:1px solid #9aa5b1}h3{font-size:10.5pt}"
           "table{border-collapse:collapse;margin:6px 0;font-size:8pt}"
           "td,th{border:1px solid #9aa5b1;padding:2px 4px;vertical-align:top}th{background:#e8edf2}"
           "code{font-size:8pt}@page{size:A4 landscape;margin:10mm}")
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>" + \
        "\n".join(out) + "</body></html>"


if __name__ == "__main__":
    main()
