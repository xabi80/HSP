"""Build MOORING-SPEC.md (and MOORING-SPEC.pdf via headless Edge, if present) from the FloatSim
results: spec_statics.json (mooring_spec.py), spec_extremes.json (run_spec_extremes.py),
spec_extreme_buoy.json and line_hardware.json (round 2). Every number in the tables comes from
those files. Run: python build_spec.py
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
EX = json.loads((HERE / "spec_extremes.json").read_text())
if (HERE / "spec_extremes_t0scan.json").exists():
    EX += json.loads((HERE / "spec_extremes_t0scan.json").read_text())
EXB = json.loads((HERE / "spec_extreme_buoy.json").read_text())
LH = json.loads((HERE / "line_hardware.json").read_text())
SAFETY, STROKE = 3.0, 1.25
NAMES = {"buoy": "Single buoy", "cluster": "Cluster (4 buoys, 45°)",
         "platform": "4×4 platform (16 buoys, 45°)"}


def runs(art: str) -> list[dict]:
    """The worst-case runs at the specified pretension (spec_statics.json's t0_scale)."""
    if art == "buoy":
        return [EXB]
    sc = ST[art]["t0_scale"]
    return [r for r in EX if r["article"] == art and abs(r["t0_scale"] - sc) < 1e-9]


def scan_rows() -> str:
    """T_min / T0 against the pretension scale: the round-2 runs (x1.0) and today's."""
    out = ["| article | T0 scale | T0 per line/leg | T = 2.35 s: T_min/T0, surge max | T = 2.65 s: T_min/T0, surge max |",
           "|---|---|---|---|---|"]
    base = [r for r in LH["stroke"] if r["applied_drift"] and r["article"] != "buoy"]
    for art in ("cluster", "platform"):
        for sc in sorted({1.0, *(r["t0_scale"] for r in EX if r["article"] == art)}):
            rr = ([r for r in base if r["article"] == art] if sc == 1.0 else
                  [r for r in EX if r["article"] == art and abs(r["t0_scale"] - sc) < 1e-9])
            by_t = {r["T"]: r for r in rr}
            cells = []
            for T in (2.35, 2.65):
                r = by_t.get(T)
                verdict = "" if r is None else ("PASS" if r["T_min_ratio"] >= 0.15 else "**FAIL**")
                cells.append("—" if r is None else
                             f"{r['T_min_ratio']:.2f} {verdict}, {r['surge_max_m']:.2f} m")
            T0 = rr[0]["T0_N"][0] if rr else float("nan")
            out.append(f"| {art} | ×{sc:g} | {T0:.2f} N | {cells[0]} | {cells[1]} |")
    return "\n".join(out)


def f(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def xyz(v: list[float]) -> str:
    return "(" + ", ".join(f"{c:+.3f}" for c in v) + ")"


def line_table(art: str) -> tuple[str, dict]:
    st = ST[art]
    rr = runs(art)
    tmax = [max(r["T_max_N"][i] for r in rr) for i in range(st["n_lines"])]
    tmin = [min(r["T_min_N"][i] for r in rr) for i in range(st["n_lines"])]
    rows = ["| line | anchor (x, y, z) m | attachment (x, y, z) m | L0 unstretched | k target (band −13 / +30 %) | pretension T0 | at-rest stretch | peak stretch | elongation capacity ×1.25 | max tension | working load ×3 |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    peak = {}
    for i, ln in enumerate(st["lines"]):
        k = ln["k_N_per_m"]
        ps = tmax[i] / k
        rows.append(
            f"| {i + 1} ({ln['body']}) | {xyz(ln['anchor_m'])} | {xyz(ln['attachment_m'])} | "
            f"{f(ln['L0_m'], 3)} m | {f(k, 2)} N/m ({f(ln['k_band_N_per_m'][0], 2)}–"
            f"{f(ln['k_band_N_per_m'][1], 2)}) | {f(ln['T0_nominal_N'], 2)} N | "
            f"{f(ln['stretch_at_rest_m'], 2)} m | {f(ps, 2)} m | {f(STROKE * ps, 2)} m "
            f"({f(100 * STROKE * ps / ln['L0_m'], 0)} % of L0) | {f(tmax[i], 1)} N | "
            f"{f(SAFETY * tmax[i], 0)} N |")
        peak[i] = (tmax[i], tmin[i], ps)
    # anchors: lines sharing an anchor point (platform bridles: 2 legs)
    anchors: dict[tuple, list[int]] = {}
    for i, ln in enumerate(st["lines"]):
        anchors.setdefault(tuple(round(c, 3) for c in ln["anchor_m"]), []).append(i)
    arows = ["| anchor (x, y, z) m | lines | anchor design load (sum of line maxima) | anchor working-load limit ×3 |",
             "|---|---|---|---|"]
    wll_max = 0.0
    for a, idx in anchors.items():
        load = sum(tmax[i] for i in idx)
        wll_max = max(wll_max, SAFETY * load)
        arows.append(f"| {xyz(list(a))} | {', '.join(str(i + 1) for i in idx)} | {f(load, 1)} N | "
                     f"**{f(SAFETY * load, 0)} N** |")
    return "\n".join(rows) + "\n\n" + "\n".join(arows), {
        "tmax": max(tmax), "tmin": min(tmin), "peak_stretch": max(p[2] for p in peak.values()),
        "wll_anchor": wll_max,
        "strain": max(STROKE * peak[i][2] / st["lines"][i]["L0_m"] for i in peak)}


def envelope_rows() -> str:
    out = ["| article | case | min / max line tension | T_min / T0 (≥ 0.15) | mean offset (range) | max tilt | min line clearance to local surface | validity |",
           "|---|---|---|---|---|---|---|---|"]
    for art in ("buoy", "cluster", "platform"):
        for r in runs(art):
            T0 = r["T0_N"][0]
            ok = "PASS" if r["T_min_ratio"] >= 0.15 else "**FAIL**"
            fov = " **> +1.0 m**" if r["surge_max_m"] > 1.0 else ""
            out.append(
                f"| {art} | H {r['H']} m, T {r['T']} s, drift sum, T0 {f(T0, 2)} N | "
                f"{f(min(r['T_min_N']), 1)} / {f(max(r['T_max_N']), 1)} N | "
                f"{f(r['T_min_ratio'], 2)} {ok} | {f(r['mean_offset_m'], 2)} m "
                f"({f(r['surge_min_m'], 2)}–{f(r['surge_max_m'], 2)}){fov} | "
                f"{f(r['tilt_max_deg'], 1)}° | {r['clearance_local_surface_min_m']:+.2f} m | "
                f"indicative (tilt > 5.7°) |")
    return "\n".join(out)


def main() -> None:
    tmpl = (HERE / "MOORING-SPEC.template.md").read_text(encoding="utf-8")
    parts, summ = {}, {}
    for art in ("buoy", "cluster", "platform"):
        parts[art], summ[art] = line_table(art)
    head = ["| | single buoy | cluster | 4×4 platform |", "|---|---|---|---|"]
    st = ST

    def row(label: str, fn) -> str:  # type: ignore[no-untyped-def]
        return f"| {label} | " + " | ".join(fn(a) for a in ("buoy", "cluster", "platform")) + " |"

    head += [
        row("lines (legs)", lambda a: str(st[a]["n_lines"])),
        row("attachment", lambda a: "spar-top collar r = 0.2 m, +0.717 m" if a == "buoy"
            else "each moored spar's pin, +0.717 m"),
        row("cord stiffness k (band)", lambda a: f"{f(st[a]['lines'][0]['k_N_per_m'], 2)} N/m "
            f"({f(st[a]['lines'][0]['k_band_N_per_m'][0], 2)}–"
            f"{f(st[a]['lines'][0]['k_band_N_per_m'][1], 2)})"),
        row("pretension T0 per line/leg", lambda a: f"{f(st[a]['lines'][0]['T0_nominal_N'], 2)} N"),
        row("unstretched length L0", lambda a: " / ".join(sorted({f(ln['L0_m'], 3) for ln in st[a]['lines']})) + " m"),
        row("at-rest stretch", lambda a: f"{f(st[a]['lines'][0]['stretch_at_rest_m'], 2)} m"),
        row("peak stretch (worst case)", lambda a: f"{f(summ[a]['peak_stretch'], 2)} m"),
        row("elongation capacity ×1.25", lambda a: f"{f(1.25 * summ[a]['peak_stretch'], 2)} m "
            f"({f(100 * summ[a]['strain'], 0)} % of L0)"),
        row("max line tension / working load ×3", lambda a: f"{f(summ[a]['tmax'], 1)} / "
            f"{f(3 * summ[a]['tmax'], 0)} N"),
        row("anchor working-load limit ×3", lambda a: f"**{f(summ[a]['wll_anchor'], 0)} N**"),
        row("min tension / T0 (≥ 0.15)", lambda a: f(min(r['T_min_ratio'] for r in runs(a)), 2)),
        row("surge / sway / yaw period", lambda a: f"{f(st[a]['periods_s']['surge'], 1)} / "
            f"{f(st[a]['periods_s']['sway'], 1)} / {f(st[a]['periods_s']['yaw'], 2)} s"),
        row("pull stiffness surge / sway", lambda a: f"{f(st[a]['pull']['surge']['K0'], 1)} / "
            f"{f(st[a]['pull']['sway']['K0'], 1)} N/m"),
        row("pull stiffness yaw", lambda a: f"{f(st[a]['pull']['yaw']['K0'], 2)} N·m/rad"),
    ]
    tank = ["| | surge period | sway period | yaw period | surge pull K / F at 0.25, 0.5 m | sway pull K / F at 0.1 m | yaw pull K / M at 5° |",
            "|---|---|---|---|---|---|---|"]
    for a in ("buoy", "cluster", "platform"):
        p = st[a]["pull"]
        tank.append(
            f"| {a} | {f(st[a]['periods_s']['surge'], 1)} s | {f(st[a]['periods_s']['sway'], 1)} s | "
            f"{f(st[a]['periods_s']['yaw'], 2)} s | {f(p['surge']['K0'], 1)} N/m / "
            f"{f(p['surge']['points'][0]['F'], 1)}, {f(p['surge']['points'][1]['F'], 1)} N | "
            f"{f(p['sway']['K0'], 1)} N/m / {f(p['sway']['points'][0]['F'], 2)} N | "
            f"{f(p['yaw']['K0'], 2)} N·m/rad / {f(p['yaw']['points'][0]['F'], 3)} N·m |")
    md = (tmpl.replace("{{HEADLINE}}", "\n".join(head))
          .replace("{{BUOY}}", parts["buoy"]).replace("{{CLUSTER}}", parts["cluster"])
          .replace("{{PLATFORM}}", parts["platform"]).replace("{{ENVELOPE}}", envelope_rows())
          .replace("{{TANK}}", "\n".join(tank)).replace("{{SCAN}}", scan_rows()))
    scales = {a: ST[a]["t0_scale"] for a in ("cluster", "platform")}
    note = ("+" + f"{round(100 * (scales['cluster'] - 1)):d} %" if len(set(scales.values())) == 1
            else f"cluster +{round(100 * (scales['cluster'] - 1)):d} % / platform "
                 f"+{round(100 * (scales['platform'] - 1)):d} %")
    md = md.replace("{{SCALE}}", note)
    assert "{{" not in md
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
        elif ln.startswith("- "):
            out.append("<ul>")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(f"<li>{_inline(lines[i][2:])}</li>")
                i += 1
            out.append("</ul>")
            continue
        elif re.match(r"\d+\. ", ln):
            out.append("<ol>")
            while i < len(lines) and re.match(r"\d+\. ", lines[i]):
                item = re.sub(r"^\d+\. ", "", lines[i])
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
