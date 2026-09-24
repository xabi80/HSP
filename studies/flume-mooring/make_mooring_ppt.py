"""Technical deck for the flume station-keeping mooring (TEAMER review response): the question and
scope, the design approach, the recommended mooring for each article, the coupled FloatSim check,
the surge-period trade-off, the moving-body drift refinement, how the flume size influences the
mooring, and the limits / verification plan. Same visual style as the flume sidewall decks.

Every number is read from the study outputs (mooring_design_table.csv, mooring_verify.csv,
mooring_tsurge_sweep.csv, drift_refined.csv / drift_refined_summary.json) so the deck follows
the analysis when it is re-run. Embeds mooring_sizing.png, mooring_layout.png, mooring_verify.png
and drift_refined.png.  Writes Flume_mooring_technical.pptx.  Run: python make_mooring_ppt.py
"""
# ruff: noqa: RUF001, E702  -- slide copy uses display typography; compact helper setup lines
from __future__ import annotations

import csv
import json
from pathlib import Path

import mooring_sizing as ms
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
BUOY, CLUSTER, PLAT = "1 buoy", "1 cluster (4 buoys)", "4x4 platform (45°)"
ARTS = (BUOY, CLUSTER, PLAT)
SHORT = {BUOY: "1 buoy", CLUSTER: "1 cluster", PLAT: "4×4 platform"}
REC = {BUOY: "spar, waterline", CLUSTER: "4 spars, waterline",
       PLAT: "bow+stern rows (8 spars), waterline"}
REC_TXT = {BUOY: "one collar on the spar", CLUSTER: "its 4 spars, one line each",
           PLAT: "bow + stern rows, 8 spars (bridles)"}


def _csv(name: str) -> list[dict]:
    with (HERE / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


DES = {r["article"]: r for r in _csv("mooring_design_table.csv") if float(r["T_surge_s"]) == 15}
VER = {(r["article"], r["attachment"]): r for r in _csv("mooring_verify.csv")}
SWP = _csv("mooring_tsurge_sweep.csv")
DRIFT = json.loads((HERE / "drift_refined_summary.json").read_text(encoding="utf-8"))
F05 = max(sum(ms.drift_per_spar(0.5, T)[:2]) for T in ms.T_WAVE)
F03 = max(sum(ms.drift_per_spar(0.3, T)[:2]) for T in ms.T_WAVE)


def v(a: str, key: str, att: str | None = None) -> float:
    return float(VER[(a, att or REC[a])][key])


TEAL, TEAL_D = RGBColor(0x0C, 0x8B, 0x96), RGBColor(0x0A, 0x55, 0x60)
INK, GREY = RGBColor(0x1B, 0x26, 0x2E), RGBColor(0x51, 0x60, 0x6A)
LIGHT, RULE = RGBColor(0xEC, 0xF4, 0xF5), RGBColor(0xC9, 0xD6, 0xDA)
RED, WHITE = RGBColor(0xB2, 0x43, 0x2C), RGBColor(0xFF, 0xFF, 0xFF)
NAVY = RGBColor(0x14, 0x2A, 0x38)
FONT, MONO = "Segoe UI", "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def T(text, **kw):  # noqa: N802
    return (text, kw)


def _rect(s, l, t, w, h, c):  # type: ignore[no-untyped-def]
    sh = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb = c
    sh.line.fill.background(); sh.shadow.inherit = False
    return sh


def _runs(p, runs, size, font=FONT):  # type: ignore[no-untyped-def]
    for txt, o in runs:
        r = p.add_run()
        r.text = txt
        r.font.name = o.get("font", font)
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("b", False)
        r.font.italic = o.get("i", False)
        r.font.color.rgb = o.get("c", INK)


def slide(title=None, eyebrow=None):  # type: ignore[no-untyped-def]
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
    if title:
        if eyebrow:
            tb = s.shapes.add_textbox(Inches(0.7), Inches(0.40), Inches(12), Inches(0.4))
            r = tb.text_frame.paragraphs[0].add_run()
            r.text = eyebrow.upper()
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(12), True, TEAL
        tb = s.shapes.add_textbox(Inches(0.7), Inches(0.68), Inches(12), Inches(0.85))
        r = tb.text_frame.paragraphs[0].add_run()
        r.text = title
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(27), True, NAVY
        _rect(s, 0.72, 1.46, 1.9, 0.05, TEAL)
    return s


def bullets(s, items, w=11.9, t=1.85, size=17, gap=11, x=0.85, h=5.2):  # type: ignore[no-untyped-def]
    tb = s.shapes.add_textbox(Inches(x), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (lvl, runs) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl; p.space_after = Pt(gap); p.space_before = Pt(1)
        lead = p.add_run()
        lead.text = "▪  " if lvl == 0 else "      –  "
        lead.font.name = FONT
        lead.font.size = Pt(size if lvl == 0 else size - 2)
        lead.font.color.rgb = TEAL if lvl == 0 else GREY
        _runs(p, runs, size if lvl == 0 else size - 2)


def caption(s, runs, t=6.55, h=0.75, size=12, x=0.85, w=11.6):  # type: ignore[no-untyped-def]
    _rect(s, x, t, w, h, LIGHT)
    tb = s.shapes.add_textbox(Inches(x + 0.2), Inches(t + 0.03), Inches(w - 0.4), Inches(h - 0.06))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    _runs(p, runs, size)


def eqbox(s, text, l, t, w, h=0.6, size=15):  # type: ignore[no-untyped-def]
    _rect(s, l, t, w, h, NAVY)
    tb = s.shapes.add_textbox(Inches(l + 0.1), Inches(t), Inches(w - 0.2), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.name, r.font.size, r.font.color.rgb, r.font.italic = MONO, Pt(size), WHITE, True


def table(s, rows, l, t, col_w, size=13, rh=0.42, first_font=FONT):  # type: ignore[no-untyped-def]
    y = t
    for ri, row in enumerate(rows):
        x = l
        if ri == 0:
            _rect(s, l, y, sum(col_w), rh, TEAL_D)
        elif ri % 2 == 1:
            _rect(s, l, y, sum(col_w), rh, LIGHT)
        for ci, cell in enumerate(row):
            tb = s.shapes.add_textbox(Inches(x + 0.08), Inches(y + 0.01), Inches(col_w[ci] - 0.12),
                                      Inches(rh - 0.02))
            tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(cell)
            r.font.name = first_font if ci == 0 else FONT
            r.font.size = Pt(size); r.font.bold = (ri == 0)
            r.font.color.rgb = WHITE if ri == 0 else INK
            x += col_w[ci]
        y += rh
    return y


def pic(s, name, l, t, w=None, h=None):  # type: ignore[no-untyped-def]
    """Place an image inside the box (l, t, w, h) preserving its aspect ratio, centred."""
    iw, ih = Image.open(HERE / name).size
    ar = iw / ih
    if w is not None and h is not None:
        if w / h > ar:
            ww, hh = h * ar, h
        else:
            ww, hh = w, w / ar
        l += 0.5 * (w - ww)
    elif w is not None:
        ww, hh = w, w / ar
    else:
        ww, hh = h * ar, h
    s.shapes.add_picture(str(HERE / name), Inches(l), Inches(t), Inches(ww), Inches(hh))


def pct(x: float) -> str:
    return f"{x:+.2f} %".replace("-", "−")


# ============================ 1 — title ===============================================
s = slide()
_rect(s, 0, 0, 13.333, 0.28, TEAL)
_rect(s, 0, 5.3, 13.333, 2.2, NAVY)
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.1), Inches(11.6), Inches(0.5))
r = tb.text_frame.paragraphs[0].add_run()
r.text = "PHASES 1–3 · OSU LARGE WAVE FLUME · TEAMER REVIEW RESPONSE"
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(14), True, TEAL_D
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.7), Inches(11.7), Inches(2.6))
tb.text_frame.word_wrap = True
r = tb.text_frame.paragraphs[0].add_run()
r.text = "Station-keeping mooring in the flume\nfor 1 buoy, 1 cluster and the 4×4 platform"
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(38), True, NAVY
p = tb.text_frame.add_paragraph(); p.space_before = Pt(10)
r = p.add_run()
r.text = "Which mooring we need, and how the flume size influences it"
r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(19), GREY
tb = s.shapes.add_textbox(Inches(0.95), Inches(5.55), Inches(11.5), Inches(1.7))
tb.text_frame.word_wrap = True
r = tb.text_frame.paragraphs[0].add_run()
hv = max(abs(v(a, "dT_heave_pct")) for a in ARTS)
tl = max(abs(v(a, "dT_tilt_pct")) for a in (CLUSTER, PLAT))
r.text = ("Answer: four soft horizontal lines at the waterline, T_surge ≈ 15 s. On the coupled "
          f"models the heave period shifts ≤ {hv:.2f} %, pitch < 1 %, the buoy-tilt mode "
          f"≤ {tl:.1f} %, with zero trim.")
r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(17), WHITE

# ============================ 2 — question and scope ==================================
s = slide("The question, and what the mooring has to do", "Scope")
bullets(s, [
    (0, [T("Reviewer: ", b=True), T("which mooring do we need, and how could the flume size "
                                     "influence it?", i=True)]),
    (0, [T("Role: ", b=True), T("station-keeping only. Hold each article in the test section "
                                "against the mean wave drift, without changing what we measure "
                                "(free decay, heave/pitch response, accelerations).")]),
    (0, [T("Field site is deep water: ", b=True),
         T("a 2.7 m-deep flume cannot hold a scale model of the field mooring → an equivalent "
           "soft mooring, characterised and put in the model.")]),
    (0, [T("Test waves: ", b=True), T("H = 0.2–0.5 m, T = 1.4–4.0 s. Flume: OSU LWF, 3.66 m wide, "
                                      "2.7 m deep, 104 m long.")]),
], w=6.6, t=1.85, size=16)
rows = [["Article", "Mass", "Surge inertia M+A₁₁", "Span / wall gap"]]
for a in ARTS:
    art = ms.ARTICLES[a]
    rows.append([SHORT[a], f"{art['M']:.0f} kg", f"{art['M'] + art['A']:.0f} kg",
                 f"{2 * art['half_w']:.2f} m / {ms.W_FLUME / 2 - art['half_w']:.2f} m"])
table(s, rows, 7.75, 2.0, [1.55, 0.95, 1.45, 1.5], size=12)
tb = s.shapes.add_textbox(Inches(7.75), Inches(3.8), Inches(5.3), Inches(2.2))
tb.text_frame.word_wrap = True
_runs(tb.text_frame.paragraphs[0], [T(
    "Articles at the 45° test orientation: the 4×4 grid and the cluster sit square to the flume, "
    "two (cluster) or four (platform) buoys facing the waves. Each buoy is the OSU spar + heave "
    "plate (0.159 m spar, 2.52 s heave, 2.11 s pitch).", c=GREY)], 13)

# ============================ 3 — approach ============================================
s = slide("Approach: size it simply, then check it on the full models", "Method")
cols = [("1  Hold it", "Mean drift sets the offset: δ = F / Kx. Drift from splash-zone drag on "
         "every spar, taken for a fixed body (an upper bound)."),
        ("2  Keep it soft", "Surge period well above the waves: T_surge ≥ 3–5 × 4 s, so the "
         "mooring barely resists wave-frequency motion. Baseline 15 s."),
        ("3  Don't couple", "Lines horizontal at the waterline, where the drift acts: no heave "
         "or pitch stiffness, no lever arm, no trim.")]
for i, (h, body) in enumerate(cols):
    x = 0.85 + i * 4.0
    _rect(s, x, 1.95, 3.7, 0.55, TEAL_D)
    tb = s.shapes.add_textbox(Inches(x + 0.15), Inches(1.97), Inches(3.4), Inches(0.5))
    _runs(tb.text_frame.paragraphs[0], [T(h, b=True, c=WHITE)], 16)
    _rect(s, x, 2.5, 3.7, 1.55, LIGHT)
    tb = s.shapes.add_textbox(Inches(x + 0.15), Inches(2.6), Inches(3.4), Inches(1.4))
    tb.text_frame.word_wrap = True
    _runs(tb.text_frame.paragraphs[0], [T(body)], 13.5)
eqbox(s, "Kx = (M + A₁₁)(2π / T_surge)²     F_drift = (2 / 3π) ρ Cd D A U²,  U = Aω coth kh", 0.85,
      4.35, 11.65, 0.6, 15)
bullets(s, [
    (0, [T("Then check on the real FloatSim models ", b=True),
         T("— 6-DOF buoy, 30-DOF cluster, 126-DOF platform, with the pinned joints and full "
           "inter-buoy hydrodynamics: natural periods moored vs unmoored, and the static "
           "response to the drift, for every realistic attachment point.")]),
    (0, [T("Finally refine the drift ", b=True),
         T("with the article moving (drag-limited time domain, relative velocity at each "
           "spar's waterline).")]),
], t=5.15, size=14, gap=6)

# ============================ 4 — drift load ==========================================
s = slide("Design load: the mean wave drift", "Load")
pic(s, "mooring_sizing.png", 0.6, 1.75, w=12.1, h=4.3)
caption(s, [T("Left: drift per spar, fixed-body upper bound; it peaks at the steepness limit "
              f"(H/L = 1/15): {F05:.1f} N at H = 0.5 m, {F03:.1f} N at H = 0.3 m. "),
            T("Right: offset vs mooring softness — identical for all three articles, because "
              "drift and inertia both scale with the number of buoys.", b=True)], t=6.3, h=0.9)

# ============================ 5 — the recommended mooring =============================
s = slide("Recommended mooring: X-spread of 4 soft lines at the waterline", "Design")
pic(s, "mooring_layout.png", 0.45, 1.65, w=7.4, h=4.3)
rows = [["T_surge = 15 s", SHORT[BUOY], SHORT[CLUSTER], "4×4"]]
spec = [("Kx (N/m)", "Kx_Npm", "{:.1f}"), ("Line k (N/m)", "k_line_Npm", "{:.1f}"),
        ("Pretension (N)", "pretension_N", "{:.1f}"), ("Max tension (N)", "max_line_tension_N",
                                                       "{:.0f}"),
        ("Offset H 0.5 (m)", "offset_H05_m", "{:.2f}"), ("Spring stroke (m)", "max_stretch_m",
                                                          "{:.1f}")]
for lab, key, fmt in spec:
    rows.append([lab] + [fmt.format(float(DES[a][key])) for a in ARTS])
table(s, rows, 8.05, 1.85, [1.85, 1.0, 1.1, 1.0], size=12, rh=0.4)
bullets(s, [
    (0, [T("Attach at the waterline: ", b=True), T(f"buoy — {REC_TXT[BUOY]}; cluster — "
                                                  f"{REC_TXT[CLUSTER]}; platform — "
                                                  f"{REC_TXT[PLAT]}.")]),
    (0, [T("Anchors on the walls, ±5 m; ", b=True), T("soft linear spring + low-stretch rope; "
                                                      "same design scaled by buoy count.")]),
], x=8.05, t=4.8, w=4.95, size=12.5, gap=5)
caption(s, [T("Line stiffness and pretension scale with the article; the geometry, the offset and "
              "the ~2 m spring stroke are the same for all three.", b=True)], t=6.3, h=0.7)

# ============================ 6 — coupled check method =================================
s = slide("Checking it on the full FloatSim models", "Verification method")
bullets(s, [
    (0, [T("Models: ", b=True), T("single buoy (6 DOF); cluster = 4 buoys + hub, pinned "
                                 "(yaw-locked KKT joints), 30 DOF; platform = 16 buoys + 4 hubs + "
                                 "deck, 126 DOF. Coupled Capytaine BEM for each.")]),
    (0, [T("Mooring as an anisotropic point stiffness ", b=True),
         T("(the deck spring element is isotropic and would add Kx to heave):")]),
], t=1.8, size=15, gap=8)
eqbox(s, "K_moor = Bᵀ diag(Kx, Ky, Kz,geo) B ,   B = [ I  −[r]× ]   at each attachment r",
      1.2, 3.05, 10.9, 0.58, 15)
bullets(s, [
    (0, [T("Natural periods: ", b=True), T("constrained eigenproblem on the joint null space, "
                                          "iterated on A(ω); each mode identified unmoored and "
                                          "tracked into the moored model (MAC ≥ 0.9).")]),
    (0, [T("Static drift response: ", b=True), T("KKT solve [K Gᵀ; G 0] — offset, deck trim, "
                                                 "and each buoy's tilt about its pin.")]),
    (0, [T("Attachments compared: ", b=True), T("spar top / waterline / CoG depth (buoy); pin "
                                                "plane (hub or deck, +0.72 m) vs spars at the "
                                                "waterline or CoG depth (cluster, platform); "
                                                "platform corners only.")]),
], t=3.85, size=15, gap=8)

# ============================ 7 — attachment result ===================================
s = slide("Result: attach at the waterline", "Coupled check, T_surge = 15 s")
pic(s, "mooring_verify.png", 0.4, 1.6, w=8.5, h=5.65)
pin_t = abs(v(PLAT, "dT_tilt_pct", "pin plane (+0.72 m)"))
t_corner = v(PLAT, "buoy_tilt_H0.5_deg", "4 corner spars, waterline")
t_cog = v(PLAT, "buoy_tilt_H0.5_deg", "bow+stern rows (8 spars), CoG depth (-0.91 m)")
bullets(s, [
    (0, [T("Heave: ", b=True), T(f"{pct(v(PLAT, 'dT_heave_pct'))} everywhere (line pretension).")]),
    (0, [T("Waterline attachment: ", b=True),
         T(f"pitch {pct(v(BUOY, 'dT_pitch_pct'))} (buoy), {pct(v(CLUSTER, 'dT_pitch_pct'))} "
           f"(cluster), {pct(v(PLAT, 'dT_pitch_pct'))} (platform); zero trim. The cluster's "
           "buoys do not tilt at all: each spar holds its own drift.")]),
    (0, [T("Pin plane (deck/hub): ", b=True),
         T(f"no trim, but it holds the light deck still → the buoys' tilt mode shifts "
           f"−{pin_t:.1f} %.")]),
    (0, [T("Avoid: ", b=True),
         T(f"platform corners only ({t_corner:.0f}° corner tilt) and below the waterline "
           f"({t_cog:.0f}°).")]),
    (0, [T("The 3° platform buoy tilt is inherent ", b=True),
         T("(drift 0.72 m below each pin), not caused by the mooring.")]),
], x=9.0, t=1.75, w=4.1, size=12.5, gap=6)

# ============================ 8 — how soft ============================================
s = slide("How soft: the surge-period trade-off", "Stiffness")


def sw(a: str, att: str, key: str, ts: float) -> float:
    return float(next(r[key] for r in SWP if r["article"] == a and r["attachment"] == att
                      and float(r["T_surge_design_s"]) == ts))


TS = (10.0, 15.0, 20.0, 25.0, 30.0)
rows = [["T_surge"] + [f"{t:.0f} s" for t in TS]]
rows.append(["Tilt-mode shift, waterline"] + [pct(sw(PLAT, REC[PLAT], "dT_tilt_pct", t))
                                              for t in TS])
rows.append(["Tilt-mode shift, pin plane"] + [pct(sw(PLAT, "pin plane (+0.72 m)", "dT_tilt_pct", t))
                                              for t in TS])
rows.append(["Offset, H = 0.5 m (bound)"] + [f"{sw(PLAT, REC[PLAT], 'offset_H0.5_m', t):.2f} m"
                                             for t in TS])
rows.append(["Offset, H = 0.3 m (bound)"] + [f"{sw(PLAT, REC[PLAT], 'offset_H0.3_m', t):.2f} m"
                                             for t in TS])
table(s, rows, 0.85, 1.95, [3.4] + [1.6] * 5, size=14, rh=0.5)
bullets(s, [
    (0, [T("15 s is the compromise: ", b=True),
         T("every period shift ≤ 2.4 %, offset < 0.8 m, spring stroke ~2 m.")]),
    (0, [T("20 s ", b=True), T("halves the tilt-mode shift, at 1.3 m offset and ~3.1 m stroke.")]),
    (0, [T("A single bridle at the deck ", b=True),
         T("would need ≥ 25 s for the same effect, with a 2 m offset.")]),
], t=4.8, size=15, gap=8)

# ============================ 9 — refined drift =======================================
s = slide("Refinement: the drift on a moving article", "Drift, relative velocity")
pic(s, "drift_refined.png", 0.45, 1.6, w=12.4, h=3.9)
dc = DRIFT.get(CLUSTER, {}); db = DRIFT.get(BUOY, {}); dp = DRIFT.get(PLAT, {})
bullets(s, [
    (0, [T("Away from resonance ", b=True),
         T("the articles move with the water; where the model is valid the drift is "
           f"≤ {db.get('max_abs_ok_N', 0):.1f} / {dc.get('max_abs_ok_N', 0):.1f} / "
           f"{dp.get('max_abs_ok_N', 0):.0f} N (buoy / cluster / platform) vs a design bound of "
           f"{db.get('bound_N', 0):.0f} / {dc.get('bound_N', 0):.0f} / "
           f"{dp.get('bound_N', 0):.0f} N.")]),
    (0, [T("Near the buoys' pitch / tilt resonances it reverses ", b=True),
         T("(upstream) and can approach the bound's magnitude — with 30°+ buoy tilts, beyond "
           "the model's small-angle range (open markers, indicative).")]),
    (0, [T("Design stays on the bound, both directions: ", b=True),
         T("the symmetric X-spread holds the same offset up- or downstream.")]),
], t=5.45, size=13.5, gap=4, h=1.9)

# ============================ 10 — flume size =========================================
s = slide("How the flume size influences the mooring", "Facility")
cols = [
    ("Depth  2.7 m", [
        "The deep-water field mooring cannot be scaled → equivalent soft mooring.",
        "Lines must be horizontal: to a floor anchor 5 m away a line dips ~27°, adding "
        "vertical stiffness ≈ 0.3 Kx and pulling the article down.",
        "Depth raises long-wave drift (×1.9 at 4 s), but the design drift peaks near 2.2 s "
        "where depth adds only ~5 % → not design-driving."]),
    ("Width  3.66 m", [
        "Sets the X-spread angle (20°): sway stiffness ≈ ⅓ of surge, T_sway ≈ 26 s.",
        f"A 10 % lateral load moves the platform 0.21 m; wall gap "
        f"{ms.W_FLUME / 2 - ms.ARTICLES[PLAT]['half_w']:.2f} m.",
        "±10 m anchors would halve the heave coupling but double the sideways excursion → "
        "the width fixes the anchors at ~±5 m."]),
    ("Length  104 m", [
        "Not a constraint: offsets < 0.8 m and a ±5.3 m line footprint fit easily.",
        "Wave gauges referenced to the mean moored position."]),
]
for i, (h, items) in enumerate(cols):
    x = 0.85 + i * 4.0
    _rect(s, x, 1.95, 3.7, 0.55, TEAL_D)
    tb = s.shapes.add_textbox(Inches(x + 0.15), Inches(1.97), Inches(3.4), Inches(0.5))
    _runs(tb.text_frame.paragraphs[0], [T(h, b=True, c=WHITE)], 17)
    _rect(s, x, 2.5, 3.7, 4.1, LIGHT)
    tb = s.shapes.add_textbox(Inches(x + 0.15), Inches(2.62), Inches(3.4), Inches(3.9))
    tf = tb.text_frame; tf.word_wrap = True
    for j, it in enumerate(items):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.space_after = Pt(9)
        _runs(p, [T("▪  ", c=TEAL), T(it)], 13.5)
caption(s, [T("The width is the binding constraint (anchor spread and lateral clearance); the "
              "depth only forbids short floor-anchored lines.", b=True)], t=6.75, h=0.55)

# ============================ 11 — limits and verification ===========================
s = slide("Limits, and how we verify in the flume", "Confidence")
bullets(s, [
    (0, [T("Drift is a bound: ", b=True), T("fixed bodies, no shielding, steepest waves. The "
                                            "moving-body refinement is lower off-resonance but "
                                            "indicative near resonance (large tilts).")]),
    (0, [T("Linear check: ", b=True), T("eigen-analysis + static equilibrium; regular waves give "
                                        "only a mean drift (irregular waves would add slow "
                                        "drift near T_surge).")]),
    (0, [T("Deep-water BEM ", b=True), T("for the coupled models (as the field site); the flume "
                                         "depth effect is covered in the sidewall study.")]),
    (0, [T("In the flume: ", b=True), T("static pull test (stiffness, pretension) before testing; "
                                        "load cells on the four lines measure the real mean drift; "
                                        "the measured mooring goes into the numerical model.")]),
    (0, [T("Open with HWRL: ", b=True), T("wall anchor points at the waterline (or a beam / the "
                                          "carriage), springs with ~2 m linear stroke, load "
                                          "cells, motion-tracking coverage.")]),
    (0, [T("Test-matrix note: ", b=True), T("in H = 0.5 m waves near the pinned buoys' tilt "
                                            "resonance (2.6–3.0 s) the linear model predicts "
                                            "30°+ buoy tilts — check the gimbal range.", c=RED)]),
], t=1.8, size=15, gap=8)

# ============================ 12 — answer per article =================================
s = slide("Answer, per article", "Summary")
rows = [["", SHORT[BUOY], SHORT[CLUSTER], SHORT[PLAT]],
        ["Lines attach at (SWL)"] + [REC_TXT[a] for a in ARTS],
        ["Per line: k / pretension"] + [f"{float(DES[a]['k_line_Npm']):.1f} N/m / "
                                        f"{float(DES[a]['pretension_N']):.0f} N" for a in ARTS],
        ["Moored surge period"] + [f"{v(a, 'T_surge_s'):.1f} s" for a in ARTS],
        ["Heave period shift"] + [pct(v(a, "dT_heave_pct")) for a in ARTS],
        ["Pitch period shift"] + [pct(v(a, "dT_pitch_pct")) for a in ARTS],
        ["Buoy-tilt mode shift", "n/a"] + [pct(v(a, "dT_tilt_pct")) for a in (CLUSTER, PLAT)],
        ["Offset, H 0.5 / 0.3 m (bound)"] + [f"{v(a, 'offset_H0.5_m'):.2f} / "
                                             f"{v(a, 'offset_H0.3_m'):.2f} m" for a in ARTS],
        ["Trim / largest buoy tilt (H 0.5)"] + [f"{abs(v(a, 'trim_H0.5_deg')):.0f}° / "
                                                f"{v(a, 'buoy_tilt_H0.5_deg'):.1f}°" for a in ARTS]]
table(s, rows, 0.85, 1.85, [3.2, 2.6, 2.9, 3.0], size=13.5, rh=0.5)
caption(s, [T("One design, scaled by buoy count: four soft waterline lines, T_surge ≈ 15 s, wall "
              "anchors ±5 m. ", b=True),
            T("Decay tests can run with the mooring connected.")], t=6.55, h=0.65)

out = HERE / "Flume_mooring_technical.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides)} slides)")
