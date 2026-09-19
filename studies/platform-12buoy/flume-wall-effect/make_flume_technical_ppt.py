"""Technical presentation of the flume sidewall-effect study (TEAMER rebuttal, engineering
audience). Denser and more quantitative than the plain-English deck (make_flume_ppt.py):
configuration + method, the weak-wavemaker / sub-cut-on physical basis, the transverse
cut-on ("ring") analysis, and the free-decay / wave-response / all-DOF results for the
16-buoy platform, honestly separating the (small) sidewall effect from the (larger,
depth-driven) facility effect. No lay "what is an N-body model" explainer slide.

Depth handling: the coupled/articulated BEM runs at deep water (finite depth is impractically
slow at the coupled panel count); free-decay is depth-robust so that is valid there. The
depth-sensitive wave response is carried by the fast native-2.7 m single-array frequency-domain
sweep (flume_wall_effect.py) + a single-DOF impedance model (floatsim_wall_1dof.py).

Embeds flume_blockage.png, ring_modes.png, floatsim_wall_rao.png, wall_vs_depth.png,
accel_multidof.png. Writes Flume_wall_effect_technical.pptx. Requires python-pptx.
"""
# ruff: noqa: RUF001, E702  -- slide copy uses display typography (en dashes, arrows, times
# sign); E702 compact multi-statement helper setup lines in a one-off deck generator.
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent

# Results (16-buoy, 0deg orientation). Free-decay + accel: articulated 21-body at deep water
# (depth-robust / near-resonance). Wave response: single-DOF + frequency-domain at 2.7 m.
R = {
    "T_heave": -0.55, "zeta": "6.4% → 6.4%", "buoy_tilt": "0.0007 → 0.0009 rad",
    "resp_band": 2.6, "resp_tail": 20,          # single-DOF heave response wall effect @2.7 m
    "exc_res": -5.4, "exc_long": -17, "A_res": -1.5,   # freq-domain heave, @2.7 m
    "pitch_surge_exc": 2.6,
    "depth_exc": "−19% to −37%",       # finite-depth effect, 2.52-4 s
    "acc_surge": 0.9, "acc_heave": 4.0, "acc_pitch": 2.2,
    "clearance_cm": 44, "span_pct": 76, "rad_frac_pct": 3.5,
    "cutoffs": "2.19 / 1.53 / 1.25 s",
}

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


def bullets(s, items, w=11.9, t=1.85, size=17, gap=11, x=0.85):  # type: ignore[no-untyped-def]
    tb = s.shapes.add_textbox(Inches(x), Inches(t), Inches(w), Inches(5.2))
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


def table(s, rows, l, t, col_w, size=14, head=True):  # type: ignore[no-untyped-def]
    y = t
    for ri, row in enumerate(rows):
        x = l; rh = 0.44
        if ri == 0 and head:
            _rect(s, l, y, sum(col_w), rh, TEAL_D)
        elif ri % 2 == 1:
            _rect(s, l, y, sum(col_w), rh, LIGHT)
        for ci, cell in enumerate(row):
            tb = s.shapes.add_textbox(Inches(x + 0.08), Inches(y + 0.02), Inches(col_w[ci] - 0.12),
                                      Inches(rh - 0.02))
            tf = tb.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
            r = p.add_run(); r.text = str(cell)
            r.font.name = MONO if ci > 0 else FONT
            r.font.size = Pt(size); r.font.bold = (ri == 0)
            r.font.color.rgb = WHITE if (ri == 0 and head) else INK
            x += col_w[ci]
        y += rh
    return y


# ============================ 1 — title ===============================================
s = slide()
_rect(s, 0, 0, 13.333, 0.28, TEAL)
_rect(s, 0, 5.3, 13.333, 2.2, NAVY)
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.1), Inches(11.6), Inches(0.5))
r = tb.text_frame.paragraphs[0].add_run()
r.text = "PHASE-3 PLATFORM · OSU LARGE WAVE FLUME · TEAMER REVIEW RESPONSE"
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(14), True, TEAL_D
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.7), Inches(11.7), Inches(2.6))
tb.text_frame.word_wrap = True
r = tb.text_frame.paragraphs[0].add_run()
r.text = "Sidewall effects on the free-decay\nand wave-sweep campaigns"
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(40), True, NAVY
p = tb.text_frame.add_paragraph(); p.space_before = Pt(10)
r = p.add_run()
r.text = "Potential-flow BEM with explicit flume walls (method of images)"
r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(19), GREY
tb = s.shapes.add_textbox(Inches(0.95), Inches(5.55), Inches(11.5), Inches(1.7))
tb.text_frame.word_wrap = True
r = tb.text_frame.paragraphs[0].add_run()
r.text = ("Finding: sidewalls do not corrupt the dynamic data. Free-decay periods shift < 0.6 %; "
          "the heave response through resonance shifts ≤ 3 % at the 2.7 m depth.")
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(17), True, WHITE
p = tb.text_frame.add_paragraph()
r = p.add_run()
r.text = ("The dominant flume artifact is the finite 2.7 m depth, not the walls. "
          "16-buoy platform (4 clusters × 4), 2.5 m to buoy centres.")
r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(13), LIGHT

# ============================ 2 — claim & finding =====================================
s = slide("Reviewer claim and our finding", "context")
_rect(s, 0.85, 1.72, 11.6, 1.1, LIGHT)
tb = s.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(0.95))
tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
r = tf.paragraphs[0].add_run()
r.text = ("“Phase 3 platform (2.50 m) inside HWRL's 3.67 m flume leaves 0.6 m side clearance, "
          "inducing severe sidewall reflections that will corrupt dynamic data.”")
r.font.name, r.font.size, r.font.italic, r.font.color.rgb = FONT, Pt(16), True, TEAL_D
bullets(s, [
    (0, [T("Severe reflections are the correct concern for a strong wavemaker or a body that "
           "spans the tank. The Phase-3 platform is neither, and we show it quantitatively.")]),
    (0, [T("Method: ", b=True), T("Capytaine potential-flow BEM with the LWF walls modelled by "
           "the method of images; every comparison is "),
         T("walls-in vs walls-out at the same depth", b=True), T(".")]),
    (0, [T("Finding: ", b=True), T("free-decay periods shift < 0.6 %; the heave response through "
           "resonance shifts ≤ 3 % at 2.7 m. "),
         T("The larger flume effect is the finite depth — a known, correctable facility "
           "property, not a sidewall artifact.", b=True, c=TEAL_D)]),
], t=3.05, gap=12)

# ============================ 3 — configuration & method ==============================
s = slide("Configuration and method", "setup")
s.shapes.add_picture(str(HERE / "flume_blockage.png"), Inches(7.5), Inches(1.75),
                     height=Inches(4.5))
bullets(s, [
    (0, [T("Test article: ", b=True), T("16-buoy platform — 4 clusters × 4 buoys (square), "
           "articulated (gimbal buoy→hub, hub→deck).")]),
    (0, [T("Hull: ", b=True), T("Phase-1 decay-correlated spar + heave plate (T ≈ 2.6 s, "
           "ζ ≈ 13 %).")]),
    (0, [T("Size: ", b=True), T("2.5 m to buoy centres; outer span 2.79 m "),
         T(f"({R['span_pct']} % of width)", b=True),
         T(f"; clearance ~0.{R['clearance_cm']} m/side.")]),
    (0, [T("Facility: ", b=True), T("OSU LWF, W = 3.66 m, h = 2.7 m.")]),
    (0, [T("Walls: ", b=True), T("method of images (mirror across y = ±W/2, ~3 levels).")]),
    (0, [T("Depth: ", b=True), T("coupled/articulated at deep water (free-decay is depth-robust); "
           "the depth-sensitive wave response is carried by the native-2.7 m frequency-domain + "
           "single-DOF models.", size=14)]),
    (0, [T("Reviewer's 0.6 m assumes 2.50 m is the outer extent; it is the buoy-centre circle, so "
           "the as-built clearance is the tighter ~0.44 m analysed here.", c=GREY, size=13)]),
], w=6.5, t=1.75, gap=8)

# ============================ 4 — physical basis ======================================
s = slide("Physical basis", "mechanism")
bullets(s, [
    (0, [T("Free-decay — weak radiator. ", b=True, c=TEAL_D),
         T("Heave radiation damping is only "), T(f"~{R['rad_frac_pct']} % of the total", b=True),
         T(" (the rest is viscous drag); the walls can perturb only that small radiated part, so "
           "the natural period and damping barely move. This is depth-robust.")]),
    (0, [T("Wave response — sub-cut-on, but depth-sensitive. ", b=True, c=TEAL_D),
         T("A corrupting cross-flume standing wave forms only at the transverse cut-ons (next "
           "slide); the operating band is below the first one. The residual wall effect on wave "
           "loads is a "), T("diffraction/blockage", b=True),
         T(" effect that grows with wavelength and shallowness — hence assessed at 2.7 m.")]),
    (0, [T("Porous, does not span the tank. ", b=True, c=TEAL_D),
         T("Water passes between the buoys and runs off along the 104 m length; only the "
           "cross-flume direction is bounded.")]),
], t=1.9, gap=16)

# ============================ 5 — ring / cut-on =======================================
s = slide("The flume “ring” — transverse cut-on modes", "ring effect")
s.shapes.add_picture(str(HERE / "ring_modes.png"), Inches(1.15), Inches(1.7),
                     width=Inches(11.0))
eqbox(s, "T_n = 2π / √( g·k_n·tanh(k_n·h) ),  k_n = nπ/W   →   "
         "T_1,2,3 = 2.19 / 1.53 / 1.25 s", 1.15, 4.7, 11.0, 0.56, size=14)
bullets(s, [
    (0, [T("Between the side walls the water sloshes across the width like a bathtub; these "
           "transverse modes have fixed (nearly depth-independent) cut-on periods set by W. ",
           size=14)]),
    (0, [T("Below T₁ = 2.19 s (incl. the ~2.6 s heave): the modes are ", size=14),
         T("evanescent", b=True, c=TEAL_D, size=14),
         T(" — they decay away from the body and cannot ring. Even at the cut-ons the weak "
           "scattering keeps the effect bounded (~±6 %); we skip those periods in the matrix.",
           size=14)]),
], t=5.4, gap=8)

# ============================ 6 — free-decay ==========================================
s = slide("Free-decay results (depth-robust)", "results 1/4")
table(s, [
    ("Quantity", "walls out → walls in", "wall effect"),
    ("Heave natural period", "2.607 → 2.593 s", f"{R['T_heave']:+.2f}%"),
    ("Heave damping ζ", R["zeta"], "unchanged"),
    ("Buoy gimbal tilt", R["buoy_tilt"], "~0 (negligible)"),
], 1.1, 2.0, [4.2, 4.0, 3.0], size=15)
bullets(s, [
    (0, [T("Full articulated 21-body FloatSim (real quadratic drag, KKT gimbal joints), "
           "walls-in vs walls-out.")]),
    (0, [T("The period shift is "), T("~20× smaller than the ±(3–5) % scatter", b=True, c=TEAL_D),
         T(" of a physical free-decay test; damping is viscous-dominated and unchanged. Radiation "
           "is depth-robust, so deep-water modelling is valid here.")]),
], t=4.05, gap=12)

# ============================ 7 — wave response @ 2.7 m ===============================
s = slide("Wave response at the 2.7 m depth", "results 2/4")
s.shapes.add_picture(str(HERE / "floatsim_wall_rao.png"), Inches(1.1), Inches(1.7),
                     width=Inches(7.3))
bullets(s, [
    (0, [T("Through resonance", b=True, c=TEAL_D), T(" (1.5–2.9 s, the dynamically "
           "important band): heave response wall effect "), T(f"≤ {R['resp_band']} %", b=True),
         T(".")]),
    (0, [T("Long-period tail", b=True, c=TEAL_D), T(" (T > 3 s): grows to "),
         T(f"~{R['resp_tail']} %", b=True),
         T(" — but off-resonance (small motion), and the finite-depth effect there is far "
           "larger (next slide).")]),
    (0, [T("Pitch / surge loads: ", b=True), T(f"≤ {R['pitch_surge_exc']} %.", )]),
], w=4.7, x=8.5, t=2.0, gap=12, size=15)
caption(s, [T("Single-DOF platform-heave impedance model on the native-2.7 m coupled-array "
             "coefficients (walls-in vs walls-out). The response tracks the wave-excitation wall "
             "effect: small through resonance, larger only in the long-period tail.")],
        t=6.5, h=0.85, size=12.5)

# ============================ 8 — walls vs depth ======================================
s = slide("Sidewalls vs finite depth — the dominant effect", "results 3/4")
s.shapes.add_picture(str(HERE / "wall_vs_depth.png"), Inches(1.6), Inches(1.75),
                     width=Inches(10.1))
caption(s, [T("At every period the "), T("finite-depth effect (red) exceeds the sidewall effect "
             "(teal)", b=True, c=RED),
            T(f"; on heave excitation the depth effect is {R['depth_exc']} at 2.5–4 s. The "
              "reviewer's concern (walls) is the smaller term; the real flume consideration is "
              "depth — a known, correctable property handled by depth-scaling.")],
        t=6.35, h=0.95, size=13)

# ============================ 9 — all-DOF accelerations ==============================
s = slide("All-DOF accelerations near resonance", "results 4/4")
s.shapes.add_picture(str(HERE / "accel_multidof.png"), Inches(1.25), Inches(1.75),
                     width=Inches(10.8))
caption(s, [T("Articulated 21-body accelerations at the deck centre and four cluster hubs, "
             "each excited DOF: surge "), T(f"≤ {R['acc_surge']} %", b=True, c=TEAL_D),
            T(", heave "), T(f"≤ {R['acc_heave']} %", b=True, c=TEAL_D),
            T(", pitch "), T(f"≤ {R['acc_pitch']} %", b=True, c=TEAL_D),
            T(". Sway/roll/yaw unexcited in head seas; peaks at the 2.19 s cut-on.")],
        t=6.35, h=0.95, size=13)

# ============================ 9b — orientation robustness ============================
s = slide("Orientation robustness — 0° vs 45°", "orientation")
s.shapes.add_picture(str(HERE / "orientation_compare.png"), Inches(1.55), Inches(1.75),
                     width=Inches(10.2))
caption(s, [T("A 90° rotation is a symmetry no-op (4-fold layout); 45° turns the platform "
             "corner-on and "), T("nearly doubles the clearance (0.44 → 0.80 m)", b=True, c=TEAL_D),
            T(" — yet the wall effect is unchanged (free-decay −0.55 % both). It is set "
              "by the "), T("bulk channel blockage", b=True, c=RED),
            T(" (total array volume vs cross-section), not the nearest-buoy clearance — so the "
              "reviewer's 0.6 m clearance is not the controlling parameter.")],
        t=6.35, h=0.95, size=12.5)

# ============================ 10 — concessions & recs ================================
s = slide("Scope, concessions and recommendations", "caveats")
bullets(s, [
    (0, [T("Transverse cut-ons ", b=True), T(f"({R['cutoffs']}): "),
         T("skip these narrow, known periods in the sweep matrix (bounded ~±6 % even there).")]),
    (0, [T("Long-period tests (T > 3 s): ", b=True),
         T("apply the standard finite-depth (and the smaller blockage) corrections — the "
           "response there is depth-dominated, not a sidewall artifact.")]),
    (0, [T("Depth of the coupled model: ", b=True),
         T("deep water (finite depth is impractically slow at the coupled panel count); free-decay "
           "is depth-robust and the 2.7 m wave response is carried by the frequency-domain + "
           "single-DOF models.")]),
    (0, [T("Mesh / walled radiation matrix: ", b=True),
         T("coarse mesh (the walls-in/out ratio is mesh-robust); the image trick makes walled B "
           "non-reciprocal, so it is PSD-projected (heave decay is robust to this).")]),
], t=1.85, gap=12, size=15)

# ============================ 11 — conclusion =========================================
s = slide("Conclusion", "conclusion")
_rect(s, 0.85, 1.9, 11.6, 0.06, TEAL)
bullets(s, [
    (0, [T("Sidewall reflections do not corrupt the Phase-3 dynamic data: free-decay periods "
           "shift < 0.6 %, and the heave response through resonance shifts ≤ 3 % at 2.7 m.")]),
    (0, [T("Accelerations at every sensor point, every excited DOF, change by "),
         T("≤ 4 %", b=True, c=TEAL_D), T(" near resonance.")]),
    (0, [T("The dominant flume consideration is the finite 2.7 m depth", b=True),
         T(" — larger than the walls, and handled by standard depth-scaling, not by the "
           "sidewalls.")]),
    (0, [T("The 2.50 m platform is compatible with the OSU Large Wave Flume.", b=True, c=TEAL_D,
           size=19)]),
], t=2.15, gap=14)

out = HERE / "Flume_wall_effect_technical.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
