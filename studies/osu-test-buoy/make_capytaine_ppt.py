"""Build a small meeting deck explaining the Capytaine diffraction & radiation
analysis (and the pre-test heave-decay prediction) for the OSU Test Buoy.

Companion to OSU-TEST-BUOY-GEOMETRY.md; embeds Capytaine_explained.png and
OSU_heave_decay_prediction.png. Writes Capytaine_analysis_OSU_buoy.pptx next to
this script. Requires python-pptx (scripting-only, not a FloatSim runtime dep).
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
TEAL = RGBColor(0x0C, 0x8B, 0x96)
TEAL_D = RGBColor(0x0A, 0x55, 0x60)
INK = RGBColor(0x25, 0x32, 0x3A)
GREY = RGBColor(0x54, 0x63, 0x6D)
LIGHT = RGBColor(0xEE, 0xF6, 0xF7)
RED = RGBColor(0xD1, 0x54, 0x3A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Segoe UI"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def _bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def _rect(slide, l, t, w, h, color):
    sh = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def _title(slide, text, eyebrow=None):
    if eyebrow:
        tb = slide.shapes.add_textbox(Inches(0.7), Inches(0.42), Inches(12), Inches(0.4))
        p = tb.text_frame.paragraphs[0]
        r = p.add_run(); r.text = eyebrow.upper()
        r.font.name = FONT; r.font.size = Pt(13); r.font.bold = True
        r.font.color.rgb = TEAL
    tb = slide.shapes.add_textbox(Inches(0.7), Inches(0.72), Inches(12), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    r.font.name = FONT; r.font.size = Pt(30); r.font.bold = True
    r.font.color.rgb = INK
    _rect(slide, 0.72, 1.58, 2.2, 0.06, TEAL)


def _bullets(slide, items, l=0.85, t=1.95, w=11.7, h=5.0, size=18, gap=8):
    """items: list of (level, runs) where runs = list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (lvl, runs) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        p.space_after = Pt(gap)
        p.space_before = Pt(2)
        bullet = "•  " if lvl == 0 else "–  "
        indent = "" if lvl == 0 else "      "
        lead = p.add_run(); lead.text = indent + bullet
        lead.font.name = FONT; lead.font.size = Pt(size)
        lead.font.color.rgb = TEAL if lvl == 0 else GREY
        for txt, opts in runs:
            r = p.add_run(); r.text = txt
            r.font.name = FONT
            r.font.size = Pt(opts.get("size", size if lvl == 0 else size - 2))
            r.font.bold = opts.get("b", False)
            r.font.italic = opts.get("i", False)
            r.font.color.rgb = opts.get("c", INK)
    return tb


def slide(title=None, eyebrow=None):
    s = prs.slides.add_slide(BLANK)
    _bg(s)
    if title:
        _title(s, title, eyebrow)
    return s


def T(text, **kw):
    return (text, kw)


# ---------------------------------------------------------------- 1. Title
s = slide()
_rect(s, 0, 0, 13.333, 7.5, WHITE)
_rect(s, 0, 5.55, 13.333, 1.95, TEAL)
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.35), Inches(11.5), Inches(2.6))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "DIFFRACTION & RADIATION ANALYSIS"
r.font.name = FONT; r.font.size = Pt(16); r.font.bold = True; r.font.color.rgb = TEAL
p2 = tf.add_paragraph(); p2.space_before = Pt(6)
r = p2.add_run(); r.text = "How we compute the buoy's\nhydrodynamics in Capytaine"
r.font.name = FONT; r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = INK
tb = s.shapes.add_textbox(Inches(0.95), Inches(5.75), Inches(11.5), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "OSU Test Buoy  ·  potential-flow BEM  →  time-domain FloatSim (Cummins)"
r.font.name = FONT; r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = WHITE
p2 = tf.add_paragraph()
r = p2.add_run(); r.text = "Two sub-problems (radiation + diffraction), their outputs, and how they drive the simulation"
r.font.name = FONT; r.font.size = Pt(14); r.font.color.rgb = LIGHT

# ---------------------------------------------------------- 2. What it computes
s = slide("What Capytaine computes", "the tool")
_bullets(s, [
    (0, [T("Boundary-element (panel) solver for "), T("linear potential-flow hydrodynamics", b=True), T(".")]),
    (0, [T("In: the wetted hull as a mesh of panels.  Out: the flow (velocity potential φ) around it in regular waves at each frequency ω.")]),
    (0, [T("“Potential flow” = ", ), T("inviscid, irrotational, small (linear)", b=True, c=TEAL_D), T(" waves & motions — the one assumption that sets both the outputs and the limits.")]),
    (0, [T("Key idea — ", ), T("superposition", b=True), T(": the wave-body problem splits into two sub-problems, solved separately and added:")]),
    (1, [T("Radiation", b=True, c=TEAL_D), T(" — the body wavemaking in still water")]),
    (1, [T("Diffraction", b=True, c=TEAL_D), T(" — the body held fixed under incoming waves")]),
], t=2.0, gap=12)

# ---------------------------------------------------------- 3. Figure
s = slide("The two sub-problems — and how they combine", "overview")
pic = s.shapes.add_picture(str(HERE / "Capytaine_explained.png"), Inches(1.75), Inches(1.85), height=Inches(5.2))

# ---------------------------------------------------------- 4. Radiation
s = slide("Radiation — “body as a wavemaker”", "sub-problem ①")
_bullets(s, [
    (0, [T("Turn incident waves ", ), T("off", b=True), T(". Force the body to oscillate in still water — one DOF at a time, at each ω.")]),
    (0, [T("It radiates its own waves; the reaction force has two pieces:")]),
    (1, [T("Added mass A(ω)", b=True, c=TEAL_D), T(" — in phase with acceleration (virtual mass of entrained water).")]),
    (1, [T("Radiation damping B(ω)", b=True, c=TEAL_D), T(" — in phase with velocity (energy carried off as radiated waves).")]),
    (0, [T("6×6 matrices per ω — captures cross-DOF coupling (e.g. heave → pitch).")]),
    (0, [T("→ This run produces the ", ), T("A(ω) and B(ω)", b=True), T(" curves.")]),
], t=2.0, gap=12)

# ---------------------------------------------------------- 5. Diffraction
s = slide("Diffraction — “body as an obstacle”", "sub-problem ②")
_bullets(s, [
    (0, [T("Hold the body ", ), T("fixed", b=True), T(". Let incident waves scatter off it.")]),
    (0, [T("Net wave-pressure force on the fixed hull = ", ), T("excitation force F_exc(ω, β)", b=True, c=TEAL_D), T("  (β = wave heading):")]),
    (1, [T("Froude–Krylov", b=True), T(" — force from the undisturbed incident-wave pressure.")]),
    (1, [T("Diffraction", b=True), T(" — correction because the body scatters the waves.")]),
    (0, [T("→ This run produces the wave forcing ", ), T("F_exc(ω)", b=True), T(" that drives the motion.")]),
    (0, [T("Plus the hydrostatic stiffness ", ), T("C", b=True), T(" (buoyancy / waterplane) from the geometry.")]),
], t=2.0, gap=12)

# ---------------------------------------------------------- 6. Cummins bridge
s = slide("From frequency domain to time-domain simulation", "the bridge")
_bullets(s, [
    (0, [T("A(ω), B(ω) can’t go straight into a time-stepper — a transient contains all frequencies at once. Bridge = the "), T("Cummins equation", b=True), T(":")]),
], t=1.9, gap=6)
_rect(s, 0.85, 2.75, 11.6, 1.15, LIGHT)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.9), Inches(11.3), Inches(0.9))
tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "(M + A∞) ẍ  +  ∫₀ᵗ K(t−τ) ẋ(τ) dτ  +  C x  =  F_exc(t)  +  F_drag(ẋ)"
r.font.name = "Cambria Math"; r.font.size = Pt(22); r.font.bold = True; r.font.color.rgb = INK
_bullets(s, [
    (0, [T("A∞", b=True, c=TEAL_D), T(" = infinite-frequency added mass.")]),
    (0, [T("K(t) = (2/π)∫ B(ω) cos(ωt) dω", b=True, c=TEAL_D), T(" — the retardation kernel = ", ), T("fluid memory", i=True), T(" of previously radiated waves.")]),
    (0, [T("Mapping:  ", ), T("Radiation → A∞ + K(t)", b=True), T("   ·   "), T("Diffraction → F_exc", b=True), T("   ·   "), T("Hydrostatics → C", b=True), T(".  FloatSim integrates this in time.")]),
], t=4.1, gap=10)

# ---------------------------------------------------------- 7. Limits
s = slide("Limits — and why the heave plate needs the tank", "caveats")
_bullets(s, [
    (0, [T("Potential flow is ", ), T("inviscid and linear", b=True), T(". Two consequences for us:")]),
    (1, [T("No viscous drag in the BEM", b=True, c=RED), T(" → all quadratic / viscous damping is added separately as ", ), T("Morison drag F_drag", b=True), T(" — not from Capytaine.")]),
    (1, [T("Panels treat every surface as solid", b=True, c=RED), T(" → ", ), T("over-predicts added mass", b=True), T(" for the open/perforated heave-plate frame (it blocks flow the real perforations pass).")]),
    (0, [T("Practical split for the OSU buoy:")]),
    (1, [T("Spar (6″ pipe)", b=True, c=TEAL_D), T(" — a clean cylinder → the BEM is reliable.")]),
    (1, [T("Heave-plate / ballast frame", b=True, c=TEAL_D), T(" — A and B are BEM ", ), T("stand-ins until the tank test measures them", b=True), T(".")]),
], t=2.0, gap=11)

# ---------------------------------------------------------- 8. Prediction
s = slide("Pre-test heave decay prediction", "what to expect")
s.shapes.add_picture(str(HERE / "OSU_heave_decay_prediction.png"), Inches(1.15), Inches(1.85), width=Inches(11.0))
_rect(s, 0.85, 6.35, 11.6, 0.82, LIGHT)
tb = s.shapes.add_textbox(Inches(1.05), Inches(6.4), Inches(11.2), Inches(0.72))
tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
for txt, opts in [
    T("Period is predictable → T ≈ 2.3–2.4 s", b=True, c=TEAL_D),
    T("   (M, C₃₃ known; only plate added mass uncertain).   ", ),
    T("Damping is the measurement", b=True, c=RED),
    T("  → ζ₁ ≈ 8–15% at 100 mm.  Test pins: period→added mass, decay→drag Cd.", ),
]:
    r = p.add_run(); r.text = txt
    r.font.name = FONT; r.font.size = Pt(14); r.font.bold = opts.get("b", False)
    r.font.color.rgb = opts.get("c", INK)

out = HERE / "Capytaine_analysis_OSU_buoy.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
