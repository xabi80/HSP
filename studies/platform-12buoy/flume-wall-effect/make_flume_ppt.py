"""Plain-English presentation of the flume sidewall-effect study (the TEAMER rebuttal).

Simple-words explainer: the concern, how we checked it (three ways), the results, the bottom
line. Embeds flume_blockage.png, articulated_summary.png, articulated_accel.png. Writes
Flume_wall_effect_explained.pptx next to this script. Requires python-pptx (scripting-only).
"""
# ruff: noqa: RUF001  -- slide copy uses display typography (en dashes, arrows, times sign).
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ORI = "_rot45"   # test orientation: 45 deg flat-on ("" = 0 deg corner-on)
TEAL, TEAL_D = RGBColor(0x0C, 0x8B, 0x96), RGBColor(0x0A, 0x55, 0x60)
INK, GREY = RGBColor(0x25, 0x32, 0x3A), RGBColor(0x54, 0x63, 0x6D)
LIGHT = RGBColor(0xEE, 0xF6, 0xF7)
RED, WHITE = RGBColor(0xD1, 0x54, 0x3A), RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Segoe UI"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def T(text, **kw):  # noqa: N802
    return (text, kw)


def _rect(s, l, t, w, h, c):  # type: ignore[no-untyped-def]
    sh = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = c
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def _runs(p, runs, default_size):  # type: ignore[no-untyped-def]
    for txt, o in runs:
        r = p.add_run()
        r.text = txt
        r.font.name = FONT
        r.font.size = Pt(o.get("size", default_size))
        r.font.bold = o.get("b", False)
        r.font.italic = o.get("i", False)
        r.font.color.rgb = o.get("c", INK)


def slide(title=None, eyebrow=None):  # type: ignore[no-untyped-def]
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = WHITE
    if title:
        if eyebrow:
            tb = s.shapes.add_textbox(Inches(0.7), Inches(0.42), Inches(12), Inches(0.4))
            r = tb.text_frame.paragraphs[0].add_run()
            r.text = eyebrow.upper()
            r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(13), True, TEAL
        tb = s.shapes.add_textbox(Inches(0.7), Inches(0.72), Inches(12), Inches(0.9))
        r = tb.text_frame.paragraphs[0].add_run()
        r.text = title
        r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(30), True, INK
        _rect(s, 0.72, 1.58, 2.2, 0.06, TEAL)
    return s


def bullets(s, items, w=11.7, t=2.0, size=19, gap=14):  # type: ignore[no-untyped-def]
    tb = s.shapes.add_textbox(Inches(0.85), Inches(t), Inches(w), Inches(5.0))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (lvl, runs) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        p.space_after = Pt(gap)
        p.space_before = Pt(2)
        lead = p.add_run()
        lead.text = "•  " if lvl == 0 else "      –  "
        lead.font.name = FONT
        lead.font.size = Pt(size if lvl == 0 else size - 3)
        lead.font.color.rgb = TEAL if lvl == 0 else GREY
        _runs(p, runs, size if lvl == 0 else size - 3)


def caption(s, runs, t=6.55, h=0.8, size=13):  # type: ignore[no-untyped-def]
    _rect(s, 0.85, t, 11.6, h, LIGHT)
    tb = s.shapes.add_textbox(Inches(1.05), Inches(t + 0.04), Inches(11.2), Inches(h - 0.08))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _runs(p, runs, size)


# 1 -- title
s = slide()
_rect(s, 0, 0, 13.333, 7.5, WHITE)
_rect(s, 0, 5.55, 13.333, 1.95, TEAL)
tb = s.shapes.add_textbox(Inches(0.9), Inches(1.5), Inches(11.5), Inches(2.6))
tb.text_frame.word_wrap = True
r = tb.text_frame.paragraphs[0].add_run()
r.text = "PLAIN-ENGLISH SUMMARY"
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(16), True, TEAL
p = tb.text_frame.add_paragraph()
p.space_before = Pt(6)
r = p.add_run()
r.text = "Will the flume walls spoil\nour platform test?"
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(42), True, INK
tb = s.shapes.add_textbox(Inches(0.95), Inches(5.8), Inches(11.5), Inches(1.4))
tb.text_frame.word_wrap = True
r = tb.text_frame.paragraphs[0].add_run()
r.text = "Short answer: no. The 2.5 m, 16-buoy platform is fine in the OSU wave flume."
r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(20), True, WHITE
p = tb.text_frame.add_paragraph()
r = p.add_run()
r.text = "Here is the concern, how we checked it, and what we found — in simple terms."
r.font.name, r.font.size, r.font.color.rgb = FONT, Pt(14), LIGHT

# 2 -- the concern
s = slide("The concern a reviewer raised", "the question")
s.shapes.add_picture(str(HERE / f"flume_blockage{ORI}.png"), Inches(7.15), Inches(1.85),
                     height=Inches(4.6))
bullets(s, [
    (0, [T("We want to test a floating platform — 16 buoys on a 2.5 m frame — in OSU's "
           "wave flume (a long tank, "), T("3.66 m wide", b=True), T(").")]),
    (0, [T("It sits square to the tank in the middle, about 0.8 m from each wall. A reviewer "
           "worried:")]),
    (1, [T("“the platform makes waves, they bounce off the side walls, come back, and "
           "corrupt the measured data.”", i=True, c=TEAL_D)]),
    (0, [T("Fair question. So we tested it.", b=True)]),
], w=6.2, t=2.1)

# 3 -- the key idea
s = slide("The key idea — our platform barely makes waves", "why it's ok")
bullets(s, [
    (0, [T("“Wall reflections” only matter if the platform "), T("makes strong waves", b=True),
         T(" to reflect.")]),
    (0, [T("Ours doesn't. It's a sparse cluster of "), T("thin floats with drag plates", b=True),
         T(" — a "), T("poor wavemaker", b=True, c=TEAL_D), T(".")]),
    (0, [T("When it bobs, almost all its energy goes into "),
         T("water friction (drag)", b=True), T(", not into radiated waves.")]),
    (0, [T("No waves out → "),
         T("nothing to bounce off the walls and come back", b=True, c=TEAL_D),
         T(". That's the whole story.")]),
], t=2.2, gap=16)
caption(s, [T("In numbers: only about "), T("3–4% ", b=True, c=TEAL_D),
            T("of the bobbing energy leaves as waves — the rest is friction. So even perfect "
              "wall reflections could barely touch the result.")], size=14)

# 4 -- how we checked
s = slide("How we checked it — three ways, each more thorough", "the method")
bullets(s, [
    (0, [T("1.  Pen-and-paper physics", b=True, c=TEAL_D),
         T(" — where reflections could build up, and by how much.")]),
    (0, [T("2.  A computer flow-simulation of one buoy", b=True, c=TEAL_D),
         T(", with and without the walls.")]),
    (0, [T("3.  A full simulation of the whole 21-piece platform", b=True, c=TEAL_D),
         T(" inside the flume — every buoy, every joint, the real drag — walls in vs walls out.")]),
    (0, [T("The full-platform model (3) is the one we trust; the simpler checks point the same "
           "way near the platform's natural bobbing period.")]),
], t=2.2, gap=16)

# 5 -- what is the platform (21 bodies)
s = slide("What is the “21-piece” platform?", "the model")
bullets(s, [
    (0, [T("The platform is not one solid raft — it's an "),
         T("articulated (jointed) structure", b=True), T(":")]),
    (1, [T("16 buoys", b=True, c=TEAL_D), T(" — the floats (spar + drag plate)")]),
    (1, [T("4 cluster hubs", b=True, c=TEAL_D), T(" — each holds 4 buoys on gimbal pins")]),
    (1, [T("1 central deck", b=True, c=TEAL_D), T(" — the 4 arms connect to it")]),
    (0, [T("16 + 4 + 1 = "), T("21 pieces", b=True), T(". The "),
         T("gimbal pins", b=True, c=TEAL_D),
         T(" let each buoy tilt on its own — that's why we model all 21, not one rigid block.")]),
], t=2.1, gap=13)

# 6 -- result 1: the bobbing (decay)
s = slide("Result 1 — the up-and-down bobbing", "results")
bullets(s, [
    (0, [T("We push the platform down and let it bob back (a “free-decay” test).")]),
    (0, [T("With the walls in vs out, the bobbing "),
         T("speed changes by about half a percent", b=True, c=TEAL_D), T(" …")]),
    (0, [T("… and "), T("how fast it settles doesn't change at all", b=True, c=TEAL_D),
         T(" (the friction is unchanged by the walls).")]),
    (0, [T("The buoys' tilting on their gimbals is "), T("barely affected", b=True), T(" too.")]),
], t=2.2, gap=15)
caption(s, [T("Half a percent is far smaller than the "),
            T("±3–5% you'd expect just from run-to-run scatter", b=True, c=TEAL_D),
            T(" in a physical test.")], size=14)

# 7 -- result 2: wave response near the natural period
s = slide("Result 2 — the response to waves", "results")
s.shapes.add_picture(str(HERE / f"articulated_summary{ORI}.png"), Inches(1.05), Inches(1.85),
                     width=Inches(11.2))
caption(s, [T("Across every wave period we'll test, the full articulated-platform simulation "
              "shows the walls change the response by "),
            T("at most about 4%", b=True, c=TEAL_D),
            T(" — and it doesn't grow at long waves.")],
        t=6.45, size=13)

# 8 -- result 3: the water depth matters more than the walls
s = slide("Result 3 — the shallow water matters more than the walls", "results")
s.shapes.add_picture(str(HERE / f"wall_vs_depth{ORI}.png"), Inches(1.35), Inches(1.95),
                     width=Inches(10.6))
caption(s, [T("The walls (teal) stay "), T("small at every wave period", b=True, c=TEAL_D),
            T("; it's the "), T("shallow 2.7 m water", b=True, c=RED),
            T(" (red) that grows at long waves — and that's a "),
            T("known, routine correction", b=True, c=TEAL_D),
            T(" every flume test already applies.")],
        t=6.4, h=0.85, size=13)

# 9 -- result 4: accelerations in every direction
s = slide("Result 4 — every direction, not just up-down", "results")
s.shapes.add_picture(str(HERE / f"accel_multidof{ORI}.png"), Inches(1.15), Inches(2.05),
                     width=Inches(11.0))
caption(s, [T("We checked the accelerometers in "),
            T("every direction they can move", b=True, c=TEAL_D),
            T(" — fore-aft, up-down, and tilt (pitch). Near the natural period, walls in vs out "
              "changes each one by "),
            T("about 4% or less", b=True, c=TEAL_D),
            T(", at the deck centre and all four cluster points. Side-to-side and twist stay "
              "essentially zero (the waves come straight down the flume).")],
        t=6.05, h=1.15, size=13)

# 9b -- why the shallow water matters more for long waves
s = slide("Why long waves feel the bottom", "the depth effect")
s.shapes.add_picture(str(HERE / "depth_effect_explained.png"), Inches(1.35), Inches(1.75),
                     width=Inches(10.6))
caption(s, [T("Under a wave, the water moves in little loops. Short waves only stir the top "
              "layer, so the flume feels like the open sea. Long waves reach all the way down, and "
              "the 2.7 m floor "), T("squashes the loops flat", b=True, c=RED),
            T(" — so there's less up-and-down push on the buoys' plates. That's the depth effect: "
              "well understood, and "), T("corrected as standard", b=True, c=TEAL_D), T(".")],
        t=5.85, h=1.15, size=13)

# 10 -- the one caveat
s = slide("The one thing to watch", "the caveat")
bullets(s, [
    (0, [T("A flume can “ring” sideways like a bathtub at a few special wave periods "),
         T("(≈ 2.2, 1.5, 1.25 s)", b=True, c=RED), T(".")]),
    (0, [T("Near those, the walls "), T("do", i=True), T(" matter — so we simply "),
         T("avoid parking a test exactly there", b=True, c=TEAL_D), T(".")]),
    (0, [T("They're narrow, known in advance, and easy to step around in the test plan.")]),
    (0, [T("And for "), T("very long waves", b=True), T(" (slower than ~3 s), the "),
         T("shallow water depth", b=True, c=RED),
         T(" is the bigger effect — corrected as standard, just like any flume test.")]),
], t=2.2, gap=14)

# 11 -- bottom line
s = slide("Bottom line", "conclusion")
bullets(s, [
    (0, [T("The 2.5 m, 16-buoy platform is "),
         T("compatible with the OSU wave flume", b=True, c=TEAL_D), T(".")]),
    (0, [T("The side walls change the measured bobbing speed by "), T("about ½%", b=True),
         T(", the damping "), T("not at all", b=True), T(", and the near-resonance wave response "
           "by "), T("only a few percent", b=True), T(".")]),
    (0, [T("The bigger flume factor is the "), T("shallow water depth", b=True, c=RED),
         T(" (not the walls) — a routine, corrected effect.")]),
    (0, [T("Reason, in one line:", b=True)]),
    (1, [T("the platform barely makes waves, so there's almost nothing for the walls to "
           "reflect — the flume test measures the real behaviour once depth is accounted for.",
           i=True, c=TEAL_D)]),
], t=2.2, gap=15)

out = HERE / "Flume_wall_effect_explained.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
