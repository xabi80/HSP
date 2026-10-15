"""PowerPoint deck of the platform payload-stability study: pictures, native charts and summary.

Every number comes from this folder's records:
- buoy_hydrostatics_vs_draft.json;
- platform_envelope_h0p0.json / _h0p5.json / _h1p0.json and platform_vs_height.json;
- platform_tilt_paths.json;
- platform_crosscheck.json;
- cluster_envelope.json and tilt_paths.json, for the comparison with the cluster.
The pictures come from platform_figures.py: sketch_platform.png, pseq_*.png and the panim_*.gif
animations, which play in PowerPoint. Charts are native PowerPoint XY charts, editable in place.

Writes Platform_payload_stability.pptx. Run: python make_platform_ppt.py
(after platform_statics.py envelope 0 / 0.5 / 1, platform_statics.py heights,
platform_crosscheck.py, platform_figures.py)
"""

# ruff: noqa: E501, RUF001  -- slide-copy strings kept whole for editing; degree signs, dashes and
# times signs in the slide text
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from deck_kit import (
    AMBER,
    DIR_COL,
    GREEN,
    ICE,
    INK,
    MUTED,
    NAVY,
    RED,
    TEAL,
    WHITE,
    arrow,
    card,
    fit_picture,
    picture,
    prs,
    slide,
    stat,
    table,
    text,
    title,
    xy_chart,
)
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import PP_ALIGN

HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


HYD = load("buoy_hydrostatics_vs_draft.json")
E0, E05, E10 = (load(f"platform_envelope_h{t}.json") for t in ("0p0", "0p5", "1p0"))
VH = load("platform_vs_height.json")
PT = load("platform_tilt_paths.json")
PX = load("platform_crosscheck.json")
CENV, CTILT = load("cluster_envelope.json"), load("tilt_paths.json")
PURPLE = RGBColor(0x7B, 0x3F, 0xA0)

# ------------------------------------------------------------------ data
G_ACC = HYD["g"]
c33 = HYD["rows"]["36"][0]["C33"]
nb = E0["n_buoys"]
m_hub, m_deck, h_deck = E0["m_hub_kg"], E0["m_deck_kg"], E0["deck_above_pins_m"]
m_plat = nb * E0["m_buoy_kg"] + 4 * m_hub + m_deck
k_tilt = PT["k_tilt_Nm_per_rad"]
k_tilt_cluster = CTILT["k_tilt_Nm_per_rad"]
tip_mh = k_tilt / G_ACC - m_deck * h_deck            # payload mass x CG height above the pins, kg m


def row(env: dict, d: float, e: float) -> dict:
    return next(r for r in env["rows"] if r["dir_deg"] == d and abs(r["offset_m"] - e) < 1e-9)


def me_at(t: str, env: dict = E0) -> float:
    """Smallest mass x offset (kg m) giving deck tilt t over the offsets (heavier payloads at
    small offsets tilt a little more per kg m: the conservative value)."""
    return min(r["m_tilt_kg"][t] * r["offset_m"] for r in env["rows"]
               if r["dir_deg"] == 0.0 and r["offset_m"] > 0 and r["m_tilt_kg"][t] is not None)


m_centred = row(E0, 0.0, 0.0)["m_limit_kg"]
m_centred_cluster = next(r["m_limit_kg"] for r in CENV["rows"] if r["offset_m"] == 0.0)
sink_per_kg_mm = 1e3 * G_ACC / (nb * c33)
dead_sink_mm = (4 * m_hub + m_deck) * sink_per_kg_mm
offsets = [round(x, 2) for x in np.arange(0.05, 1.2001, 0.05)]
worst_pct = max(100 * (row(E0, 0.0, e)["m_limit_kg"] - row(E0, 45.0, e)["m_limit_kg"])
                / row(E0, 0.0, e)["m_limit_kg"] for e in offsets)
all_rows = [r for env in (E0, E05, E10) for r in env["rows"]] + VH["rows"]
gimbal_min = min(r["critical_state"]["min_gimbal_net_Nm_per_rad"] for r in all_rows)
hub_resid = max(r["max_hub_moment_residual_Nm"] for r in all_rows)
gap_min = min(r["critical_state"]["min_axis_gap_m"] for r in all_rows)
h_switch = tip_mh / m_centred - h_deck              # CG height where tipping overtakes awash
edge = row(E0, 45.0, 1.2)
r_lo, r_hi = row(E0, 0.0, 0.2), row(E10, 0.0, 0.2)
m_tip_1m = row(E10, 0.0, 0.0)["m_limit_kg"]
ratio = k_tilt / k_tilt_cluster


def h_max(m: float) -> float:
    return tip_mh / m - h_deck


# ================================================================== 1 title
s = slide(dark=True, notes=(
    "This deck repeats the cluster study for the 2.4 m platform: how much mass can sit on the deck, "
    "and where, before its stability is compromised. Every number comes from the study's records; "
    "the drawings are to scale and come from the solved equilibria. The animations play in "
    "PowerPoint's slide show."))
text(s, 0.8, 1.6, 7.4, 0.5, "FLUME TEST ARTICLES · 4×4 PLATFORM", size=14, bold=True, color=ICE)
text(s, 0.8, 2.1, 7.4, 2.2, "How much payload can the platform carry?", size=44, bold=True,
     color=WHITE)
text(s, 0.8, 4.35, 7.2, 1.2, "A mass on the deck, moved around and raised: the stability "
     "envelope and the physics behind it", size=20, color=ICE)
text(s, 0.8, 6.5, 7.2, 0.4, "Payload-stability study · 2026-10-15 · Capytaine hydrostatics, "
     "large-angle statics, FloatSim cross-check", size=12, color=ICE)
card(s, 7.9, 0.9, 4.95, 5.7, WHITE)
fit_picture(s, "sketch_platform.png", 8.0, 1.0, 4.75, 5.5)

# ================================================================== 2 bottom line
s = slide(notes=(
    f"Four numbers carry the story. A centred mass only sinks the platform: {m_centred:.0f} kg puts "
    "all sixteen pins at the waterline. Off-centre, the whole deck tilts on its four clusters, and it "
    f"is about thirty times stiffer than the cluster's hub: about {me_at('5'):.0f} kg·m of mass "
    f"times offset per 5 degrees. The new physics is height: a payload whose mass times CG height "
    f"above the pins exceeds about {tip_mh:.0f} kg·m tips the deck over even when centred. And at "
    f"the edge, 1.2 m towards a corner, the limit is {edge['m_limit_kg']:.0f} kg."))
title(s, "The bottom line", "Payload on the deck, free-floating in calm water, gimbals free, "
      "pins kept dry")
stat(s, 0.6, 1.9, 2.9, f"{m_centred:.0f} kg", "centred until all 16 pins reach the waterline: the "
     "platform only sinks")
stat(s, 3.7, 1.9, 2.9, f"{me_at('5'):.0f} kg·m", "mass × offset per 5° of deck tilt: "
     f"{ratio:.0f}× stiffer than the cluster's hub", color=AMBER)
stat(s, 6.8, 1.9, 2.9, f"{tip_mh:.0f} kg·m", "mass × CG height above the pins that tips the deck, "
     "even centred", color=RED)
stat(s, 9.9, 1.9, 2.85, f"{edge['m_limit_kg']:.0f} kg", "at the deck edge, 1.2 m towards a "
     f"corner (deck at {edge['tilt_at_limit_deg']:.0f}°)", color=PURPLE)
text(s, 0.6, 4.0, 12.1, 0.5, "What limits the platform", size=20, bold=True, color=NAVY)
text(s, 0.6, 4.55, 12.1, 2.6, [
    [("Centred, on the deck: ", {"bold": True}), ("reserve buoyancy. ", {}),
     (f"Each kg sinks it {sink_per_kg_mm:.2f} mm; at {m_centred:.1f} kg every pin is awash.", {})],
    [("Off-centre: ", {"bold": True}),
     ("one cluster's four pins go awash together; worst towards a hub, on the diagonal. ", {}),
     (f"E.g. {row(E0, 45.0, 0.6)['m_limit_kg']:.0f} kg at 0.6 m, {edge['m_limit_kg']:.0f} kg at 1.2 m.", {})],
    [("Raised payload: ", {"bold": True}),
     (f"keep mass × CG height above the pins below {tip_mh:.0f} kg·m, or the deck tips over "
      f"(e.g. {m_tip_1m:.0f} kg with its CG 1 m above the deck).", {})],
    [("For operation: ", {"bold": True, "color": TEAL}),
     (f"≤ 5° of deck tilt ↔ mass × offset ≤ {me_at('5'):.0f} kg·m (≤ 10° ↔ {me_at('10'):.0f} kg·m), "
      "for a payload on the deck.", {"color": TEAL})],
], size=16, gap=7)

# ================================================================== 3 the platform
s = slide(notes=(
    "The platform is the rev D test platform: sixteen OSU buoys on a 0.8 m grid, in four clusters "
    "of four. Each buoy hangs from its cluster's hub on a gimbal at the spar top, exactly as in the "
    "cluster; each hub hangs from the deck frame on another gimbal, at the hub centre. The deck "
    "frame sits 0.18 m above the pins. The study puts a point mass on the deck, moves it "
    "horizontally out to the outer buoy line, and also raises its centre of gravity. Same "
    "decisions as for the cluster: free-floating, calm water, gimbals free, pins dry."))
title(s, "The platform and the question", "Four clusters hung from a deck on gimbals; a mass on "
      "the deck, moved parallel to the waterline")
fit_picture(s, "sketch_platform.png", 0.5, 1.55, 6.4, 5.7)
text(s, 7.1, 1.75, 5.7, 0.5, "The question", size=20, bold=True, color=NAVY)
text(s, 7.1, 2.25, 5.7, 1.2, "How much mass can sit on the deck, and how far from the centre, "
     "before stability is compromised? Then: an envelope of mass and centre of gravity.", size=16)
text(s, 7.1, 3.55, 5.7, 0.5, "The set-up", size=20, bold=True, color=NAVY)
text(s, 7.1, 4.05, 5.7, 3.2, [
    [("Mass: ", {"bold": True}), ("a point mass on the deck frame (+0.90 m), moved "
                                  "horizontally up to 1.2 m; its CG also raised 0.5 and 1.0 m.", {})],
    [("Water: ", {"bold": True}), ("calm; the platform free-floating (no mooring).", {})],
    [("Gimbals: ", {"bold": True}), ("free, buoys to hubs and hubs to the deck; no stops.", {})],
    [("Limit: ", {"bold": True}), ("loss of restoring, or a pin reaching the water.", {})],
    [("Platform: ", {"bold": True}), (f"{m_plat:.1f} kg unloaded: {nb} × {E0['m_buoy_kg']} kg buoys "
                                      f"+ 4 × {m_hub:.0f} kg hubs + {m_deck:.0f} kg deck.", {})],
], size=15, gap=6)

# ================================================================== 4 how it carries the load
s = slide(notes=(
    "How the platform carries a load, and how it was computed. The key simplification is exact, "
    "not assumed: each hub receives the deck's load at its own gimbal, which sits at the hub centre "
    "in the plane of its four pins. That load has no lever arm to tilt the hub, so the hub hangs "
    "level and its four buoys sink together. Each cluster then acts as one vertical spring under the "
    "deck, and only the deck tilts. The hub tilts are still in the stability check. The method is "
    "the cluster's: the same Capytaine draft table, the statics extended to the deck and four hubs, "
    "and FloatSim's own platform model checking the small-angle range."))
title(s, "How the platform carries a load", "The deck rests on four clusters; each cluster acts as "
      "one spring")
steps = [("1  Capytaine at every draft",
          "The cluster study's buoy table is reused as it is: buoyancy, pin load and tilt stiffness "
          "for sinkage −0.95 … +0.72 m.", TEAL),
         ("2  Large-angle statics",
          "Deck + 4 hubs + 16 hanging buoys, exact at any deck tilt: 11 coordinates, equilibrium at "
          "the minimum of potential energy, stability from its stiffness.", AMBER),
         ("3  FloatSim cross-check",
          "FloatSim's own platform model (16 buoys, coupled BEM, joints), solved statically: "
          "identical sinkage and tilt at small angles.", GREEN)]
for i, (h, b, col) in enumerate(steps):
    x = 0.6 + i * 4.2
    card(s, x, 1.75, 3.7, 2.5)
    text(s, x + 0.25, 1.95, 3.2, 0.55, h, size=18, bold=True, color=col)
    text(s, x + 0.25, 2.55, 3.2, 1.7, b, size=14)
    if i < 2:
        arrow(s, x + 3.75, 2.85)
card(s, 0.6, 4.55, 12.1, 2.55, RGBColor(0xF3, 0xF6, 0xF7))
text(s, 0.85, 4.7, 11.6, 0.5, "Why each hub stays level", size=20, bold=True, color=NAVY)
text(s, 0.85, 5.25, 11.6, 1.8, [
    [("The deck's load reaches each hub at its gimbal: ", {"bold": True}),
     ("the hub centre, in the plane of its four pins. It has no lever arm to tilt the hub, so the "
      "hub hangs level and its four buoys sink together.", {})],
    [("So each cluster is one vertical spring under the deck: ", {"bold": True}),
     (f"4 × {c33:.0f} N/m, 1.13 m from the centre. Only the deck tilts. Checked, not assumed: the "
      f"hub moment residual is ≤ {hub_resid:.0e} N·m and the hub tilts are in the stability "
      "check.", {})]], size=15, gap=8)

# ================================================================== 5 centred
s = slide(notes=(
    f"Scenario one: the mass in the centre of the deck. Nothing tilts; the sixteen buoys share the "
    f"load and the platform sinks {sink_per_kg_mm:.2f} mm per kilogram, a quarter of the cluster's "
    f"rate because there are four times the buoys. The deck and hubs alone sink it {dead_sink_mm:.0f} "
    f"mm. At {m_centred:.1f} kg all sixteen pins reach the waterline: about four clusters' worth, "
    "less the deck."))
title(s, "A centred mass only sinks the platform", "All sixteen buoys share the load; nothing tilts")
fit_picture(s, "pseq_centred.png", 0.6, 1.55, 12.1, 3.75)
stat(s, 0.6, 5.55, 3.8, f"{sink_per_kg_mm:.2f} mm", f"of sinkage per kg of payload ({c33:.0f} N/m "
     "per buoy, sixteen buoys)", h=1.5)
stat(s, 4.75, 5.55, 3.8, f"{m_centred:.1f} kg", "on the deck puts all sixteen pins at the "
     "waterline", color=RED, h=1.5)
stat(s, 8.9, 5.55, 3.8, f"≈ 4 × {m_centred_cluster:.0f} kg", "the cluster's centred limit, four "
     f"times, less the {m_deck:.0f} kg deck", color=GREEN, h=1.5)

# ================================================================== 6 off-centre
r15 = row(E0, 45.0, 1.2)
m5_edge = r15["m_tilt_kg"]["5"]
s = slide(notes=(
    "Scenario two: the mass towards a corner, 1.2 m out on the diagonal, the worst direction. The "
    "deck tilts as a whole on its four clusters; the hubs stay level, so each cluster's buoys move "
    "together. The resistance is the clusters' difference in sinkage times their distance squared: "
    f"{k_tilt:.0f} N·m per radian, thirty times the cluster's hub, because sixteen buoys work at "
    f"about 0.8 m instead of four at 0.3 m. {m5_edge:.0f} kg at the corner tilts the deck 5 degrees; "
    f"at {r15['m_limit_kg']:.1f} kg the near cluster's pins reach the water with the deck at "
    f"{r15['tilt_at_limit_deg']:.0f} degrees."))
title(s, "Off-centre, the whole deck tilts", "1.2 m towards a corner: the hubs stay level; the "
      "clusters' difference in sinkage resists the tilt")
fit_picture(s, "pseq_offset.png", 0.6, 1.55, 12.1, 3.75)
stat(s, 0.6, 5.5, 3.8, f"{k_tilt:.0f} N·m/rad", f"deck tilt stiffness: {ratio:.0f}× the cluster's "
     "hub (16 buoys at ~0.8 m)", h=1.6)
stat(s, 4.75, 5.5, 3.8, f"{m5_edge:.0f} kg → 5°", "at the corner, 1.2 m out on the diagonal",
     color=AMBER, h=1.6)
stat(s, 8.9, 5.5, 3.8, f"{r15['m_limit_kg']:.1f} kg", f"the near cluster's pins awash, deck at "
     f"{r15['tilt_at_limit_deg']:.0f}° tilt", color=RED, h=1.6)

# ================================================================== 7 tilt vs m x e
lin_me = np.linspace(0.0, 80.0, 30)
series = []
for p in PT["paths"]:
    raised = p["h_pay_m"] > 0
    series.append({"name": f"{p['offset_m']:.1f} m, CG {'1 m above' if raised else 'on'} the deck",
                   "x": p["me_kgm"], "y": p["tilt_deg"], "color": PURPLE if raised else RED,
                   "width": 1.75, "dash": MSO_LINE_DASH_STYLE.DASH if raised else None})
series.append({"name": f"small-angle: θ = m g e / {k_tilt:.0f} N·m/rad", "x": lin_me,
               "y": np.degrees(lin_me * PT["g"] / k_tilt), "color": INK, "width": 1.5,
               "dash": MSO_LINE_DASH_STYLE.ROUND_DOT})
s = slide(notes=(
    "Why mass times offset describes the tilt. With the mass on the deck (red), the branches at "
    "three offsets almost fall on one curve: the payload's moment tilts the deck, against the "
    f"clusters' {k_tilt:.0f} N·m per radian. They separate a little because a heavy mass also sits "
    "above the pins and helps the tilt along. With the CG raised a metre (purple, dashed) that "
    "effect is large: the same moment tilts the deck much more, and most for the heaviest masses."))
title(s, "The tilt follows mass × offset", "Payload on the deck: one curve; raised a metre: the "
      "heavy masses tilt the deck further")
xy_chart(s, 0.5, 1.55, 7.6, 5.6, series, "mass × offset (kg·m)", "deck tilt (°)", (0, 80), (0, 30),
         legend=True, title_text="Deck tilt against payload moment, towards a corner")
text(s, 8.4, 1.6, 4.4, 0.5, "Reading it (CG on the deck)", size=20, bold=True, color=NAVY)
r10 = row(E0, 0.0, 1.0)
cells = [["Deck tilt", "mass × offset", "e.g. at 1.0 m"]]
for t in ("2", "5", "10", "20"):
    cells.append([f"{t}°", f"{me_at(t):.1f} kg·m", f"{r10['m_tilt_kg'][t]:.1f} kg"])
table(s, 8.4, 2.2, 4.4, 2.3, cells, size=14)
text(s, 8.4, 4.75, 4.4, 2.4, [
    [("mass × offset: ", {"bold": True}), ("the smallest over the offsets, so a safe reading; "
                                          "the same in every direction.", {})],
    [("Raised payloads ", {"bold": True}), ("tilt further for the same moment: by "
                                            f"1 / (1 − m·H / {tip_mh:.0f} kg·m), H = CG height above "
                                            "the pins.", {})]], size=13, color=MUTED, gap=8)

# ================================================================== 8 raised payload
s = slide(notes=(
    "Scenario three, the new physics: the same kind of offset load, but with the payload's centre "
    "of gravity a metre above the deck. As the deck tilts, the raised mass moves sideways over the "
    "low side and adds to the tipping moment. At 0.2 m off centre the limit falls from "
    f"{r_lo['m_limit_kg']:.0f} kg with the mass on the deck to {r_hi['m_limit_kg']:.0f} kg raised, "
    f"and the deck reaches {r_hi['tilt_at_limit_deg']:.0f} degrees instead of "
    f"{r_lo['tilt_at_limit_deg']:.0f}."))
title(s, "A raised payload tips the deck further", "CG 1.0 m above the deck, 0.2 m off centre: the "
      "raised mass leans out over the low side")
fit_picture(s, "pseq_raised.png", 0.6, 1.55, 12.1, 3.75)
stat(s, 0.6, 5.5, 3.8, f"{r_hi['m_limit_kg']:.0f} kg", f"limit at 0.2 m with the CG 1 m up "
     f"(on the deck: {r_lo['m_limit_kg']:.0f} kg)", color=RED, h=1.6)
stat(s, 4.75, 5.5, 3.8, f"{r_hi['tilt_at_limit_deg']:.0f}°", f"deck tilt at that limit (on the "
     f"deck: {r_lo['tilt_at_limit_deg']:.0f}°)", color=AMBER, h=1.6)
stat(s, 8.9, 5.5, 3.8, f"{m_tip_1m:.0f} kg", "centred, CG 1 m up: the level deck tips over before "
     "any pin is awash", color=PURPLE, h=1.6)

# ================================================================== 9 CG height
hv = VH["rows"]
hh = np.linspace(0.0, 2.0, 41)
s = slide(notes=(
    "How high can the payload's centre of gravity be? With the mass centred, the pins limit it to "
    f"{m_centred:.0f} kg while the CG is low. Above about {h_switch:.2f} m over the deck, the level "
    "deck tips over first. Tilting the deck by a small angle moves a mass at height H above the pins "
    "sideways by H times the angle: a tipping moment m g H per radian, against the clusters' "
    f"{k_tilt:.0f}. They balance at m times H equal to {tip_mh:.0f} kg·m, the dashed curve, which the "
    "full calculation follows exactly."))
title(s, "How high can the payload's CG be?", "Centred payload: its limit against the height of "
      "its CG above the deck")
xy_chart(s, 0.5, 1.55, 7.6, 5.6, [
    {"name": "limit (large-angle statics)", "x": [r["h_pay_m"] for r in hv],
     "y": [r["m_limit_kg"] for r in hv], "color": NAVY, "width": 3},
    {"name": f"tips over: m (h + {h_deck:.2f} m) = {tip_mh:.0f} kg·m", "x": hh,
     "y": [tip_mh / (h + h_deck) for h in hh], "color": RED, "width": 1.5,
     "dash": MSO_LINE_DASH_STYLE.DASH},
    {"name": "all pins awash", "x": [0.0, 2.0], "y": [m_centred, m_centred], "color": TEAL,
     "width": 1.5, "dash": MSO_LINE_DASH_STYLE.ROUND_DOT}],
    "payload CG height above the deck frame (m)", "centred payload limit (kg)", (0, 2), (0, 250),
    title_text="Centred payload: pins awash, or the deck tips over")
text(s, 8.4, 1.6, 4.4, 0.5, "The rule", size=20, bold=True, color=NAVY)
card(s, 8.4, 2.15, 4.4, 1.25)
text(s, 8.6, 2.25, 4.0, 1.1, [[("mass × CG height above the pins", {"bold": True})],
                              [(f"≤ {tip_mh:.0f} kg·m", {"bold": True, "size": 26, "color": RED})]],
     size=15, gap=2)
cells = [["Payload", "CG at most above the deck"]]
for m in (100.0, 150.0, 200.0):
    cells.append([f"{m:.0f} kg", f"{h_max(m):.2f} m"])
table(s, 8.4, 3.6, 4.4, 1.6, cells, size=14)
text(s, 8.4, 5.4, 4.4, 1.8, [
    [("Why: ", {"bold": True}), ("a raised mass moves sideways as the deck tilts; past this, its "
                                 f"tipping moment beats the clusters' {k_tilt:.0f} N·m/rad.", {})],
    [("A stability limit, ", {"bold": True}), ("not a working value: near it, small offsets "
                                               "give large tilts.", {})]], size=13, color=MUTED, gap=6)

# ================================================================== 10 animations
s = slide(notes=(
    "The three loadings animated, the mass rising from zero to the limit. They play in slide show "
    "(and in normal view in recent PowerPoint). Watch the centred platform sink straight down, the "
    "deck tilt towards the loaded corner with the hubs staying level, and the raised mass lean out "
    "over the low side."))
title(s, "Watch it happen", "The mass rising from zero to the limit (animations play in slide "
      "show)")
for i, (g, lab) in enumerate((("panim_centred.gif", "Centred, on the deck"),
                              ("panim_offset.gif", "1.2 m towards a corner"),
                              ("panim_raised.gif", "CG 1 m up, 0.2 m off centre"))):
    x = 0.55 + i * 4.15
    picture(s, g, x, 1.55, w=4.0)
    text(s, x, 5.95, 4.0, 0.4, lab, size=16, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

# ================================================================== 11 envelope
env_series = []
for d in (0.0, 15.0, 30.0, 45.0):
    rr = [r for r in E0["rows"] if r["dir_deg"] == d]
    lab = {0.0: "0° (between hubs)", 45.0: "45° (towards a hub)"}.get(d, f"{d:.0f}°")
    env_series.append({"name": f"limit, {lab}", "x": [r["offset_m"] for r in rr],
                       "y": [r["m_limit_kg"] for r in rr], "color": DIR_COL[d], "width": 2.5})
for env, lab, dash in ((E05, "0.5", MSO_LINE_DASH_STYLE.DASH), (E10, "1.0", MSO_LINE_DASH_STYLE.ROUND_DOT)):
    rr = [r for r in env["rows"] if r["dir_deg"] == 45.0]
    env_series.append({"name": f"45°, CG {lab} m above the deck", "x": [r["offset_m"] for r in rr],
                       "y": [r["m_limit_kg"] for r in rr], "color": PURPLE, "width": 1.75,
                       "dash": dash})
r0 = [r for r in E0["rows"] if r["dir_deg"] == 0.0 and r["offset_m"] > 0]
TILT_SHADE = {"5": INK, "10": MUTED, "20": RGBColor(0xA8, 0xB4, 0xBA)}
for t, w_ in (("5", 1.75), ("10", 1.5), ("20", 1.5)):
    pts = [(r["offset_m"], r["m_tilt_kg"][t]) for r in r0 if r["m_tilt_kg"][t] is not None]
    env_series.append({"name": f"deck tilt {t}°", "x": [a for a, _ in pts],
                       "y": [b for _, b in pts], "color": TILT_SHADE[t], "width": w_,
                       "dash": MSO_LINE_DASH_STYLE.DASH})
s = slide(notes=(
    "The envelope. Each solid line is the payload limit for one direction of the offset, with the "
    "mass on the deck: every limit is a cluster's pins reaching the water. Towards a hub, on the "
    f"diagonal (red), is the worst direction, by up to {worst_pct:.0f} percent. Purple: the worst "
    "direction with the CG raised half a metre and a metre. The grey dashed lines are deck tilts of "
    "5, 10 and 20 degrees, far inside the limits."))
title(s, "The payload envelope", "Payload mass against its offset from the deck centre: limits "
      "(solid), raised CG (purple) and deck tilt (dashed)")
xy_chart(s, 0.5, 1.5, 8.6, 5.75, env_series, "payload offset from the deck centre (m)",
         "payload mass on the deck (kg)", (0, 1.2), (0, 225), title_text=None)
text(s, 9.4, 1.6, 3.4, 0.5, "Reading it", size=20, bold=True, color=NAVY)
text(s, 9.4, 2.15, 3.4, 5.0, [
    [("Below a solid line: ", {"bold": True}), ("stable and pins dry, for that direction.", {})],
    [("Worst direction: ", {"bold": True}), (f"towards a hub (red), by up to {worst_pct:.0f} %.", {})],
    [("Every limit: ", {"bold": True}), ("one cluster's pins awash; the deck never tips with the "
                                         "mass on it.", {})],
    [("Purple: ", {"bold": True}), ("CG 0.5 m and 1.0 m above the deck: lower limits, and "
                                    "a tip-over when centred at 1.0 m.", {})],
    [("Dashed: ", {"bold": True}), ("deck tilt 5° (darkest), 10°, 20°.", {})],
], size=14, gap=9)

# ================================================================== 12 mass-CG envelope
cg_series = []
for d in (0.0, 45.0):
    rr = [r for r in E0["rows"] if r["dir_deg"] == d]
    lab = {0.0: "0° (between hubs)", 45.0: "45° (towards a hub)"}[d]
    cg_series.append({"name": f"limit, {lab}",
                      "x": [1e3 * r["cg_at_limit"]["cg_offset_m"] for r in rr],
                      "y": [r["cg_at_limit"]["total_mass_kg"] for r in rr], "color": DIR_COL[d],
                      "width": 2.5})
for t in ("5", "10", "20"):
    pts = [(r["offset_m"], r["m_tilt_kg"][t]) for r in r0 if r["m_tilt_kg"][t] is not None]
    cg_series.append({"name": f"deck tilt {t}°",
                      "x": [1e3 * m * e / (m_plat + m) for e, m in pts],
                      "y": [m_plat + m for _, m in pts], "color": TILT_SHADE[t], "width": 1.5,
                      "dash": MSO_LINE_DASH_STYLE.DASH})
s = slide(notes=(
    f"The same envelope as the whole platform's mass and centre of gravity. The platform weighs "
    f"{m_plat:.0f} kg unloaded. Its overall centre of gravity can move a few centimetres before the "
    "deck tilts 5 degrees, and roughly 15 to 20 centimetres before the limit, with the payload on "
    "the deck."))
title(s, "Mass and CG envelope of the whole platform", f"Total mass against the horizontal offset "
      f"of the platform's CG (unloaded {m_plat:.1f} kg, level geometry, payload on the deck)")
xy_chart(s, 0.5, 1.5, 8.6, 5.75, cg_series, "horizontal offset of the platform's CG (mm)",
         "total platform mass (kg)", (0, 200), (350, 580))
text(s, 9.4, 1.6, 3.4, 0.5, "At the limit", size=20, bold=True, color=NAVY)
cells = [["Payload at", "Total mass", "CG offset"]]
for e in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2):
    r = row(E0, 45.0, e)
    cells.append([f"{e:.1f} m", f"{r['cg_at_limit']['total_mass_kg']:.0f} kg",
                  f"{1e3 * r['cg_at_limit']['cg_offset_m']:.0f} mm"])
table(s, 9.4, 2.2, 3.4, 3.4, cells, size=13)
text(s, 9.4, 5.8, 3.4, 1.4, "Worst direction (towards a hub). The CG offset is that of the whole "
     "platform with the payload, before it tilts.", size=12, color=MUTED)

# ================================================================== 13 cluster vs platform
c_r5 = next(r for r in CENV["rows"] if r["dir_deg"] == 0.0 and r["offset_m"] == 0.1)
c_me5 = c_r5["m_tilt_kg"]["5"] * 0.1
c_m = 4 * CENV["m_buoy_kg"] + CENV["m_hub_kg"]
s = slide(notes=(
    "The cluster and the platform side by side. The platform carries about four clusters' worth "
    "centred, but it is thirty times stiffer in tilt, because its sixteen buoys work at about 0.8 m "
    "from the centre instead of 0.3 m. So off-centre loads that would roll a cluster's hub over only "
    "tilt the deck a few degrees. What the platform adds is height: its deck sits above the pins, "
    "so a raised payload can tip it, which the cluster with its mass in the pin plane never sees."))
title(s, "Cluster and platform, side by side", "Same buoys, same method; the platform spreads the "
      "buoys out and lifts the payload above the pins")
cells = [["", "Cluster (4 buoys)", "Platform (16 buoys)"],
         ["Unloaded mass", f"{c_m:.1f} kg", f"{m_plat:.1f} kg"],
         ["Centred limit (pins awash)", f"{m_centred_cluster:.1f} kg", f"{m_centred:.1f} kg"],
         ["Sinkage per kg", f"{1e3 * G_ACC / (4 * c33):.1f} mm", f"{sink_per_kg_mm:.2f} mm"],
         ["Tilt stiffness", f"{k_tilt_cluster:.0f} N·m/rad (hub)", f"{k_tilt:.0f} N·m/rad (deck)"],
         ["Mass × offset for 5° of tilt", f"{c_me5:.2f} kg·m", f"{me_at('5'):.1f} kg·m"],
         ["What limits off-centre loads", "a pin awash; beyond ~0.3 m the hub rolls over",
          "a cluster's pins awash, at every offset"],
         ["Worst direction", "towards a buoy", "towards a hub (the diagonal)"],
         ["Raised payload", "not run (mass in the pin plane)",
          f"mass × CG height above the pins ≤ {tip_mh:.0f} kg·m"],
         ["Buoys on their gimbals", "never flop", f"never flop (margin ≥ {gimbal_min:.0f} N·m/rad)"]]
table(s, 0.6, 1.65, 12.1, 5.2, cells, size=15)

# ================================================================== 14 validation
xs = PX["rows"]
d_like = max(abs(x["floatsim"]["tilt_deg"] - x["statics_like_for_like"]["tilt_deg"]) for x in xs)
d_real = max(100 * (x["statics_real_heights"]["tilt_deg"] / x["floatsim"]["tilt_deg"] - 1)
             for x in xs if x["floatsim"]["tilt_deg"] > 0)
s = slide(notes=(
    "How far to trust it. FloatSim, solving its own sixteen-buoy model of the platform with its "
    "joints, gives exactly the statics' sinkage and tilt when every weight is put in the pin plane, "
    "which is all FloatSim's linear model can represent. With the deck and payload at their real "
    f"heights the statics tilt up to {d_real:.1f} percent more: that is the weight-on-top effect "
    "FloatSim leaves out, small for a payload on the deck, large for a raised one."))
title(s, "Cross-checks", "FloatSim's own platform model against the large-angle statics at small "
      "tilt")
stat(s, 0.6, 1.8, 3.6, f"{d_like:.3f}°", "largest tilt difference, like for like (every weight "
     "in the pin plane)", color=GREEN, h=1.9)
stat(s, 0.6, 3.95, 3.6, f"+{d_real:.1f} %", "more tilt with the real deck and payload heights: "
     "the weight on top FloatSim omits", color=AMBER, h=1.9)
cells = [["Payload", "Offset, direction", "FloatSim sinkage / tilt", "Statics, like for like",
          "Statics, real heights"]]
for x in xs:
    f, a, b = x["floatsim"], x["statics_like_for_like"], x["statics_real_heights"]
    cells.append([f"{x['m_kg']:.0f} kg", f"{x['offset_m']:.1f} m, {x['dir_deg']:.0f}°",
                  f"{-1e3 * f['heave_m']:.1f} mm / {f['tilt_deg']:.3f}°",
                  f"{-1e3 * a['heave_m']:.1f} mm / {a['tilt_deg']:.3f}°",
                  f"{-1e3 * b['heave_m']:.1f} mm / {b['tilt_deg']:.3f}°"])
table(s, 4.6, 1.8, 8.1, 0.45 * len(cells), cells, size=13)
text(s, 4.6, 1.95 + 0.45 * len(cells), 8.1, 1.2, f"Both on FloatSim's {PX['nt']}-panel hydrostatics "
     f"(C33 {PX['floatsim_c33_N_per_m']:.1f} N/m per buoy). Sinkage is measured from the unloaded "
     "draft and includes the deck's and hubs' 14 kg.", size=12, color=MUTED)
card(s, 4.6, 5.75, 8.1, 1.6, RGBColor(0xF3, 0xF6, 0xF7))
text(s, 4.85, 5.85, 7.6, 1.45, [
    [("Like for like: ", {"bold": True}),
     ("FloatSim's linear model has no weight term, so the statics put every weight in the pin "
      "plane too; then the two agree exactly.", {})],
    [("Real heights: ", {"bold": True}),
     ("the deck 0.18 m above the pins and the payload on it. The extra tilt is the m g h effect "
      "FloatSim omits: small for these loads, but the effect that tips a raised payload.", {})]],
    size=13, gap=6)

# ================================================================== 15 what it means
s = slide(notes=(
    "What to take away. Keep heavy payloads low and central: then the platform carries about two "
    "hundred kilograms before its pins are awash. Plan off-centre loads by deck tilt. Watch the "
    "height of the payload's centre of gravity, which is the platform's new limit. The analysis is "
    "static and in calm water; waves, the mooring and the hardware would all reduce the margins."))
title(s, "What it means for loading the platform")
card(s, 0.6, 1.5, 5.95, 4.6)
text(s, 0.85, 1.7, 5.5, 0.5, "Guidance", size=20, bold=True, color=TEAL)
text(s, 0.85, 2.3, 5.5, 4.2, [
    [("Keep the mass central and low. ", {"bold": True}), (f"Centred on the deck: {m_centred:.0f} kg "
     f"before the pins are awash; it only sinks, {sink_per_kg_mm:.2f} mm per kg.", {})],
    [("Plan off-centre loads by deck tilt. ", {"bold": True}),
     (f"5° ↔ {me_at('5'):.0f} kg·m, 10° ↔ {me_at('10'):.0f} kg·m, in every direction.", {})],
    [("Mind the payload's CG height. ", {"bold": True}),
     (f"Mass × CG height above the pins must stay well below {tip_mh:.0f} kg·m: e.g. 150 kg with "
      f"its CG no more than {h_max(150.0):.1f} m above the deck, as a stability limit.", {})],
    [("Worst direction: ", {"bold": True}), (f"towards a hub, by up to {worst_pct:.0f} % on the "
                                             "limit.", {})]], size=15, gap=9)
card(s, 6.8, 1.5, 5.95, 4.6, RGBColor(0xF6, 0xEE, 0xE8))
text(s, 7.05, 1.7, 5.5, 0.5, "Not included", size=20, bold=True, color=RED)
text(s, 7.05, 2.3, 5.5, 4.2, [
    [("Waves and dynamics: ", {"bold": True}), ("calm-water statics; waves reduce the margins, "
                                                "most for raised payloads.", {})],
    [("The mooring: ", {"bold": True}), ("free-floating; the mooring's lean and line loads are "
                                         "not included.", {})],
    [("Payload shape: ", {"bold": True}), ("a point mass; a wide payload spreads its load, "
                                           "which the envelope treats as its CG.", {})],
    [("Hardware: ", {"bold": True}), ("gimbal travel, the deck's own stiffness and payload "
                                      f"retention are not modelled. The buoys never come close "
                                      f"to touching ({gap_min:.2f} m between axes at worst).", {})]],
    size=15, gap=9)

out = HERE / "Platform_payload_stability.pptx"
prs.save(str(out))
print(f"wrote {out.name} ({len(prs.slides)} slides)")
