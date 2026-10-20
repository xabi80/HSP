"""PowerPoint deck: what a payload shift (a static tilt) does to the cluster and the platform in
waves. A summary of the tilted-dynamics study, with pictures, native charts and guidance.

Every number comes from this folder's records:
- tilted_cases.json: the equilibria, pin loads and C55;
- tilted_dynamics_preload.json: FloatSim with its weight preload, the results shown;
- tilted_dynamics.json: FloatSim without the preload, for the comparison;
- the accelerations at the payload (acc_payload in tilted_dynamics_preload.json): vertical, and
  lateral (horizontal, along the waves) as the payload feels it.
The pictures come from tilted_figures.py (wseq_cluster.png, wseq_platform.png). The charts are
native PowerPoint XY charts, editable in place.

Writes Payload_tilt_in_waves.pptx. Run: python make_tilt_waves_ppt.py
(after tilted_dynamics.py summary [--preload] and tilted_figures.py)
"""

# ruff: noqa: E501, E702, RUF001  -- slide-copy strings kept whole for editing; compact data
# lines; degree signs, dashes and times signs in the slide text
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from deck_kit import (
    AMBER,
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
from pptx.util import Inches

HERE = Path(__file__).resolve().parent


def load(name: str):  # type: ignore[no-untyped-def]
    return json.loads((HERE / name).read_text(encoding="utf-8"))


CASES = {(c["article"], c["case"]): c for c in load("tilted_cases.json")["cases"]}
NEW = {(d["article"], d["case"]): d for d in load("tilted_dynamics_preload.json")}
OLD = {(d["article"], d["case"]): d for d in load("tilted_dynamics.json")}
GREY = RGBColor(0x8A, 0x95, 0x9C)
STYLE = {"unloaded": (GREY, "unloaded"), "centred": (TEAL, "payload centred"),
         "tilt5": (AMBER, "shifted: 5° tilt"), "tilt10": (RED, "shifted: 10° tilt")}
ARTS = {"cluster": "Cluster, 12.5 kg on the hub", "platform": "Platform, 50 kg on the deck"}
NAMES = ("unloaded", "centred", "tilt5", "tilt10")


# ------------------------------------------------------------------ data
def run(d: dict, T: float) -> dict:
    return next(r for r in d["runs"] if abs(r["T_s"] - T) < 1e-9)


def rock_period(art: str, name: str, new: bool = True) -> float:
    d = NEW[(art, name)] if new else OLD[(art, name)]
    return d["periods_preload" if new else "periods"]["buoy_tilt"]["T_s"]


def series(art: str, key: str, scale: float = 1.0, old: bool = False, names=NAMES) -> list:  # type: ignore[no-untyped-def]
    out = []
    for n in names:
        d = (OLD if old else NEW)[(art, n)]
        col, lab = STYLE[n]
        out.append({"name": (f"{lab} (without preload)" if old else lab),
                    "x": [r["T_s"] for r in d["runs"]], "y": [scale * r[key] for r in d["runs"]],
                    "color": col, "width": 1.25 if old else 2.5,
                    "dash": MSO_LINE_DASH_STYLE.DASH if old else None})
    return out


gimbal_peak = {a: max(max(r["gimbal_deg_max"] for r in NEW[(a, n)]["runs"]) for n in NAMES[1:])
               for a in ARTS}
gimbal_peak_min = {a: min(max(r["gimbal_deg_max"] for r in NEW[(a, n)]["runs"]) for n in NAMES[1:])
                   for a in ARTS}
old_at42 = max(run(OLD[(a, n)], 4.2)["gimbal_deg_max"] for a in ARTS for n in NAMES[1:])
fb_worst = {a: min(min(r["min_freeboard_m"] for r in NEW[(a, n)]["runs"]) for n in NAMES) for a in ARTS}
fb_static10 = {a: min(NEW[(a, "tilt10")]["static_freeboard_m"]) for a in ARTS}
fb_loss10 = {a: fb_static10[a] - min(r["min_freeboard_m"] for r in NEW[(a, "tilt10")]["runs"]) for a in ARTS}
h_awash = {a: 0.1 * fb_static10[a] / fb_loss10[a] for a in ARTS}   # linear extrapolation
heave_period = {a: (NEW[(a, "unloaded")]["periods_preload"]["heave"]["T_s"],
               NEW[(a, "centred")]["periods_preload"]["heave"]["T_s"]) for a in ARTS}
pitch265 = {a: [run(NEW[(a, n)], 2.65)["pitch_deg"] for n in NAMES[1:]] for a in ARTS}
offsets = {a: [CASES[(a, n)]["offset_m"] for n in ("tilt5", "tilt10")] for a in ARTS}
# The heave and hub / deck pitch resonances of the loaded cases (whole-article motions), as
# opposed to the buoys' rocking on their gimbals (~4.2 s)
hp_periods = [NEW[(a, n)]["periods_preload"][m]["T_s"] for a in ARTS for n in NAMES[1:]
              for m in ("heave", "pitch")]
hp_band = f"{min(hp_periods):.1f}–{max(hp_periods):.1f} s"


def net(b: dict) -> float:
    return b["C55_Nm_per_rad"] - b["PL_Nm_per_rad"]


cl10 = CASES[("cluster", "tilt10")]["buoys"]
near = max(cl10, key=lambda b: b["sink_m"]); far = min(cl10, key=lambda b: b["sink_m"])
k_level = net(CASES[("cluster", "centred")]["buoys"][0])
t_level = rock_period("cluster", "centred")
r42_cl = run(NEW[("cluster", "tilt10")], 4.2)["gimbal_deg"]
r42_pl = run(NEW[("platform", "tilt10")], 4.2)["gimbal_deg"]

# Accelerations at the payload (m/s^2): vertical az; lateral (along the waves) as felt, fx
G_ACC = 9.806


def acc_max(art: str, name: str, comp: str, periods: tuple[float, ...] | None = None) -> float:
    return max(r["acc_payload"][comp] for r in NEW[(art, name)]["runs"]
               if periods is None or any(abs(r["T_s"] - T) < 1e-9 for T in periods))


def static_lat(art: str, name: str) -> float:
    return G_ACC * float(np.sin(np.radians(NEW[(art, name)]["tilt_deg"])))


def total_lat(art: str, name: str) -> float:
    return max(static_lat(art, name) + r["acc_payload"]["fx"] for r in NEW[(art, name)]["runs"])


def acc_series(art: str, comp: str) -> list:
    return [{"name": STYLE[n][1], "x": [r["T_s"] for r in NEW[(art, n)]["runs"]],
             "y": [r["acc_payload"][comp] for r in NEW[(art, n)]["runs"]], "color": STYLE[n][0],
             "width": 2.5} for n in NAMES]


az_top = max(acc_max(a, n, "az") for a in ARTS for n in NAMES)
az_loaded = {a: max(acc_max(a, n, "az") for n in NAMES[1:]) for a in ARTS}
fx_short = {a: run(NEW[(a, "centred")], 1.5)["acc_payload"]["fx"] for a in ARTS}
fx_res = {a: (acc_max(a, "centred", "fx", (2.35, 2.65)), acc_max(a, "tilt10", "fx", (2.35, 2.65)))
          for a in ARTS}
fx42 = [run(NEW[(a, n)], 4.2)["acc_payload"]["fx"] for a in ARTS for n in NAMES[1:]]
fx_waves = max(acc_max(a, n, "fx") for a in ARTS for n in NAMES[1:])
lat_total = {a: total_lat(a, "tilt10") for a in ARTS}
st10 = static_lat("platform", "tilt10")

# ================================================================== 1 title
s = slide(dark=True, notes=(
    "This deck summarises what shifting a payload does to the cluster and the platform in waves. "
    "The payload is shifted until the hub or deck tilts 5 and 10 degrees, and both are run in six "
    "regular wave periods plus one at 4.2 s. All results are FloatSim, with the hydrodynamics "
    "recomputed at each tilted position and FloatSim's new weight preload switched on."))
text(s, 0.8, 0.55, 11.8, 0.5, "FLUME TEST ARTICLES · PAYLOAD SHIFT IN WAVES", size=14, bold=True, color=ICE)
text(s, 0.8, 1.0, 11.8, 1.0, "Shifting the payload: what tilt does in waves", size=40, bold=True, color=WHITE)
text(s, 0.8, 1.95, 11.8, 0.9, "Cluster (12.5 kg) and platform (50 kg), level vs 5° vs 10° of tilt, "
     "regular waves H = 0.1 m, T = 1.5–5 s", size=19, color=ICE)
card(s, 0.8, 3.0, 11.75, 3.75, WHITE)
fit_picture(s, "wseq_platform.png", 0.95, 3.1, 11.45, 3.55)
text(s, 0.8, 6.9, 11.8, 0.4, "Payload-stability study · 2026-10-18 · FloatSim with the weight preload, "
     "BEM re-run at each tilted equilibrium", size=12, color=ICE)

# ================================================================== 2 bottom line
s = slide(notes=(
    "Four numbers. With a payload on board, the buoys' rocking on their gimbals resonates at about "
    "4.2 seconds, and there H = 0.1 m waves rock them 11 to 12.6 degrees. The tilt itself barely "
    "moves that resonance: the payload's mass sets it. And the pins stay well clear of the water at "
    "this wave height: the tightest case is the platform at 10 degrees, with 282 mm left."))
title(s, "The bottom line", "H = 0.1 m regular waves along the shift; FloatSim with its weight preload")
stat(s, 0.6, 1.9, 2.9, "≈ 4.2 s", f"the buoys' rocking resonance with a payload ({rock_period('cluster', 'tilt10'):.2f} s cluster, "
     f"{rock_period('platform', 'tilt10'):.2f} s platform)", color=RED)
stat(s, 3.7, 1.9, 2.9, f"{gimbal_peak_min['platform']:.0f}–{max(gimbal_peak.values()):.1f}°", "of rocking on the gimbals "
     "there, level or tilted", color=AMBER)
stat(s, 6.8, 1.9, 2.9, f"{rock_period('cluster', 'tilt10'):.2f} s",
     f"the cluster's rocking at 10° tilt, vs {rock_period('cluster', 'centred'):.2f} s level: the "
     "tilt barely moves it", color=TEAL)
stat(s, 9.9, 1.9, 2.85, f"{1e3 * fb_worst['platform']:.0f} mm", "lowest pin freeboard in the waves "
     "(platform, 10°): no pin awash", color=GREEN)
text(s, 0.6, 4.0, 12.1, 0.5, "What the tilt does", size=20, bold=True, color=NAVY)
text(s, 0.6, 4.55, 12.1, 2.7, [
    [("Heave: ", {"bold": True}), (f"the mass, not the tilt — the payload lengthens the heave period "
     f"{heave_period['cluster'][0]:.2f} → {heave_period['cluster'][1]:.2f} s; tilt changes heave by ≤ 6 %.", {})],
    [("Hub / deck tilting: ", {"bold": True}), (f"near the article's heave and tilt resonance ({hp_band}; "
     "the 2.65 s wave) the tilt turns heave into pitch: "
     f"{pitch265['cluster'][0]:.1f} → {pitch265['cluster'][2]:.1f}° on the cluster hub, "
     f"{pitch265['platform'][0]:.1f} → {pitch265['platform'][2]:.1f}° on the platform deck, level → 10°.", {})],
    [("Freeboard: ", {"bold": True}), ("the tilt takes still-water freeboard from the low side; the "
     f"waves take ~{1e3 * min(fb_loss10.values()):.0f}–100 mm more at worst.", {})],
    [("Buoys' own pitch: ", {"bold": True, "color": RED}), ("each buoy rocking on its gimbal — the "
     "flume record's 2.76–2.9 s pitch resonance — moves to ~4.2 s with a payload, whatever the tilt: "
     "keep a loaded article out of waves near 4.2 s.", {"color": RED})],
    [("Accelerations at the payload: ", {"bold": True}), (f"vertical ≤ {az_top / G_ACC:.2f} g; lateral "
     f"(along the waves) ≤ {fx_waves:.1f} m/s² from the waves, but a 10° tilt adds a constant "
     f"{st10:.2f} m/s² ({st10 / G_ACC:.2f} g): up to {max(lat_total.values()) / G_ACC:.2f} g in total.", {})],
], size=15, gap=6)

# ================================================================== 3 the cases
s = slide(notes=(
    "The cases. On the cluster, 12.5 kg — a quarter of the platform's 50 kg — sits on the hub; on "
    "the platform 50 kg sits on the deck. Each is first centred, then shifted along the wave "
    "direction until the hub or deck tilts 5 and 10 degrees. The pictures are the solved static "
    "equilibria, to scale: the loaded side sinks, the other side rises, and the lowest pin loses "
    "freeboard. The unloaded article is run as well, for reference."))
title(s, "The cases", "The payload centred, then shifted along the waves until the hub / deck tilts 5° and 10°")
fit_picture(s, "wseq_cluster.png", 0.5, 1.5, 6.1, 2.9)
fit_picture(s, "wseq_platform.png", 6.75, 1.5, 6.1, 2.9)
text(s, 0.5, 4.45, 6.1, 0.4, "Cluster: 12.5 kg on the hub (50 kg / 4)", size=16, bold=True, color=NAVY)
text(s, 6.75, 4.45, 6.1, 0.4, "Platform: 50 kg on the deck", size=16, bold=True, color=NAVY)
cells = [["", "Cluster", "Platform"],
         ["Shift for 5° / 10°", f"{1e3 * offsets['cluster'][0]:.0f} / {1e3 * offsets['cluster'][1]:.0f} mm",
          f"{offsets['platform'][0]:.2f} / {offsets['platform'][1]:.2f} m"]]
for n, lab in (("centred", "Lowest pin, still water: centred"), ("tilt5", "… 5°"), ("tilt10", "… 10°")):
    cells.append([lab, f"{1e3 * min(NEW[('cluster', n)]['static_freeboard_m']):.0f} mm",
                  f"{1e3 * min(NEW[('platform', n)]['static_freeboard_m']):.0f} mm"])
table(s, 0.5, 5.0, 12.3, 2.1, cells, size=14)

# ================================================================== 4 method
s = slide(notes=(
    "How it was computed. The large-angle statics give each case's equilibrium: where every buoy "
    "floats and how much load each pin carries. Capytaine recomputes the coupled hydrodynamics of "
    "the buoys at exactly those positions. FloatSim then runs regular waves about that equilibrium, "
    "with its new weight preload so the joints carry their static loads. A very soft horizontal "
    "hold stops the free article drifting; it barely touches the wave response."))
title(s, "How it was computed", "Statics → hydrodynamics at the tilted geometry → FloatSim in waves")
steps = [("1  Static equilibrium", "Large-angle statics: each buoy's draft, each pin's load, the "
          "payload offset for exactly 5° and 10°.", TEAL),
         ("2  BEM at the tilt", "Capytaine re-run with every buoy at its own position and draft, "
          "with its own hydrostatics (16 buoys: ~30 min a case).", AMBER),
         ("3  FloatSim in waves", "Regular waves H = 0.1 m along the shift, T = 1.5 / 2 / 2.35 / "
          "2.65 / 3.5 / 4.2 / 5 s; drag relative to the waves.", GREEN),
         ("4  Weight preload", "FloatSim's new opt-in: the joints carry the static loads, so the "
          "load on each pin tips its buoy (see next slide).", RED)]
for i, (h, b, col) in enumerate(steps):
    x = 0.55 + i * 3.17
    card(s, x, 1.8, 2.85, 3.0)
    text(s, x + 0.2, 1.95, 2.5, 0.8, h, size=17, bold=True, color=col)
    text(s, x + 0.2, 2.7, 2.5, 2.1, b, size=14)
    if i < 3:
        arrow(s, x + 2.88, 3.05, w=0.28, h=0.28)
text(s, 0.6, 5.2, 12.1, 1.9, [
    [("Free-floating ", {"bold": True}), ("with a very soft horizontal hold (surge period ~45 s) "
     "against start-up drift; ~1 % effect on the wave response at 5 s, less below.", {})],
    [("Checks: ", {"bold": True}), ("FloatSim's own static solve reproduces the statics geometry to "
     "1e-15 m in all 8 cases; its natural periods match an independent closed-form estimate.", {})]],
    size=14, color=MUTED, gap=6)

# ================================================================== 5 weight preload
cells = [["Case", "Rocking period, no preload", "With preload", "Pin load", "P·L / C55"]]
for a in ARTS:
    for n, lab in (("unloaded", "unloaded"), ("centred", "payload centred"), ("tilt10", "10° tilt")):
        bs = CASES[(a, n)]["buoys"]; b = max(bs, key=lambda x: x["PL_Nm_per_rad"])
        cells.append([f"{a.capitalize()}, {lab}", f"{rock_period(a, n, False):.2f} s", f"{rock_period(a, n):.2f} s",
                      f"{b['pin_load_N']:.0f} N", f"{b['PL_Nm_per_rad']:.0f} / {b['C55_Nm_per_rad']:.0f}"])
s = slide(notes=(
    "Why the weight preload changes the answer. Each buoy hangs from its gimbal at the top of the "
    "spar, 1.6 m above its centre of gravity. The hub, deck and payload press down on that pin. When "
    "the buoy leans, the pin moves sideways and the load on it pushes the buoy further over: it "
    "takes P times L off the buoy's righting stiffness. FloatSim used to leave the weights out of "
    "its statics, so the pins carried nothing and the buoys looked far too stiff. With the "
    "preload, a payload slows the buoys' rocking from about 2.85 to about 4.2 seconds."))
title(s, "Why FloatSim's weight preload changes the answer", "The load on each pin tips its buoy: "
      "it takes P × L off the buoy's righting stiffness")
text(s, 0.6, 1.7, 5.2, 4.5, [
    [("The buoy hangs from its pin, ", {"bold": True}), ("1.62 m above its centre of gravity. The "
     "hub, deck and payload press down on the pin.", {})],
    [("When the buoy leans, ", {"bold": True}), ("the pin and its load move sideways and push it "
     "further over: net righting = C55 − P × L.", {})],
    [("FloatSim used to leave weights out of its statics, ", {"bold": True}), ("so the pins "
     "carried no load and the buoys looked too stiff.", {})],
    [("With the preload, ", {"bold": True, "color": RED}), ("a payload slows the buoys' rocking from "
     "~2.85 to ~4.2 s — into the wave band.", {"color": RED})]], size=15, gap=10)
tb = table(s, 6.1, 1.75, 6.7, 3.6, cells, size=13)
for j, wd in enumerate((2.15, 1.35, 1.1, 0.95, 1.15)):
    tb.columns[j].width = Inches(wd)
text(s, 6.1, 6.0, 6.7, 1.2, "P·L / C55 in N·m/rad, the most loaded buoy. Heave, deck pitch and pin "
     "freeboard change little with the preload; the buoys' rocking changes completely.", size=12,
     color=MUTED)

# ================================================================== 6-9 charts against period
CHARTS = [
    ("heave_m", 1e3, "heave amplitude (mm)", (0, 80),
     "Heave: the payload's mass matters, the tilt hardly does",
     "Heave of the hub / deck centre against wave period",
     (f"The payload lengthens the heave period ({heave_period['cluster'][0]:.2f} → {heave_period['cluster'][1]:.2f} s "
      f"cluster, {heave_period['platform'][0]:.2f} → {heave_period['platform'][1]:.2f} s platform), so heave falls a little "
      "below resonance. 5° and 10° of tilt change heave by ≤ 6 %."), False),
    ("pitch_deg", 1.0, "pitch amplitude (°)", (0, 4),
     "Near the heave resonance, the tilt turns heave into pitch",
     f"Hub / deck pitch (the whole article tilting; not the buoys' own pitch) — resonates with heave at {hp_band}",
     (f"At 2.65 s the cluster hub pitches {pitch265['cluster'][0]:.1f} → {pitch265['cluster'][1]:.1f} → "
      f"{pitch265['cluster'][2]:.1f}° and the platform deck {pitch265['platform'][0]:.1f} → "
      f"{pitch265['platform'][1]:.1f} → {pitch265['platform'][2]:.1f}° (level → 5° → 10°): the shifted "
      "mass couples the article's heave and pitch, which resonate together at "
      f"{hp_band}. Away from it, including at the buoys' rocking resonance (4.2 s), the tilt adds "
      "almost no pitch."), False),
    ("min_freeboard_m", 1e3, "smallest pin freeboard (mm)", (0, 700),
     "The tilt costs freeboard; the waves take a little more",
     "Smallest pin freeboard during the wave cycle",
     (f"The lowest pins lose still-water freeboard as the article tilts; the waves take "
      f"~{1e3 * fb_loss10['platform']:.0f} mm more at worst (2.65 s). No pin comes near the water at "
      f"H = 0.1 m; linearly, at 10° a pin would reach it only at H ≈ {h_awash['platform']:.2f} m (platform) "
      f"or {h_awash['cluster']:.2f} m (cluster)."), False),
    ("gimbal_deg_max", 1.0, "largest gimbal angle (°)", (0, 14),
     "With a payload, the buoys rock hard at ~4.2 s",
     "The buoys' own pitch on their gimbals (dashed: FloatSim without the weight preload, payload centred)",
     (f"With the preload, a payload puts the buoys' rocking resonance at ~4.2 s, where they rock "
      f"{gimbal_peak_min['platform']:.0f}–{max(gimbal_peak.values()):.1f}° — level or tilted. Without the "
      f"preload FloatSim gave ≤ {old_at42:.1f}° there and a peak at 2.65 s — the flume record's 2.76–2.9 s "
      "pitch resonance, which leaves out the load on the pins."), True),
]
for key, sc, yt, yr, ttl, sub, msg, show_old in CHARTS:
    s = slide(notes=msg)
    title(s, ttl, sub)
    for i, a in enumerate(ARTS):
        ser = series(a, key, sc)
        if show_old:
            ser += series(a, key, sc, old=True, names=("centred",))
        x = 0.5 + i * 6.25
        xy_chart(s, x, 1.55, 6.1, 4.55, ser, "wave period (s)", yt, (1.0, 5.5), yr,
                 legend=(i == 1), title_text=ARTS[a])
    text(s, 0.6, 6.25, 12.1, 1.0, msg, size=14, color=INK)

# ================================================================== accelerations: vertical
s = slide(notes=(
    "Accelerations at the payload, where it sits in each case. Vertical first: small on both "
    "articles, at most about four hundredths of g, peaking near the heave resonance around 2.65 s. "
    "The payload's mass lowers them a little and the tilt hardly changes them."))
title(s, "Accelerations at the payload: vertical is small", "Vertical acceleration amplitude at the "
      "payload against wave period, H = 0.1 m (1 g = 9.81 m/s²)")
for i, a in enumerate(ARTS):
    xy_chart(s, 0.5 + i * 6.25, 1.55, 6.1, 4.55, acc_series(a, "az"), "wave period (s)",
             "vertical acceleration (m/s²)", (1.0, 5.5), (0, 0.5), legend=(i == 1), title_text=ARTS[a])
text(s, 0.6, 6.25, 12.1, 1.0, f"At most {az_top:.2f} m/s² ({az_top / G_ACC:.2f} g), unloaded, near the "
     f"heave resonance; with the payload {az_loaded['cluster']:.2f} (cluster) and "
     f"{az_loaded['platform']:.2f} m/s² (platform). Both articles ride the same buoys' heave, so they "
     "are alike, and the tilt changes it little.", size=14, color=INK)

# ================================================================== accelerations: lateral
s = slide(notes=(
    "Lateral acceleration, horizontal along the waves, as the payload feels it: the deck's motion "
    "plus gravity swinging along the deck as it pitches. The platform is much calmer in short "
    "waves because its 2.4 m span averages them out. Near the heave resonance the tilt adds lateral "
    "load through the coupled pitch. And at 4.2 s, with a payload, both feel about half a metre "
    "per second squared: the buoys rocking on their gimbals make the hub or deck slide along the "
    "waves."))
title(s, "Lateral acceleration (along the waves)", "As the payload feels it: the deck's motion plus "
      "gravity swinging along the deck as it pitches")
for i, a in enumerate(ARTS):
    xy_chart(s, 0.5 + i * 6.25, 1.55, 6.1, 4.3, acc_series(a, "fx"), "wave period (s)",
             "lateral acceleration (m/s²)", (1.0, 5.5), (0, 1.2), legend=(i == 1), title_text=ARTS[a])
text(s, 0.6, 5.95, 12.1, 1.4, [
    [("Short waves: ", {"bold": True}), (f"the platform's 2.4 m span averages them out: "
     f"{fx_short['platform']:.2f} against {fx_short['cluster']:.2f} m/s² at 1.5 s (payload centred).", {})],
    [("Near the heave resonance (2.35–2.65 s): ", {"bold": True}), ("the tilt adds lateral load: "
     f"cluster {fx_res['cluster'][0]:.2f} → {fx_res['cluster'][1]:.2f}, platform {fx_res['platform'][0]:.2f} "
     f"→ {fx_res['platform'][1]:.2f} m/s², level → 10°.", {})],
    [("At 4.2 s, with a payload: ", {"bold": True}), (f"{min(fx42):.1f}–{max(fx42):.1f} m/s²: the buoys "
     "rocking on their gimbals slide the hub / deck along the waves.", {})]], size=13, gap=4)

# ================================================================== accelerations: total lateral
cells = [["", "Vertical, max", "Lateral from the waves, max", "Static tilt, g·sin θ", "Total lateral, max"]]
for a in ARTS:
    for n, lab in (("unloaded", "unloaded"), ("centred", "payload centred"), ("tilt5", "5° tilt"),
                   ("tilt10", "10° tilt")):
        tot = total_lat(a, n)
        cells.append([f"{a.capitalize()}, {lab}", f"{acc_max(a, n, 'az'):.2f}",
                      f"{acc_max(a, n, 'fx'):.2f}", f"{static_lat(a, n):.2f}",
                      f"{tot:.2f} ({tot / G_ACC:.2f} g)"])
s = slide(notes=(
    "For the payload's lateral restraint, the static tilt is the biggest load. Tilted 10 degrees, "
    "the payload feels a constant 1.7 metres per second squared along the deck from gravity "
    "alone, about 0.17 g, before any wave. The waves add at most about 0.8 on top, for a total of "
    "about a quarter of g at H = 0.1 m."))
title(s, "For the payload, the static tilt is the biggest lateral load", "Maximum accelerations at the "
      "payload over the seven wave periods, m/s² (H = 0.1 m)")
tb = table(s, 0.6, 1.75, 8.2, 4.4, cells, size=13)
for j, wd in enumerate((2.3, 1.2, 1.7, 1.4, 1.6)):
    tb.columns[j].width = Inches(wd)
stat(s, 9.2, 1.75, 3.55, f"{st10:.2f} m/s²", f"along the deck from a 10° tilt alone "
     f"({st10 / G_ACC:.2f} g), before any wave", color=RED, h=1.7)
stat(s, 9.2, 3.65, 3.55, f"{lat_total['cluster'] / G_ACC:.2f} / {lat_total['platform'] / G_ACC:.2f} g",
     "worst total lateral at 10°: cluster / platform", color=AMBER, h=1.7)
text(s, 9.2, 5.55, 3.55, 1.6, "Keeping the payload level matters more for its lateral restraint "
     "than the wave motion does.", size=13, color=MUTED)

# ================================================================== 10 one period
s = slide(notes=(
    "Wouldn't each buoy rock at its own period once the load is shifted? On its own, yes: at 10 "
    "degrees the cluster's loaded pair is softer and would rock at about 4.45 seconds, the other "
    "pair at about 4.0. But they all hang from the same hub. In the mode the waves drive, every "
    "buoy leans the same way while the hub slides under them, so they rock together at one "
    "period, set by the sum of their stiffnesses. Shifting the payload softens one side and "
    "stiffens the other by almost the same amount, so the sum, and the period, hardly change."))
title(s, "One rocking period for all the buoys", "Each buoy alone would differ; tied to one hub "
      "they rock together")
cells = [["Cluster at 10°", "Sinkage", "Load on pin", "Net stiffness", "Period if alone"]]
for lab, b in (("Loaded pair", near), ("Other pair", far)):
    cells.append([lab, f"{1e3 * b['sink_m']:.0f} mm", f"{b['pin_load_N']:.0f} N", f"{net(b):.0f} N·m/rad",
                  f"~{t_level * np.sqrt(k_level / net(b)):.2f} s"])
cells.append(["Level (all four)", f"{1e3 * CASES[('cluster', 'centred')]['buoys'][0]['sink_m']:.0f} mm",
              f"{CASES[('cluster', 'centred')]['buoys'][0]['pin_load_N']:.0f} N", f"{k_level:.0f} N·m/rad",
              f"{t_level:.2f} s"])
table(s, 0.6, 1.75, 6.4, 2.0, cells, size=13)
text(s, 0.6, 4.0, 6.4, 3.2, [
    [("The hub carries all the pins together: ", {"bold": True}), ("in the mode the waves drive, "
     "every buoy leans the same way, so they share one period.", {})],
    [("Its stiffness is the sum of theirs: ", {"bold": True}),
     (f"2 × {net(near):.0f} + 2 × {net(far):.0f} = {2 * (net(near) + net(far)):.0f}, against "
      f"4 × {k_level:.0f} = {4 * k_level:.0f} level → {rock_period('cluster', 'tilt10'):.2f} vs "
      f"{rock_period('cluster', 'centred'):.2f} s.", {})],
    [("Why: ", {"bold": True}), ("a buoy's stiffness falls almost linearly as it sinks; the shift "
     "sinks one side and lifts the other by the same amount.", {})]], size=14, gap=8)
stat(s, 7.4, 1.75, 5.35, f"{min(r42_cl):.1f}–{max(r42_cl):.1f}°", "the four cluster buoys' rocking "
     "at 4.2 s, 10° tilt: almost equal", color=RED, h=1.55)
stat(s, 7.4, 3.5, 5.35, f"{min(r42_pl):.1f}–{max(r42_pl):.1f}°", "all 16 platform buoys at 4.2 s, "
     "10° tilt", color=AMBER, h=1.55)
text(s, 7.4, 5.25, 5.35, 1.9, "The buoys can rock against each other only by sliding sideways "
     "relative to one another — those modes sit near 9 s. A cross-wise twisting pair (3.7 / 4.4 s) "
     "needs waves from the side to be driven.", size=13, color=MUTED)

# ================================================================== 11 what it means
s = slide(notes=(
    "What to take away. With a payload on board, the period to watch is about 4.2 seconds, where "
    "the buoys rock hard on their gimbals, and that holds whether the payload is level or not. "
    "The tilt itself matters for freeboard and for pitch near the heave resonance. The analysis is "
    "linear and in regular waves along the shift; the large rocking angles are past FloatSim's "
    "small-angle range, so their size is indicative."))
title(s, "What it means for testing with a payload")
card(s, 0.6, 1.5, 5.95, 4.7)
text(s, 0.85, 1.7, 5.5, 0.5, "Guidance", size=20, bold=True, color=TEAL)
text(s, 0.85, 2.3, 5.5, 3.9, [
    [("Avoid waves near 4.2 s with a payload on board: ", {"bold": True}),
     (f"the buoys rock {gimbal_peak_min['platform']:.0f}–{max(gimbal_peak.values()):.1f}° on their gimbals "
      "at H = 0.1 m, level or tilted.", {})],
    [("The mass sets the rocking; the tilt sets the freeboard: ", {"bold": True}),
     ("keep the payload level to keep the low side's pins dry.", {})],
    [("Near the heave resonance (2.6–2.8 s) ", {"bold": True}), ("a tilted payload adds pitch: up to "
     f"{pitch265['cluster'][2]:.1f}° (cluster) and {pitch265['platform'][2]:.1f}° (platform) at 10°.", {})],
    [("Pins stay dry at H = 0.1 m ", {"bold": True}), (f"in every case (≥ {1e3 * fb_worst['platform']:.0f} mm).", {})],
    [("Lateral restraint: ", {"bold": True}), (f"design for the static tilt plus waves: up to "
     f"{max(lat_total.values()) / G_ACC:.2f} g at 10°, of which {st10 / G_ACC:.2f} g is the tilt alone.", {})]],
    size=14, gap=7)
card(s, 6.8, 1.5, 5.95, 4.7, RGBColor(0xF6, 0xEE, 0xE8))
text(s, 7.05, 1.7, 5.5, 0.5, "Not included", size=20, bold=True, color=RED)
text(s, 7.05, 2.3, 5.5, 3.9, [
    [("Large rocking: ", {"bold": True}), ("11–12.6° is past FloatSim's small-angle range (5.7°); the "
     "size is indicative, the resonance period is not. Gimbal travel is not modelled.", {})],
    [("Waves: ", {"bold": True}), ("regular, along the shift only; waves from the side drive other "
     "modes.", {})],
    [("Periods: ", {"bold": True}), ("seven points; the peaks between them are not resolved.", {})],
    [("Model: ", {"bold": True}), ("linear about each equilibrium; 12-panel BEM (~0.05 s on the "
     "periods); free-floating with a soft hold, not the flume mooring.", {})]], size=15, gap=9)

# ================================================================== 12 appendix: FloatSim change
s = slide(notes=(
    "For reference: the FloatSim change behind these results. The weight preload is opt-in. It adds "
    "every body's weight and each floating body's buoyancy, solves the static equilibrium through "
    "the joints, carries the joint loads in the dynamics, and refuses any model whose loads leave "
    "it without restoring. It was validated against a closed-form pinned float, against the full "
    "test suite with the switch off, and against this study's hand statics on the real cluster."))
title(s, "Appendix: FloatSim's weight preload", "Opt-in `simulation.weight_preload: true` "
      "(main 05128b7, dd889f2; docs/weight-preload-plan.md)")
cells = [["Check", "Result"],
         ["Gate A: a float pinned at its top, T = 2π√((I + mL²)/(C55 − P L))",
          "pin force exact; periods within 1e-4 (9.98 / 4.62 / 5.93 s); finite-amplitude correction matches"],
         ["Gate B: switch off", "unchanged: 786 fast tests pass"],
         ["Gate C: the flume cluster vs the study's statics",
          "157.94 mm sinkage (10 kg), 0.869° tilt (1 kg at 0.1 m), a quarter of the load per pin"],
         ["Geometric stiffness K_geo", "= the closed form (−P L) on the pinned float and on the cluster; symmetric"],
         ["This study", "static solve keeps the statics geometry to 1e-15 m; periods = the closed-form estimate"]]
table(s, 0.6, 1.8, 12.1, 4.2, cells, size=13)
text(s, 0.6, 6.2, 12.1, 1.0, "The flume mooring and wall-effect records were run without the preload: "
     "their buoy-rocking results are the ones affected (the unloaded platform's rocking mode is 3.10 s, "
     "not 2.81 s).", size=13, color=MUTED)

out = HERE / "Payload_tilt_in_waves.pptx"
prs.save(str(out))
print(f"wrote {out.name} ({len(prs.slides)} slides)")
