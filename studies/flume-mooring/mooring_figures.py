"""Figures for MOORING-SPEC rev C (flume-mooring, 2026-09-25): every figure is generated from the
committed geometry and results, so it cannot disagree with the spec tables. No coordinate is
entered by hand:

- article geometry: the FloatSim decks (floatsim_decks.deck: body reference points, joints, the
  spar Morison members and the heave-plate member) and articulated_wall's pin / CoG levels;
- flume: mooring_sizing.W_FLUME / H_FLUME, mooring_verify.L_ANCHOR;
- anchors, attachment points, pretension: spec_statics.json (mooring_spec.py);
- calm equilibrium and line profiles: FloatSim's settle (moored_equilibrium.json) and
  line_hardware.line_states (the catenary closed form at FloatSim's line force);
- the design rationale: attachment_sweep.json through build_spec.sweep_selection (the same rows
  as the spec's sweep table);
- offsets: attachment_design.json (operational set) and extreme_set.json (extreme set).

  (a) plan_<article>.png       plan view, to scale
  (b) elevation_<article>.png  elevation looking across the flume, to scale (+ sag inset)
  (c) rationale.png            tilt-period shift vs attachment height (the sweep)
  (d) offsets.png              H = 0.5 m mean positions, both cord sets, vs the tracking window

``spotcheck`` reads back, from the drawn figures, one anchor coordinate and the attachment depth
per article and compares them with the tables in MOORING-SPEC.md.

Run: python mooring_figures.py [spotcheck]
"""
# ruff: noqa: E402, RUF001  -- sys.path bootstrap first; labels print the multiplication sign
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import attachment_sweep as asw
import build_spec as bs
import floatsim_decks as fd
import line_hardware as lh
import mooring_sizing as ms
import mooring_verify as mv

FIG = HERE / "figs"
ARTS = ("buoy", "cluster", "platform")
W2 = ms.W_FLUME / 2
XA = mv.L_ANCHOR
WINDOW = (-0.3, 1.0)                                  # tracking window, surge (F6)
C = {"op": "#1b7f79", "ext": "#c8553d", "wall": "#39424e", "water": "#e6f2f5",
     "art": "#26323f", "win": "#f2c14e", "grey": "#8a949e"}
RECORD: dict[str, dict] = {}                          # what each figure drew, for the spot-check


def xyz(v) -> str:  # type: ignore[no-untyped-def]
    return bs.xyz(list(v))


def geometry(art: str) -> dict:
    """Article geometry from the FloatSim deck."""
    dk = fd.deck(art)
    buoys = [b for b in dk.bodies if b.hydro_body_label or b.hydro_database]
    el = buoys[0].drag_elements
    plate = next(e for e in el if e.type == "plate")
    spar = [e for e in el if e.type == "morison_member"]
    z0 = buoys[0].reference_point[2]
    ref = {b.name: np.asarray(b.reference_point, dtype=float) for b in dk.bodies}
    segs = [(ref[j.body_a] + np.asarray(j.attach_a_body), ref[j.body_b]) for j in dk.joints]
    return {"xy": np.array([b.reference_point[:2] for b in buoys], dtype=float),
            "plate_r": plate.radius, "plate_z": z0 + plate.center[2], "plate_t": plate.thickness,
            "spar_r": spar[0].diameter / 2,
            "spar_bot": z0 + min(min(e.node_a[2], e.node_b[2]) for e in spar),
            "top": fd.aw.ZH, "cog": fd.aw.ZB, "segs": segs}


def _hull(g: dict, dx: float = 0.0) -> np.ndarray:
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    pts = np.concatenate([np.column_stack([x + dx + g["plate_r"] * np.cos(th),
                                           y + g["plate_r"] * np.sin(th)]) for x, y in g["xy"]])
    return pts


def _footprint(ax, g: dict, dx: float = 0.0, color: str = C["art"], lw: float = 1.0,  # type: ignore[no-untyped-def]
               ls: str = "-", fill: bool = True, collar: np.ndarray | None = None) -> None:
    for x, y in g["xy"]:
        ax.add_patch(Circle((x + dx, y), g["plate_r"], fc="white" if fill else "none", ec=color,
                            lw=lw, ls=ls, zorder=4))
        ax.add_patch(Circle((x + dx, y), g["spar_r"], fc=color if fill else "none", ec=color,
                            lw=lw * 0.8, ls=ls, zorder=5))
    for a, b in g["segs"]:
        ax.plot([a[0] + dx, b[0] + dx], [a[1], b[1]], color=color, lw=2.2 * lw, ls=ls,
                solid_capstyle="round", zorder=6)
    if collar is not None:
        for p in collar:
            ax.plot([g["xy"][0, 0] + dx, p[0] + dx], [g["xy"][0, 1], p[1]], color=color,
                    lw=2.0 * lw, ls=ls, zorder=6)


def _flume(ax, xlim: float = 5.6) -> None:  # type: ignore[no-untyped-def]
    ax.add_patch(Rectangle((-xlim, -W2), 2 * xlim, 2 * W2, fc=C["water"], ec="none", zorder=0))
    for s in (-1, 1):
        ax.plot([-xlim, xlim], [s * W2, s * W2], color=C["wall"], lw=3, zorder=2)
        ax.text(0.0, s * (W2 + 0.12), f"flume wall, y = {s * W2:+.2f} m", ha="center",
                va="bottom" if s > 0 else "top", fontsize=7, color=C["wall"])
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-W2 - 0.45, W2 + 0.45)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m), down-flume")
    ax.set_ylabel("y (m)")


def _window(ax, g: dict) -> None:  # type: ignore[no-untyped-def]
    pts = np.concatenate([_hull(g, WINDOW[0]), _hull(g, WINDOW[1])])
    h = ConvexHull(pts)
    ax.add_patch(Polygon(pts[h.vertices], fc=C["win"], alpha=0.28, ec=C["win"], lw=0.8,
                         zorder=1, label=f"tracking window, surge {WINDOW[0]:+.1f} … "
                         f"{WINDOW[1]:+.1f} m"))


def _wave_arrow(ax) -> None:  # type: ignore[no-untyped-def]
    ax.annotate("", xy=(-4.1, 0.0), xytext=(-5.3, 0.0),
                arrowprops={"arrowstyle": "-|>", "color": C["wall"], "lw": 1.6})
    ax.text(-4.7, 0.12, "waves", ha="center", fontsize=8, color=C["wall"])


def plan(art: str, st: dict) -> None:
    g = geometry(art)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    _flume(ax)
    _window(ax, g)
    lines = st["lines"]
    collar = np.array([ln["attachment_m"][:2] for ln in lines]) if art == "buoy" else None
    _footprint(ax, g, collar=collar)
    for ln in lines:
        a, p = ln["anchor_m"], ln["attachment_m"]
        ax.plot([p[0], a[0]], [p[1], a[1]], color=C["op"], lw=1.1, zorder=3)
    anchors = sorted({tuple(ln["anchor_m"]) for ln in lines})
    for a in anchors:
        ax.plot(a[0], a[1], marker="^" if a[1] < 0 else "v", ms=9, color=C["wall"], zorder=7,
                gid="anchor")
        ax.text(a[0] - np.sign(a[0]) * 0.08, a[1] - np.sign(a[1]) * 0.22, xyz(a),
                ha="right" if a[0] > 0 else "left", va="center", fontsize=7.5, zorder=8,
                gid="anchor_label", bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1.0})
    # wall clearance: the footprint's widest point to the wall
    far = g["xy"][np.argmax(np.abs(g["xy"][:, 1]))]
    y_edge = far[1] + np.sign(far[1]) * g["plate_r"]
    clr = W2 - abs(y_edge)
    ax.annotate("", xy=(far[0], np.sign(far[1]) * W2), xytext=(far[0], y_edge),
                arrowprops={"arrowstyle": "<->", "color": C["ext"], "lw": 1.1})
    ax.text(far[0] + 0.08, (y_edge + np.sign(far[1]) * W2) / 2, f"{clr:.2f} m to the wall",
            fontsize=7.5, color=C["ext"], va="center")
    _wave_arrow(ax)
    zatt = lines[0]["attachment_m"][2]
    ax.set_title(f"{bs.NAMES[art]}: plan (to scale). {len(lines)} lines to the wall anchors; "
                 f"attachments and anchors at z = {zatt:+.2f} m (under water)", fontsize=9)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / f"plan_{art}.png", dpi=180)
    RECORD.setdefault(art, {})["plan_anchor_labels"] = [
        t.get_text() for t in ax.texts if t.get_gid() == "anchor_label"]
    RECORD[art]["plan_anchor_xy"] = [tuple(np.round(ln.get_xydata()[0], 3))
                                     for ln in ax.lines if ln.get_gid() == "anchor"]
    plt.close(fig)


def elevation(art: str, st: dict) -> None:
    g = geometry(art)
    des = asw.chosen(art)
    dk, xi, _lf, _lines = lh._setup(art, opts=des["opts"], tag=des["tag"])
    prof = lh.line_states(dk, xi, n_pts=61)
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    x0, x1, zb, zt = -5.6, 5.6, -1.6, 1.4
    ax.add_patch(Rectangle((x0, zb), x1 - x0, -zb, fc=C["water"], ec="none", zorder=0))
    right = ax.get_yaxis_transform()                  # x in axes fraction, z in data
    for zz, lab, col, ls in (
            (0.0, "SWL  z = 0", "#3a7ca5", "-"),
            (g["top"], (f"pin plane +{g['top']:.3f}\n(rev A attachment)" if art != "buoy" else
                        f"spar top +{g['top']:.3f}\n(rev A collar)"), C["grey"], "--"),
            (g["cog"], f"buoy CoG {g['cog']:+.3f}", C["grey"], ":")):
        ax.axhline(zz, color=col, lw=1.0 if ls == "-" else 0.8, ls=ls, zorder=1)
        ax.text(1.008, zz, lab, transform=right, fontsize=7.5, color=col, va="center",
                clip_on=False)
    for x, _y in g["xy"]:
        ax.add_patch(Rectangle((x - g["spar_r"], g["spar_bot"]), 2 * g["spar_r"],
                               g["top"] - g["spar_bot"], fc="white", ec=C["art"], lw=0.9,
                               alpha=0.85, zorder=4))
        ax.plot([x, x], [g["spar_bot"], g["plate_z"]], color=C["art"], lw=0.6, zorder=4)
        ax.add_patch(Rectangle((x - g["plate_r"], g["plate_z"] - g["plate_t"] / 2),
                               2 * g["plate_r"], g["plate_t"], fc=C["art"], ec=C["art"], lw=1.6,
                               zorder=5))
        ax.plot(x, g["cog"], marker="+", color=C["ext"], ms=6, zorder=6)
    for a, b in g["segs"]:
        ax.plot([a[0], b[0]], [g["top"], g["top"]], color=C["art"], lw=2.4, zorder=6)
    for ln in prof:
        p = ln["pts"]
        ax.plot(p[:, 0], p[:, 2], color=C["op"], lw=1.0, zorder=3)
    atts = st["lines"]
    if art == "buoy":                                # the collar, projected on the x-z plane
        xs = [ln["attachment_m"][0] for ln in atts]
        ax.plot([min(xs), max(xs)], [atts[0]["attachment_m"][2]] * 2, color=C["art"], lw=2.4,
                zorder=6)
    for ln in atts:
        a, p = ln["anchor_m"], ln["attachment_m"]
        ax.plot(p[0], p[2], "o", ms=3.5, color=C["op"], zorder=7, gid="attachment")
        ax.plot(a[0], a[2], marker=">" if a[0] < 0 else "<", ms=8, color=C["wall"], zorder=7,
                gid="anchor")
    for s in (-1, 1):
        a = next(ln["anchor_m"] for ln in atts if np.sign(ln["anchor_m"][0]) == s)
        ax.text(a[0] - s * 0.05, a[2] - 0.14,
                f"anchors on the side walls\n(y = ±{abs(a[1]):.2f} m), z = {a[2]:+.2f} m",
                ha="right" if s > 0 else "left", va="top", fontsize=7, gid="anchor_label",
                bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1.0})
    zatt = atts[0]["attachment_m"][2]
    rel = (f"attachment z = {zatt:+.2f} m: {abs(zatt):.2f} m below SWL,\n"
           f"{g['top'] - zatt:.3f} m below the {'pin' if art != 'buoy' else 'spar top'}, "
           f"{zatt - g['cog']:.3f} m above the CoG")
    xl = min(ln["attachment_m"][0] for ln in atts)
    ax.annotate(rel, xy=(xl, zatt), xytext=(-4.9, 0.95), fontsize=7.5, va="center",
                color=C["op"], arrowprops={"arrowstyle": "->", "color": C["op"], "lw": 0.8},
                gid="depth_label")
    # sag inset: the line's drop below its straight chord, mm (the main view is to scale)
    ins = ax.inset_axes([0.70, 0.64, 0.27, 0.27])
    p = prof[0]["pts"]
    s_ = np.linalg.norm(p - p[0], axis=1)
    chord = p[0] + np.outer(s_ / s_[-1], p[-1] - p[0])
    sag_mm = 1000 * (chord[:, 2] - p[:, 2])
    ins.plot(s_, sag_mm, color=C["op"], lw=1.2)
    ins.fill_between(s_, 0, sag_mm, color=C["op"], alpha=0.15)
    ins.set_title(f"sag below the chord at T0: max {sag_mm.max():.0f} mm", fontsize=6.5)
    ins.tick_params(labelsize=6)
    ins.set_xlabel("along the line (m)", fontsize=6)
    ins.set_ylabel("mm", fontsize=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(zb, zt)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m), down-flume")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{bs.NAMES[art]}: elevation looking across the flume (to scale); lines at "
                 f"pretension (FloatSim calm equilibrium, catenary profile)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"elevation_{art}.png", dpi=180)
    RECORD.setdefault(art, {})["elev_attach_z"] = sorted({
        round(float(ln.get_xydata()[0, 1]), 3) for ln in ax.lines if ln.get_gid() == "attachment"})
    RECORD[art]["elev_depth_label"] = next(t.get_text() for t in ax.texts
                                           if t.get_gid() == "depth_label")
    plt.close(fig)


def rationale() -> None:
    sel = bs.sweep_selection()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    for i, art in enumerate(ARTS):
        top, bot = axes[0, i], axes[1, i]
        lim = bs.lim(art)
        top.axhspan(-lim, lim, color=C["op"], alpha=0.12, label=f"±ζ/3 = ±{lim:.2f} %")
        rows = [s for s in sel if s["article"] == art]
        for ks, mk in ((1.0, "o"), (0.5, "s")):
            rr = sorted((s for s in rows if s["k_scale"] == ks), key=lambda s: s["height"])
            z = [s["height"] for s in rr]
            top.plot(z, [s["row"]["tilt_shift_pct"] for s in rr], color=C["art"], lw=1,
                     ls="-" if ks == 1.0 else "--", label=f"k ×{ks:g}")
            for s in rr:
                top.plot(s["height"], s["row"]["tilt_shift_pct"], mk, ms=6,
                         mfc=C["art"] if s["passes"] else "white", mec=C["art"])
                bot.plot(s["height"], s["row"]["T0_line_N"], mk, ms=5,
                         mfc=C["art"] if s["passes"] else "white", mec=C["art"])
        pin = next(s["row"] for s in rows if s["row"]["name"] == "pin" and s["k_scale"] == 1.0)
        top.annotate(f"rev A: pin plane\n{pin['tilt_shift_pct']:+.1f} %",
                     xy=(pin["height"], pin["tilt_shift_pct"]),
                     xytext=(pin["height"] - 0.75, pin["tilt_shift_pct"] + 0.5), fontsize=7.5,
                     arrowprops={"arrowstyle": "->", "lw": 0.8})
        d = bs.design(art)
        for ax_, y in ((top, d["tilt_shift_pct"]), (bot, d["T0_line_N"])):
            ax_.plot(d["height"], y, "*", ms=15, color=C["ext"], zorder=9,
                     label="chosen (rev B / C operational)" if ax_ is top else None)
        cap = sorted((r for r in bs.AS[art] if "article" in r and r.get("t0_is_static_cap")),
                     key=lambda r: r["height"])
        if cap:
            zc = sorted({r["height"] for r in cap})
            bot.plot(zc, [next(r["T0_line_N"] for r in cap if r["height"] == z) for z in zc],
                     color=C["ext"], lw=1.2,
                     label="calm static tilt = 1° (shaded: above 1°, fails)")
            bot.fill_between(zc, [next(r["T0_line_N"] for r in cap if r["height"] == z)
                                  for z in zc], 100, color=C["ext"], alpha=0.08)
        else:
            bot.text(0.5, 0.9, "the collar balances the pretension:\nno static-tilt limit",
                     transform=bot.transAxes, ha="center", va="top", fontsize=8)
        top.set_title(bs.NAMES[art], fontsize=10)
        top.set_ylim(min(-10.0, pin["tilt_shift_pct"] - 1), 1.5)
        bot.set_yscale("log")
        bot.set_ylim(0.3, 30)
        for ax_ in (top, bot):
            for zz, lab in ((0.0, "SWL"), (fd.aw.ZH, "pin"), (fd.aw.ZB, "CoG")):
                ax_.axvline(zz, color=C["grey"], lw=0.6, ls=":")
                if ax_ is bot:
                    ax_.text(zz, 0.33, lab, fontsize=7, color=C["grey"], ha="center")
            ax_.grid(alpha=0.25)
        bot.set_xlabel("attachment height z (m)")
        top.legend(fontsize=7, loc="lower left")
        if cap:
            bot.legend(fontsize=7, loc="upper right")
    axes[0, 0].set_ylabel("tilt-period shift vs unmoored (%)")
    axes[1, 0].set_ylabel("pretension T0 per line/leg (N)")
    fig.suptitle("Design rationale (attachment_sweep.py): each point is the sweep table's row, "
                 "i.e. the most pretension allowed at that height (filled = passes every hard "
                 "criterion)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "rationale.png", dpi=170)
    plt.close(fig)


def offsets(st: dict) -> None:
    ad = json.loads((HERE / "attachment_design.json").read_text())
    ex = json.loads((HERE / "extreme_set.json").read_text())
    bsr = json.loads((HERE / "buoy_stability_check.json").read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 12.5))
    for ax, art in zip(axes, ARTS, strict=True):
        g = geometry(art)
        _flume(ax)
        _window(ax, g)
        _footprint(ax, g, color=C["grey"], lw=0.8)
        for ln in st[art]["lines"]:
            ax.plot(ln["anchor_m"][0], ln["anchor_m"][1], "^", ms=7, color=C["wall"])
        cases = []
        if art == "buoy":
            for r in bsr:
                if r["H"] == 0.5 and r["state"] != "diverges":
                    cases.append((r, C["op"], f"the buoy's one cord set, H 0.5 m T {r['T']} s"))
        else:
            m = ex["choice"][art]["m_criterion4"]
            for r in ad["extremes"]:
                if r["article"] == art and r["H"] == 0.5:
                    cases.append((r, C["ext"], f"operational set, H 0.5 m T {r['T']} s"))
            for r in ex["final"]:
                if r["article"] == art:
                    cases.append((r, C["op"], f"extreme set (k ×{m:g}), H 0.5 m T {r['T']} s"))
        for r, col, lab in cases:
            ls = "-" if r["T"] in (2.35, 3.0) else "--"
            _footprint(ax, g, dx=r["mean_offset_m"], color=col, lw=0.9, ls=ls, fill=False)
            ax.plot([], [], color=col, ls=ls, label=f"{lab}: mean {r['mean_offset_m']:+.2f} m "
                    f"(max {r['surge_max_m']:+.2f} m)")
        _wave_arrow(ax)
        ax.set_title(f"{bs.NAMES[art]}: calm position (grey) and H = 0.5 m mean positions "
                     "(FloatSim, drift sum)", fontsize=9)
        if art == "buoy":
            ax.text(-5.3, W2 - 0.25,
                    "H 0.5 m at T 2.35 / 2.65 s: FloatSim diverges (no prediction)",
                    fontsize=7, color=C["ext"])
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
        ax.set_xlim(-5.6, 5.6)
    fig.tight_layout()
    fig.savefig(FIG / "offsets.png", dpi=160)
    plt.close(fig)


def spotcheck() -> list[str]:
    """One anchor coordinate and the attachment depth per article, read back from the drawn
    figures, against the per-article line tables of MOORING-SPEC.md."""
    md = (HERE / "MOORING-SPEC.md").read_text(encoding="utf-8")
    out = []
    heads = {"buoy": "Single buoy", "cluster": "Cluster", "platform": "4×4 platform"}
    for art, head in heads.items():
        sec = md[md.index(f"### 2{'abc'[ARTS.index(art)]}. {head}"):]
        row = next(ln for ln in sec.splitlines() if ln.startswith("| 1 ("))
        cells = [c.strip() for c in row.strip("|").split("|")]
        anchor_tbl, att_tbl = cells[1], cells[2]
        z_tbl = float(re.findall(r"[-+]\d+\.\d+", att_tbl)[2])
        rec = RECORD[art]
        a_xy = tuple(float(v) for v in re.findall(r"[-+]\d+\.\d+", anchor_tbl)[:2])
        ok_label = anchor_tbl in rec["plan_anchor_labels"]
        ok_xy = any(abs(x - a_xy[0]) < 5e-4 and abs(y - a_xy[1]) < 5e-4
                    for x, y in rec["plan_anchor_xy"])
        ok_z = any(abs(z - z_tbl) < 5e-4 for z in rec["elev_attach_z"])
        ok_lab = f"attachment z = {z_tbl:+.2f} m" in rec["elev_depth_label"]
        verdict = "MATCH" if (ok_label and ok_xy and ok_z and ok_lab) else "MISMATCH"
        out.append(f"{art}: table line 1 anchor {anchor_tbl}, attachment z {z_tbl:+.3f} m | "
                   f"plan label found {ok_label}, plan marker at it {ok_xy}; elevation markers "
                   f"{rec['elev_attach_z']}, depth label '{rec['elev_depth_label'][:34]}…' -> "
                   f"{verdict}")
    return out


def layout() -> None:
    """The three plan views stacked (the 3D viewer's layout panel)."""
    ims = [plt.imread(FIG / f"plan_{a}.png") for a in ARTS]
    w = max(i.shape[1] for i in ims)
    fig = plt.figure(figsize=(w / 180, sum(i.shape[0] for i in ims) / 180), dpi=180)
    y = 1.0
    for im in ims:
        h = im.shape[0] / sum(i.shape[0] for i in ims)
        ax = fig.add_axes((0, y - h, im.shape[1] / w, h))
        ax.imshow(im)
        ax.axis("off")
        y -= h
    fig.savefig(FIG / "layout.png", dpi=110)
    plt.close(fig)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    FIG.mkdir(exist_ok=True)
    st = json.loads((HERE / "spec_statics.json").read_text())
    for art in ARTS:
        plan(art, st[art])
        elevation(art, st[art])
    layout()
    rationale()
    if (HERE / "extreme_set.json").exists() and "final" in json.loads(
            (HERE / "extreme_set.json").read_text()):
        offsets(st)
    if sys.argv[1:] == ["spotcheck"]:
        for line in spotcheck():
            print(line)
    print("figures:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
