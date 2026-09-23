"""Recommended flume station-keeping mooring for the three test articles: line design table and
plan/side-view layout schematic in the OSU LWF (3.66 m wide, 2.7 m deep).

Design (from mooring_sizing.py + the coupled check in mooring_verify.py): an X-spread of 4
horizontal lines at the still-water line (SWL) to wall anchors 5 m up- and downstream, each line
a soft linear spring in series with low-stretch rope; the lines attach at the SWL of the spar(s):
  * 1 buoy     -- one collar on the spar at the SWL (all 4 lines);
  * 1 cluster  -- the upstream and downstream buoy spars (2 lines each);
  * 4x4 platform (45 deg) -- the upstream and downstream rows of 4 spars (each line ends in a
    2-leg bridle to 2 spars).
Per-line stiffness, pretension, pre-stretch and the stroke the spring must provide are tabulated
for T_surge = 15 s (baseline) and 20 s (softer option), at the conservative H = 0.5 m drift.

Writes mooring_design_table.csv + mooring_layout.png.  Run: python mooring_layout.py
"""
# ruff: noqa: E702, RUF001  -- compact numeric lines; display typography in labels
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mooring_sizing as ms
import numpy as np
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent
L_ANCHOR, A_WAVE = 5.0, 0.25
T_DESIGN = (15.0, 20.0)
WL, PIN, DECK = 0.0, 0.717, 0.90                 # z of SWL, pins, deck frame (m)
SPAR_BOT, PLATE_Z, PLATE_T = -0.967, -1.383, 0.03
S = 1.25 / 1.5
CLUSTER = np.array([(0.5 * S * np.cos(a), 0.5 * S * np.sin(a))
                    for a in np.deg2rad([180.0, 90.0, 0.0, 270.0])])
PLATFORM = np.array([(x, y) for x in (-0.884, -0.295, 0.295, 0.884)
                     for y in (-0.884, -0.295, 0.295, 0.884)])
TEAL, RUST, INK, SEA = "#0c8b96", "#b2432c", "#23313a", "#dcecef"


def design_table() -> list[dict]:
    f_spar = max(sum(ms.drift_per_spar(0.5, T)[:2]) for T in ms.T_WAVE)
    rows = []
    for name, a in ms.ARTICLES.items():
        for ts in T_DESIGN:
            Kx = (a["M"] + a["A"]) * (2 * np.pi / ts) ** 2
            F = a["n"] * f_spar; delta = F / Kx
            xs = ms.xspread(Kx, delta, A_WAVE, L_ANCHOR)
            ca = np.cos(np.radians(xs["alpha_deg"]))
            pre = xs["T0"] / xs["kl"]; stroke = pre + (delta + A_WAVE) * ca
            rows.append(dict(article=name, T_surge_s=ts, Kx_Npm=round(Kx, 2),
                             drift_H05_N=round(float(F), 1), offset_H05_m=round(delta, 2),
                             line_len_m=round(xs["ell"], 2),
                             line_angle_deg=round(xs["alpha_deg"], 1),
                             k_line_Npm=round(xs["kl"], 2), pretension_N=round(xs["T0"], 1),
                             pre_stretch_m=round(pre, 2), max_stretch_m=round(stroke, 2),
                             max_line_tension_N=round(xs["kl"] * stroke, 1),
                             Ky_Npm=round(xs["Ky"], 2), Kz_geo_Npm=round(xs["Kz"], 2)))
    return rows


def _flume(ax, title):
    W = ms.W_FLUME
    ax.add_patch(Rectangle((-6.2, -W / 2), 12.4, W, color=SEA, zorder=0))
    for y in (-W / 2, W / 2):
        ax.plot([-6.2, 6.2], [y, y], color=INK, lw=3, solid_capstyle="butt")
    ax.annotate("", xy=(-3.3, 0.0), xytext=(-5.6, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="0.45", lw=2.0))
    ax.text(-4.45, 0.12, "waves", ha="center", va="bottom", fontsize=8.5, color="0.35")
    ax.set_xlim(-6.2, 6.2); ax.set_ylim(-W / 2 - 0.35, W / 2 + 0.55); ax.set_aspect("equal")
    ax.set_xticks([-5, -2.5, 0, 2.5, 5]); ax.set_yticks([-1.83, 0, 1.83])
    ax.tick_params(labelsize=7.5); ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")


def _buoys(ax, xy):
    for x, y in xy:
        ax.add_patch(Circle((x, y), ms.PLATE_R, fc="white", ec=INK, lw=0.9, zorder=3))
        ax.add_patch(Circle((x, y), ms.SPAR_D / 2, fc=INK, ec=INK, zorder=4))


def _line(ax, p_att, p_anchor, bridle=None):
    """Line from wall anchor to attachment; optional 2-leg bridle from a ring to 2 spars."""
    end = p_att if bridle is None else bridle[0]
    ax.plot([p_anchor[0], end[0]], [p_anchor[1], end[1]], color=TEAL, lw=1.4, zorder=2)
    # spring symbol near the anchor
    t = np.linspace(0.08, 0.30, 40); d = np.subtract(end, p_anchor); nrm = np.array([-d[1], d[0]])
    nrm = nrm / np.hypot(*nrm)
    zz = np.array(p_anchor)[None, :] + t[:, None] * d[None, :] \
        + 0.09 * np.sign(np.sin(t * 180))[:, None] * nrm[None, :]
    ax.plot(zz[:, 0], zz[:, 1], color=RUST, lw=1.3, zorder=2)
    ax.plot(*p_anchor, marker="s", ms=6, color=INK, zorder=5)
    if bridle is not None:
        for q in bridle[1]:
            ax.plot([end[0], q[0]], [end[1], q[1]], color=TEAL, lw=1.0, zorder=2)
        ax.plot(*end, marker="o", ms=3.5, color=TEAL, zorder=5)


def plan_views(axs):
    W2 = ms.W_FLUME / 2
    anchors = [(-L_ANCHOR, W2), (-L_ANCHOR, -W2), (L_ANCHOR, W2), (L_ANCHOR, -W2)]
    # 1 buoy
    _flume(axs[0], "1 buoy — one collar at the SWL")
    _buoys(axs[0], [(0.0, 0.0)])
    for a in anchors:
        _line(axs[0], (0.0, 0.0), a)
    # cluster: hub + arms, lines to the upstream (-x) and downstream (+x) spars
    _flume(axs[1], "1 cluster — upstream + downstream spars at the SWL")
    for x, y in CLUSTER:
        axs[1].plot([0, x], [0, y], color="0.55", lw=1.2, zorder=1)
    axs[1].add_patch(Circle((0, 0), 0.05, color="0.45", zorder=3))
    _buoys(axs[1], CLUSTER)
    up, dn = tuple(CLUSTER[0]), tuple(CLUSTER[2])
    for a in anchors:
        _line(axs[1], up if a[0] < 0 else dn, a)
    # platform: deck frame + rows; each line -> ring -> 2-leg bridle to 2 spars of its half-row
    _flume(axs[2], "4×4 platform (45°) — upstream + downstream rows of 4 spars at the SWL")
    hub = [(-0.589, -0.589), (-0.589, 0.589), (0.589, 0.589), (0.589, -0.589), (-0.589, -0.589)]
    axs[2].plot(*zip(*hub, strict=True), color="0.55", lw=1.4, zorder=1)
    _buoys(axs[2], PLATFORM)
    for a in anchors:
        xr = -0.884 if a[0] < 0 else 0.884
        legs = [(xr, 0.295 * np.sign(a[1])), (xr, 0.884 * np.sign(a[1]))]
        ring = (xr + np.sign(a[0]) * 0.75, 0.59 * np.sign(a[1]))
        _line(axs[2], None, a, bridle=(ring, legs))
    y_edge = 0.884 + ms.PLATE_R
    axs[2].annotate("", xy=(0.0, W2), xytext=(0.0, y_edge),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9, shrinkA=0, shrinkB=0))
    axs[2].text(0.08, 0.5 * (W2 + y_edge), f"{W2 - y_edge:.2f} m", va="center", fontsize=8,
                color="0.3")
    for ax in axs:
        ax.text(5.0, W2 + 0.12, "wall anchor at SWL", ha="center", fontsize=7.5, color="0.3")


def side_view(ax, rows):
    ax.add_patch(Rectangle((-6.2, -ms.H_FLUME), 12.4, ms.H_FLUME, color=SEA, zorder=0))
    ax.plot([-6.2, 6.2], [0, 0], color="#4f8fa0", lw=1.2)
    ax.plot([-6.2, 6.2], [-ms.H_FLUME] * 2, color=INK, lw=3)
    for x in (-0.884, -0.295, 0.295, 0.884):
        ax.add_patch(Rectangle((x - ms.SPAR_D / 2, SPAR_BOT), ms.SPAR_D, PIN - SPAR_BOT,
                               fc="white", ec=INK, lw=0.9, zorder=3))
        ax.add_patch(Rectangle((x - ms.PLATE_R, PLATE_Z - PLATE_T / 2), 2 * ms.PLATE_R, PLATE_T,
                               fc=INK, zorder=3))
        ax.plot([x, x], [SPAR_BOT, PLATE_Z], color=INK, lw=1.0, zorder=3)
        ax.add_patch(Circle((x, PIN), 0.045, fc="white", ec=RUST, lw=1.4, zorder=5))
    ax.add_patch(Rectangle((-0.95, PIN + 0.04), 1.9, DECK - PIN, fc="0.8", ec="0.45", zorder=2))
    for sx, xa in ((-0.884, -L_ANCHOR), (0.884, L_ANCHOR)):
        _line(ax, (sx, 0.0), (xa, 0.0))
    ax.annotate("", xy=(1.95, -0.14), xytext=(1.12, -0.14),
                arrowprops=dict(arrowstyle="-|>", color=RUST, lw=2))
    ax.text(1.12, -0.24, "mean drift acts\nat the SWL", ha="left", va="top", fontsize=8,
            color=RUST)
    ax.text(0.0, DECK + 0.1, "deck frame", ha="center", fontsize=8, color="0.3")
    ax.text(1.05, PIN, "pins +0.72 m", va="center", fontsize=8, color=RUST)
    ax.text(-6.0, -ms.H_FLUME + 0.1, "flume floor, 2.7 m", fontsize=8, color="0.3")
    ax.text(-5.0, 0.12, "lines horizontal at the SWL", fontsize=8, color=TEAL)
    r15 = {r["article"]: r for r in rows if r["T_surge_s"] == 15.0}
    txt = "\n".join(f"{a.split(' (')[0]}: k = {r['k_line_Npm']:.1f} N/m, T₀ = "
                    f"{r['pretension_N']:.0f} N" for a, r in r15.items())
    ax.text(2.1, -1.2, f"per line, T_surge = 15 s\n{txt}", ha="left", va="top", fontsize=8,
            color=INK, linespacing=1.4)
    ax.set_xlim(-6.2, 6.2); ax.set_ylim(-ms.H_FLUME - 0.1, 1.25); ax.set_aspect("equal")
    ax.tick_params(labelsize=7.5)
    ax.set_title("4×4 platform, side view — lines at the SWL, not at the pins/deck",
                 fontsize=10.5, fontweight="bold", loc="left")


def main() -> None:
    rows = design_table()
    with (HERE / "mooring_design_table.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    for r in rows:
        print(r)
    fig, ax = plt.subplots(2, 2, figsize=(15, 8.2), gridspec_kw=dict(hspace=0.28, wspace=0.1))
    plan_views([ax[0, 0], ax[0, 1], ax[1, 0]])
    side_view(ax[1, 1], rows)
    fig.suptitle("Flume station-keeping mooring — X-spread of 4 soft horizontal lines to wall "
                 "anchors ±5 m (to scale, m)", fontsize=12.5, fontweight="bold", y=0.99)
    fig.savefig(HERE / "mooring_layout.png", dpi=130, bbox_inches="tight")
    print("wrote mooring_design_table.csv, mooring_layout.png")


if __name__ == "__main__":
    main()
