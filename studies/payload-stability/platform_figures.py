"""Pictures of the platform's payload physics, drawn to scale from the solved equilibria
(platform_statics.py), for the platform payload deck.

Side views, all from ONE fixed direction (VIEW_DEG = atan(1/4), 14 deg off the grid). From there
the 16 spars of the 4 x 4 grid project evenly, 0.19 m apart, so none hides behind another. The
payload always appears on the right. As in payload_figures.py:
- every buoy hangs vertically from its gimbal pin;
- nearer buoys are drawn darker;
- a top-view inset shows the buoys, the hubs, the deck, the payload and the viewing direction.
Each hub hangs level from its gimbal under the deck: the hub arms run from the hub centre to its
four pins, and a post joins the hub to the deck frame (+0.183 m).
- sketch_platform.png: the unloaded platform, labelled, with the loads of a payload on the deck.
- pseq_centred.png, pseq_offset.png, pseq_raised.png: still sequences of three loadings.
  - centred, CG on the deck: the platform only sinks, until all pins are awash at 212.7 kg;
  - 1.2 m towards a hub (the diagonal), CG on the deck: the deck tilts, until that hub's four
    pins are awash;
  - CG 1.0 m above the deck, 0.2 m off centre: the raised mass tips the deck much further.
- panim_centred.gif, panim_offset.gif, panim_raised.gif: the same loadings animated.
- platform_tilt_paths.json: deck tilt against mass x offset along several branches.

Run: python platform_figures.py
"""

# ruff: noqa: E702  -- compact drawing lines, as payload_figures.py
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cluster_statics as cs
import matplotlib.pyplot as plt
import numpy as np
import platform_statics as ps
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle

HERE = Path(__file__).resolve().parent
SPAR_R, PLATE_R = 0.07965, 0.1437
PIN_TO_SPAR_BOT = 0.717 + 0.967          # pin (spar top) down to the spar bottom
PIN_TO_PLATE = 0.717 + 1.383             # pin down to the heave plate
DECK_HALF = 1.2                          # deck drawn to the outer buoy-centre line
WATER, WATER_EDGE = "#d9eef2", "#5aa9b8"
HULL, HUB, DECK, PAYLOAD, LIMIT = "#1f3b4d", "#3a4852", "#6b7f8c", "#c9772b", "#b2432c"
VIEW_DEG = float(np.degrees(np.arctan(0.25)))   # camera direction for every picture
XL, YL = (-1.8, 1.8), (-2.35, 2.55)
XL_SKETCH = (-2.9, 2.9)                  # the labelled sketch: margins for the labels
CASES = {
    "centred": {"e": 0.0, "dir": 0.0, "h": 0.0},
    "offset": {"e": 1.2, "dir": 45.0, "h": 0.0},
    "raised": {"e": 0.2, "dir": 0.0, "h": 1.0},
}
B = cs.Buoy("36")


def _unloaded() -> np.ndarray:
    q, _ = ps.solve(ps.payload(0.0, 0.0, 0.0, 0.0), B, np.zeros(3))
    return np.asarray(q)


Q0 = _unloaded()                                # deck and hubs settled, no payload
Z_PIN0 = float(cs.S_TOP + Q0[0])


def view_axes(d: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal view direction u (left-right on the page) and depth v (towards the viewer)."""
    u = np.array([np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d)), 0.0])
    return u, np.array([-u[1], u[0], 0.0])


def geometry(q: np.ndarray, ld: ps.Load) -> dict:
    R = cs.rot(q[1], q[2])[0]
    o = np.array([0.0, 0.0, cs.S_TOP + q[0]])               # deck's pin-plane centre
    hubs = o + ps.HUB @ R.T
    pins = np.concatenate([h + ps.PIN for h in hubs])        # hubs hang level
    c = DECK_HALF * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
    corners = o + np.column_stack([c, np.full(4, ld.hd)]) @ R.T
    tops = o + np.column_stack([ps.HUB[:, :2], np.full(4, ld.hd)]) @ R.T
    foot = o + R @ np.array([ld.r[0], ld.r[1], ld.hd])
    return {"hubs": hubs, "pins": pins, "corners": corners, "tops": tops,
            "payload": o + R @ ld.r, "foot": foot}


def plan_inset(ax, ld: ps.Load, u: np.ndarray, v: np.ndarray,  # type: ignore[no-untyped-def]
               box: tuple[float, float, float, float]) -> None:
    """Top view of the (level) platform: deck, hubs, buoys, payload and the viewing direction."""
    ins = ax.inset_axes(box)
    ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_edgecolor("0.7"); sp.set_linewidth(0.8)
    ins.add_patch(Rectangle((-DECK_HALF, -DECK_HALF), 2 * DECK_HALF, 2 * DECK_HALF, fill=False,
                            ec=DECK, lw=1.0, zorder=1))
    for h in ps.HUB:
        for p in ps.PIN:
            ins.plot([h[0], h[0] + p[0]], [h[1], h[1] + p[1]], color=HUB, lw=1.0, zorder=2)
            ins.add_patch(Circle((h[0] + p[0], h[1] + p[1]), PLATE_R, color=HULL, alpha=0.85,
                                 zorder=3))
        ins.plot(h[0], h[1], "s", color=HUB, ms=3, zorder=4)
    ins.add_patch(Rectangle((ld.r[0] - 0.13, ld.r[1] - 0.13), 0.26, 0.26, color=PAYLOAD,
                            zorder=5))
    tail = 2.1 * v; head = 1.6 * v                         # the viewer stands on the +v side
    ins.add_patch(FancyArrowPatch(tuple(tail[:2]), tuple(head[:2]), arrowstyle="-|>",
                                  mutation_scale=9, color=LIMIT, lw=1.5, zorder=6))
    lab = tail[:2] - 0.75 * u[:2]
    ins.text(lab[0], lab[1], "view", fontsize=7, color=LIMIT, ha="center", va="center")
    ins.set_xlim(-2.2, 2.2); ins.set_ylim(-2.2, 2.2); ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("top view", fontsize=7, color="0.4", pad=2)


def draw(ax, q: np.ndarray, ld: ps.Load, note: str = "",  # type: ignore[no-untyped-def]
         labels: bool = False) -> None:
    ax.clear()
    u, v = view_axes(VIEW_DEG)
    g = geometry(q, ld)
    xl = XL_SKETCH if labels else XL
    ax.add_patch(Rectangle((xl[0], -2.6), xl[1] - xl[0], 2.6, color=WATER, zorder=0))
    ax.axhline(0.0, color=WATER_EDGE, lw=1.2, zorder=1)
    if not labels:                                             # unloaded pin height, for scale
        ax.axhline(Z_PIN0, color="0.55", lw=0.8, ls="--", zorder=1)
        ax.text(XL[0] + 0.05, Z_PIN0 + 0.03, "unloaded pins", fontsize=7.5, color="0.45")
    depth = g["pins"] @ v
    span = max(np.ptp(depth), 1e-9)
    for i in np.argsort(depth):                                # far buoys first
        p = g["pins"][i]; h = p @ u
        shade = (depth[i] - depth.min()) / span
        col = to_rgba(HULL, 0.3 + 0.65 * shade)
        z0 = 2 + shade
        ax.add_patch(Rectangle((h - SPAR_R, p[2] - PIN_TO_SPAR_BOT), 2 * SPAR_R, PIN_TO_SPAR_BOT,
                               color=col, zorder=z0))
        ax.add_patch(Rectangle((h - PLATE_R, p[2] - PIN_TO_PLATE - 0.015), 2 * PLATE_R, 0.03,
                               color=col, zorder=z0))
        ax.plot([h, h], [p[2] - PIN_TO_PLATE, p[2] - PIN_TO_SPAR_BOT], color=col, lw=1.0, zorder=z0)
        ax.plot(h, p[2], "o", ms=4, mfc="white", mec=HUB, mew=1.2, zorder=6)
    for c, hc in enumerate(g["hubs"]):
        for p in g["pins"][4 * c:4 * c + 4]:
            ax.plot([hc @ u, p @ u], [hc[2], p[2]], color=HUB, lw=2, solid_capstyle="round",
                    zorder=5)
        t = g["tops"][c]
        ax.plot([hc @ u, t @ u], [hc[2], t[2]], color=DECK, lw=2.2, zorder=5)
    xy = np.column_stack([g["corners"] @ u, g["corners"][:, 2]])
    hull = xy[np.argsort(np.arctan2(xy[:, 1] - xy[:, 1].mean(), xy[:, 0] - xy[:, 0].mean()))]
    ax.add_patch(Polygon(hull, closed=True, fc=to_rgba(DECK, 0.35), ec=DECK, lw=1.6, zorder=6))
    if ld.m > 0:
        pp, ft = g["payload"], g["foot"]
        s = 0.06 + 0.009 * np.sqrt(ld.m)
        if np.linalg.norm(pp - ft) > 1e-6:                      # a raised payload on its stand
            ax.plot([ft @ u, pp @ u], [ft[2], pp[2]], color=PAYLOAD, lw=2.5, zorder=7)
        ax.add_patch(Rectangle((pp @ u - s, pp[2] - 0.7 * s), 2 * s, 1.4 * s, color=PAYLOAD,
                               zorder=8))
        ax.text(pp @ u, pp[2] + 0.7 * s + 0.05, f"{ld.m:.1f} kg", ha="center", fontsize=10,
                color=PAYLOAD, fontweight="bold", zorder=9)
    tilt = ps.tilt_deg(ps.full(q))
    fb = cs.S_TOP + q[0] + (ps.HUB @ cs.rot(q[1], q[2])[0].T)[:, 2].min()   # lowest pin height
    if not labels:
        ax.text(XL[0] + 0.05, YL[1] - 0.15, f"deck tilt {tilt:4.1f}°", fontsize=11,
                fontweight="bold", color=HUB)
        ax.text(XL[0] + 0.05, YL[1] - 0.36, f"lowest pin {1e3 * fb:4.0f} mm above water",
                fontsize=9.5, color=LIMIT if fb <= cs.AWASH_TOL else HUB)
    if note:
        ax.text(XL[0] + 0.05, YL[1] - 0.57, note, fontsize=9.5, color=LIMIT, fontweight="bold")
    if labels:
        annotate(ax, g, u)
    plan_inset(ax, ld, u, v, (0.0, 0.01, 0.2, 0.2) if labels else (0.73, 0.71, 0.26, 0.26))
    ax.set_xlim(*xl); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.axis("off")


def annotate(ax, g: dict, u: np.ndarray) -> None:  # type: ignore[no-untyped-def]
    """Labels and loads on the unloaded sketch."""
    kw = {"fontsize": 9, "color": HUB}
    arrow = {"arrowstyle": "-", "color": HUB, "lw": 0.8}
    right = int(np.argmax(g["hubs"] @ u)); hc = g["hubs"][right]; top = g["tops"][right]
    p0 = g["pins"][int(np.argmax(g["pins"] @ u))]; h0 = p0 @ u
    deck_mid = g["corners"].mean(axis=0)
    ax.annotate("deck frame, 6 kg, +0.90 m\n(the payload sits on it)",
                (deck_mid @ u - 0.9, deck_mid[2]), (-2.85, 1.9), arrowprops=arrow, **kw)
    ax.annotate("4 hubs, 2 kg each, hang level:\ngimbal to the deck at the hub\n"
                "centre, in the pin plane", (hc @ u, hc[2]), (0.75, 1.75), arrowprops=arrow, **kw)
    ax.annotate("", (top @ u, top[2]), (1.2, 1.75), arrowprops=arrow)
    ax.annotate("gimbal pins at the\nspar tops (+0.717 m),\n16 buoys", (h0, p0[2]),
                (1.68, 0.35), arrowprops=arrow, **kw)
    ax.annotate("heave plate\nØ 0.29 m", (h0 + PLATE_R, p0[2] - PIN_TO_PLATE), (1.75, -1.75),
                arrowprops=arrow, **kw)
    ax.text(-2.85, 0.05, "still water level", fontsize=9, color=WATER_EDGE)
    mid = deck_mid
    ax.add_patch(FancyArrowPatch((mid @ u, mid[2] + 0.55), (mid @ u, mid[2] + 0.08),
                                 arrowstyle="-|>", mutation_scale=14, color=PAYLOAD, lw=2,
                                 zorder=9))
    ax.text(mid @ u + 0.06, mid[2] + 0.3, "payload\nweight", fontsize=9, color=PAYLOAD)
    pl = g["pins"][int(np.argmin(g["pins"] @ u))]; hl = pl @ u
    ax.add_patch(FancyArrowPatch((hl - 0.3, pl[2] - 1.05), (hl - 0.3, pl[2] - 0.6),
                                 arrowstyle="-|>", mutation_scale=14, color=WATER_EDGE, lw=2,
                                 zorder=9))
    ax.text(-2.85, pl[2] - 1.0, "extra buoyancy\nas it sinks", fontsize=9, color="#2d7f8f")


def path_states(case: dict) -> tuple[list[dict], callable, float, str]:  # type: ignore[valid-type]
    def load_at(m: float) -> ps.Load:
        return ps.payload(m, case["e"], case["dir"], case["h"])

    br = ps.branch(load_at, B)
    path = br["path"]
    m_aw = cs.awash(path, br)
    lim = br["m_crit_kg"] if m_aw is None else min(m_aw, br["m_crit_kg"])
    note = "pins awash" if m_aw is not None and lim == m_aw else "deck tips over"
    keep = [p for p in path if p["m_kg"] <= lim + 1e-9]
    q_lim, _ = ps.solve(load_at(lim), B, np.asarray(keep[-1]["q"]))
    if q_lim is not None:                                      # the exact limit state
        keep.append(ps.state(q_lim, load_at(lim), B))
    return keep, load_at, lim, note


def nearest(path: list[dict], m: float) -> dict:
    return min(path, key=lambda p: abs(p["m_kg"] - m))


def sequence(name: str, masses: list[float]) -> None:
    path, load_at, lim, note = path_states(CASES[name])
    masses = [*masses, lim]
    fig, axs = plt.subplots(1, len(masses), figsize=(3.3 * len(masses), 4.9))
    for ax, m in zip(axs, masses, strict=True):
        st = nearest(path, m)
        draw(ax, np.array(st["q"]), load_at(st["m_kg"]), note=note if m == lim else "")
    fig.tight_layout(); fig.savefig(HERE / f"pseq_{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def animate(name: str) -> None:
    path, load_at, lim, note = path_states(CASES[name])
    masses = [*np.linspace(0.0, lim, 40), *[lim] * 12]
    fig, ax = plt.subplots(figsize=(5.6, 6.0))

    def frame(k: int) -> None:
        st = nearest(path, masses[k])
        draw(ax, np.array(st["q"]), load_at(st["m_kg"]), note=note if k >= 40 else "")

    anim = FuncAnimation(fig, frame, frames=len(masses))
    anim.save(HERE / f"panim_{name}.gif", writer=PillowWriter(fps=8), dpi=95)
    plt.close(fig)


def tilt_paths() -> None:
    out = []
    for h in (0.0, 1.0):
        for r in (0.2, 0.6, 1.2):
            def load_at(m: float, r: float = r, h: float = h) -> ps.Load:
                return ps.payload(m, r, 45.0, h)
            br = ps.branch(load_at, B)
            m_aw = cs.awash(br["path"], br)
            lim = br["m_crit_kg"] if m_aw is None else min(m_aw, br["m_crit_kg"])
            path = [p for p in br["path"] if p["m_kg"] <= lim + 1e-9]
            out.append({"h_pay_m": h, "dir_deg": 45.0, "offset_m": r, "m_limit_kg": lim,
                        "me_kgm": [p["m_kg"] * r for p in path],
                        "tilt_deg": [p["tilt_deg"] for p in path]})
    c33 = cs.TABLE["rows"]["36"][0]["C33"]
    k_tilt = 4 * c33 * float(np.sum(ps.HUB[:, 0] ** 2))      # clusters' differential sinkage
    (HERE / "platform_tilt_paths.json").write_text(
        json.dumps({"k_tilt_Nm_per_rad": k_tilt, "g": cs.G_ACC, "paths": out}, indent=1),
        encoding="utf-8")


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 8.0))
    draw(ax, Q0, ps.payload(0.0, 0.0, 0.0, 0.0), labels=True)
    fig.tight_layout(); fig.savefig(HERE / "sketch_platform.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    sequence("centred", [0.0, 70.0, 140.0])
    sequence("offset", [0.0, 15.0, 40.0])
    sequence("raised", [0.0, 40.0, 80.0])
    for name in CASES:
        animate(name)
    tilt_paths()
    print("wrote sketch_platform.png, pseq_*.png, panim_*.gif, platform_tilt_paths.json")


if __name__ == "__main__":
    main()
