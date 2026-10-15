"""Pictures of the cluster's payload physics, drawn to scale from the solved equilibria
(cluster_statics.py), for the payload-stability deck.

Side views, all from ONE fixed direction, 22.5 deg off the buoy diagonals (VIEW_DEG): seen
square-on or along a diagonal, buoys hide behind each other, but from 22.5 deg the four spars
spread evenly across the picture. The payload always appears on the right. Every buoy hangs
vertically from its gimbal pin; nearer buoys are drawn darker; the hub arms run from the hub centre
to the four pins. A top-view inset shows the four buoys, the payload position and the viewing
direction.
- sketch_cluster.png: the unloaded cluster, labelled, with the loads of a payload on the hub.
- seq_centred.png, seq_offset.png, seq_rollover.png: still sequences of three loadings.
  - centred: the mass only sinks the cluster, until the pins are awash at 54.7 kg;
  - 0.2 m towards a buoy: the seesaw, until the first pin is awash;
  - 0.4 m between buoys: the hub rolls onto its side.
- anim_centred.gif, anim_offset.gif, anim_rollover.gif: the same loadings animated (mass rising).
- tilt_paths.json: hub tilt against mass x offset along several branches (for the deck's chart).

Run: python payload_figures.py
"""

# ruff: noqa: E702  -- compact drawing lines; degree signs and dashes in labels
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cluster_statics as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
SPAR_R, PLATE_R = 0.07965, 0.1437
PIN_TO_SPAR_BOT = 0.717 + 0.967          # pin (spar top) down to the spar bottom
PIN_TO_PLATE = 0.717 + 1.383             # pin down to the heave plate
PIN_TO_COG = cs.L_PIN
WATER, WATER_EDGE = "#d9eef2", "#5aa9b8"
HULL, HUB, PAYLOAD, LIMIT = "#1f3b4d", "#3a4852", "#c9772b", "#b2432c"
VIEW_DEG = 22.5                          # camera direction for every picture (see the docstring)
CASES = {
    "centred": {"e": 0.0, "dir": 0.0, "title": "Centred mass: the cluster only sinks"},
    "offset": {"e": 0.2, "dir": 45.0, "title": "0.2 m towards a buoy: the seesaw"},
    "rollover": {"e": 0.4, "dir": 0.0, "title": "0.4 m between buoys: the hub rolls over"},
}


def _unloaded_pin_z() -> float:
    q0, _ = cs.solve(0.0, np.zeros(3), cs.Buoy("36"), np.zeros(3))
    return float(cs.S_TOP + q0[0])


Z_PIN0 = _unloaded_pin_z()                  # pins with only the hub's weight on them


def offset_vec(e: float, d: float) -> np.ndarray:
    return np.array([e * np.cos(np.deg2rad(d)), e * np.sin(np.deg2rad(d)), 0.0])


def view_axes(d: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal view direction u (left-right on the page) and depth v (towards the viewer)."""
    u = np.array([np.cos(np.deg2rad(d)), np.sin(np.deg2rad(d)), 0.0])
    return u, np.array([-u[1], u[0], 0.0])


def geometry(q: np.ndarray, e: np.ndarray) -> dict:
    R = cs.rot(q[1], q[2])[0]
    hub = np.array([0.0, 0.0, cs.S_TOP + q[0]])
    return {"hub": hub, "pins": hub + cs.Q_PIN @ R.T, "payload": hub + R @ e}


def plan_inset(ax, e: np.ndarray, u: np.ndarray, v: np.ndarray,  # type: ignore[no-untyped-def]
               box: tuple[float, float, float, float]) -> None:
    """Top view of the (level) cluster: buoys, hub arms, payload and the viewing direction."""
    ins = ax.inset_axes(box)
    ins.set_facecolor("white")
    for sp in ins.spines.values():
        sp.set_edgecolor("0.7"); sp.set_linewidth(0.8)
    for x, y, _ in cs.Q_PIN:
        ins.plot([0, x], [0, y], color=HUB, lw=1.6, zorder=2)
        ins.add_patch(Circle((x, y), PLATE_R, color=HULL, alpha=0.85, zorder=3))
    ins.add_patch(Rectangle((e[0] - 0.06, e[1] - 0.06), 0.12, 0.12, color=PAYLOAD, zorder=4))
    tail = 0.95 * v; head = 0.6 * v                       # the viewer stands on the +v side
    ins.add_patch(FancyArrowPatch(tuple(tail[:2]), tuple(head[:2]), arrowstyle="-|>",
                                  mutation_scale=10, color=LIMIT, lw=1.6, zorder=5))
    lab = tail[:2] + 0.32 * u[:2]
    ins.text(lab[0], lab[1], "view", fontsize=7, color=LIMIT, ha="center", va="center")
    ins.set_xlim(-1.15, 1.15); ins.set_ylim(-1.15, 1.15); ins.set_aspect("equal")
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("top view", fontsize=7, color="0.4", pad=2)


def draw(ax, q: np.ndarray, m: float, e: np.ndarray, note: str = "",  # type: ignore[no-untyped-def]
         labels: bool = False) -> None:
    ax.clear()
    u, v = view_axes(VIEW_DEG)
    g = geometry(q, e)
    ax.add_patch(Rectangle((-1.2, -2.6), 2.4, 2.6, color=WATER, zorder=0))
    ax.axhline(0.0, color=WATER_EDGE, lw=1.2, zorder=1)
    if not labels:                                             # unloaded pin height, for scale
        ax.axhline(Z_PIN0, color="0.55", lw=0.8, ls="--", zorder=1)
        ax.text(-1.15, Z_PIN0 + 0.03, "unloaded", fontsize=7.5, color="0.45")
    order = np.argsort(g["pins"] @ v)                       # far buoys first
    depth = g["pins"] @ v
    span = max(np.ptp(depth), 1e-9)
    for i in order:
        p = g["pins"][i]; h = p @ u
        shade = 0.35 + 0.65 * (depth[i] - depth.min()) / span if np.ptp(depth) > 1e-6 else 1.0
        col = to_rgba(HULL, 0.35 + 0.6 * shade)
        ax.add_patch(Rectangle((h - SPAR_R, p[2] - PIN_TO_SPAR_BOT), 2 * SPAR_R, PIN_TO_SPAR_BOT,
                               color=col, zorder=2 + shade))
        ax.add_patch(Rectangle((h - PLATE_R, p[2] - PIN_TO_PLATE - 0.015), 2 * PLATE_R, 0.03,
                               color=col, zorder=2 + shade))
        ax.plot([h, h], [p[2] - PIN_TO_PLATE, p[2] - PIN_TO_SPAR_BOT], color=col, lw=1.2)
        ax.plot(h, p[2], "o", ms=6, mfc="white", mec=HUB, mew=1.6, zorder=6)
    hc = g["hub"]
    for p in g["pins"]:
        ax.plot([hc @ u, p @ u], [hc[2], p[2]], color=HUB, lw=3, solid_capstyle="round", zorder=5)
    if m > 0:
        pp = g["payload"]
        s = 0.05 + 0.012 * np.sqrt(m)
        ax.add_patch(Rectangle((pp @ u - s, pp[2]), 2 * s, 1.4 * s, color=PAYLOAD, zorder=7))
        ax.text(pp @ u, pp[2] + 1.4 * s + 0.04, f"{m:.1f} kg", ha="center", fontsize=10,
                color=PAYLOAD, fontweight="bold", zorder=8)
    tilt = cs.tilt_deg(q)
    fb = cs.S_TOP - cs.sinkage(q).max()
    awash = fb <= cs.AWASH_TOL
    if not labels:
        ax.text(-1.12, 1.33, f"hub tilt {tilt:4.1f}°", fontsize=11, fontweight="bold", color=HUB)
        ax.text(-1.12, 1.17, f"lowest pin {1e3 * fb:4.0f} mm above water",
                fontsize=9.5, color=LIMIT if awash else HUB)
    if note:
        ax.text(-1.12, 1.01, note, fontsize=9.5, color=LIMIT, fontweight="bold")
    if labels:
        annotate(ax, g, u)
    plan_inset(ax, e, u, v, (0.01, 0.01, 0.27, 0.27) if labels else (0.72, 0.6, 0.27, 0.27))
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-2.45, 1.45); ax.set_aspect("equal"); ax.axis("off")


def annotate(ax, g: dict, u: np.ndarray) -> None:  # type: ignore[no-untyped-def]
    """Labels and loads on the unloaded sketch."""
    kw = {"fontsize": 9, "color": HUB}
    hc = g["hub"]; p0 = g["pins"][np.argmax(g["pins"] @ u)]; h0 = p0 @ u
    ax.annotate("hub (2 kg) + payload,\nin the pin plane", (hc @ u, hc[2]), (-1.12, 1.22),
                arrowprops={"arrowstyle": "-", "color": HUB, "lw": 0.8}, **kw)
    ax.annotate("gimbal pin at the spar top\n(+0.717 m): roll and pitch free",
                (h0, p0[2]), (0.32, 1.22), arrowprops={"arrowstyle": "-", "color": HUB, "lw": 0.8},
                **kw)
    ax.annotate("spar Ø 0.16 m", (h0 + SPAR_R, p0[2] - 1.2), (0.55, -0.9),
                arrowprops={"arrowstyle": "-", "color": HUB, "lw": 0.8}, **kw)
    ax.annotate("heave plate Ø 0.29 m", (h0 + PLATE_R, p0[2] - PIN_TO_PLATE), (0.5, -2.3),
                arrowprops={"arrowstyle": "-", "color": HUB, "lw": 0.8}, **kw)
    ax.plot(h0, p0[2] - PIN_TO_COG, "+", color=LIMIT, ms=10, mew=2, zorder=8)
    ax.annotate("buoy CoG, 1.62 m below the pin", (h0, p0[2] - PIN_TO_COG), (0.5, -1.2),
                arrowprops={"arrowstyle": "-", "color": HUB, "lw": 0.8}, **kw)
    ax.text(-1.12, 0.05, "still water level", fontsize=9, color=WATER_EDGE)
    ax.add_patch(FancyArrowPatch((hc @ u, hc[2] + 0.42), (hc @ u, hc[2] + 0.12),
                                 arrowstyle="-|>", mutation_scale=14, color=PAYLOAD, lw=2,
                                 zorder=9))
    ax.text(hc @ u - 0.05, hc[2] + 0.22, "payload\nweight", fontsize=9, color=PAYLOAD,
            ha="right")
    pl = g["pins"][np.argmin(g["pins"] @ u)]; hl = pl @ u
    ax.add_patch(FancyArrowPatch((hl, pl[2] - 0.95), (hl, pl[2] - 0.6), arrowstyle="-|>",
                                 mutation_scale=14, color=WATER_EDGE, lw=2, zorder=9))
    ax.text(-1.15, pl[2] - 1.05, "extra buoyancy\nas it sinks", fontsize=9, color="#2d7f8f")


def path_states(case: dict) -> tuple[list[dict], np.ndarray, float, str]:
    b = cs.Buoy("36")
    e = offset_vec(case["e"], case["dir"])
    br = cs.branch(e, b)
    path = br["path"]
    m_aw = cs.awash(path, br)
    lim = br["m_crit_kg"] if m_aw is None else min(m_aw, br["m_crit_kg"])
    note = "pin awash" if m_aw is not None and lim == m_aw else "hub rolls onto its side"
    q0, _ = cs.solve(0.0, e, b, np.zeros(3))
    path = [{**path[0], "q": list(q0)}, *path[1:]]          # m = 0: the hub weight settled
    keep = [p for p in path if p["m_kg"] <= lim + 1e-9]
    q_lim, _ = cs.solve(lim, e, b, np.asarray(keep[-1]["q"]))
    if q_lim is not None:                                      # the exact limit state
        keep.append(cs.state(q_lim, lim, e, b))
    return keep, e, lim, note


def nearest(path: list[dict], m: float) -> dict:
    return min(path, key=lambda p: abs(p["m_kg"] - m))


def sequence(name: str, masses: list[float]) -> None:
    case = CASES[name]
    path, e, lim, note = path_states(case)
    masses = [*masses, lim]
    fig, axs = plt.subplots(1, len(masses), figsize=(3.1 * len(masses), 4.6))
    for ax, m in zip(axs, masses, strict=True):
        st = nearest(path, m)
        draw(ax, np.array(st["q"]), st["m_kg"], e, note=note if m == lim else "")
    fig.tight_layout(); fig.savefig(HERE / f"seq_{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def animate(name: str) -> None:
    case = CASES[name]
    path, e, lim, note = path_states(case)
    masses = [*np.linspace(0.0, lim, 40), *[lim] * 12]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))

    def frame(k: int) -> None:
        st = nearest(path, masses[k])
        draw(ax, np.array(st["q"]), st["m_kg"], e, note=note if k >= 40 else "")

    anim = FuncAnimation(fig, frame, frames=len(masses))
    anim.save(HERE / f"anim_{name}.gif", writer=PillowWriter(fps=8), dpi=95)
    plt.close(fig)


def tilt_paths() -> None:
    b = cs.Buoy("36"); out = []
    for d in (0.0, 45.0):
        for r in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
            br = cs.branch(offset_vec(r, d), b)
            m_aw = cs.awash(br["path"], br)             # each branch ends at its own limit
            lim = br["m_crit_kg"] if m_aw is None else min(m_aw, br["m_crit_kg"])
            path = [p for p in br["path"] if p["m_kg"] <= lim + 1e-9]
            out.append({"dir_deg": d, "offset_m": r, "m_limit_kg": lim,
                        "me_kgm": [p["m_kg"] * r for p in path],
                        "tilt_deg": [p["tilt_deg"] for p in path]})
    c33 = cs.TABLE["rows"]["36"][-1]["C33"]
    k_tilt = c33 * float(np.sum(cs.Q_PIN[:, 0] ** 2))         # differential sinkage, N m/rad
    (HERE / "tilt_paths.json").write_text(json.dumps({"k_tilt_Nm_per_rad": k_tilt, "g": cs.G_ACC,
                                                      "paths": out}, indent=1), encoding="utf-8")


def main() -> None:
    b = cs.Buoy("36")
    q0, _ = cs.solve(0.0, np.zeros(3), b, np.zeros(3))
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    draw(ax, np.asarray(q0), 0.0, np.zeros(3), labels=True)
    fig.tight_layout(); fig.savefig(HERE / "sketch_cluster.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    sequence("centred", [0.0, 20.0, 40.0])
    sequence("offset", [0.0, 3.0, 12.0])
    sequence("rollover", [0.0, 3.0, 8.0])
    for name in CASES:
        animate(name)
    tilt_paths()
    print("wrote sketch_cluster.png, seq_*.png, anim_*.gif, tilt_paths.json")


if __name__ == "__main__":
    main()
