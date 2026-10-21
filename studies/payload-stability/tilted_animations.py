"""Animations of the cluster and the platform moving in regular waves, for the tilt-in-waves deck.

Two cases, each with the payload (12.5 kg on the cluster's hub, 50 kg on the platform's deck):
centred (level), and shifted to a 10 deg static tilt. The waves are H = 0.1 m along the shift, at
two key periods:
- 2.65 s: the heave and hub / deck-pitch resonance, where the tilt turns heave into pitch;
- 4.2 s: the buoys' rocking resonance on their gimbals.
The motion is FloatSim's, with its weight preload (tilted_dynamics.py history ... --preload).

Every body is drawn where FloatSim puts it, at true scale (no exaggeration), over the last full
wave cycle, in real time:
- each buoy at its own position and rotation (the rocking buoys lean);
- the hub arms out to the moving pins;
- the deck and payload, tilting;
- the wave surface passing.
The view is the static pictures' fixed view: 22.5 deg off the diagonals for the cluster, 14 deg
off the grid for the platform, so every buoy shows.

Writes wanim_<article>_<case>_T<T>.gif from tilted_anim/<article>_<case>_T<T>.npz.
Run: python tilted_animations.py
"""

# ruff: noqa: E702  -- compact drawing lines
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SPAR_R, PLATE_R = 0.07965, 0.1437
SPAR_BOT, PLATE_Z, PIN_Z = -0.967 + 0.907, -1.383 + 0.907, 1.624   # buoy body frame (CoG origin)
DECK_HALF, H_DECK, S_TOP = 1.2, 0.183, 0.717
WATER, WATER_EDGE = "#d9eef2", "#5aa9b8"
HULL, HUB, DECK, PAYLOAD, LIMIT = "#1f3b4d", "#3a4852", "#6b7f8c", "#c9772b", "#b2432c"
VIEW = {"cluster": 22.5, "platform": float(np.degrees(np.arctan(0.25)))}
LIM = {"cluster": ((-1.5, 1.5), (-2.55, 1.45)), "platform": ((-2.1, 2.1), (-2.45, 1.95))}
CASES = ("centred", "tilt10")
PERIODS = (2.65, 4.2)
N_FRAMES = 36
LABEL = {("cluster", "centred"): "Cluster, 12.5 kg centred", ("cluster", "tilt10"):
         "Cluster, 12.5 kg shifted to 10°", ("platform", "centred"): "Platform, 50 kg centred",
         ("platform", "tilt10"): "Platform, 50 kg shifted to 10°"}
ABOUT = {"centred": "about level", "tilt10": "about its 10° tilt"}


def rot0(q: list[float]) -> np.ndarray:
    """Static tilt from the statics coordinates: R = Ry(theta) Rx(phi)."""
    return Rotation.from_euler("YX", [q[2], q[1]]).as_matrix()


def layout(article: str, case: dict) -> dict:
    """Body indices and the points that move with the top body (hub / deck), at equilibrium."""
    if article == "cluster":
        buoy = [0, 1, 2, 3]; hubs = [4]; owner = [4] * 4; top = 4
        hub_centre = [np.asarray(case["hubs"][0]["centre"])]
        deck, posts = None, []
    else:
        buoy = [5 * (k // 4) + k % 4 for k in range(16)]; hubs = [4, 9, 14, 19]
        owner = [5 * (k // 4) + 4 for k in range(16)]; top = 20
        hub_centre = [np.asarray(h["centre"]) for h in case["hubs"]]
        R0 = rot0(case["q"]); o = np.array([0.0, 0.0, S_TOP + case["q"][0]])
        c = DECK_HALF * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
        deck = o + np.column_stack([c, np.full(4, H_DECK)]) @ R0.T
        posts = [o + R0 @ np.array([h[0] - o[0], h[1] - o[1], H_DECK]) for h in hub_centre]
    return {"buoy": buoy, "hubs": hubs, "owner": owner, "top": top, "hub_centre": hub_centre,
            "deck": deck, "posts": posts, "payload": np.asarray(case["payload"]),
            "m": case["m_kg"]}


def pose(refs: np.ndarray, x: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference position and rotation matrix of body b at one instant."""
    return refs[b] + x[6 * b:6 * b + 3], Rotation.from_rotvec(x[6 * b + 3:6 * b + 6]).as_matrix()


def carried(refs: np.ndarray, x: np.ndarray, b: int, p_eq: np.ndarray) -> np.ndarray:
    """Where a point fixed to body b (at p_eq at equilibrium) is now."""
    r, R = pose(refs, x, b)
    return r + R @ (p_eq - refs[b])


def draw(ax, article: str, case: str, lay: dict, refs: np.ndarray,  # type: ignore[no-untyped-def]
         x: np.ndarray, t: float, wave: dict, amp: dict) -> None:
    ax.clear()
    a = np.radians(VIEW[article]); u = np.array([np.cos(a), np.sin(a), 0.0])
    v = np.array([-u[1], u[0], 0.0])
    (x0, x1), (y0, y1) = LIM[article]
    hh = np.linspace(x0, x1, 200)
    eta = wave["A"] * np.cos(wave["w"] * t - wave["k"] * hh / np.cos(a))   # surface at y = 0
    ax.fill_between(hh, y0 - 0.2, eta, color=WATER, zorder=0)
    ax.plot(hh, eta, color=WATER_EDGE, lw=1.4, zorder=1)
    pins = {}
    depth = {b: float(pose(refs, x, b)[0] @ v) for b in lay["buoy"]}
    span = max(np.ptp(list(depth.values())), 1e-9)
    for b in sorted(lay["buoy"], key=lambda k: depth[k]):                   # far buoys first
        r, R = pose(refs, x, b)
        shade = (depth[b] - min(depth.values())) / span
        col = to_rgba(HULL, 0.3 + 0.65 * shade); z = 2 + shade

        def pt(s: float, zb: float, r=r, R=R) -> tuple[float, float]:
            p = r + R @ (s * u + np.array([0.0, 0.0, zb]))
            return float(p @ u), float(p[2])
        ax.add_patch(Polygon([pt(SPAR_R, SPAR_BOT), pt(-SPAR_R, SPAR_BOT), pt(-SPAR_R, PIN_Z),
                              pt(SPAR_R, PIN_Z)], closed=True, color=col, zorder=z))
        ax.plot(*zip(pt(-PLATE_R, PLATE_Z), pt(PLATE_R, PLATE_Z), strict=True), color=col, lw=3,
                zorder=z, solid_capstyle="butt")
        ax.plot(*zip(pt(0, PLATE_Z), pt(0, SPAR_BOT), strict=True), color=col, lw=1.0, zorder=z)
        pins[b] = r + R @ np.array([0.0, 0.0, PIN_Z])
    for i, h in enumerate(lay["hubs"]):
        hc = carried(refs, x, h, lay["hub_centre"][i])
        for b, o in zip(lay["buoy"], lay["owner"], strict=True):
            if o == h:
                ax.plot([hc @ u, pins[b] @ u], [hc[2], pins[b][2]], color=HUB, lw=2.4,
                        solid_capstyle="round", zorder=5)
        if lay["posts"]:
            pt = carried(refs, x, lay["top"], lay["posts"][i])
            ax.plot([hc @ u, pt @ u], [hc[2], pt[2]], color=DECK, lw=2.2, zorder=5)
    for p in pins.values():
        ax.plot(p @ u, p[2], "o", ms=4.5, mfc="white", mec=HUB, mew=1.2, zorder=6)
    if lay["deck"] is not None:
        cs = np.array([carried(refs, x, lay["top"], c) for c in lay["deck"]])
        xy = np.column_stack([cs @ u, cs[:, 2]])
        hull = xy[np.argsort(np.arctan2(xy[:, 1] - xy[:, 1].mean(), xy[:, 0] - xy[:, 0].mean()))]
        ax.add_patch(Polygon(hull, closed=True, fc=to_rgba(DECK, 0.35), ec=DECK, lw=1.6, zorder=6))
    pp = carried(refs, x, lay["top"], lay["payload"])
    sz = 0.06 + 0.009 * np.sqrt(lay["m"])
    ax.add_patch(Rectangle((pp @ u - sz, pp[2] - 0.7 * sz), 2 * sz, 1.4 * sz, color=PAYLOAD,
                           zorder=8))
    ax.add_patch(FancyArrowPatch((x0 + 0.1, y0 + 0.25), (x0 + 0.75, y0 + 0.25), arrowstyle="-|>",
                                 mutation_scale=14, color=WATER_EDGE, lw=2, zorder=9))
    ax.text(x0 + 0.1, y0 + 0.36, "waves", fontsize=9, color="#2d7f8f", zorder=9)
    ax.text(x0 + 0.05, y1 - 0.12, LABEL[(article, case)], fontsize=11.5, fontweight="bold",
            color=HUB)
    ax.text(x0 + 0.05, y1 - 0.32, f"T = {wave['T']:g} s, H = 0.1 m · true scale, real time",
            fontsize=9.5, color=HUB)
    what = "hub" if article == "cluster" else "deck"
    ax.text(x0 + 0.05, y1 - 0.52, f"{what} pitch ±{amp['pitch']:.1f}° {ABOUT[case]}",
            fontsize=9.5, color=LIMIT)
    ax.text(x0 + 0.05, y1 - 0.72, f"buoys rock ±{amp['rock']:.1f}° on their pins", fontsize=9.5,
            color=LIMIT)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")


def main() -> None:
    raw = json.loads((HERE / "tilted_cases.json").read_text(encoding="utf-8"))["cases"]
    cases = {(c["article"], c["case"]): c for c in raw}
    dyn = json.loads((HERE / "tilted_dynamics_preload.json").read_text(encoding="utf-8"))
    runs = {(d["article"], d["case"]): d for d in dyn}
    for article, case in ((a, c) for a in ("cluster", "platform") for c in CASES):
        lay = layout(article, cases[(article, case)])
        for T in PERIODS:
            d = np.load(HERE / "tilted_anim" / f"{article}_{case}_T{T:g}.npz")
            t_all, xi, refs = d["t"], d["xi"], d["refs"]
            wave = {"A": float(d["amplitude"]), "w": float(d["omega"]), "k": float(d["wavenumber"]),
                    "T": T}
            run = next(r for r in runs[(article, case)]["runs"] if abs(r["T_s"] - T) < 1e-9)
            amp = {"pitch": run["pitch_deg"], "rock": run["gimbal_deg_max"]}
            tf = t_all[-1] - T + T * np.arange(N_FRAMES) / N_FRAMES          # the last cycle
            frames = np.column_stack([np.interp(tf, t_all, xi[:, j]) for j in range(xi.shape[1])])
            fig, ax = plt.subplots(figsize=(5.6, 5.4))

            def frame(i: int, frames=frames, tf=tf, lay=lay, refs=refs, wave=wave, amp=amp,
                      ax=ax, article=article, case=case) -> None:
                draw(ax, article, case, lay, refs, frames[i], float(tf[i]), wave, amp)

            anim = FuncAnimation(fig, frame, frames=N_FRAMES)
            out = HERE / f"wanim_{article}_{case}_T{T:g}.gif"
            anim.save(out, writer=PillowWriter(fps=round(N_FRAMES / T)), dpi=90)
            plt.close(fig)
            print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
