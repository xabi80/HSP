"""Finite-depth effect explainer figure, from Airy (linear) wave theory -- method-independent.

Why the flume depth matters more at long periods: a wave only "feels" the bottom once its
wavelength is long compared with the 2.7 m depth (h/lambda small). Short waves are confined near
the surface and behave as in deep water; long waves reach the floor, which forces the vertical
particle velocity to zero, flattening the circular orbits into ellipses and shrinking the vertical
motion that lifts the heave plate. The plate sits at z = -1.38 m in 2.7 m of water, so it sees a
lot of that attenuation.

Panels: (a) deep-water orbits and (b) 2.7 m orbits at a long period, with the heave plate and its
vertical-motion amplitude; (c) the plate's vertical orbital amplitude at 2.7 m relative to deep
water vs period (the finite-depth effect), with the BEM depth-effect points overlaid -- they
agree, which is the independent corroboration of the BEM depth effect.

Writes depth_effect_explained.png. Run: python depth_effect_plots.py
"""
# ruff: noqa: E702, RUF001  -- compact plotting lines; figure text uses display typography
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
G, H, A, Z_PLATE, Z_SPAR_BOT = 9.806, 2.7, 0.42, -1.383, -0.967
WATER, TEAL, DOT = "#2A6FBF", "#0F6E56", "#185FA5"
FLOOR, ORBIT, RED = "#7A8994", "#8a97a3", "#b2432c"
# BEM finite-depth effect on heave excitation (2.7 m vs deep), from flume_wall_effect.py
BEM_T = np.array([1.40, 1.53, 1.80, 2.19, 2.52, 3.00, 3.50, 4.00])
BEM_PCT = np.array([+0.5, -0.4, -4.2, -12.2, -19.3, -27.3, -33.0, -36.6])


def k_fin(T: float) -> float:
    w = 2 * np.pi / T; k = w * w / G
    for _ in range(90):
        k = w * w / (G * np.tanh(k * H))
    return k


def v_fin(z, k):
    return A * np.sinh(k * (z + H)) / np.sinh(k * H)


def u_fin(z, k):
    return A * np.cosh(k * (z + H)) / np.sinh(k * H)


def a_deep(z, k):
    return A * np.exp(k * z)


def orbit_panel(ax, T, mode):
    k = (2 * np.pi / T) ** 2 / G if mode == "deep" else k_fin(T)
    xs = np.linspace(0, 6.6, 300)
    eta = A * np.cos(k * xs)
    ax.fill_between(xs, eta, -3.2, color=WATER, alpha=0.10)
    ax.plot(xs, eta, color=WATER, lw=2)
    if mode == "flume":
        ax.axhline(-H, color=FLOOR, lw=2.2)
        ax.fill_between([0, 6.6], -H, -3.2, color=FLOOR, alpha=0.25, hatch="///", lw=0)
        ax.text(0.15, -H + 0.08, "bottom −2.7 m", fontsize=9, color=FLOOR)
    else:
        ax.text(0.15, -3.05, "deep — bottom far below", fontsize=9, color=FLOOR)
    for x0 in [1.0, 2.6, 4.2, 5.8]:
        for z0 in [-0.35, -0.95, -1.55, -2.15]:
            if mode == "flume" and z0 < -H + 0.05:
                continue
            ua = a_deep(z0, k) if mode == "deep" else u_fin(z0, k)
            va = a_deep(z0, k) if mode == "deep" else v_fin(z0, k)
            th = np.linspace(0, 2 * np.pi, 80)
            ax.plot(x0 + ua * np.cos(th), z0 + va * np.sin(th), color=ORBIT, lw=1)
            ph = k * x0
            ax.plot(x0 - ua * np.sin(ph), z0 + va * np.cos(ph), "o", color=DOT, ms=4)
    vap = a_deep(Z_PLATE, k) if mode == "deep" else v_fin(Z_PLATE, k)
    xb = 3.4
    ax.plot([xb, xb], [0.15, Z_SPAR_BOT], color=TEAL, lw=4, solid_capstyle="round")
    ax.plot([xb - 0.36, xb + 0.36], [Z_PLATE, Z_PLATE], color=TEAL, lw=5, solid_capstyle="round")
    ax.plot(xb, 0.15, "o", color=TEAL, ms=7)
    ax.annotate("", xy=(xb + 0.55, Z_PLATE + vap), xytext=(xb + 0.55, Z_PLATE - vap),
                arrowprops=dict(arrowstyle="<->", color=TEAL, lw=1.4))
    ax.text(xb + 0.68, Z_PLATE, f"plate\nmotion\n±{vap:.2f} m", fontsize=8.5, color=TEAL,
            va="center")
    ax.set_xlim(0, 6.6); ax.set_ylim(-3.2, 0.9)
    ax.set_xticks([]); ax.set_yticks([0, -1, -2, -2.7] if mode == "flume" else [0, -1, -2, -3])
    ax.set_ylabel("depth (m)", fontsize=9)
    ax.set_title(("Deep water (open sea)" if mode == "deep" else "OSU flume — 2.7 m water")
                 + f"   T = {T:.1f} s", fontsize=10.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    T_show = 3.5
    fig = plt.figure(figsize=(13.2, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], wspace=0.28)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    orbit_panel(axa, T_show, "deep")
    orbit_panel(axb, T_show, "flume")

    Ts = np.linspace(1.3, 4.5, 200)
    ratio = np.array([100 * (v_fin(Z_PLATE, k_fin(T))
                             / a_deep(Z_PLATE, (2 * np.pi / T) ** 2 / G) - 1) for T in Ts])
    hl = np.array([H / (2 * np.pi / k_fin(T)) for T in Ts])
    axc.axvspan(1.3, Ts[np.argmax(hl < 0.5)], color="0.93", zorder=0)
    axc.text(1.42, -7, "deep\n(h/λ > 0.5)", fontsize=8.5, color="0.4", va="top")
    axc.text(3.0, 3, "intermediate — wave feels the bottom", fontsize=8.5, color="0.4", va="top")
    axc.plot(Ts, ratio, color=TEAL, lw=2.2,
             label="Airy theory: plate vertical motion, 2.7 m vs deep")
    axc.plot(BEM_T, BEM_PCT, "s", color=RED, ms=6, label="BEM: heave excitation, 2.7 m vs deep")
    axc.axvline(2.6, color=TEAL, lw=1, ls=":")
    axc.text(2.64, -9, "natural\nperiod", fontsize=8, color=TEAL, va="top")
    axc.axhline(0, color="0.5", lw=0.8)
    axc.set_xlim(1.3, 4.5); axc.set_ylim(-44, 6)
    axc.set_xlabel("wave period T (s)"); axc.set_ylabel("finite-depth effect (%)")
    axc.set_title("How much: 2.7 m vs deep water", fontsize=10.5, fontweight="bold")
    axc.legend(fontsize=7.8, loc="lower left"); axc.grid(alpha=0.3)
    axc.spines[["top", "right"]].set_visible(False)
    fig.suptitle("The finite-depth effect — long waves reach the 2.7 m bottom, which flattens "
                 "the orbits and shrinks the vertical motion at the heave plate", fontsize=11.5,
                 fontweight="bold", y=1.02)
    fig.savefig(HERE / "depth_effect_explained.png", dpi=130, bbox_inches="tight")
    print("wrote depth_effect_explained.png")


if __name__ == "__main__":
    main()
