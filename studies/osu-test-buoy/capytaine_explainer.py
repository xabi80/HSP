"""Explainer schematic for the Capytaine diffraction & radiation analysis.

A teaching/meeting figure (not a computation): the linear wave-body problem splits
by superposition into two BEM sub-problems, whose outputs feed the FloatSim
time-domain Cummins equation.

  (1) RADIATION  — body forced to oscillate in still water -> added mass A(ω),
                   radiation damping B(ω).
  (2) DIFFRACTION — body held fixed under incident waves -> excitation force
                    F_exc(ω,β) = Froude-Krylov + diffraction.
  assembly       -> (M+A_inf) x'' + ∫ K(t-τ) x'(τ) dτ + C x = F_exc(t) + F_drag(x')
                    with K(t) = (2/π)∫ B(ω) cos(ωt) dω. Viscous F_drag is added
                    separately (Morison) — potential flow cannot see it.

Writes Capytaine_explained.png next to this script. See OSU-TEST-BUOY-GEOMETRY.md
("Capytaine diffraction & radiation analysis — how to explain it").
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

CY, WV, INK, PL = "#0c8b96", "#2f74c0", "#25323a", "#54636d"


def _buoy(ax, x0, z0):
    ax.add_patch(Rectangle((x0 - 0.04, z0), 0.08, 0.9, facecolor=CY, edgecolor=INK, lw=1.2, zorder=5))
    ax.add_patch(Rectangle((x0 - 0.14, z0 - 0.06), 0.28, 0.06, facecolor=PL, edgecolor=INK, lw=1, zorder=5))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    fig = plt.figure(figsize=(13.5, 7.6))

    # --- (1) Radiation ---
    ax = fig.add_axes([0.04, 0.42, 0.44, 0.5])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, 1.3)
    ax.axis("off")
    ax.axhline(0.75, color=WV, lw=1, alpha=.5)
    for r in (0.30, 0.50, 0.70, 0.92):
        x = np.linspace(-0.98, 0.98, 200)
        ax.plot(x, 0.75 + 0.05 * np.cos(2 * np.pi * x / 0.4) * np.exp(-((abs(x) - r) * 6) ** 2),
                color=WV, lw=1, alpha=.55)
    _buoy(ax, 0, 0.05)
    ax.add_patch(FancyArrowPatch((0.28, 0.35), (0.28, 0.75), arrowstyle="<->",
                                 color="#d1543a", lw=2, mutation_scale=14))
    ax.text(0.33, 0.55, "forced\noscillation", color="#d1543a", fontsize=9, va="center")
    ax.set_title("(1) RADIATION  -  body as a WAVEMAKER", fontsize=11, fontweight="bold", color=CY)
    ax.text(0, -0.45, "Hold the waves off. Force the body to move in still water\n"
            "(each DOF, each ω). It radiates its own waves.\n"
            "-> Added mass A(ω)  +  Radiation damping B(ω)", ha="center", fontsize=9.5)

    # --- (2) Diffraction ---
    ax = fig.add_axes([0.52, 0.42, 0.44, 0.5])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, 1.3)
    ax.axis("off")
    x = np.linspace(-1, 1, 300)
    ax.plot(x, 0.75 + 0.06 * np.cos(2 * np.pi * x / 0.35), color=WV, lw=1.6)
    _buoy(ax, 0, 0.05)
    ax.add_patch(FancyArrowPatch((-0.9, 0.9), (-0.55, 0.9), arrowstyle="-|>",
                                 color=WV, lw=2.4, mutation_scale=16))
    ax.text(-0.72, 1.0, "incident waves", color=WV, fontsize=9, ha="center")
    ax.text(0.0, -0.12, "FIXED", ha="center", fontsize=9, color="#d1543a", fontweight="bold")
    ax.set_title("(2) DIFFRACTION  -  body as an OBSTACLE", fontsize=11, fontweight="bold", color=CY)
    ax.text(0, -0.45, "Hold the body FIXED. Let waves scatter off it.\n"
            "Force on the fixed body = incident + scattered pressure.\n"
            "-> Excitation force  F_exc(ω,β) = Froude-Krylov + diffraction", ha="center", fontsize=9.5)

    # --- assembly ---
    ax = fig.add_axes([0.06, 0.02, 0.88, 0.30])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyArrowPatch((0.28, 0.98), (0.42, 0.72), arrowstyle="-|>",
                                 color=PL, lw=1.8, mutation_scale=14))
    ax.add_patch(FancyArrowPatch((0.72, 0.98), (0.58, 0.72), arrowstyle="-|>",
                                 color=PL, lw=1.8, mutation_scale=14))
    ax.add_patch(Rectangle((0.06, 0.12), 0.88, 0.56, facecolor="#eef6f7", edgecolor=CY, lw=1.5))
    ax.text(0.5, 0.58, "Time-domain equation of motion  (Cummins):",
            ha="center", fontsize=10.5, fontweight="bold", color=INK)
    ax.text(0.5, 0.37,
            r"$(M + A_\infty)\,\ddot{x}\;+\;\int_0^{t} K(t-\tau)\,\dot{x}(\tau)\,d\tau\;+\;"
            r"C\,x\;=\;F_{exc}(t)\;+\;F_{drag}(\dot x)$", ha="center", fontsize=15, color=INK)
    ax.text(0.5, 0.19,
            r"$A_\infty$ = high-ω added mass   ·   $K(t)=\frac{2}{\pi}\!\int B(\omega)\cos\omega t\,d\omega$"
            r" (radiation memory)   ·   $C$ = hydrostatic stiffness   ·   "
            r"$F_{drag}$ = viscous (Morison) — NOT from the BEM", ha="center", fontsize=9, color=PL)

    fig.suptitle("What Capytaine computes — linear potential-flow BEM on the wetted hull (panels)\n"
                 "the wave-body problem splits by superposition into two sub-problems that add up",
                 fontsize=12, y=1.0)
    out = Path(__file__).resolve().parent / "Capytaine_explained.png"
    fig.savefig(out, dpi=145, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
