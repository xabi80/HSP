"""Figures for the flume sidewall-effect study (see flume_wall_effect.py, REBUTTAL-sidewall.md).

Fast, geometry-only figures (blockage plan view + transverse-mode regime map) are generated
here directly. The BEM figures (rebuttal_summary.png, rebuttal_heave_rao.png) come from the
``run_sweep`` results and are committed as artifacts alongside this script.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import flume_wall_effect as fw
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def fig_blockage(path: str = "flume_blockage.png") -> None:
    """Plan view to scale: the 16-buoy platform (2.5 m to centres) in the LWF."""
    cen = fw.buoy_centers()
    half = fw.FLUME_W / 2
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    ax.axvspan(-half - 0.25, -half, color="0.55")
    ax.axvspan(half, half + 0.25, color="0.55")
    ax.axvline(-half, color="k", lw=1)
    ax.axvline(half, color="k", lw=1)
    for x, y in cen:
        ax.add_patch(Circle((x, y), fw.PLATE_R, fc="#9FE1CB", ec="#0F6E56", lw=1.1, alpha=0.85))
        ax.add_patch(Circle((x, y), fw.SPAR_R, fc="#0c8b96", ec="k", lw=0.5))
    clr = fw.clearance()
    maxx = np.abs(cen[:, 0]).max() + fw.PLATE_R
    ax.annotate("", xy=(-half, -1.9), xytext=(-maxx, -1.9),
                arrowprops=dict(arrowstyle="<->", color="#d1543a", lw=1.6))
    ax.text((-half - maxx) / 2, -2.05, f"{clr * 100:.0f} cm", color="#d1543a",
            ha="center", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-2.15, 2.05)
    ax.grid(alpha=0.25)
    ax.set_xlabel(f"across-flume (m)  —  LWF width {fw.FLUME_W} m")
    ax.set_ylabel("along-flume (m)")
    ax.set_title(f"16-buoy platform (2.5 m to centres) in the OSU LWF\n"
                 f"outer span {2 * maxx:.2f} m ({200 * maxx / fw.FLUME_W:.0f}% of width) "
                 f"-> {clr * 100:.0f} cm/side", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


def fig_regime(path: str = "flume_regime_map.png") -> None:
    """Where the walls matter across wave period: transverse cut-on map."""
    t1, t2, t3 = fw.transverse_cutoff_periods()
    fig, ax = plt.subplots(figsize=(12, 3.9))
    ax.axvspan(t1, 4.2, color="#E1F5EE")
    ax.axvspan(1.05, t1, color="#FAEEDA")
    for t, n in [(t1, 1), (t2, 2), (t3, 3)]:
        ax.axvline(t, color="#d1543a", ls="--", lw=1.6)
        ax.text(t, 0.82, f"n={n}\n{t:.2f}s", color="#a3312d", ha="center",
                fontsize=9, fontweight="bold")
    ax.axvline(fw.T_HEAVE, color="#185fa5", lw=2.2)
    ax.text(fw.T_HEAVE, 0.40, "heave\nresonance\n2.52 s", color="#185fa5",
            ha="center", fontsize=9, fontweight="bold")
    ax.text((t1 + 4.2) / 2, 0.16,
            "SUB-CUTOFF (T > 2.2 s): transverse modes evanescent\n-> wall effect small & smooth",
            ha="center", fontsize=10, color="#0f6e56")
    ax.text((1.05 + t1) / 2, 0.16, "WALL-SENSITIVE:\ntransverse sloshing\nmodes propagate",
            ha="center", fontsize=9.5, color="#854f0b")
    ax.set_xlim(1.05, 4.2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("wave period T (s)")
    ax.set_title(f"Where the LWF walls affect the RAO — transverse-mode map "
                 f"(W={fw.FLUME_W} m, h={fw.FLUME_H} m)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    fig_blockage()
    fig_regime()
    print("wrote flume_blockage.png, flume_regime_map.png")
