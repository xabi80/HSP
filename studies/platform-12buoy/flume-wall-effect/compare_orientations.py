"""0deg (square, flat-on) vs 45deg (diagonal, corner-on) orientation comparison for the
16-buoy platform in the flume. Rotating the platform 90deg is a 4-fold-symmetric no-op; 45deg
is the genuinely different orientation, and it turns the platform corner-on to the side walls,
nearly doubling the clearance (0.44 -> 0.80 m/side). So 0deg is the WIDEST / worst orientation
and 45deg should show a smaller wall effect -- this quantifies it.

Reads each orientation's outputs (needs both to have been run):
  articulated_decay[_rot45].json   -> free-decay heave period shift
  sweep_results[_rot45].npy        -> heave wall + depth excitation, and the single-DOF response

Writes orientation_compare.png + prints the comparison table.
Run after both orientations' analysis chains: python compare_orientations.py
"""
# ruff: noqa: E702
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
C33_BUOY, N_BUOY, T_HEAVE, ZETA = 194.5, 16, 2.6, 0.10
S = 1.25 / 1.5
PLATE_R, FLUME_W = 0.1437, 3.66
TEAL, RED = "#0c8b96", "#b2432c"


def clearance(rot_deg: float) -> float:
    ang = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]) + rot_deg)
    cy = []
    for pc in ang:
        for tb in ang:
            cy.append(S * np.sin(pc) + 0.5 * S * np.sin(tb))
    return FLUME_W / 2 - (np.max(np.abs(cy)) + PLATE_R)


def single_dof_response(suf: str):
    """(T, response wall-effect %) from the 2.7 m sweep, single-DOF platform-heave impedance."""
    rows = np.load(HERE / f"sweep_results{suf}.npy", allow_pickle=True)
    T = np.array([r[0] for r in rows])
    w = 2 * np.pi / T
    A_o = np.array([r[1][("A", "He")] for r in rows])
    B_o = np.array([r[1][("B", "He")] for r in rows])
    F_o = np.array([r[1][("F", "He")] for r in rows])
    A_w = np.array([r[2][("A", "He")] for r in rows])
    B_w = np.array([r[2][("B", "He")] for r in rows])
    F_w = np.array([r[2][("F", "He")] for r in rows])
    C = N_BUOY * C33_BUOY
    wn = 2 * np.pi / T_HEAVE
    A_n = np.interp(wn, w[::-1], A_o[::-1])
    M = C / wn**2 - A_n
    Bv = 2 * ZETA * np.sqrt(C * (M + A_n))
    r_o = np.abs(F_o) / np.abs(C - (M + A_o) * w**2 + 1j * w * (B_o + Bv))
    r_w = np.abs(F_w) / np.abs(C - (M + A_w) * w**2 + 1j * w * (B_w + Bv))
    return T, 100 * (r_w / r_o - 1)


def decay_shift(suf: str) -> float:
    d = json.loads((HERE / f"articulated_decay{suf}.json").read_text())
    return 100 * (d["walled"]["heave_T"] / d["open"]["heave_T"] - 1)


def band_max(T, pct):
    b = (T >= 1.5) & (T <= 2.9)
    return np.max(np.abs(pct[b]))


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    T0, p0 = single_dof_response("")
    T45, p45 = single_dof_response("_rot45")
    print(" orientation |  clearance | free-decay dT | response |wall| (1.5-2.9 s)")
    for name, suf, pct in [("0deg (flat-on)", "", p0), ("45deg (corner)", "_rot45", p45)]:
        print(f"  {name:14s} | {clearance(0 if not suf else 45) * 100:5.0f} cm |"
              f"   {decay_shift(suf):+6.2f}%    |   {band_max(T0 if not suf else T45, pct):5.1f}%")

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(T0, p0, "o-", color=RED, lw=2, label="0° (flat-on, 0.44 m/side clearance)")
    ax.plot(T45, p45, "s--", color=TEAL, lw=2, label="45° (corner-on, 0.80 m/side clearance)")
    ax.axhspan(-5, 5, color="0.9", zorder=0)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvspan(2.09, 2.29, color="0.82", alpha=0.6, zorder=0)
    ax.set_xlabel("wave period T (s)")
    ax.set_ylabel("heave response wall effect at 2.7 m (%)")
    ax.set_title("Orientation effect — 0° (flat-on) vs 45° (corner-on)\n"
                 "doubling the clearance (0.44→0.80 m) leaves the wall effect unchanged — "
                 "it is set by bulk blockage, not clearance",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "orientation_compare.png", dpi=130, bbox_inches="tight")
    print("\nwrote orientation_compare.png")


if __name__ == "__main__":
    main()
