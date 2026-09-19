"""0deg (square, flat-on) vs 45deg (diagonal, corner-on) orientation comparison for the
16-buoy platform in the flume. Rotating the platform 90deg is a 4-fold-symmetric no-op; 45deg
is the genuinely different orientation, and it turns the platform corner-on to the side walls,
nearly doubling the clearance (0.44 -> 0.80 m/side). This quantifies whether that changes the
(small) wall effect -- it does not, because the effect is set by bulk blockage, not clearance.

The sidewall effect is read from the AUTHORITATIVE coupled BEM (open vs walled excitation on the
platform-heave mode); the single-array frequency-domain method under-converges at long periods.

Writes orientation_compare.png + prints the comparison table. Needs both orientations' coupled
BEM .nc files and articulated_decay json. Run: python compare_orientations.py
"""
# ruff: noqa: E702
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
S = 1.25 / 1.5
PLATE_R, FLUME_W = 0.1437, 3.66
TEAL, RED = "#0c8b96", "#b2432c"


def clearance(rot_deg: float) -> float:
    ang = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]) + rot_deg)
    cy = [S * np.sin(pc) + 0.5 * S * np.sin(tb) for pc in ang for tb in ang]
    return FLUME_W / 2 - (np.max(np.abs(cy)) + PLATE_R)


def coupled_wall(suf: str):
    """(T, wall% on platform-heave excitation) from the coupled BEM open vs walled."""
    o = xr.load_dataset(HERE / f"coupled_osu_open{suf}.nc")
    w = xr.load_dataset(HERE / f"coupled_osu_walled{suf}.nc")
    dofs = [str(x) for x in o.influenced_dof.values]
    he = [i for i, d in enumerate(dofs) if d.endswith("__Heave")]
    om = o.omega.values

    def exc(ds):
        f = ds.excitation_force.values
        return np.abs((f[0] + 1j * f[1])[:, 0, he].sum(axis=1))
    m = np.isfinite(om) & (om > 0.2) & (om < 4.0)     # 1.5-30 s band
    T = 2 * np.pi / om[m]
    pct = 100 * (exc(w)[m] / exc(o)[m] - 1)
    idx = np.argsort(T)
    return T[idx], pct[idx]


def decay_shift(suf: str) -> float:
    d = json.loads((HERE / f"articulated_decay{suf}.json").read_text())
    return 100 * (d["walled"]["heave_T"] / d["open"]["heave_T"] - 1)


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    T0, p0 = coupled_wall("")
    T45, p45 = coupled_wall("_rot45")
    band = (T0 >= 2.0) & (T0 <= 4.0)
    b45 = (T45 >= 2.0) & (T45 <= 4.0)
    print(" orientation |  clearance | free-decay dT | max |wall effect on exc| (2-4 s)")
    print(f"  0deg  (flat-on)  | {clearance(0) * 100:5.0f} cm |   {decay_shift(''):+6.2f}%    |"
          f"   {np.max(np.abs(p0[band])):5.1f}%")
    dt45 = decay_shift("_rot45")
    print(f"  45deg (corner)   | {clearance(45) * 100:5.0f} cm |   {dt45:+6.2f}%    |"
          f"   {np.max(np.abs(p45[b45])):5.1f}%")

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.plot(T0, p0, "o-", color=RED, lw=2, label="0° (flat-on, 0.44 m/side clearance)")
    ax.plot(T45, p45, "s--", color=TEAL, lw=2, label="45° (corner-on, 0.80 m/side clearance)")
    ax.axhspan(-5, 5, color="0.9", zorder=0)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlim(1.8, 4.1)
    ax.set_xlabel("wave period T (s)")
    ax.set_ylabel("sidewall effect on heave excitation (%)")
    ax.set_title("Orientation effect — 0° (flat-on) vs 45° (corner-on), coupled BEM\n"
                 "doubling the clearance (0.44→0.80 m) leaves the small wall effect unchanged — "
                 "it is set by bulk blockage, not clearance",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "orientation_compare.png", dpi=130, bbox_inches="tight")
    print("\nwrote orientation_compare.png")


if __name__ == "__main__":
    main()
