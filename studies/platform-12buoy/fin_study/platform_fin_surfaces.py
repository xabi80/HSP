"""3D response surfaces for the 12-buoy platform fin study: buoy heave RAO and
vertical (Nz) acceleration amplitude over the full (wave period T, wave height H)
grid, one column per fin {0.215, 0.15, none}. Channel is buoy 7 (a payload buoy),
matching the 2D per-model plot. Two figures -- Cd_n=5 (operational) and Cd_n=1
(light drag); the no-fin has a single bottom-cap config, shown in both. z-limits
are shared across the three fins within each row so the fins are directly
comparable (bigger fin -> lower surface). Reads rao_summary_platform_fin*.csv.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PL = Path(__file__).resolve().parent
_TN = {"0215": 3.15, "015": 2.66, "none": 2.50}  # platform heave natural period per fin
_LBL = {"0215": "fin 0.215 m", "015": "fin 0.15 m", "none": "no fin (+cap)"}
_FINS = ["0215", "015", "none"]


def _cfg(fin: str, cd: str) -> str:
    return "none_cap" if fin == "none" else f"{fin}_Cd{cd}"


def load(fin: str, cd: str):  # type: ignore[no-untyped-def]
    rows = list(csv.DictReader((_PL / f"rao_summary_platform_fin{_cfg(fin, cd)}.csv").open()))
    H = sorted({float(r["height_m"]) for r in rows})
    T = sorted({float(r["period_s"]) for r in rows})
    R = np.full((len(H), len(T)), np.nan)
    A = np.full((len(H), len(T)), np.nan)
    for r in rows:
        i, j = H.index(float(r["height_m"])), T.index(float(r["period_s"]))
        R[i, j] = float(r["rao_buoy"])
        A[i, j] = float(r["acc_buoy_amp"])
    return H, T, R, A


def _surf(ax, T, H, Z, zlabel, title, zlim, tn, rao):  # type: ignore[no-untyped-def]
    Tg, Hg = np.meshgrid(T, H)
    ax.plot_surface(Tg, Hg, Z, cmap="viridis", vmin=zlim[0], vmax=zlim[1], edgecolor="k",
                    linewidth=0.25, rstride=1, cstride=1, antialiased=True, alpha=0.95)
    ax.scatter(Tg.ravel(), Hg.ravel(), Z.ravel(), color="crimson", s=9, depthshade=False)
    ax.plot([tn, tn], [min(H), max(H)], [zlim[0], zlim[0]], color="gray", ls=":", lw=1.3)
    if rao:
        ax.plot([min(T), max(T)], [max(H), max(H)], [1.0, 1.0], "r--", lw=1.0, alpha=0.7)
    ax.set_xlabel("T (s)", fontsize=8, labelpad=1)
    ax.set_ylabel("H (m)", fontsize=8, labelpad=1)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=2)
    ax.set_zlim(*zlim)
    ax.set_title(title, fontsize=9, pad=2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=24, azim=-58)


def make_fig(cd: str) -> None:
    data = {f: load(f, cd) for f in _FINS}
    rmax = max(np.nanmax(data[f][2]) for f in _FINS)
    amax = max(np.nanmax(data[f][3]) for f in _FINS)
    fig = plt.figure(figsize=(16, 9.5))
    for col, f in enumerate(_FINS):
        H, T, R, A = data[f]
        ij = np.unravel_index(np.nanargmax(R), R.shape)
        rt = f"{_LBL[f]} -- heave RAO\npk {np.nanmax(R):.2f}@T={T[ij[1]]:.3g}s H={H[ij[0]]:.2g}m"
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        _surf(ax, T, H, R, "RAO", rt, (0.0, 1.05 * rmax), _TN[f], True)
        ij2 = np.unravel_index(np.nanargmax(A), A.shape)
        at = f"{_LBL[f]} -- Nz accel\npk {np.nanmax(A):.3f}@T={T[ij2[1]]:.3g}s H={H[ij2[0]]:.2g}m"
        ax2 = fig.add_subplot(2, 3, col + 4, projection="3d")
        _surf(ax2, T, H, A, "Nz acc (m/s^2)", at, (0.0, 1.05 * amax), _TN[f], False)
    fig.suptitle(
        f"12-buoy platform (buoy 7): heave RAO (top) and Nz acceleration (bottom) over "
        f"(T, H) per fin -- Cd_n={cd}\n"
        "gray dotted = fin heave natural period; red dashed = RAO 1; "
        "H 0.04-0.12 m = 2-6 m full-scale @ 1:50", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = _PL / f"platform_fin_surfaces_Cd{cd}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out.name}  (RAO zmax {rmax:.2f}, accel zmax {amax:.3f})")


if __name__ == "__main__":
    make_fig("5")
    make_fig("1")  # no-fin uses its single bottom-cap config in both figures
