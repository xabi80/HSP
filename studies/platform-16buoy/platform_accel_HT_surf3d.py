"""3D surface plots of PLATFORM-centre heave acceleration (acc_center_amp, m/s^2) over
the wave height H and period T grid, one panel per config, 12-buoy vs 16-buoy side by
side. 3D companion to platform_accel_HT_maps.py (same data, surface view).

Z (height + colour) = platform accel. Z-scale shared within a row (12 vs 16 comparable);
it differs between rows because the fin lowers the whole level.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

_R = Path("studies")
_OUT = _R / "platform-16buoy/fin_study/platform_accel_HT_surf3d.png"
_12 = _R / "platform-12buoy/fin_study"
_16 = _R / "platform-16buoy/fin_study"

_CONFIGS = [
    ("no fin (+cap)", "none_cap"),
    ("0.15 m fin · Cd=5", "015_Cd5"),
    ("0.15 m fin · Cd=1", "015_Cd1"),
    ("0.215 m fin · Cd=5", "0215_Cd5"),
    ("0.215 m fin · Cd=1", "0215_Cd1"),
]


def _load(path: Path):  # type: ignore[no-untyped-def]
    rows = list(csv.DictReader(path.open()))
    Hs = sorted({float(r["height_m"]) for r in rows})
    Ts = sorted({float(r["period_s"]) for r in rows})
    Z = np.full((len(Hs), len(Ts)), np.nan)
    for r in rows:
        Z[Hs.index(float(r["height_m"])), Ts.index(float(r["period_s"]))] = float(r["acc_center_amp"])
    return np.array(Hs), np.array(Ts), Z


def main() -> None:
    fig = plt.figure(figsize=(13, 20))
    for row, (title, cfg) in enumerate(_CONFIGS):
        data = {}
        for col, (folder, pfx) in enumerate(
            [(_12, "rao_summary_platform_fin"), (_16, "rao_summary_platform16_fin")]):
            p = folder / f"{pfx}{cfg}.csv"
            data[col] = _load(p) if p.exists() else None
        vmax = max(np.nanmax(d[2]) for d in data.values() if d is not None)
        for col, mdl in enumerate(("12-buoy", "16-buoy")):
            ax = fig.add_subplot(len(_CONFIGS), 2, 2 * row + col + 1, projection="3d")
            if data[col] is None:
                ax.set_axis_off(); continue
            Hs, Ts, Z = data[col]
            TT, HH = np.meshgrid(Ts, Hs)
            surf = ax.plot_surface(TT, HH, Z, cmap=cm.inferno, vmin=0, vmax=vmax,
                                   rstride=1, cstride=1, edgecolor="k", linewidth=0.15,
                                   antialiased=True, alpha=0.97)
            ip = np.unravel_index(np.nanargmax(Z), Z.shape)
            ax.scatter([Ts[ip[1]]], [Hs[ip[0]]], [np.nanmax(Z)], color="#39cdd6",
                       s=55, edgecolor="k", linewidth=0.6, depthshade=False)
            ax.set_zlim(0, vmax * 1.02)
            ax.set_title(f"{mdl} · {title}\npeak {np.nanmax(Z):.2f} m/s² @ T={Ts[ip[1]]:.2f}s, "
                         f"H={Hs[ip[0]]:.2f}m", fontsize=8.5, pad=0)
            ax.set_xlabel("T (s)", fontsize=8, labelpad=1)
            ax.set_ylabel("H (m)", fontsize=8, labelpad=1)
            ax.set_zlabel("accel (m/s²)", fontsize=8, labelpad=1)
            ax.tick_params(labelsize=6.5)
            ax.view_init(elev=26, azim=-122)
            fig.colorbar(surf, ax=ax, pad=0.10, shrink=0.55, aspect=12).ax.tick_params(labelsize=6)
    fig.suptitle("Platform-centre heave acceleration surfaces — 12-buoy vs 16-buoy\n"
                 "one panel per fin/Cd config · Z-scale shared within each row · ● = peak",
                 fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.986))
    fig.savefig(_OUT, dpi=125)
    plt.close(fig)
    print("wrote", _OUT)


if __name__ == "__main__":
    main()
