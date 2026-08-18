"""3D surface plots of CENTRE/reference-point heave acceleration (acc_center_amp,
m/s^2) over the wave height H and period T grid, across ALL FOUR models
(single buoy / 3-cluster / 12-buoy / 16-buoy), one row per fin/Cd config.

"Centre" = the model's reference point: the buoy itself (single), the cluster centre
(3-cluster), the platform centre (12- & 16-buoy). Z-scale shared within a row so the
four models are directly comparable; it differs between rows (the fin lowers the level).
Reads the fan summary CSVs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

_R = Path("studies")
# metric: "center" (platform/centre reference point) or "buoy" (representative buoy-7 top)
_WHICH = sys.argv[1] if len(sys.argv) > 1 else "center"
_METRIC = "acc_buoy_amp" if _WHICH == "buoy" else "acc_center_amp"
_OUT = _R / ("platform-16buoy/fin_study/"
             + ("accel_buoy_HT_surf3d_4models.png" if _WHICH == "buoy"
                else "accel_HT_surf3d_4models.png"))
_WHATLBL = ("representative buoy (buoy-7 top; the single buoy for 'single')"
            if _WHICH == "buoy" else
            "centre/reference point (buoy for single, cluster centre for 3-cluster, "
            "platform for 12/16)")

# (column label, folder, filename prefix)
_MODELS = [
    ("single buoy", _R / "spar-fin-decay/fin_study", "rao_summary_fin"),
    ("3-cluster", _R / "cluster-3buoy-rigid/fin_study", "rao_summary_cluster_fin"),
    ("12-buoy", _R / "platform-12buoy/fin_study", "rao_summary_platform_fin"),
    ("16-buoy", _R / "platform-16buoy/fin_study", "rao_summary_platform16_fin"),
]
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
        Z[Hs.index(float(r["height_m"])), Ts.index(float(r["period_s"]))] = float(r[_METRIC])
    return np.array(Hs), np.array(Ts), Z


def main() -> None:
    nrow, ncol = len(_CONFIGS), len(_MODELS)
    fig = plt.figure(figsize=(21, 23))
    for row, (title, cfg) in enumerate(_CONFIGS):
        loaded = []
        for _, folder, pfx in _MODELS:
            p = folder / f"{pfx}{cfg}.csv"
            loaded.append(_load(p) if p.exists() else None)
        vmax = max(np.nanmax(d[2]) for d in loaded if d is not None)
        row_axes = []
        for col, (mdl, _, _) in enumerate(_MODELS):
            ax = fig.add_subplot(nrow, ncol, ncol * row + col + 1, projection="3d")
            row_axes.append(ax)
            d = loaded[col]
            if d is None:
                ax.set_axis_off(); continue
            Hs, Ts, Z = d
            TT, HH = np.meshgrid(Ts, Hs)
            ax.plot_surface(TT, HH, Z, cmap=cm.inferno, vmin=0, vmax=vmax, rstride=1, cstride=1,
                            edgecolor="k", linewidth=0.12, antialiased=True, alpha=0.97)
            ip = np.unravel_index(np.nanargmax(Z), Z.shape)
            ax.scatter([Ts[ip[1]]], [Hs[ip[0]]], [np.nanmax(Z)], color="#39cdd6", s=45,
                       edgecolor="k", linewidth=0.5, depthshade=False)
            ax.set_zlim(0, vmax * 1.02)
            head = f"{mdl} · {title}" if col == 0 else mdl
            ax.set_title(f"{head}\npeak {np.nanmax(Z):.2f} @ T={Ts[ip[1]]:.2f}, H={Hs[ip[0]]:.2f}",
                         fontsize=8, pad=0)
            ax.set_xlabel("T (s)", fontsize=7.5, labelpad=0)
            ax.set_ylabel("H (m)", fontsize=7.5, labelpad=0)
            ax.set_zlabel("accel (m/s²)", fontsize=7.5, labelpad=0)
            ax.tick_params(labelsize=6)
            ax.view_init(elev=26, azim=-122)
        _ = row_axes  # z-axis carries the scale; colour is only for surface shading
    what = "Representative-buoy" if _WHICH == "buoy" else "Centre/reference-point"
    fig.suptitle(f"{what} heave acceleration surfaces — single / 3-cluster / 12-buoy / 16-buoy\n"
                 f"one row per fin/Cd config · Z-scale shared within each row · ● = peak · "
                 f"({_WHATLBL})", fontsize=13, y=0.995)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.965, bottom=0.02, wspace=0.02, hspace=0.14)
    fig.savefig(_OUT, dpi=115)
    plt.close(fig)
    print("wrote", _OUT)


if __name__ == "__main__":
    main()
