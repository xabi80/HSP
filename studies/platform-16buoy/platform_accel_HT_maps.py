"""2D maps of PLATFORM-centre heave acceleration (acc_center_amp, m/s^2) over the
wave height H and period T grid, one panel per config, 12-buoy vs 16-buoy side by side.

Configs (rows): no-fin (+cap), 0.15 m fin (Cd 5 / 1), 0.215 m fin (Cd 5 / 1).
Colour scale is shared within a row (12 vs 16 directly comparable); it differs between
rows because the fin lowers the whole level. Reads the fan summary CSVs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_R = Path("studies")
_OUT = _R / "platform-16buoy/fin_study/platform_accel_HT_maps.png"
_12 = _R / "platform-12buoy/fin_study"
_16 = _R / "platform-16buoy/fin_study"

# (row title, cfg key)
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
    fig, axes = plt.subplots(len(_CONFIGS), 2, figsize=(11.5, 16.5))
    for row, (title, cfg) in enumerate(_CONFIGS):
        data = {}
        for col, (mdl, folder, pfx) in enumerate(
            [("12-buoy", _12, "rao_summary_platform_fin"),
             ("16-buoy", _16, "rao_summary_platform16_fin")]):
            p = folder / f"{pfx}{cfg}.csv"
            data[col] = _load(p) if p.exists() else None
        vmax = max(np.nanmax(d[2]) for d in data.values() if d is not None)
        for col, mdl in enumerate(("12-buoy", "16-buoy")):
            ax = axes[row, col]
            if data[col] is None:
                ax.set_axis_off(); continue
            Hs, Ts, Z = data[col]
            pcm = ax.pcolormesh(Ts, Hs, Z, shading="gouraud", cmap="inferno", vmin=0, vmax=vmax)
            cs = ax.contour(Ts, Hs, Z, colors="white", linewidths=0.5, alpha=0.55)
            ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
            # mark the peak
            ip = np.unravel_index(np.nanargmax(Z), Z.shape)
            ax.plot(Ts[ip[1]], Hs[ip[0]], "*", color="#39cdd6", ms=13, mec="k", mew=0.6)
            ax.set_title(f"{mdl} · {title}   (peak {np.nanmax(Z):.2f} m/s² @ "
                         f"T={Ts[ip[1]]:.2f}s, H={Hs[ip[0]]:.2f}m)", fontsize=8.5)
            ax.set_xlabel("wave period T (s)", fontsize=8)
            ax.set_ylabel("wave height H (m)", fontsize=8)
            ax.tick_params(labelsize=7)
            cb = fig.colorbar(pcm, ax=ax, pad=0.02)
            cb.set_label("platform accel (m/s²)", fontsize=7)
            cb.ax.tick_params(labelsize=6)
    fig.suptitle("Platform-centre heave acceleration vs wave height & period — 12-buoy vs 16-buoy\n"
                 "one panel per fin/Cd config · colour scale shared within each row (12 vs 16 "
                 "comparable) · ★ = peak", fontsize=11, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(_OUT, dpi=135)
    plt.close(fig)
    print("wrote", _OUT)


if __name__ == "__main__":
    main()
