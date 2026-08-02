"""M11b PR8 follow-up: 3D surfaces of heave RAO and vertical (Nz) acceleration
amplitude vs (wave height, wave period), comparing the operational plate
Cd_n=5.0 fan (pr8_fan_out/) with the light Cd_n=1.0 fan (pr8_fan_cd1_out/).

For EACH of the three recorded acceleration buoys (buoy1 cluster A, buoy4
cluster B, buoy7 cluster C) and the platform centre, two figures are written:
RAO(H,T) and Nz-accel(H,T), each a 2x2 grid {location} x {Cd}, z-scale shared
per location so the drag contrast is honest.

RAO from rao_summary.csv; Nz-accel amplitude = 0.5*(max-min) of the heave-accel
channel over each case's steady window (the whole case CSV is that window)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_BASE = Path(__file__).resolve().parent
_CD5 = _BASE / "pr8_fan_out"
_CD1 = _BASE / "pr8_fan_cd1_out"
_OUTDIR = _BASE / "pr8_fan_cd_compare"
_TN = 3.1435  # undamped M/K heave natural period (Cd-invariant)

# (buoy id, accel-channel prefix in case CSVs, RAO column in summary, label)
_BUOYS = [
    (1, "buoy1_clusterA", "rao_buoy1_heave", "buoy 1 (cluster A)"),
    (4, "buoy4_clusterB", "rao_buoy4_heave", "buoy 4 (cluster B)"),
    (7, "buoy7_clusterC", "rao_buoy7_heave", "buoy 7 (cluster C)"),
]


def _tag(h: float, p: float) -> str:
    return f"H{h:g}_T{p:g}".replace(".", "p")


def _accel_amp(cols: dict[str, np.ndarray], name: str) -> float:
    v = cols[name]
    return 0.5 * (v.max() - v.min())


def load(fan: Path) -> dict:
    summ = list(csv.DictReader((fan / "rao_summary.csv").open()))
    H = sorted({float(r["height_m"]) for r in summ})
    T = sorted({float(r["period_s"]) for r in summ})
    shape = (len(H), len(T))
    rao_p = np.full(shape, np.nan)
    acc_p = np.full(shape, np.nan)
    rao_b = {bid: np.full(shape, np.nan) for bid, *_ in _BUOYS}
    acc_b = {bid: np.full(shape, np.nan) for bid, *_ in _BUOYS}
    for r in summ:
        i, j = H.index(float(r["height_m"])), T.index(float(r["period_s"]))
        rao_p[i, j] = float(r["rao_platform_heave"])
        for bid, _chan, raocol, _lab in _BUOYS:
            rao_b[bid][i, j] = float(r[raocol])
        crows = list(csv.DictReader(
            (fan / f"case_{_tag(float(r['height_m']), float(r['period_s']))}.csv").open()))
        cols = {k: np.array([float(rr[k]) for rr in crows]) for k in crows[0]}
        acc_p[i, j] = _accel_amp(cols, "platform_heave_acc_mps2")
        for bid, chan, _raocol, _lab in _BUOYS:
            acc_b[bid][i, j] = _accel_amp(cols, f"{chan}_heave_acc_mps2")
    return {"H": H, "T": T, "rao_p": rao_p, "acc_p": acc_p, "rao_b": rao_b, "acc_b": acc_b}


def _surface(ax, T, H, Z, zlabel, title, zlim, rao=False):
    Tg, Hg = np.meshgrid(T, H)
    ax.plot_surface(Tg, Hg, Z, cmap="viridis", vmin=zlim[0], vmax=zlim[1],
                    edgecolor="k", linewidth=0.25, rstride=1, cstride=1,
                    antialiased=True, alpha=0.95)
    ax.scatter(Tg.ravel(), Hg.ravel(), Z.ravel(), color="crimson", s=14, depthshade=False)
    ax.plot([_TN, _TN], [min(H), max(H)], [zlim[0], zlim[0]], color="gray", ls=":", lw=1.3)
    if rao:
        ax.plot([min(T), max(T)], [max(H), max(H)], [1.0, 1.0], "r--", lw=1.0, alpha=0.7)
    ax.set_xlabel("period T (s)", fontsize=8, labelpad=1)
    ax.set_ylabel("height H (m)", fontsize=8, labelpad=1)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=2)
    ax.set_zlim(*zlim)
    ax.set_title(title, fontsize=10, pad=2)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=24, azim=-58)


def make_fig(d5: dict, d1: dict, kind: str, buoy: tuple, out: Path) -> None:
    bid, _chan, _raocol, blabel = buoy
    if kind == "rao":
        pkey, bkey_of = "rao_p", "rao_b"
        zlabel, unit = "heave RAO (= |z| / (H/2))", ""
        suptitle = (f"12-buoy platform: heave RAO(H, T) -- platform centre & {blabel}\n"
                    "Cd_n=5.0 (operational) vs Cd_n=1.0 (light drag)")
    else:
        pkey, bkey_of = "acc_p", "acc_b"
        zlabel, unit = "Nz accel amp (m/s^2)", " m/s^2"
        suptitle = (f"12-buoy platform: vertical (Nz) acceleration amplitude(H, T) -- "
                    f"platform centre & {blabel}\nCd_n=5.0 vs Cd_n=1.0")
    rows = [("platform centre", d5[pkey], d1[pkey]),
            (blabel, d5[bkey_of][bid], d1[bkey_of][bid])]
    fig = plt.figure(figsize=(13, 10))
    for ri, (loc, Z5, Z1) in enumerate(rows):
        zmax = max(np.nanmax(Z5), np.nanmax(Z1))
        zlim = (0.0, 1.05 * zmax)
        for ci, (d, Z, cd) in enumerate([(d5, Z5, "5.0"), (d1, Z1, "1.0")]):
            ax = fig.add_subplot(2, 2, 2 * ri + ci + 1, projection="3d")
            ij = np.unravel_index(np.nanargmax(Z), Z.shape)
            title = (f"{loc} -- Cd_n={cd}\npeak {float(np.nanmax(Z)):.3g}{unit} @ "
                     f"T={d['T'][ij[1]]:.3g}s, H={d['H'][ij[0]]:.2g}m")
            _surface(ax, d["T"], d["H"], Z, zlabel, title, zlim, rao=(kind == "rao"))
    fig.suptitle(suptitle, fontsize=13, y=0.99)
    fig.text(0.5, 0.02, "gray dotted line = 3.14 s undamped natural period (Cd-invariant)"
             + ("  |  red dashed = RAO 1" if kind == "rao" else ""),
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=140)
    print(f"wrote {out.name}")


def main() -> None:
    _OUTDIR.mkdir(exist_ok=True)
    # remove any stale un-suffixed figures from the earlier single-buoy run
    for stale in ("rao_Cd5_vs_Cd1.png", "nz_accel_Cd5_vs_Cd1.png"):
        (_OUTDIR / stale).unlink(missing_ok=True)
    d5, d1 = load(_CD5), load(_CD1)
    for buoy in _BUOYS:
        bid = buoy[0]
        make_fig(d5, d1, "rao", buoy, _OUTDIR / f"rao_Cd5_vs_Cd1_buoy{bid}.png")
        make_fig(d5, d1, "acc", buoy, _OUTDIR / f"nz_accel_Cd5_vs_Cd1_buoy{bid}.png")
    print("\npeak table (T s, H m):")
    for name, d in [("Cd=5.0", d5), ("Cd=1.0", d1)]:
        print(f"  {name}:")
        entries = [("RAO platform", d["rao_p"]), ("Nz-acc platform", d["acc_p"])]
        for bid, _c, _r, lab in _BUOYS:
            entries += [(f"RAO {lab}", d["rao_b"][bid]), (f"Nz-acc {lab}", d["acc_b"][bid])]
        for lab, Z in entries:
            ij = np.unravel_index(np.nanargmax(Z), Z.shape)
            print(f"    {lab:22s} {float(np.nanmax(Z)):8.4f} @ T={d['T'][ij[1]]:.3f} H={d['H'][ij[0]]:.2f}")


if __name__ == "__main__":
    main()
