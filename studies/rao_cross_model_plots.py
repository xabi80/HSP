"""Cross-model heave RAO + vertical (Nz) acceleration comparison at the corrected
model-scale heights (0.04..0.12 m). Reads the three uniform-schema fans:

  single   spar-fin-decay/sparfin_rao_out/
  cluster  cluster-3buoy-rigid/cluster_rao_out/
  platform platform-12buoy/platform_rao_lowh_out/

Each rao_summary_Cd{5,1}.csv has columns height_m, period_s, rao_center,
acc_center_amp, rao_buoy, acc_buoy_amp (+ peak/settle). "center" = platform ref
/ cluster hub / the buoy; "buoy" = a representative buoy (platform buoy7, cluster
buoy1, single = the buoy).

Outputs (studies/rao_cross_model_out/):
  <model>_rao.png / <model>_nz_accel.png -- per-model Cd5-vs-Cd1 surfaces
  cross_model_rao.png / cross_model_nz_accel.png -- 1-vs-3-vs-12 overlays
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_BASE = Path(__file__).resolve().parent
_OUT = _BASE / "rao_cross_model_out"
# (key, label, out_dir, T_n, has_distinct_center)
_SD = _BASE / "spar-fin-decay" / "sparfin_rao_out"
_CLD = _BASE / "cluster-3buoy-rigid" / "cluster_rao_out"
_PLD = _BASE / "platform-12buoy" / "platform_rao_lowh_out"
_MODELS = [
    ("single", "1 buoy", _SD, 2.966, False),
    ("cluster", "3-buoy cluster", _CLD, 3.106, True),
    ("platform", "12-buoy platform", _PLD, 3.143, True),
]
_CDS = [("5", "Cd_n=5.0 (operational)"), ("1", "Cd_n=1.0 (light drag)")]


def load(out_dir: Path, cd: str) -> dict:
    rows = list(csv.DictReader((out_dir / f"rao_summary_Cd{cd}.csv").open()))
    H = sorted({float(r["height_m"]) for r in rows})
    T = sorted({float(r["period_s"]) for r in rows})
    g = {
        k: np.full((len(H), len(T)), np.nan)
        for k in ("rao_center", "acc_center_amp", "rao_buoy", "acc_buoy_amp")
    }
    for r in rows:
        i, j = H.index(float(r["height_m"])), T.index(float(r["period_s"]))
        for k in g:
            g[k][i, j] = float(r[k])
    return {"H": H, "T": T, **g}


def _surf(ax, T, H, Z, zlabel, title, zlim, tn, rao):  # type: ignore[no-untyped-def]
    Tg, Hg = np.meshgrid(T, H)
    ax.plot_surface(
        Tg,
        Hg,
        Z,
        cmap="viridis",
        vmin=zlim[0],
        vmax=zlim[1],
        edgecolor="k",
        linewidth=0.25,
        rstride=1,
        cstride=1,
        antialiased=True,
        alpha=0.95,
    )
    ax.scatter(Tg.ravel(), Hg.ravel(), Z.ravel(), color="crimson", s=12, depthshade=False)
    ax.plot([tn, tn], [min(H), max(H)], [zlim[0], zlim[0]], color="gray", ls=":", lw=1.3)
    if rao:
        ax.plot([min(T), max(T)], [max(H), max(H)], [1.0, 1.0], "r--", lw=1.0, alpha=0.7)
    ax.set_xlabel("T (s)", fontsize=8, labelpad=1)
    ax.set_ylabel("H (m)", fontsize=8, labelpad=1)
    ax.set_zlabel(zlabel, fontsize=8, labelpad=2)
    ax.set_zlim(*zlim)
    ax.set_title(title, fontsize=9, pad=1)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=24, azim=-58)


def per_model_fig(key, label, out_dir, tn, has_ctr, kind):  # type: ignore[no-untyped-def]
    d = {cd: load(out_dir, cd) for cd, _ in _CDS}
    rao = kind == "rao"
    ck = "rao_center" if rao else "acc_center_amp"
    bk = "rao_buoy" if rao else "acc_buoy_amp"
    zlab = "heave RAO" if rao else "Nz accel amp (m/s^2)"
    unit = "" if rao else " m/s^2"
    locs = [("centre", ck), ("buoy", bk)] if has_ctr else [("buoy", bk)]
    nrow = len(locs)
    fig = plt.figure(figsize=(12, 4.7 * nrow))
    for ri, (locname, gk) in enumerate(locs):
        zmax = max(np.nanmax(d[cd][gk]) for cd, _ in _CDS)
        zlim = (0.0, 1.05 * zmax)
        for ci, (cd, cdlabel) in enumerate(_CDS):
            ax = fig.add_subplot(nrow, 2, 2 * ri + ci + 1, projection="3d")
            Z = d[cd][gk]
            ij = np.unravel_index(np.nanargmax(Z), Z.shape)
            ttl = (
                f"{locname} -- {cdlabel}\npeak {float(np.nanmax(Z)):.3g}{unit} @ "
                f"T={d[cd]['T'][ij[1]]:.3g}s H={d[cd]['H'][ij[0]]:.2g}m"
            )
            _surf(ax, d[cd]["T"], d[cd]["H"], Z, zlab, ttl, zlim, tn, rao)
    kindname = "heave RAO(H,T)" if rao else "vertical (Nz) acceleration amplitude(H,T)"
    fig.suptitle(
        f"{label}: {kindname} -- Cd_n=5.0 vs 1.0   (heights 0.04-0.12 m = 2-6 m "
        f"full-scale @1:50; T_n={tn:.2f} s)",
        fontsize=12,
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "gray dotted = heave natural period" + ("  |  red dashed = RAO 1" if rao else ""),
        ha="center",
        fontsize=8,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = _OUT / f"{key}_{'rao' if rao else 'nz_accel'}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out.name}")


def cross_model_fig(kind, h_slice=0.08):  # type: ignore[no-untyped-def]
    rao = kind == "rao"
    bk = "rao_buoy" if rao else "acc_buoy_amp"
    ylab = "heave RAO (buoy)" if rao else "buoy Nz accel amp (m/s^2)"
    colors = {"single": "#1f77b4", "cluster": "#2ca02c", "platform": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (cd, cdlabel) in zip(axes, _CDS, strict=False):
        for _key, label, out_dir, tn, _hc in _MODELS:
            d = load(out_dir, cd)
            i = d["H"].index(min(d["H"], key=lambda x: abs(x - h_slice)))
            T = np.array(d["T"])
            ax.plot(
                T,
                d[bk][i, :],
                "-o",
                color=colors[_key],
                ms=4,
                lw=1.8,
                label=f"{label} (T_n={tn:.2f}s)",
            )
            ax.axvline(tn, color=colors[_key], ls=":", lw=1.0, alpha=0.6)
        if rao:
            ax.axhline(1.0, color="gray", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(f"{cdlabel}", fontsize=11)
        ax.set_xlabel("wave period T (s)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    kindname = "buoy heave RAO" if rao else "buoy vertical (Nz) acceleration amplitude"
    fig.suptitle(
        f"1-vs-3-vs-12 {kindname} vs period at H={h_slice:g} m "
        f"(= {h_slice*50:.0f} m full-scale @1:50)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = _OUT / f"cross_model_{'rao' if rao else 'nz_accel'}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}")


def main() -> None:
    _OUT.mkdir(exist_ok=True)
    for key, label, _o, tn, hc in _MODELS:
        per_model_fig(key, label, _o, tn, hc, "rao")
        per_model_fig(key, label, _o, tn, hc, "acc")
    cross_model_fig("rao")
    cross_model_fig("acc")
    # numeric summary
    print("\npeak buoy RAO / peak buoy Nz-accel (m/s^2) per model per Cd:")
    for _key, label, out_dir, _tn, _hc in _MODELS:
        line = f"  {label:18s}"
        for cd, _ in _CDS:
            d = load(out_dir, cd)
            pr = float(np.nanmax(d["rao_buoy"]))
            pa = float(np.nanmax(d["acc_buoy_amp"]))
            line += f"  | Cd{cd}: RAO {pr:5.2f}  acc {pa:5.3f}"
        print(line)


if __name__ == "__main__":
    main()
