"""Cluster fin-size sensitivity plots + a single-vs-cluster cross-model summary.
Reads the cluster fin summaries (this dir) and the single-buoy fin summaries
(../../spar-fin-decay/fin_study/). Writes cluster_fin_sensitivity.png and
fin_single_vs_cluster.png."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_CL = Path(__file__).resolve().parent
_SG = _CL.parent.parent / "spar-fin-decay" / "fin_study"
_TN_CL = {"0215": 3.12, "015": 2.63, "none": 2.47}
_TN_SG = {"0215": 2.99, "015": 2.48, "none": 2.31}
_COL = {"0215": "#d62728", "015": "#ff7f0e", "none": "#7f7f7f"}
_LBL = {"0215": "fin 0.215 m", "015": "fin 0.15 m", "none": "no fin (+bottom-cap)"}


def row_at(d: Path, prefix: str, cfg: str, key: str, h: float):
    rows = list(csv.DictReader((d / f"rao_summary_{prefix}fin{cfg}.csv").open()))
    sub = sorted(
        (r for r in rows if abs(float(r["height_m"]) - h) < 1e-9),
        key=lambda r: float(r["period_s"]),
    )
    return (np.array([float(r["period_s"]) for r in sub]), np.array([float(r[key]) for r in sub]))


def _peak(d: Path, prefix: str, cfg: str, key: str) -> float:
    rows = list(csv.DictReader((d / f"rao_summary_{prefix}fin{cfg}.csv").open()))
    return max(float(r[key]) for r in rows)


def cluster_panel(ax, kind, cd, h):  # type: ignore[no-untyped-def]
    key = "rao_buoy" if kind == "rao" else "acc_buoy_amp"  # buoy1 -- the payload body
    for fin in ("0215", "015"):
        T, v = row_at(_CL, "cluster_", f"{fin}_Cd{cd}", key, h)
        ax.plot(T, v, "-o", ms=4, color=_COL[fin], label=_LBL[fin])
        ax.axvline(_TN_CL[fin], color=_COL[fin], ls=":", lw=0.9, alpha=0.6)
    T, v = row_at(_CL, "cluster_", "none_cap", key, h)
    ax.plot(T, v, "-o", ms=4, color=_COL["none"], label=_LBL["none"])
    ax.axvline(_TN_CL["none"], color=_COL["none"], ls=":", lw=0.9, alpha=0.6)
    if kind == "rao":
        ax.axhline(1.0, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.set_title(
        f"{'heave RAO' if kind == 'rao' else 'Nz accel amp (m/s^2)'} " f"-- Cd_n={cd} (H={h:g} m)",
        fontsize=10,
    )
    ax.set_xlabel("wave period T (s)")
    ax.set_ylabel("RAO" if kind == "rao" else "Nz accel amp (m/s^2)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


def make_cluster_fig() -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    cluster_panel(ax[0, 0], "rao", "5", 0.04)
    cluster_panel(ax[0, 1], "rao", "1", 0.04)
    cluster_panel(ax[1, 0], "acc", "5", 0.12)
    cluster_panel(ax[1, 1], "acc", "1", 0.12)
    fig.suptitle(
        "3-buoy cluster (buoy 1 -- a payload buoy): fin-size sensitivity of heave RAO "
        "(top, H=0.04 m) and Nz acceleration (bottom, H=0.12 m)\nrigorous coupled BEM "
        "per fin; dotted = each fin's cluster heave natural period",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_CL / "cluster_fin_sensitivity.png", dpi=140)
    plt.close(fig)
    print("wrote cluster_fin_sensitivity.png")


def make_cross_model() -> None:
    fins = ["0215", "015", "none"]
    x = [0.215, 0.15, 0.084]  # no-fin plotted at the spar/bottom-cap radius
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Compare the BUOY (payload body) across models -- "center" is the buoy for
    # the single model but the hub node for the cluster, which is not comparable.
    for k, (_kind, key, ylab) in enumerate(
        [
            ("rao", "rao_buoy", "peak buoy heave RAO"),
            ("acc", "acc_buoy_amp", "peak buoy Nz accel (m/s^2)"),
        ]
    ):
        sg = [_peak(_SG, "", f"{f}_Cd5" if f != "none" else "none_cap", key) for f in fins]
        cl = [_peak(_CL, "cluster_", f"{f}_Cd5" if f != "none" else "none_cap", key) for f in fins]
        ax[k].plot(x, sg, "-o", color="#1f77b4", ms=7, label="single buoy")
        ax[k].plot(x, cl, "-s", color="#2ca02c", ms=7, label="3-buoy cluster")
        for xi, s, c in zip(x, sg, cl, strict=False):
            ax[k].annotate(
                f"{s:.2f}",
                (xi, s),
                fontsize=7,
                color="#1f77b4",
                textcoords="offset points",
                xytext=(4, 4),
            )
            ax[k].annotate(
                f"{c:.2f}",
                (xi, c),
                fontsize=7,
                color="#2ca02c",
                textcoords="offset points",
                xytext=(4, -10),
            )
        ax[k].set_xlabel("fin radius (m)  [no-fin at spar r=0.084]")
        ax[k].set_ylabel(ylab)
        ax[k].set_title(f"{ylab} vs fin size (Cd_n=5)", fontsize=10)
        ax[k].invert_xaxis()  # shrinking fin -> right
        ax[k].grid(alpha=0.25)
        ax[k].legend(fontsize=9)
    fig.suptitle(
        "Fin-size sensitivity of BUOY heave: single buoy vs 3-buoy cluster "
        "(smaller fin -> right = higher motion, both models)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_CL / "fin_single_vs_cluster.png", dpi=150)
    plt.close(fig)
    print("wrote fin_single_vs_cluster.png")
    print("\npeak BUOY (Cd5) single / cluster:")
    for f in fins:
        cfg = f"{f}_Cd5" if f != "none" else "none_cap"
        print(
            f"  {_LBL[f]:22s}: RAO {_peak(_SG, '', cfg, 'rao_buoy'):.2f}/"
            f"{_peak(_CL, 'cluster_', cfg, 'rao_buoy'):.2f}  "
            f"accel {_peak(_SG, '', cfg, 'acc_buoy_amp'):.3f}/"
            f"{_peak(_CL, 'cluster_', cfg, 'acc_buoy_amp'):.3f}"
        )


if __name__ == "__main__":
    make_cluster_fig()
    make_cross_model()
