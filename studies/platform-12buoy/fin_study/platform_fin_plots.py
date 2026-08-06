"""Platform (12-buoy) fin-size sensitivity plots + a 1-vs-3-vs-12 cross-model
summary. Reads the platform fin summaries (this dir), the cluster fin summaries
(../../cluster-3buoy-rigid/fin_study/) and the single-buoy fin summaries
(../../spar-fin-decay/fin_study/). Writes platform_fin_sensitivity.png and
fin_single_vs_cluster_vs_platform.png."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PL = Path(__file__).resolve().parent
_CL = _PL.parent.parent / "cluster-3buoy-rigid" / "fin_study"
_SG = _PL.parent.parent / "spar-fin-decay" / "fin_study"
# heave natural periods (constrained collective-heave mode), per model per fin
_TN_PL = {"0215": 3.15, "015": 2.66, "none": 2.50}  # constrained collective-heave mode (smoke)
_TN_CL = {"0215": 3.12, "015": 2.63, "none": 2.47}
_TN_SG = {"0215": 2.99, "015": 2.48, "none": 2.31}
_COL = {"0215": "#d62728", "015": "#ff7f0e", "none": "#7f7f7f"}
_LBL = {"0215": "fin 0.215 m", "015": "fin 0.15 m", "none": "no fin (+bottom-cap)"}


def _cfg(model_prefix: str, fin: str, cd: str) -> str:
    """Config token in the summary filename. Platform/cluster no-fin => '<fin>_cap';
    finned => '<fin>_Cd<cd>'. Single-buoy uses the same tokens without a prefix."""
    return f"{fin}_cap" if fin == "none" else f"{fin}_Cd{cd}"


def row_at(d: Path, prefix: str, cfg: str, key: str, h: float):  # type: ignore[no-untyped-def]
    rows = list(csv.DictReader((d / f"rao_summary_{prefix}fin{cfg}.csv").open()))
    sub = sorted(
        (r for r in rows if abs(float(r["height_m"]) - h) < 1e-9),
        key=lambda r: float(r["period_s"]),
    )
    return (np.array([float(r["period_s"]) for r in sub]), np.array([float(r[key]) for r in sub]))


def _peak(d: Path, prefix: str, cfg: str, key: str) -> float:
    rows = list(csv.DictReader((d / f"rao_summary_{prefix}fin{cfg}.csv").open()))
    return max(float(r[key]) for r in rows)


def platform_panel(ax, kind, cd, h):  # type: ignore[no-untyped-def]
    key = "rao_buoy" if kind == "rao" else "acc_buoy_amp"  # buoy7 -- the payload body
    for fin in ("0215", "015"):
        T, v = row_at(_PL, "platform_", f"{fin}_Cd{cd}", key, h)
        ax.plot(T, v, "-o", ms=4, color=_COL[fin], label=_LBL[fin])
        ax.axvline(_TN_PL[fin], color=_COL[fin], ls=":", lw=0.9, alpha=0.6)
    T, v = row_at(_PL, "platform_", "none_cap", key, h)
    ax.plot(T, v, "-o", ms=4, color=_COL["none"], label=_LBL["none"])
    ax.axvline(_TN_PL["none"], color=_COL["none"], ls=":", lw=0.9, alpha=0.6)
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


def make_platform_fig() -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    platform_panel(ax[0, 0], "rao", "5", 0.04)
    platform_panel(ax[0, 1], "rao", "1", 0.04)
    platform_panel(ax[1, 0], "acc", "5", 0.12)
    platform_panel(ax[1, 1], "acc", "1", 0.12)
    fig.suptitle(
        "12-buoy platform (buoy 7 -- a payload buoy): fin-size sensitivity of heave RAO "
        "(top, H=0.04 m) and Nz acceleration (bottom, H=0.12 m)\nrigorous coupled "
        "72-DOF BEM per fin; dotted = each fin's platform heave natural period",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_PL / "platform_fin_sensitivity.png", dpi=140)
    plt.close(fig)
    print("wrote platform_fin_sensitivity.png")


def make_cross_model() -> None:
    fins = ["0215", "015", "none"]
    x = [0.215, 0.15, 0.084]  # no-fin plotted at the spar/bottom-cap radius
    models = [
        ("single buoy", _SG, "", "#1f77b4", "-o"),
        ("3-buoy cluster", _CL, "cluster_", "#2ca02c", "-s"),
        ("12-buoy platform", _PL, "platform_", "#d62728", "-^"),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    # Compare the BUOY across all models (the payload body, same physical point
    # in every model). "center" would be the buoy for the single model but the
    # hub / platform reference node for the cluster / platform -- not comparable.
    for k, (_kind, key, ylab) in enumerate(
        [
            ("rao", "rao_buoy", "peak buoy heave RAO"),
            ("acc", "acc_buoy_amp", "peak buoy Nz accel (m/s^2)"),
        ]
    ):
        for mlabel, d, prefix, col, mk in models:
            vals = [_peak(d, prefix, _cfg(prefix, f, "5"), key) for f in fins]
            ax[k].plot(x, vals, mk, color=col, ms=7, label=mlabel)
            for xi, v in zip(x, vals, strict=True):
                ax[k].annotate(
                    f"{v:.2f}",
                    (xi, v),
                    fontsize=7,
                    color=col,
                    textcoords="offset points",
                    xytext=(4, 4),
                )
        ax[k].set_xlabel("fin radius (m)  [no-fin at spar r=0.084]")
        ax[k].set_ylabel(ylab)
        ax[k].set_title(f"{ylab} vs fin size (Cd_n=5)", fontsize=10)
        ax[k].invert_xaxis()  # shrinking fin -> right
        ax[k].grid(alpha=0.25)
        ax[k].legend(fontsize=9)
    fig.suptitle(
        "Fin-size sensitivity of BUOY heave: single buoy vs 3-buoy cluster vs "
        "12-buoy platform\n(same physical body -- a buoy -- in every model; "
        "smaller fin -> right = higher motion)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_PL / "fin_single_vs_cluster_vs_platform.png", dpi=150)
    plt.close(fig)
    print("wrote fin_single_vs_cluster_vs_platform.png")
    print("\npeak BUOY (Cd5) single / cluster / platform:")
    for f in fins:
        row = f"  {_LBL[f]:22s}:"
        for _m, d, prefix, _c, _mk in models:
            cfg = _cfg(prefix, f, "5")
            row += (
                f"  RAO {_peak(d, prefix, cfg, 'rao_buoy'):.2f}"
                f" acc {_peak(d, prefix, cfg, 'acc_buoy_amp'):.3f} |"
            )
        print(row)


if __name__ == "__main__":
    make_platform_fig()
    make_cross_model()
