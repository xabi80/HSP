"""Plots for the 16-buoy platform fin study + the 1/3/12/16 cross-model
comparison. Reads rao_summary_platform16_fin*.csv. Two figures:

  platform16_fin_sensitivity.png -- buoy-7 heave RAO and Nz-acceleration vs wave
    period for each fin (Cd_n 5 and 1), at the peak (lowest) wave height, with
    each fin's resonance marked; shows where the peak sits on the grid.
  fin_single_vs_cluster_vs_12_vs_16.png -- peak buoy heave RAO and Nz-accel per
    fin across the four models (single / 3-cluster / 12-platform / 16-platform),
    Cd_n=5, using the documented 1/3/12 peaks (FIN-SENSITIVITY.md) + the new 16.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_STUDY = _HERE / "fin_study"
_LBL = {"0215": "fin 0.215 m", "015": "fin 0.15 m", "none": "no fin (+cap)"}
_COL = {"0215": "#0c8b96", "015": "#c9772b", "none": "#d1543a"}

# Documented peaks from FIN-SENSITIVITY.md (Cd_n=5): single / 3-cluster / 12-platform.
_DOC_RAO = {"0215": [1.62, 1.73, 1.84], "015": [1.97, 2.23, 2.37], "none": [3.30, 3.81, 4.27]}
_DOC_ACC = {"0215": [0.25, 0.24, 0.25], "015": [0.44, 0.44, 0.47], "none": [0.84, 0.86, 0.92]}


def _load(cfg: str):  # type: ignore[no-untyped-def]
    rows = list(csv.DictReader((_STUDY / f"rao_summary_platform16_fin{cfg}.csv").open()))
    return rows


def _peak_height_series(rows):  # type: ignore[no-untyped-def]
    """RAO_buoy & acc_buoy vs period at the height that maximises peak RAO."""
    pk = max(rows, key=lambda r: float(r["rao_buoy"]))
    H = pk["height_m"]
    hr = sorted((r for r in rows if r["height_m"] == H), key=lambda r: float(r["period_s"]))
    T = [float(r["period_s"]) for r in hr]
    rao = [float(r["rao_buoy"]) for r in hr]
    acc = [float(r["acc_buoy_amp"]) for r in hr]
    return H, T, rao, acc


def _peak(cfg: str, key: str) -> float:
    return max(float(r[key]) for r in _load(cfg))


def sensitivity_fig() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for col, cd in enumerate(("Cd5", "Cd1")):
        for fin in ("0215", "015", "none"):
            cfg = f"{fin}_cap" if fin == "none" else f"{fin}_{cd}"
            if not (_STUDY / f"rao_summary_platform16_fin{cfg}.csv").exists():
                continue
            H, T, rao, acc = _peak_height_series(_load(cfg))
            ip = int(np.argmax(rao))
            for row, y, lab in ((0, rao, "RAO"), (1, acc, "Nz")):
                ax = axes[row, col]
                ax.plot(T, y, "o-", color=_COL[fin], lw=1.8, ms=4,
                        label=f"{_LBL[fin]} (H={H} m)")
                ax.plot(T[ip] if row == 0 else T[int(np.argmax(acc))],
                        max(rao) if row == 0 else max(acc), "*", color=_COL[fin], ms=13)
        axes[0, col].set_title(f"buoy-7 heave RAO -- Cd_n={cd[2:]}", fontsize=10)
        axes[1, col].set_title(f"buoy-7 Nz acceleration -- Cd_n={cd[2:]}", fontsize=10)
        axes[1, col].set_xlabel("wave period T (s)")
    axes[0, 0].set_ylabel("RAO")
    axes[1, 0].set_ylabel("Nz accel (m/s^2)")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].axhline(1.0, color="r", ls="--", lw=0.8, alpha=0.6)
    fig.suptitle("16-buoy platform (4 clusters x 4 buoys) -- fin sensitivity: buoy-7 heave RAO + "
                 "Nz accel vs period at the peak height\n(* = peak; grid extended to 3.8 s so the "
                 "0.215-fin resonance is captured)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = _STUDY / "platform16_fin_sensitivity.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out.name}", flush=True)


def cross_model_fig() -> None:
    models = ["single", "3-cluster", "12-buoy", "16-buoy"]
    x = np.arange(len(models))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for fin in ("0215", "015", "none"):
        cfg = f"{fin}_cap" if fin == "none" else f"{fin}_Cd5"
        rao16 = _peak(cfg, "rao_buoy")
        acc16 = _peak(cfg, "acc_buoy_amp")
        a1.plot(x, [*_DOC_RAO[fin], rao16], "o-", color=_COL[fin], lw=2, ms=7, label=_LBL[fin])
        a2.plot(x, [*_DOC_ACC[fin], acc16], "o-", color=_COL[fin], lw=2, ms=7, label=_LBL[fin])
    for ax, ttl, yl in ((a1, "peak buoy heave RAO", "RAO"),
                        (a2, "peak buoy Nz acceleration", "m/s^2")):
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_title(ttl, fontsize=11)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    a1.axhline(1.0, color="r", ls="--", lw=0.8, alpha=0.6)
    fig.suptitle("Cross-model peak buoy response (Cd_n=5): single / 3-cluster / 12-buoy / 16-buoy "
                 "platform\n1/3/12 from FIN-SENSITIVITY.md; 16 this study", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = _STUDY / "fin_single_vs_cluster_vs_12_vs_16.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out.name}", flush=True)


def main() -> None:
    sensitivity_fig()
    cross_model_fig()
    print("\nPeak buoy response (16-buoy):")
    for fin in ("0215", "015", "none"):
        for cd in (("Cd5", "Cd1") if fin != "none" else ("cap",)):
            cfg = f"{fin}_cap" if fin == "none" else f"{fin}_{cd}"
            if (_STUDY / f"rao_summary_platform16_fin{cfg}.csv").exists():
                print(f"  {cfg:10} peak RAO {_peak(cfg, 'rao_buoy'):.2f}  "
                      f"peak Nz {_peak(cfg, 'acc_buoy_amp'):.3f} m/s^2")


if __name__ == "__main__":
    main()
