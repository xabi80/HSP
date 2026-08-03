"""Fin-sensitivity plots for the single buoy: heave RAO and vertical (Nz)
acceleration vs wave period for fin radius {0.215, 0.15, none}, at Cd_n 5 and 1.
RAO shown at H=0.04 m (peak-revealing, least damped); accel at H=0.12 m (accel
peaks at large amplitude). No-fin has no plate (spar-only, amplitude-independent)
and diverges near its 2.31 s resonance -- plotted with the gap marked."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_D = Path(__file__).resolve().parent
_TN = {"0215": 2.99, "015": 2.48, "none": 2.31}
_COL = {"0215": "#d62728", "015": "#ff7f0e", "none": "#7f7f7f"}
_LBL = {"0215": "fin 0.215 m", "015": "fin 0.15 m", "none": "no fin (+spar bottom-cap)"}


def row_at(cfg: str, key: str, h: float):
    rows = list(csv.DictReader((_D / f"rao_summary_{cfg}.csv").open()))
    sub = [r for r in rows if abs(float(r["height_m"]) - h) < 1e-9]
    sub.sort(key=lambda r: float(r["period_s"]))
    T = np.array([float(r["period_s"]) for r in sub])
    v = np.array([float(r[key]) for r in sub])
    return T, v


def panel(ax, kind, cd, h):  # type: ignore[no-untyped-def]
    key = "rao_center" if kind == "rao" else "acc_center_amp"
    for fin in ("0215", "015"):
        T, v = row_at(f"fin{fin}_Cd{cd}", key, h)
        ax.plot(T, v, "-o", ms=4, color=_COL[fin], label=_LBL[fin])
        ax.axvline(_TN[fin], color=_COL[fin], ls=":", lw=0.9, alpha=0.6)
    # no-fin: uses the spar's flat-bottom form drag (bottom-cap plate at r=R_spar)
    # so it converges; the PURE no-fin idealization (zero heave damping) diverges.
    T, v = row_at("finnone_cap", key, h)
    ax.plot(T, v, "-o", ms=4, color=_COL["none"], label=_LBL["none"])
    ax.axvline(_TN["none"], color=_COL["none"], ls=":", lw=0.9, alpha=0.6)
    if np.isnan(v).any():
        tg = T[np.isnan(v)]
        ax.axvspan(tg.min() - 0.1, tg.max() + 0.1, color="gray", alpha=0.12)
        ax.text(tg.mean(), ax.get_ylim()[1] * 0.5, "no-fin\ndiverges\n(undamped)",
                ha="center", va="center", fontsize=7, color="gray")
    if kind == "rao":
        ax.axhline(1.0, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.set_title(f"{'heave RAO' if kind=='rao' else 'Nz accel amp (m/s^2)'} "
                 f"-- Cd_n={cd} (H={h:g} m)", fontsize=10)
    ax.set_xlabel("wave period T (s)")
    ax.set_ylabel("RAO" if kind == "rao" else "Nz accel amp (m/s^2)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)


fig, ax = plt.subplots(2, 2, figsize=(13, 9))
panel(ax[0, 0], "rao", "5", 0.04)
panel(ax[0, 1], "rao", "1", 0.04)
panel(ax[1, 0], "acc", "5", 0.12)
panel(ax[1, 1], "acc", "1", 0.12)
fig.suptitle("Single buoy: fin-size sensitivity of heave RAO (top, H=0.04 m) and Nz "
             "acceleration (bottom, H=0.12 m)\nrigorous BEM per fin size; dotted = each fin's "
             "heave natural period; no-fin includes the spar's small bottom-cap drag "
             "(pure no-fin is undamped)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(_D / "fin_sensitivity.png", dpi=140)
print("wrote fin_sensitivity.png")

# peak-vs-fin summary
print("\npeak (over the H,T grid) per fin per Cd:")
fins = [("0.215", "0215"), ("0.15", "015")]
for cd in ("5", "1"):
    for flab, ftag in fins:
        rows = list(csv.DictReader((_D / f"rao_summary_fin{ftag}_Cd{cd}.csv").open()))
        pr = max(float(r["rao_center"]) for r in rows)
        pa = max(float(r["acc_center_amp"]) for r in rows)
        print(f"  Cd{cd} fin {flab}: peak RAO {pr:.2f}, peak Nz-accel {pa:.3f} m/s^2")
rows = list(csv.DictReader((_D / "rao_summary_finnone_cap.csv").open()))
pr = max(float(r["rao_center"]) for r in rows)
pa = max(float(r["acc_center_amp"]) for r in rows)
print(f"  no-fin (+spar bottom-cap drag): peak RAO {pr:.2f}, peak Nz-accel {pa:.3f} m/s^2 "
      f"(pure no-fin -- zero heave damping -- diverges at resonance)")
