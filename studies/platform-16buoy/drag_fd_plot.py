"""Figure for the drag-limited FD resolution of the 12-vs-16 buoy RAO reversal.
Numbers are the validated drag_fd.py outputs (T=2.50s, H=0.04, no-fin, Cd_n=5),
hardcoded here as documented results (same convention as platform16_fin_plots.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_OUT = Path(__file__).resolve().parent / "fin_study" / "drag_fd_validation.png"

# validated drag_fd.py results
RAD = {12: 53.9, 16: 24.9}          # radiation-only FD (drag off)
DRAG = {12: 4.273, 16: 2.798}       # equivalent-linearized-drag FD
MEAS = {12: 4.27, 16: 2.80}         # nonlinear time-domain fan
PLAT = {12: 3.139, 16: 3.107}       # platform heave RAO (drag-limited)
EXC = {12: 1.628, 16: 1.648}        # per-buoy wave excitation
SUMB = {12: 100.3, 16: 132.9}       # total linearized drag damping


def main() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(2)
    lbl = ["12-buoy", "16-buoy"]

    # Panel 1: radiation-only vs drag-limited FD vs measured (log y -- rad is ~20x)
    w = 0.26
    for k, (d, col, name) in enumerate([
        (RAD, "#9aa7b1", "radiation-only FD"),
        (DRAG, "#0c8b96", "drag-limited FD"),
        (MEAS, "#d1543a", "measured (fan)"),
    ]):
        a1.bar(x + (k - 1) * w, [d[12], d[16]], w, color=col, label=name)
    a1.set_yscale("log")
    a1.set_xticks(x); a1.set_xticklabels(lbl)
    a1.set_ylabel("buoy heave RAO (log)")
    a1.set_title("Drag-limited FD reproduces the measured RAO to <0.1%\n"
                 "(radiation-only is ~13-20x too high)", fontsize=10)
    a1.legend(fontsize=9); a1.grid(True, axis="y", alpha=0.3)
    for xi, N in zip(x, (12, 16)):
        a1.annotate(f"{DRAG[N]:.2f}≈{MEAS[N]:.2f}", (xi, MEAS[N]), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8, color="#d1543a")

    # Panel 2: decomposition -- buoy vs platform, ratio flip
    a2.bar(x - 0.2, [DRAG[12], DRAG[16]], 0.38, color="#0c8b96", label="buoy heave RAO")
    a2.bar(x + 0.2, [PLAT[12], PLAT[16]], 0.38, color="#c9772b", label="platform heave RAO")
    a2.set_xticks(x); a2.set_xticklabels(lbl)
    a2.set_ylabel("heave RAO")
    a2.set_title("Same platform response; buoys lose their overshoot\n"
                 "buoy/platform: 1.36 (12) -> 0.90 (16)", fontsize=10)
    a2.legend(fontsize=9); a2.grid(True, axis="y", alpha=0.3)
    for xi, N in zip(x, (12, 16)):
        a2.annotate(f"ratio {DRAG[N] / PLAT[N]:.2f}", (xi, max(DRAG[N], PLAT[N])),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
        a2.annotate(f"exc/buoy {EXC[N]:.2f}\nΣdrag {SUMB[N]:.0f}", (xi, 0.15),
                    ha="center", fontsize=8, color="#555")

    fig.suptitle("No-fin buoy heave RAO reversal (12 -> 16 buoy), resolved by the "
                 "equivalent-linearized-drag FD (drag_fd.py)\nT=2.50s, H=0.04, Cd_n=5: "
                 "drag-limited; excitation & platform unchanged; the denser cluster damps the "
                 "buoys' articulation overshoot", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(_OUT, dpi=140)
    plt.close(fig)
    print(f"wrote {_OUT.name}")


if __name__ == "__main__":
    main()
