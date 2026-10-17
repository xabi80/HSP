"""Effect of a mass-shift tilt on the dynamics in waves (tilted_dynamics.json, tilted_cases.json).

Rows: the cluster (12.5 kg on the hub) and the platform (50 kg on the deck). Lines: unloaded,
payload centred (level), payload shifted to 5 deg and 10 deg of static tilt. Regular waves,
H = 0.1 m, along the shift. Columns, against the wave period:
(a) heave amplitude of the hub / deck centre;
(b) pitch amplitude of the hub / deck;
(c) smallest pin freeboard during the wave cycle. Dashed lines: the still-water freeboard. A
    pin is awash where the curve reaches 0;
(d) largest gimbal angle (a buoy rocking relative to its hub). FloatSim leaves out the pin load's
    tipping of the buoys (P L), so on the loaded side this column is optimistic.

Writes tilted_dynamics.png. Run: python tilted_dynamics_plot.py
"""

# ruff: noqa: E702  -- compact figure-assembly lines
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
STYLE = {"unloaded": ("#8a959c", "unloaded", "o"),
         "centred": ("#0c8b96", "payload centred (level)", "s"),
         "tilt5": ("#c9772b", "shifted: 5° tilt", "^"),
         "tilt10": ("#b2432c", "shifted: 10° tilt", "D")}
ROWS = {"cluster": "Cluster, 12.5 kg on the hub", "platform": "Platform, 50 kg on the deck"}


def main() -> None:
    data = json.loads((HERE / "tilted_dynamics.json").read_text(encoding="utf-8"))
    arts = [a for a in ROWS if any(d["article"] == a for d in data)]
    fig, axs = plt.subplots(len(arts), 4, figsize=(19, 4.6 * len(arts)), squeeze=False)
    for i, art in enumerate(arts):
        for name, (col, lab, mk) in STYLE.items():
            d = next((x for x in data if x["article"] == art and x["case"] == name), None)
            if d is None:
                continue
            T = np.array([r["T_s"] for r in d["runs"]])
            kw = {"color": col, "marker": mk, "ms": 5, "lw": 1.8, "label": lab}
            axs[i, 0].plot(T, [1e3 * r["heave_m"] for r in d["runs"]], **kw)
            axs[i, 1].plot(T, [r["pitch_deg"] for r in d["runs"]], **kw)
            axs[i, 2].plot(T, [1e3 * r["min_freeboard_m"] for r in d["runs"]], **kw)
            axs[i, 2].axhline(1e3 * min(d["static_freeboard_m"]), color=col, lw=0.9, ls="--")
            axs[i, 3].plot(T, [r["gimbal_deg_max"] for r in d["runs"]], **kw)
            p = d["periods"]
            for ax in axs[i, :2]:
                ax.axvline(p["heave"]["T_s"] if ax is axs[i, 0] else p["pitch"]["T_s"],
                           color=col, lw=0.6, ls=":")
        what = "hub" if art == "cluster" else "deck"
        axs[i, 0].set_ylabel(f"{ROWS[art]}\n\n{what} heave amplitude (mm)")
        axs[i, 1].set_ylabel(f"{what} pitch amplitude (°)")
        axs[i, 2].set_ylabel("smallest pin freeboard in the wave (mm)")
        axs[i, 3].set_ylabel("largest gimbal angle (°)")
        axs[i, 2].axhline(0.0, color="k", lw=1.0)
        for ax in axs[i]:
            ax.grid(alpha=0.3); ax.set_xlim(1.3, 5.2)
        axs[i, 0].legend(fontsize=8.5)
    for j, t in enumerate(("(a) heave of the hub / deck centre", "(b) pitch of the hub / deck",
                           "(c) smallest pin freeboard (dashed: still water)",
                           "(d) buoy rocking on its gimbal (no P·L: optimistic)")):
        axs[0, j].set_title(t, fontsize=10.5)
        axs[-1, j].set_xlabel("wave period (s)")
    fig.suptitle("Mass-shift tilt in regular waves, H = 0.1 m along the shift — FloatSim at each "
                 "static equilibrium (BEM re-run at the tilted geometry). Dotted: natural periods.",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "tilted_dynamics.png", dpi=125, bbox_inches="tight")
    print("wrote tilted_dynamics.png")


if __name__ == "__main__":
    main()
