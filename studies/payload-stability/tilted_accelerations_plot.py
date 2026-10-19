"""Accelerations at the payload, cluster against platform, for the mass-shift cases in regular
waves (tilted_dynamics_preload.json: FloatSim with its weight preload, H = 0.1 m along the shift).

Rows: the cluster (12.5 kg on the hub) and the platform (50 kg on the deck). Lines: unloaded,
payload centred, shifted to 5 deg and 10 deg. The point is the payload (for the unloaded article,
where a centred payload would sit). Columns, against the wave period:
(a) vertical acceleration amplitude;
(b) lateral acceleration (horizontal, along the waves) as the payload feels it, in the deck's
    own frame: the inertial acceleration along the waves, minus g times the dynamic pitch (the
    deck's tilting swings gravity along it). Dotted: the inertial part alone;
(c) the total lateral load the payload feels: (b) plus the static tilt's constant g sin(tilt)
    (0.85 m/s^2 at 5 deg, 1.70 at 10 deg).
All are first-harmonic amplitudes, in m/s^2 (right axis: g).

Writes tilted_accelerations.png. Run: python tilted_accelerations_plot.py
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
G = 9.806
STYLE = {"unloaded": ("#8a959c", "unloaded", "o"),
         "centred": ("#0c8b96", "payload centred", "s"),
         "tilt5": ("#c9772b", "shifted: 5° tilt", "^"),
         "tilt10": ("#b2432c", "shifted: 10° tilt", "D")}
ROWS = {"cluster": "Cluster, 12.5 kg on the hub", "platform": "Platform, 50 kg on the deck"}


def main() -> None:
    rows = json.loads((HERE / "tilted_dynamics_preload.json").read_text(encoding="utf-8"))
    data = {(d["article"], d["case"]): d for d in rows}
    fig, axs = plt.subplots(2, 3, figsize=(18.5, 9.6), sharey="col")
    for i, art in enumerate(ROWS):
        for name, (col, lab, mk) in STYLE.items():
            d = data[(art, name)]
            T = np.array([r["T_s"] for r in d["runs"]])
            acc = [r["acc_payload"] for r in d["runs"]]
            static = G * np.sin(np.radians(d["tilt_deg"]))
            kw = {"color": col, "marker": mk, "ms": 5, "lw": 1.9, "label": lab}
            axs[i, 0].plot(T, [a["az"] for a in acc], **kw)
            axs[i, 1].plot(T, [a["fx"] for a in acc], **kw)
            axs[i, 1].plot(T, [a["ax"] for a in acc], color=col, lw=0.9, ls=":")
            axs[i, 2].plot(T, [static + a["fx"] for a in acc], **kw)
            if static > 0:
                axs[i, 2].axhline(static, color=col, lw=0.8, ls="--")
        axs[i, 0].set_ylabel(f"{ROWS[art]}\n\nacceleration amplitude (m/s²)")
        for ax in axs[i]:
            ax.grid(alpha=0.3); ax.set_xlim(1.3, 5.2)
            sec = ax.secondary_yaxis("right", functions=(lambda a: a / G, lambda g: g * G))
            sec.set_ylabel("g", fontsize=9)
        axs[i, 0].legend(fontsize=8.5)
    for j, t in enumerate(("(a) vertical, at the payload",
                           "(b) lateral (along the waves), as felt; dotted: inertial",
                           "(c) total lateral: (b) + the static tilt's g·sin θ (dashed)")):
        axs[0, j].set_title(t, fontsize=10.5)
        axs[-1, j].set_xlabel("wave period (s)")
    fig.suptitle("Accelerations at the payload in regular waves, H = 0.1 m along the shift — "
                 "FloatSim with its weight preload", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "tilted_accelerations.png", dpi=125, bbox_inches="tight")
    print("wrote tilted_accelerations.png")


if __name__ == "__main__":
    main()
