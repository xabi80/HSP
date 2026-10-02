"""Platform-size comparison, flat-on (45 deg): buoy centres on the 2.5 m circle vs a 2.4 m square.

The 2.4 m square (PLAT_SQUARE_M=2.4) is the 2.5 m-circle layout scaled by 1.358. Its 0.8 m grid
puts the outer heave plates 0.486 m from each wall, against 0.802 m for the 2.5 m circle. The
corner-on (0 deg) comparison of compare_orientations.py cannot be repeated for it: corner-on, an
outer plate would reach 1 cm into the wall.

This script quantifies what the larger platform does to the sidewall effect:
- the sidewall effect on the platform-heave excitation, from the AUTHORITATIVE coupled BEM (open
  vs walled; the same reading as compare_orientations.py);
- the free-decay heave period shift (articulated_wall.py decay).
It also prints the sidewall effect at the periods of the README's sidewall-vs-depth table.

Needs, for both sizes: coupled_osu_{open,walled}<suf>.nc and articulated_decay<suf>.json, with
<suf> = _rot45 and _rot45_sq2p4. Writes size_compare_rot45.png and prints the table.
Run: python compare_sizes.py
"""

# ruff: noqa: E702  -- compact figure-assembly lines, as compare_orientations.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from compare_orientations import coupled_wall

HERE = Path(__file__).resolve().parent
PLATE_R, FLUME_W = 0.1437, 3.66
TEAL, RED = "#0c8b96", "#b2432c"
SIZES = {"_rot45": ("2.5 m circle", 1.25 / 1.5),
         "_rot45_sq2p4": ("2.4 m square", 2.4 * np.sqrt(2) / 3)}
TABLE_T = [2.52, 3.0, 3.5, 4.0]


def geometry(s: float) -> tuple[float, float]:
    """(span with plates, clearance per side) flat-on, for cluster arm s."""
    ang = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]) + 45.0)
    cy = [s * np.sin(pc) + 0.5 * s * np.sin(tb) for pc in ang for tb in ang]
    span = 2 * (np.max(np.abs(cy)) + PLATE_R)
    return float(span), float((FLUME_W - span) / 2)


def decay(suf: str) -> dict:
    d = json.loads((HERE / f"articulated_decay{suf}.json").read_text())
    return {"T_open": d["open"]["heave_T"], "T_walled": d["walled"]["heave_T"],
            "dT_pct": 100 * (d["walled"]["heave_T"] / d["open"]["heave_T"] - 1)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    rows = {}
    for suf, (name, s) in SIZES.items():
        T, p = coupled_wall(suf)
        band = (T >= 2.0) & (T <= 4.0)
        span, clr = geometry(s)
        rows[suf] = {"name": name, "span_m": span, "clearance_m": clr,
                     "width_pct": 100 * span / FLUME_W, "T": T, "pct": p,
                     "max_exc_2_4s_pct": float(np.max(np.abs(p[band]))),
                     "exc_at_table_T_pct": [float(np.interp(t, T, p)) for t in TABLE_T],
                     **decay(suf)}
    print(" platform (flat-on) | span (plates) | clearance | width | free-decay T open→walled "
          "| max |exc. wall effect| 2-4 s | exc. wall effect at 2.52/3.0/3.5/4.0 s")
    for r in rows.values():
        print(f" {r['name']:18} | {r['span_m']:6.3f} m | {r['clearance_m']:6.3f} m | "
              f"{r['width_pct']:4.1f} % | {r['T_open']:.3f} → {r['T_walled']:.3f} s "
              f"({r['dT_pct']:+.2f} %) | {r['max_exc_2_4s_pct']:4.1f} % | "
              + " / ".join(f"{v:+.1f}" for v in r["exc_at_table_T_pct"]) + " %")
    out = {k: {kk: vv for kk, vv in v.items() if kk not in ("T", "pct")} for k, v in rows.items()}
    (HERE / "size_compare_rot45.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    for r, col, mk in zip(rows.values(), (TEAL, RED), ("s--", "o-"), strict=True):
        ax.plot(r["T"], r["pct"], mk, color=col, lw=2,
                label=f"{r['name']} ({r['span_m']:.2f} m with plates, "
                      f"{r['clearance_m']:.2f} m/side)")
    ax.axhspan(-5, 5, color="0.9", zorder=0)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlim(1.8, 4.1)
    ax.set_xlabel("wave period T (s)")
    ax.set_ylabel("sidewall effect on heave excitation (%)")
    ax.set_title("Platform size, flat-on (45°) — 2.5 m circle vs 2.4 m square, coupled BEM",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "size_compare_rot45.png", dpi=130, bbox_inches="tight")
    print("\nwrote size_compare_rot45.png and size_compare_rot45.json")


if __name__ == "__main__":
    main()
