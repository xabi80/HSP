"""Mass and CG envelope of the 4-buoy cluster with a payload on the hub (cluster_envelope.json).

(a) Payload mass limit against the payload's offset from the hub centre, per direction.
    - The limit is reached when a pin goes awash, or when the hub rolls onto its side.
    - Dashed lines: hub tilt 5 / 10 / 20 / 30 deg. They are the same in every direction.
    - Dotted line: buoys touching (axes closer than a heave-plate diameter), the most
      conservative direction.
    - Shaded: inside every direction's limit.
(b) The same limits as total cluster mass against the horizontal offset of the whole cluster's CG
    (level geometry, design frame).

Writes cluster_envelope.png. Run: python envelope_plot.py
"""

# ruff: noqa: E702, RUF001  -- compact figure-assembly lines; en dashes / degree signs in labels
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {0.0: "#0c8b96", 15.0: "#6a8a2b", 30.0: "#c9772b", 45.0: "#b2432c"}
DIR_LABEL = {0.0: "0° (between buoys)", 15.0: "15°", 30.0: "30°", 45.0: "45° (towards a buoy)"}
TILT_STYLE = {"5": ("#555555", 1.4), "10": ("#555555", 1.1), "20": ("#888888", 1.0),
              "30": ("#aaaaaa", 0.9)}


def main() -> None:
    d = json.loads((HERE / "cluster_envelope.json").read_text(encoding="utf-8"))
    rows = d["rows"]
    m_buoys = 4 * d["m_buoy_kg"] + d["m_hub_kg"]
    dirs = sorted({r["dir_deg"] for r in rows})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    lim_all = []
    for dd in dirs:
        rr = [r for r in rows if r["dir_deg"] == dd]
        e = np.array([r["offset_m"] for r in rr]); m = np.array([r["m_limit_kg"] for r in rr])
        awash = np.array([r["limit_reason"] == "pin awash" for r in rr])
        a1.plot(e, m, "-", color=COLORS[dd], lw=2, label=DIR_LABEL[dd])
        a1.plot(e[awash], m[awash], "o", color=COLORS[dd], ms=4)
        a1.plot(e[~awash], m[~awash], "^", color=COLORS[dd], ms=4, mfc="white")
        lim_all.append(m)
        cg = np.array([1e3 * r["cg_at_limit"]["cg_offset_m"] for r in rr])
        mt = np.array([r["cg_at_limit"]["total_mass_kg"] for r in rr])
        a2.plot(cg, mt, "-", color=COLORS[dd], lw=2, label=DIR_LABEL[dd])
        a2.plot(cg[awash], mt[awash], "o", color=COLORS[dd], ms=4)
        a2.plot(cg[~awash], mt[~awash], "^", color=COLORS[dd], ms=4, mfc="white")
    e0 = np.array([r["offset_m"] for r in rows if r["dir_deg"] == dirs[0]])
    env = np.min(np.vstack(lim_all), axis=0)
    a1.fill_between(e0, 0, env, color="#0c8b96", alpha=0.10, zorder=0)
    # (b) region reachable with offsets up to the largest swept (e_max) and inside every limit:
    # up the e_max line from no payload to its limit, then back along the worst boundary
    e_max, m_emax = e0[-1], env[-1]
    mm = np.linspace(0.0, m_emax, 40)
    up = np.column_stack([1e3 * mm * e_max / (m_buoys + mm), m_buoys + mm])
    back = np.column_stack([1e3 * env * e0 / (m_buoys + env), m_buoys + env])[::-1]
    a2.fill(*np.vstack([up, back]).T, color="#0c8b96", alpha=0.10, zorder=0)
    a2.plot(up[:, 0], up[:, 1], "-", color="0.6", lw=0.8)
    a2.text(up[len(up) // 5, 0] + 2, up[len(up) // 5, 1] + 1.5, f"payload at {e_max:.1f} m",
            fontsize=8, color="0.4")

    r0 = [r for r in rows if r["dir_deg"] == dirs[0] and r["offset_m"] > 0]
    for t, (col, lw) in TILT_STYLE.items():
        pts = [(r["offset_m"], r["m_tilt_kg"][t]) for r in r0 if r["m_tilt_kg"][t] is not None]
        if not pts:
            continue
        e, m = np.array(pts).T
        a1.plot(e, m, "--", color=col, lw=lw)
        a1.text(e[-1] + 0.005, m[-1], f"{t}° tilt", fontsize=8, color=col, va="center")
        cg = 1e3 * m * e / (m_buoys + m)
        a2.plot(cg, m_buoys + m, "--", color=col, lw=lw)
        a2.text(cg[-1] + 3, m_buoys + m[-1], f"{t}°", fontsize=8, color=col, va="center")
    touch = [(r["offset_m"], min(x["m_buoys_touch_kg"] for x in rows
                                 if x["offset_m"] == r["offset_m"] and x["m_buoys_touch_kg"]))
             for r in r0 if any(x["m_buoys_touch_kg"] for x in rows
                                if x["offset_m"] == r["offset_m"])]
    if touch:
        e, m = np.array(touch).T
        a1.plot(e, m, ":", color="#7b3fa0", lw=1.6, label="buoys may touch (worst direction)")

    a1.set_xlabel("payload offset from the hub centre, in the pin plane (m)")
    a1.set_ylabel("payload mass on the hub (kg)")
    a1.set_title("(a) Payload limit: ● pin awash, △ hub rolls onto its side", fontsize=10.5)
    a1.set_xlim(0, 0.56); a1.set_ylim(0, 60); a1.grid(alpha=0.3); a1.legend(fontsize=8.5)
    a1.axvline(d["r_buoy_m"], color="0.6", lw=0.8)
    a1.text(d["r_buoy_m"] + 0.004, 57, "buoy centres", fontsize=8, color="0.4")
    a2.set_xlabel("horizontal offset of the whole cluster's CG (mm, level geometry)")
    a2.set_ylabel("total cluster mass (kg)")
    a2.set_title(f"(b) Mass–CG envelope (unloaded cluster {m_buoys:.1f} kg)", fontsize=10.5)
    a2.set_ylim(m_buoys - 2, m_buoys + 60); a2.grid(alpha=0.3); a2.legend(fontsize=8.5)
    fig.suptitle("4-buoy cluster, payload on the hub (pin plane), calm water, free-floating — "
                 "large-angle hydrostatics (Capytaine draft table, NT 36)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "cluster_envelope.png", dpi=130, bbox_inches="tight")
    print("wrote cluster_envelope.png")


if __name__ == "__main__":
    main()
