"""Mass and CG envelope of the 4x4 platform with a payload on the deck (platform_envelope_h*.json,
platform_vs_height.json).

(a) Payload mass limit against the payload's offset from the deck centre, per direction, with the
    payload CG on the deck frame.
    - Every limit is a cluster's four pins reaching the waterline.
    - Grey dashed lines: deck tilt 5 / 10 / 20 deg. They are the same in every direction.
    - Thin lines: the worst direction with the payload CG raised 0.5 / 1.0 m above the deck.
    - Shaded: inside every direction's limit.
(b) The same limits as total platform mass against the horizontal offset of the whole platform's
    CG (level geometry, design frame).
(c) Centred payload: the limit against the payload CG height above the deck.
    - Pins awash while the CG is low.
    - Above about 0.7 m, the level deck loses its tilt restoring first. That limit follows
      m (h + 0.183 m) g = K_tilt, the dashed curve.

Writes platform_envelope.png. Run: python platform_envelope_plot.py
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
DIR_LABEL = {0.0: "0° (between hubs, along the grid)", 15.0: "15°", 30.0: "30°",
             45.0: "45° (towards a hub, the diagonal)"}
TILT_STYLE = {"5": ("#555555", 1.3), "10": ("#777777", 1.1), "20": ("#999999", 1.0)}
RAISED = {"0.5": ("#7b3fa0", "--"), "1.0": ("#7b3fa0", ":")}


def load(tag: str) -> dict:
    return json.loads((HERE / f"platform_envelope_h{tag}.json").read_text(encoding="utf-8"))


def k_tilt(d: dict) -> float:
    """Deck tilt stiffness from the clusters' heave springs (N m/rad), NT 36 table."""
    tab = json.loads((HERE / "buoy_hydrostatics_vs_draft.json").read_text(encoding="utf-8"))
    c33 = tab["rows"]["36"][0]["C33"]
    return 4 * c33 * 4 * (d["r_hub_m"] / np.sqrt(2.0)) ** 2


def main() -> None:
    d = load("0p0"); rows = d["rows"]
    m0 = d["n_buoys"] * d["m_buoy_kg"] + 4 * d["m_hub_kg"] + d["m_deck_kg"]
    dirs = sorted({r["dir_deg"] for r in rows})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 5.8))

    lim_all = []
    for dd in dirs:
        rr = [r for r in rows if r["dir_deg"] == dd]
        e = np.array([r["offset_m"] for r in rr]); m = np.array([r["m_limit_kg"] for r in rr])
        a1.plot(e, m, "-", color=COLORS[dd], lw=2, label=DIR_LABEL[dd])
        lim_all.append(m)
        cg = np.array([1e3 * r["cg_at_limit"]["cg_offset_m"] for r in rr])
        mt = np.array([r["cg_at_limit"]["total_mass_kg"] for r in rr])
        a2.plot(cg, mt, "-", color=COLORS[dd], lw=2, label=DIR_LABEL[dd])
    e0 = np.array([r["offset_m"] for r in rows if r["dir_deg"] == dirs[0]])
    env = np.min(np.vstack(lim_all), axis=0)
    a1.fill_between(e0, 0, env, color="#0c8b96", alpha=0.10, zorder=0)
    e_max, m_emax = e0[-1], env[-1]
    mm = np.linspace(0.0, m_emax, 40)
    up = np.column_stack([1e3 * mm * e_max / (m0 + mm), m0 + mm])
    back = np.column_stack([1e3 * env * e0 / (m0 + env), m0 + env])[::-1]
    a2.fill(*np.vstack([up, back]).T, color="#0c8b96", alpha=0.10, zorder=0)
    a2.plot(up[:, 0], up[:, 1], "-", color="0.6", lw=0.8)
    a2.text(up[len(up) // 3, 0] + 4, up[len(up) // 3, 1], f"payload at {e_max:.1f} m",
            fontsize=8, color="0.4")

    r0 = [r for r in rows if r["dir_deg"] == dirs[0] and r["offset_m"] > 0]
    for t, (col, lw) in TILT_STYLE.items():
        pts = [(r["offset_m"], r["m_tilt_kg"][t]) for r in r0 if r["m_tilt_kg"][t] is not None]
        e, m = np.array(pts).T
        a1.plot(e, m, "--", color=col, lw=lw)
        a1.text(e[-1] + 0.012, m[-1], f"{t}° tilt", fontsize=8, color=col, va="center")
        cg = 1e3 * m * e / (m0 + m)
        a2.plot(cg, m0 + m, "--", color=col, lw=lw)
        a2.text(cg[-1] + 3, m0 + m[-1], f"{t}°", fontsize=8, color=col, va="center")
    for tag, (col, ls) in RAISED.items():
        rr = [r for r in load(tag.replace(".", "p"))["rows"] if r["dir_deg"] == 45.0]
        a1.plot([r["offset_m"] for r in rr], [r["m_limit_kg"] for r in rr], ls, color=col,
                lw=1.4, label=f"45°, payload CG {tag} m above the deck")

    a1.set_xlabel("payload offset from the deck centre (m)")
    a1.set_ylabel("payload mass on the deck (kg)")
    a1.set_title("(a) Payload limit (payload CG on the deck frame): a cluster's pins awash",
                 fontsize=10.5)
    a1.set_xlim(0, 1.32); a1.set_ylim(0, 230); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    a1.axvline(d["r_hub_m"], color="0.6", lw=0.8)
    a1.text(d["r_hub_m"] + 0.008, 220, "hub centres\n(diagonal)", fontsize=8, color="0.4", va="top")
    a2.set_xlabel("horizontal offset of the whole platform's CG (mm, level geometry)")
    a2.set_ylabel("total platform mass (kg)")
    a2.set_title(f"(b) Mass–CG envelope (unloaded platform {m0:.1f} kg)", fontsize=10.5)
    a2.set_ylim(m0 - 5, m0 + 230); a2.grid(alpha=0.3); a2.legend(fontsize=8)

    hv = json.loads((HERE / "platform_vs_height.json").read_text(encoding="utf-8"))["rows"]
    h = np.array([r["h_pay_m"] for r in hv]); m = np.array([r["m_limit_kg"] for r in hv])
    awash = np.array([r["limit_reason"] == "pin awash" for r in hv])
    a3.plot(h, m, "-", color="#1f3b4d", lw=2)
    a3.plot(h[awash], m[awash], "o", color="#1f3b4d", ms=5, label="pins awash")
    a3.plot(h[~awash], m[~awash], "^", color="#b2432c", ms=6, mfc="white",
            label="the level deck loses its tilt restoring")
    hh = np.linspace(0.0, 2.0, 200); kt = k_tilt(d)
    mt = (kt / d.get("g", 9.806) - d["m_deck_kg"] * d["deck_above_pins_m"]) / (
        hh + d["deck_above_pins_m"])
    a3.plot(hh, mt, "--", color="#b2432c", lw=1,
            label=f"m (h + {d['deck_above_pins_m']:.3f} m) g = {kt:.0f} N m/rad")
    a3.set_xlabel("payload CG height above the deck frame (m)")
    a3.set_ylabel("centred payload limit (kg)")
    a3.set_title("(c) Centred payload: raising its CG", fontsize=10.5)
    a3.set_xlim(0, 2.0); a3.set_ylim(0, 230); a3.grid(alpha=0.3); a3.legend(fontsize=8)
    fig.suptitle("4x4 platform (2.4 m), payload on the deck, calm water, free-floating — "
                 "large-angle hydrostatics (Capytaine draft table, NT 36)",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "platform_envelope.png", dpi=130, bbox_inches="tight")
    print("wrote platform_envelope.png")


if __name__ == "__main__":
    main()
