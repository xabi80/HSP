"""Statics for MOORING-SPEC.md (2026-09-25): the final design of each article, from FloatSim.

Design: pin-level soft lines (0.3 N/m), anchors at the walls in the pin plane; the single buoy at
T0 = 4.0 N per line on its r = 0.2 m radial collar; the cluster and platform with their
pretension raised 20 % over the Phase C design (line stiffness k kept; line_hardware
.design_opts). Per article:
  * every line: anchor and attachment point (TRUE flume frame: origin at the article centre on
    the flume centreline at still water, x down-flume with the waves, z up), unstretched length
    L0, stiffness k and its -13 % / +30 % acceptance band, nominal and FloatSim at-rest tension,
    at-rest stretch;
  * pull stiffness (surge, sway, yaw) from the FloatSim catenary force against a rigid offset
    (tank_predictions), and the pull force at 0.25 / 0.5 m and 5 deg;
  * surge / sway / yaw periods: the round-1 FloatSim decays (tank_rows/decay_*) scaled by
    sqrt(K_decay / K_now), i.e. the same effective mass; the buoy's yaw from its FloatSim inertia
    and collar stiffness (its decay cannot be run: STATE-FORCE-LAG-NEGATIVE-DAMPING).
Peak tensions and stretches come from the targeted FloatSim runs (spec_extremes.json,
spec_extreme_buoy.json) and are merged by the report.

Writes spec_statics.json.  Run: python mooring_spec.py
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import line_hardware as lh
import tank_predictions as tp

OUT = HERE / "spec_statics.json"
# pretension over the Phase C design, set by the targeted worst-case runs (slack pass:
# T_min >= 0.15 T0 at H = 0.5 m, T = 2.35 and 2.65 s, drift sum); +20 % failed
T0_SCALE = {"buoy": 1.0, "cluster": 1.6, "platform": 1.6}
BAND = (0.87, 1.30)
I_ZZ_BUOY = 0.063                                  # FloatSim deck (articulated_wall.IZZ)


def article(art: str) -> dict:
    sc = T0_SCALE[art]
    dk, xi_eq, lf, lines = lh._setup(art, t0_scale=sc)
    ref = np.array([b.reference_point for b in dk.bodies], dtype=float)
    st = lh.line_states(dk, xi_eq)
    rows = []
    for ln, s in zip(lines, st, strict=True):
        b = ln["body"]
        fair = ref[b] + xi_eq[6 * b:6 * b + 3] + ln["fairlead"]
        rows.append({"body": dk.bodies[b].name, "anchor_m": np.round(ln["anchor"], 4).tolist(),
                     "attachment_m": np.round(fair, 4).tolist(), "L0_m": ln["L0"],
                     "chord_m": ln["chord"], "k_N_per_m": ln["k"],
                     "k_band_N_per_m": [BAND[0] * ln["k"], BAND[1] * ln["k"]],
                     "T0_nominal_N": ln["T0"], "T_at_rest_N": s["T"],
                     "stretch_at_rest_m": s["T"] / s["k"]})
    z = np.zeros(xi_eq.size)

    def res(xi: np.ndarray) -> np.ndarray:
        return tp._resultant(dk, xi, np.sum([f(0.0, xi, z) for f in lf], axis=0))

    pull: dict = {}
    for mode, j, h, pts in (("surge", 0, 1e-3, (0.25, 0.5, 1.0)), ("sway", 1, 1e-3, (0.1, 0.25)),
                            ("yaw", 3, np.radians(0.1), (np.radians(5.0), np.radians(10.0)))):
        Rp, Rm = res(tp.rigid(dk, xi_eq, mode, h)), res(tp.rigid(dk, xi_eq, mode, -h))
        K = -(Rp[j] - Rm[j]) / (2 * h)
        F0 = res(xi_eq)[j]
        pull[mode] = {"K0": float(K), "points": [
            {"offset": float(p), "F": float(F0 - res(tp.rigid(dk, xi_eq, mode, p))[j])}
            for p in pts]}
    periods = {}
    for mode in ("surge", "sway", "yaw"):
        if art == "buoy" and mode == "yaw":
            periods[mode] = float(2 * np.pi * np.sqrt(I_ZZ_BUOY / pull["yaw"]["K0"]))
            continue
        dec = json.loads((HERE / "tank_rows" / f"decay_{art}_{mode}.json").read_text())
        K_dec = json.loads((HERE / "tank_rows" / f"pull_{art}.json").read_text())[mode]["K0"]
        periods[mode] = float(dec["T_s"] * np.sqrt(K_dec / pull[mode]["K0"]))
    out = {"article": art, "t0_scale": sc, "n_lines": len(lines), "lines": rows, "pull": pull,
           "periods_s": periods}
    print(f"{art:8s} T0 x{sc}: {len(lines)} lines, T0 {rows[0]['T0_nominal_N']:.2f} N "
          f"(at rest {rows[0]['T_at_rest_N']:.2f}), L0 {min(r['L0_m'] for r in rows):.3f}-"
          f"{max(r['L0_m'] for r in rows):.3f} m, k {rows[0]['k_N_per_m']:.3f}, at-rest stretch "
          f"{rows[0]['stretch_at_rest_m']:.2f} m | K surge {pull['surge']['K0']:.2f} sway "
          f"{pull['sway']['K0']:.2f} N/m yaw {pull['yaw']['K0']:.3f} N m/rad | T surge "
          f"{periods['surge']:.2f} sway {periods['sway']:.2f} yaw {periods['yaw']:.2f} s",
          flush=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    res = {a: article(a) for a in ("buoy", "cluster", "platform")}
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
