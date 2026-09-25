"""Statics for MOORING-SPEC.md (2026-09-25, rev B): the design of each article, from FloatSim.

Design (rev B): the attachment-height design chosen by attachment_sweep.py's decision rule --
submerged soft lines (0.02 N/m in water), each anchor on the wall at its attachment's depth, the
pretension and stiffness of the chosen row (attachment_sweep.json "choice"), the calm
equilibrium from FloatSim's settle of those lines (articulated) or the checked static solve (the
single buoy on its r = 0.2 m radial collar). Rev A's pin-plane lines with T0 x1.6 are REVERTED.
Per article:
  * every line: anchor and attachment point (TRUE flume frame: origin at the article centre on
    the flume centreline at still water, x down-flume with the waves, z up), unstretched length
    L0, stiffness k and its -13 % / +30 % acceptance band, nominal and FloatSim at-rest tension,
    at-rest stretch;
  * pull stiffness (surge, sway, yaw) from the FloatSim catenary force against a rigid offset
    (tank_predictions), and the pull force at 0.25 / 0.5 m and 5 deg;
  * surge / sway / yaw periods: the round-1 FloatSim decays (tank_rows/decay_*) scaled by
    sqrt(K_decay / K_now), i.e. the same effective mass; the buoy's yaw from its FloatSim inertia
    and collar stiffness (its decay cannot be run: STATE-FORCE-LAG-NEGATIVE-DAMPING).
Peak tensions and stretches come from the targeted FloatSim runs (attachment_design.json,
buoy_stability_check.json) and are merged by the report.

Rev C adds the extreme cord set (extreme_set.py): ``python mooring_spec.py extreme`` writes
spec_statics_extreme.json (cluster and platform; the buoy has one cord set).

Writes spec_statics.json.  Run: python mooring_spec.py [extreme]
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

import attachment_sweep as asw
import line_hardware as lh
import tank_predictions as tp

OUT = HERE / "spec_statics.json"
OUT_EXT = HERE / "spec_statics_extreme.json"      # rev C: the extreme cord set
BAND = (0.87, 1.30)
I_ZZ_BUOY = 0.063                                  # FloatSim deck (articulated_wall.IZZ)


def article(art: str, ext: dict | None = None) -> dict:
    """The operational set (rev B's choice), or with ``ext`` = {"opts", "tag", "m"} the extreme
    set (extreme_set.py: the same anchors, attachments and nominal T0, k x m; its own settle)."""
    des = asw.chosen(art)
    if ext is not None:
        des = {**des, "opts": ext["opts"], "tag": ext["tag"]}
    dk, xi_eq, lf, lines = lh._setup(art, opts=des["opts"], tag=des["tag"],
                                     reuse_eq=bool(ext and ext.get("reuse_eq")))
    sc = des["row"]["t0_scale"]
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
    buoys = [k for k, b in enumerate(dk.bodies) if b.hydro_body_label or b.hydro_database]
    tilt = max(float(np.degrees(np.hypot(xi_eq[6 * k + 3], xi_eq[6 * k + 4]))) for k in buoys)
    out = {"article": art, "t0_scale": sc, "n_lines": len(lines), "lines": rows, "pull": pull,
           "periods_s": periods, "design": des["row"], "tag": des["tag"],
           "static_tilt_settle_deg": tilt}
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
    if sys.argv[1:] == ["extreme"]:
        import extreme_set as es
        ch = json.loads(es.OUT.read_text())["choice"]
        res = {}
        for a in es.ARTS:
            m = ch[a]["m_criterion4"]
            # at-rest-matched pretension: the operational settle is the calm equilibrium
            # (accepted within FloatSim's equilibrium tolerance; extreme_set.py)
            ext = {"opts": es.opts(a, m, rest=True), "tag": asw.chosen(a)["tag"],
                   "reuse_eq": True}
            res[a] = {**article(a, ext), "m": m, "rest_ratio": es.rest_ratio(a, m)}
        OUT_EXT.write_text(json.dumps(res, indent=1, default=float))
        return
    res = {a: article(a) for a in ("buoy", "cluster", "platform")}
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
