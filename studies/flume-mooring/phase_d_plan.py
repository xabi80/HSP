"""Phase D case list and compute cost (flume-mooring), from the confirmed matrix and Phase C.

Rules (DESIGN-BASIS.md, confirmed 2026-09-24):
  * T = 1.4-3.5 s; every period >= 5 % clear of the sloshing periods 1.25 / 1.53 / 2.19 s;
  * fine step across the tilt/pitch resonances from resonance_bandwidth.json (item 3), spanning
    every article's half-power band at H = 0.04 m plus one step each side; 0.10 s elsewhere (the
    heave bands are ~0.6 s wide, so 0.10 s meets the same <= dT/3 rule there);
  * operational H = 0.04 / 0.08 / 0.12 m at every period; extreme H = 0.2 / 0.35 / 0.5 m at a
    subset (loads), dropped where H/lambda > 0.08 (finite depth 2.7 m) or where the recorded drift
    bound tilts the articles past the 3 deg mean-tilt criterion (flagged, not run);
  * criterion 6 reference (restrained free article, 60 s) at every operational case, the 120 s
    insensitivity check on a subset; moored decays (heave, tilt; surge / sway / yaw are item 4).

Cost: FloatSim wall minutes per case measured in Phase C (bandwidth_rows/: the same harness and
settle, 17 cases in parallel on this 32-core machine), scaled by simulated duration; the total is
divided by the same 17-way parallelism.

Writes phase_d_plan.json.  Run: python phase_d_plan.py
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mooring_sizing as ms

OUT = HERE / "phase_d_plan.json"
SLOSH = (1.25, 1.53, 2.19)
H_OP, H_EX = (0.04, 0.08, 0.12), (0.2, 0.35, 0.5)
K_TILT = {"buoy": 70.771, "cluster": 75.36, "platform": 81.45}   # N m/rad (record, §B)
ARM_PIN = 0.717                                                  # m, spar SWL to pin
TILT_MAX = 3.0
CAP = ms.STEEP_MATRIX
# ~0.1 s outside the fine band, hugging the slosh exclusions (1.4535-1.6065, 2.0805-2.2995 s)
COARSE_T = (1.40, 1.45, 1.65, 1.75, 1.85, 1.95, 2.05, 2.30, 2.90, 3.00, 3.10, 3.20, 3.35, 3.50)
# extreme (load) subset: both ends, each side of the 2.19 s exclusion, the resonances (§C3)
EXTREME_T = (1.40, 1.85, 2.05, 2.35, 2.55, 2.65, 2.75, 3.00, 3.50)
ARTICLES = ("buoy", "cluster", "platform")
SETTLE_OP, SETTLE_EX, KEEP = 120.0, 180.0, 4   # s; extreme cases settle the ~15 s surge longer
RAMP = 15.0
PARALLEL = 17                                  # as measured (the rates include this load)


def clear_of_slosh(T: float) -> bool:
    return all(abs(T - s) >= 0.05 * s for s in SLOSH)


def steepness(H: float, T: float) -> float:
    return H * ms.k_fin(T) / (2 * np.pi)


def periods(step: float, band: tuple[float, float]) -> tuple[list[float], list[float]]:
    """The fine band covers every H = 0.04 m tilt/pitch half-power band plus one step each side."""
    lo = np.floor((band[0] - step) / step) * step
    fine = [round(float(t), 3) for t in np.arange(lo, band[1] + 1.5 * step, step)]
    allp = sorted({*fine, *(t for t in COARSE_T if not fine[0] <= t <= fine[-1])})
    assert all(clear_of_slosh(t) for t in allp), [t for t in allp if not clear_of_slosh(t)]
    return allp, fine


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    bw = json.loads((HERE / "resonance_bandwidth.json").read_text())
    step = float(bw["step_s"])
    bands = [c["pitch"] for k, c in bw["cases"].items()
             if k.endswith("@H0.04") and c["pitch"]["ok"]]
    band = (min(b["T_lo"] for b in bands), max(b["T_hi"] for b in bands))
    allp, fine = periods(step, band)
    # measured wall minutes per simulated second (item 3 rows: ramp 15 + settle 120 + 4 T)
    rate = {}
    for art in ARTICLES:
        rows = [json.loads(p.read_text())
                for p in (HERE / "bandwidth_rows").glob(f"{art}_H*.json")]
        rate[art] = float(np.median([r["wall_min"] / (RAMP + 120.0 + 4 * r["T"]) for r in rows]))
    cases, flagged = [], []
    for art in ARTICLES:
        for T in allp:
            for H in H_OP:
                cases.append({"article": art, "kind": "operational", "H": H, "T": T,
                              "sim_s": RAMP + SETTLE_OP + KEEP * T})
                cases.append({"article": art, "kind": "reference_60s", "H": H, "T": T,
                              "sim_s": RAMP + SETTLE_OP + KEEP * T})
        assert set(EXTREME_T) <= set(allp)
        for T in EXTREME_T:
            for H in H_EX:
                s = steepness(H, T)
                F = sum(ms.drift_per_spar(H, T, steep=CAP)[:2])
                tilt = float(np.degrees(F * ARM_PIN / K_TILT[art]))
                why = (f"H/lambda {s:.3f} > {CAP:.2f}" if s > CAP else
                       f"mean tilt {tilt:.2f} deg > {TILT_MAX:g} deg (drift {F:.2f} N/spar)"
                       if tilt > TILT_MAX else None)
                row = {"article": art, "kind": "extreme", "H": H, "T": T, "H_over_lambda": s,
                       "drift_N_per_spar": F, "mean_tilt_deg": tilt,
                       "sim_s": RAMP + SETTLE_EX + 30.0 + KEEP * T}
                (flagged if why else cases).append({**row, **({"flag": why} if why else {})})
        for T in (2.55, 3.0):                       # criterion-6 insensitivity check (120 s)
            cases.append({"article": art, "kind": "reference_120s", "H": 0.08, "T": T,
                          "sim_s": RAMP + SETTLE_OP + KEEP * T})
        for mode in ("heave", "tilt"):
            cases.append({"article": art, "kind": f"decay_{mode}", "H": 0.0, "T": 0.0,
                          "sim_s": 60.0})
    cases.append({"article": "platform", "kind": "D0_no_applied_drift", "H": 0.12, "T": 2.65,
                  "sim_s": RAMP + SETTLE_OP + KEEP * 2.65})
    for c in cases:
        c["wall_min"] = c["sim_s"] * rate[c["article"]]
    summary = {}
    for art in ARTICLES:
        cs = [c for c in cases if c["article"] == art]
        tot = sum(c["wall_min"] for c in cs)
        summary[art] = {"n": len(cs), "case_min": tot, "wall_h_at_parallel": tot / PARALLEL / 60}
    res = {"step_s": step, "fine_band_s": band, "periods_s": allp, "fine_periods_s": fine,
           "rate_min_per_sim_s": rate, "summary": summary, "cases": cases, "flagged": flagged}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"step {step} s, fine band {band[0]:.3f}-{band[1]:.3f} s")
    print(f"{len(allp)} periods: {allp}")
    for k, v in summary.items():
        print(f"{k:8s}: {v['n']} cases, {v['case_min']:.0f} case-min, "
              f"{v['wall_h_at_parallel']:.1f} h at {PARALLEL} in parallel")
    for f in flagged:
        print(f"  flagged {f['article']:8s} H {f['H']} T {f['T']}: {f['flag']}")


if __name__ == "__main__":
    main()
