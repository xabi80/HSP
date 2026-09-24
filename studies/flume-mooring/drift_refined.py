"""Refined (moving-body) mean drift vs the fixed-body upper bound, for the three articles.

Reads drift_rows/*.json from drift_td.py (ratio R = F_rel / F_fixed at each period, deep-water
kinematics) and applies R to the flume-depth fixed-body splash-zone drift of mooring_sizing.py:
    F_refined(T) = R(T) * n_spar * F_drag,fixed(H, T; h = 2.7 m).
Cases whose largest buoy tilt exceeds 10 deg lie beyond the small-angle validity of the linear
joint / hydrostatic model and are marked indicative.

Writes drift_refined.csv + drift_refined.png.  Run: python drift_refined.py
"""
# ruff: noqa: E702, RUF001  -- compact plotting lines; display typography in labels
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mooring_sizing as ms
import numpy as np

HERE = Path(__file__).resolve().parent
TILT_OK_DEG = 10.0
ARTS = {"1 buoy": ("buoy", 1), "1 cluster (4 buoys)": ("cluster", 4),
        "4x4 platform (45°)": ("platform", 16)}
COL = {"1 buoy": "#0c8b96", "1 cluster (4 buoys)": "#6a3d9a", "4x4 platform (45°)": "#b2432c"}


def f_fixed(n: int, H: float, T: float) -> float:
    return n * float(ms.drift_per_spar(H, T)[0])


def load(H: float = 0.5) -> list[dict]:
    rows = []
    for f in sorted((HERE / "drift_rows").glob("*.json")):
        r = json.loads(f.read_text())
        if abs(r["H_req"] - H) > 1e-9:
            continue
        n = r["n_spar"]
        ff = f_fixed(n, H, r["T"])
        rows.append(dict(article=r["article"], T_s=r["T"], H_used_m=round(r["H_used"], 3),
                         R=round(r["R"], 3), F_fixed_flume_N=round(ff, 2),
                         F_refined_N=round(r["R"] * ff, 2),
                         max_buoy_tilt_deg=round(float(np.degrees(r["max_tilt_rad"])), 1),
                         small_angle_ok=bool(np.degrees(r["max_tilt_rad"]) <= TILT_OK_DEG)))
    rows.sort(key=lambda r: (list(ARTS).index(r["article"]), r["T_s"]))
    return rows


def summary(rows: list[dict]) -> dict:
    out = {}
    for a, (_key, n) in ARTS.items():
        rr = [r for r in rows if r["article"] == a]
        if not rr:
            continue
        bound = max(f_fixed(n, 0.5, T) + n * ms.drift_per_spar(0.5, T)[1] for T in ms.T_WAVE)
        ok = [r for r in rr if r["small_angle_ok"]]
        out[a] = dict(bound_N=round(float(bound), 1),
                      max_down_N=round(max(r["F_refined_N"] for r in rr), 1),
                      max_up_N=round(min(r["F_refined_N"] for r in rr), 1),
                      max_abs_ok_N=round(max((abs(r["F_refined_N"]) for r in ok), default=0.0), 1),
                      max_abs_ratio=round(float(max(abs(r["F_refined_N"]) for r in rr) / bound),
                                          2),
                      n_cases=len(rr))
    return out


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    rows = load()
    with (HERE / "drift_refined.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    summ = summary(rows)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.9), sharex=True)
    Tc = ms.T_WAVE
    for a_, (a, (_key, n)) in zip(ax, ARTS.items(), strict=True):
        fb = [f_fixed(n, 0.5, T) for T in Tc]
        a_.plot(Tc, fb, color="0.35", lw=2, label="fixed-body upper bound (design basis)")
        a_.plot(Tc, [-v for v in fb], color="0.35", lw=1, ls=":", label="− upper bound")
        a_.axhline(0, color="0.5", lw=0.8)
        rr = [r for r in rows if r["article"] == a]
        if rr:
            T_ = [r["T_s"] for r in rr]; F_ = [r["F_refined_N"] for r in rr]
            a_.plot(T_, F_, "-", color=COL[a], lw=1.2, alpha=0.6)
            ok = [r for r in rr if r["small_angle_ok"]]
            bad = [r for r in rr if not r["small_angle_ok"]]
            a_.plot([r["T_s"] for r in ok], [r["F_refined_N"] for r in ok], "o", color=COL[a],
                    ms=7, label="moving body (relative velocity)")
            a_.plot([r["T_s"] for r in bad], [r["F_refined_N"] for r in bad], "o", color=COL[a],
                    mfc="white", ms=7, label=f"… buoy tilt > {TILT_OK_DEG:.0f}° (indicative)")
        a_.set_title(f"{a} — H = 0.5 m (steepness-capped)", fontsize=11, fontweight="bold")
        a_.set_xlabel("wave period T (s)"); a_.grid(alpha=0.3)
        a_.text(0.02, 0.03, "negative = drift points upstream", transform=a_.transAxes,
                fontsize=8, color="0.35")
    ax[0].set_ylabel("mean drift force (N), + = downstream")
    ax[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Mean wave drift: fixed-body upper bound vs the moving article (drag-limited "
                 "FloatSim time domain, relative velocity at each spar's waterline)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "drift_refined.png", dpi=130, bbox_inches="tight")
    (HERE / "drift_refined_summary.json").write_text(json.dumps(summ, indent=1))
    for a, s in summ.items():
        print(a, s)


if __name__ == "__main__":
    main()
