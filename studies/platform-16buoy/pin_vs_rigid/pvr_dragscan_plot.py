"""Tabulate + plot the drag-sensitivity scan (pvr_dragscan.py rows, pvr_td.py baseline at
Cd x1, pvr_fd.py radiation-only limit).

Effective drag scale s_eff = Cd_scale * H / 0.10 m (quadratic drag relative to the linear forces
scales with Cd * H, so a small wave stands in for a very small Cd).

One-DOF check at the 3.2 s resonance: radiation (linear) + drag (quadratic, equivalent-
linearised, proportional to Cd * amplitude) gives  X = X_lin / (1 + c * s_eff * X),  with c
calibrated on the Cd x1 run alone; it then predicts the reduced-Cd runs.

Writes pvr_dragscan_summary.csv + pvr_dragscan.png.  Run: python pvr_dragscan_plot.py
"""
# ruff: noqa: E702, RUF001  -- compact plotting lines; display typography in labels
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
PERIODS = (3.0, 3.2, 3.5)
COL = {3.0: "#6a3d9a", 3.2: "#b2432c", 3.5: "#0c8b96"}
PIN, RIG = "#0c8b96", "#b2432c"


def _csv(name: str) -> list[dict]:
    with (_HERE / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load() -> tuple[list[dict], dict]:
    """Paired pin/rigid rows keyed by (scale, H, T), plus the FD limit per T."""
    raw: dict = {}
    for r in _csv("pvr_td_summary.csv"):          # Cd x1 baseline, H = 0.10 m
        T = float(r["T"])
        if T in PERIODS:
            raw[(1.0, 0.1, T, r["config"])] = dict(
                heave=float(r["heave_RAO"]), pitch=float(r["pitch_RAO"]),
                acc=float(r["acc_heave_peak"]) / 0.05, diverged=False)
    for f in sorted((_HERE / "pvr_drag_rows").glob("*.json")):
        r = json.loads(f.read_text())
        raw[(r["scale"], r["height"], r["T"], r["config"])] = dict(
            heave=r["heave"], pitch=r["pitch"], acc=r["acc_heave"] / (0.5 * r["height"]),
            diverged=bool(r.get("diverged", False)))
    fd = {float(r["T_s"]): r for r in _csv("pvr_fd_summary.csv")}
    fd = {T: {c: (float(r["heave_RAO"]), float(r["pitch_RAO_radpm"])) for c, r in
              ((row["config"], row) for row in _csv("pvr_fd_summary.csv")
               if abs(float(row["T_s"]) - T) < 1e-9)} for T in PERIODS if T in fd}
    rows = []
    for (s, H, T, cfg) in sorted(raw):
        if cfg != "artic" or (s, H, T, "rigid") not in raw:
            continue
        a, b = raw[(s, H, T, "artic")], raw[(s, H, T, "rigid")]
        ok = not (a["diverged"] or b["diverged"])
        rows.append(dict(
            T_s=T, cd_scale=s, height_m=H, drag_scale_eff=round(s * H / 0.1, 6),
            heave_artic=round(a["heave"], 4), heave_rigid=round(b["heave"], 4),
            heave_gap_pct=round(100 * (b["heave"] / a["heave"] - 1), 2) if ok else "",
            pitch_artic_radpm=round(a["pitch"], 4), pitch_rigid_radpm=round(b["pitch"], 4),
            pitch_ratio=round(b["pitch"] / a["pitch"], 3) if ok else "",
            acc_heave_artic_per_m=round(a["acc"], 3), acc_heave_rigid_per_m=round(b["acc"], 3),
            note=("artic run diverged (motions beyond the small-angle joint range)"
                  if a["diverged"] else "large-amplitude" if s * H / 0.1 < 0.005 and H >= 0.1
                  else "")))
    for T in PERIODS:
        (ha, pa), (hr, pr) = fd[T]["artic"], fd[T]["rigid"]
        rows.append(dict(T_s=T, cd_scale=0.0, height_m="", drag_scale_eff=0.0,
                         heave_artic=round(ha, 4), heave_rigid=round(hr, 4),
                         heave_gap_pct=round(100 * (hr / ha - 1), 2),
                         pitch_artic_radpm=round(pa, 4), pitch_rigid_radpm=round(pr, 4),
                         pitch_ratio=round(pr / pa, 3), acc_heave_artic_per_m="",
                         acc_heave_rigid_per_m="", note="radiation-only FD (no drag)"))
    rows.sort(key=lambda r: (r["T_s"], -r["drag_scale_eff"], str(r["height_m"])))
    return rows, fd


def one_dof(x_lin: float, x_1: float, s: np.ndarray) -> np.ndarray:
    """Resonant amplitude with radiation + equivalent-linearised quadratic drag."""
    c = (x_lin / x_1 - 1.0) / x_1
    a = c * np.asarray(s, float)
    return np.where(a > 0, (-1 + np.sqrt(1 + 4 * a * x_lin)) / (2 * np.maximum(a, 1e-300)),
                    x_lin)


def main() -> None:
    rows, fd = load()
    with (_HERE / "pvr_dragscan_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    td = [r for r in rows if r["drag_scale_eff"] > 0]
    s_mod = np.geomspace(1e-4, 1.0, 200)
    x_lin = fd[3.2]["artic"][0], fd[3.2]["rigid"][0]
    base = next(r for r in td if r["T_s"] == 3.2 and r["cd_scale"] == 1.0)
    mp = one_dof(x_lin[0], base["heave_artic"], s_mod)
    mr = one_dof(x_lin[1], base["heave_rigid"], s_mod)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
    # (a) resonant heave vs drag
    r32 = [r for r in td if r["T_s"] == 3.2]
    for r in r32:
        big = r["note"] != ""
        if not r["note"].startswith("artic"):
            ax[0].plot(r["drag_scale_eff"], r["heave_artic"], "o", color=PIN, ms=7,
                       mfc="none" if big else PIN)
        ax[0].plot(r["drag_scale_eff"], r["heave_rigid"], "s", color=RIG, ms=7,
                   mfc="none" if big else RIG)
    ax[0].plot(s_mod, mp, color=PIN, lw=1.4, label="one-DOF model, pin (fit at Cd ×1 only)")
    ax[0].plot(s_mod, mr, color=RIG, lw=1.4, ls="--", label="one-DOF model, rigid")
    ax[0].axhline(x_lin[0], color="0.4", ls=":", lw=1.2)
    ax[0].text(1.2e-4, x_lin[0] + 0.25, "radiation-only FD (no drag): 10.9 / 10.8", fontsize=8,
               color="0.3")
    ax[0].plot([], [], "o", color=PIN, label="TD pin"); ax[0].plot([], [], "s", color=RIG,
                                                                     label="TD rigid")
    ax[0].plot([], [], "s", color=RIG, mfc="none", label="TD at H = 0.1 m, large motion")
    ax[0].text(1.2e-4, 0.35, "pin at Cd ×0.001, H = 0.1 m diverged (motions beyond the\n"
               "small-angle joint range): not shown; the H = 0.01 m twin is plotted",
               fontsize=7.5, color="0.3")
    ax[0].set_title("(a) Heave RAO at the 3.2 s resonance", fontsize=11, fontweight="bold")
    ax[0].set_ylabel("heave RAO (m/m)"); ax[0].set_ylim(0, 12)
    ax[0].legend(fontsize=7.8, loc="center left", bbox_to_anchor=(0.0, 0.42))
    # (b) heave gap, (c) pitch ratio
    for T in PERIODS:
        rr = sorted((r for r in td if r["T_s"] == T and r["heave_gap_pct"] != ""
                     and r["note"] == ""), key=lambda r: r["drag_scale_eff"])
        s = [r["drag_scale_eff"] for r in rr]
        ax[1].plot(s, [r["heave_gap_pct"] for r in rr], "o-", color=COL[T], lw=1.6,
                   label=f"TD, T = {T:.1f} s")
        ax[2].plot(s, [r["pitch_ratio"] for r in rr], "o-", color=COL[T], lw=1.6,
                   label=f"TD, T = {T:.1f} s")
        (_, pa), (_, pr) = fd[T]["artic"], fd[T]["rigid"]
        ax[2].axhline(pr / pa, color=COL[T], ls=":", lw=1.1)
    ax[1].plot(s_mod, 100 * (mr / mp - 1), color=COL[3.2], ls="--", lw=1.2,
               label="one-DOF model, T = 3.2 s")
    ax[1].axhspan(-1, 1, color="0.85", zorder=0)
    ax[1].text(1.3e-2, 0.3, "radiation-only FD (no drag): |gap| < 1 %", fontsize=8, color="0.3")
    low = [r for r in td if r["T_s"] == 3.2 and r["drag_scale_eff"] < 0.005
           and r["heave_gap_pct"] != ""]
    for r in low:   # 400 s cap: amplitude still drifting ~0.5 %/window -> +-1.5 % band
        ax[1].errorbar(r["drag_scale_eff"], r["heave_gap_pct"], yerr=1.5, color=COL[3.2],
                       capsize=4, lw=1.2)
        ax[1].text(r["drag_scale_eff"] * 1.35, r["heave_gap_pct"] + 0.5,
                   "not fully settled\n(±1.5 %)", fontsize=7.5, color="0.3")
    ax[1].set_title("(b) Rigid-vs-pin heave gap", fontsize=11, fontweight="bold")
    ax[1].set_ylabel("heave RAO rigid / pin − 1 (%)"); ax[1].legend(fontsize=8, loc="lower left")
    ax[2].plot([], [], ":", color="0.4", label="radiation-only FD (per period)")
    ax[2].set_title("(c) Rigid / pin deck-pitch ratio", fontsize=11, fontweight="bold")
    ax[2].set_ylabel("pitch RAO rigid / pin"); ax[2].set_ylim(0, 3.0)
    ax[2].legend(fontsize=8, loc="lower left")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_xlim(1e-4, 1.6); a_.grid(alpha=0.3, which="both", lw=0.5)
        a_.set_xlabel("effective drag scale  Cd_scale × H / 0.1 m")
    fig.suptitle("Pin vs rigid — drag sensitivity: the heave gap is drag-mediated and closes only "
                 "once drag is a small part of the damping (Cd ≈ 1000× lower at resonance)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(_HERE / "pvr_dragscan.png", dpi=130, bbox_inches="tight")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
