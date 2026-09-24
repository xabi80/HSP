"""Why do the buoys pitch so much near resonance? FloatSim's answer (all runs are FloatSim decks
from floatsim_decks: deck bodies + BEM through the driver, the deck's Morison drag elements).

1. Single-buoy pitch free decay (unmoored, calm water) from 2, 5 and 10 deg: period and damping
   ratio per amplitude (log decrement), with and without the drag elements.
2. Single buoy in regular waves, H = 0.05 m, T = 2.2-3.5 s, unmoored: pitch amplitude with the
   deck's drag, with every Cd x2 and x4, and without drag. H = 0.05 m keeps the resonant pitch
   inside FloatSim's small-angle range (beyond ~13 deg the lone free buoy's yaw goes unstable:
   small-angle kinematics + spar drag, no yaw restraint or inertia to speak of).
3. Cluster in-phase buoy tilt at its resonance (T = 2.9 s, H = 0.1 m): with and without drag.

Writes pitch_check.json + pitch_check.png.  Run: python pitch_check.py
"""
# ruff: noqa: E402, E702, RUF001  -- sys.path bootstrap first; compact lines; typography
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import floatsim_decks as fd

from floatsim.solver.newmark import integrate_cummins

OUT = HERE / "pitch_check.json"
H_SWEEP = 0.05
T_SWEEP = (2.2, 2.4, 2.5, 2.6, 2.7, 2.76, 2.85, 2.95, 3.1, 3.3, 3.5)
VARIANTS = (("drag", 1.0), ("drag x2", 2.0), ("drag x4", 4.0), ("no drag", None))
COL = {"drag": "#0c8b96", "drag x2": "#0c8b96", "drag x4": "#0c8b96", "no drag": "#b2432c"}
LS = {"drag": "-", "drag x2": "--", "drag x4": "-.", "no drag": "-"}


def fit_amp(t, x, w):  # type: ignore[no-untyped-def]
    D = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(D, x, rcond=None)
    return float(np.hypot(c[0], c[1]))


def decay(theta0_deg: float, with_drag: bool) -> dict:
    dk = fd.deck("buoy", drag=with_drag, pitch0=np.radians(theta0_deg))
    setup, _ = fd.build(dk, "buoy", solve_equilibrium=False)
    r = integrate_cummins(lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0,
                          xi_dot0=setup.xi_dot0, duration=30.0, dt=fd.DT, rho_inf=0.8,
                          state_force=setup.state_force)
    th = r.xi[:, 4]
    pk, _ = find_peaks(np.abs(th), distance=int(0.8 / fd.DT)); pk = pk[:7]
    amp = np.abs(th[pk])
    zeta = float(np.log(amp[0] / amp[2]) / (2 * np.pi)) if len(amp) > 2 else None
    return {"theta0_deg": theta0_deg, "drag": with_drag,
            "T_s": float(2 * np.mean(np.diff(r.t[pk][:6]))), "zeta": zeta}


def forced(article: str, dk, T: float, H: float) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    setup, hd = fd.build(dk, article)
    r = fd.run_wave(setup, hd, dk, T, H, 45.0, 6)
    msk = r.t >= r.t[-1] - 6 * T
    w = 2 * np.pi / T
    buoys = [k for k, b in enumerate(dk.bodies) if b.hydro_body_label or b.hydro_database]
    amp = max(fit_amp(r.t[msk], r.xi[msk, 6 * k + 4] - setup.xi0[6 * k + 4], w) for k in buoys)
    yaw = max(float(np.abs(r.xi[msk, 6 * k + 5]).max()) for k in buoys)
    return float(np.degrees(amp)), float(np.degrees(yaw))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if res.get("source") != "floatsim":
        res = {"source": "floatsim"}
    if "decay" not in res:
        res["decay"] = [decay(a, d) for d in (True, False) for a in (2, 5, 10)]
        for dd in res["decay"]:
            print(f"decay {dd['theta0_deg']:>2}° drag={dd['drag']}: T {dd['T_s']:.3f} s, "
                  f"zeta {dd['zeta']:.4f}", flush=True)
        OUT.write_text(json.dumps(res, indent=1))
    if "buoy" not in res:
        res["buoy"] = {}
        for lab, sc in VARIANTS:
            rows = []
            dk = fd.deck("buoy", drag=sc is not None, cd_scale=sc or 1.0)
            for T in T_SWEEP:
                th, yaw = forced("buoy", dk, T, H_SWEEP)
                rows.append({"T": T, "pitch_deg": th, "yaw_deg": yaw})
                print(f"buoy {lab:>8} T {T:.2f}: pitch {th:7.1f}°  (yaw {yaw:.1e}°)", flush=True)
            res["buoy"][lab] = rows
            OUT.write_text(json.dumps(res, indent=1))
    if "cluster" not in res:
        res["cluster"] = {}
        for lab, drag in (("drag", True), ("no drag", False)):
            th, _ = forced("cluster", fd.deck("cluster", drag=drag), 2.9, 0.1)
            res["cluster"][lab] = th
            print(f"cluster {lab}: buoy tilt {th:.1f}° at T 2.9 s, H 0.1 m", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    plot(res)


def plot(res: dict) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1, 1.35, 0.8]))
    dd = [d for d in res["decay"] if d["drag"]]; d0 = [d for d in res["decay"] if not d["drag"]]
    ax[0].plot([d["theta0_deg"] for d in dd], [100 * d["zeta"] for d in dd], "o-",
               color=COL["drag"], lw=2, label="with Morison drag")
    ax[0].plot([d["theta0_deg"] for d in d0], [100 * d["zeta"] for d in d0], "s--",
               color=COL["no drag"], lw=1.6, label="radiation only")
    ax[0].axhline(12, color="0.45", ls=":", lw=1.2)
    ax[0].text(2.1, 12.4, "heave damping (field decay) ≈ 12–13 %", fontsize=8, color="0.35")
    ax[0].set_xlabel("initial pitch (deg)"); ax[0].set_ylabel("pitch damping ratio ζ (%)")
    ax[0].set_title(f"(a) Single-buoy pitch free decay, T = {dd[0]['T_s']:.2f} s",
                    fontsize=11, fontweight="bold")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8.5, loc="center right"); ax[0].set_ylim(0, 16)
    for lab, _sc in VARIANTS:
        rr = res["buoy"][lab]
        ok = [r for r in rr if r["yaw_deg"] < 1.0]
        bad = [r for r in rr if r["yaw_deg"] >= 1.0]
        ax[1].plot([r["T"] for r in ok], [r["pitch_deg"] for r in ok], LS[lab], marker="o",
                   color=COL[lab], lw=1.8, ms=5,
                   label=lab + (" (× = FloatSim diverges)" if bad else ""))
        if bad:
            ax[1].plot([r["T"] for r in bad], [min(r["pitch_deg"], 90) for r in bad], "x",
                       color=COL[lab], ms=9, mew=2)
    ax[1].axhline(13, color="0.5", ls=":", lw=1)
    ax[1].text(2.21, 14, "≈ small-angle limit of the lone free buoy in FloatSim", fontsize=8,
               color="0.35")
    ax[1].set_yscale("log"); ax[1].set_xlabel("wave period T (s)")
    ax[1].set_ylabel(f"pitch amplitude (deg), H = {H_SWEEP} m")
    ax[1].set_title("(b) Single buoy in waves, unmoored (FloatSim)", fontsize=11,
                    fontweight="bold")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8, loc="upper right")
    cl = res.get("cluster", {})
    if cl:
        labs = ["drag", "no drag"]
        ax[2].bar(range(2), [cl[k] for k in labs], color=[COL[k] for k in labs])
        for i, k in enumerate(labs):
            ax[2].text(i, cl[k], f"{cl[k]:.0f}°", ha="center", va="bottom", fontsize=9)
        ax[2].set_xticks(range(2)); ax[2].set_xticklabels(["with drag", "no drag"])
        ax[2].set_ylabel("buoy tilt amplitude (deg)")
        ax[2].set_title("(c) Cluster tilt resonance, T = 2.9 s, H = 0.1 m", fontsize=11,
                        fontweight="bold")
        ax[2].grid(alpha=0.3, axis="y")
    fig.suptitle("Pitch in FloatSim: Morison drag is included but damps pitch weakly, and the "
                 "pitch / tilt resonance (2.76–2.92 s) sits in the wave band", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "pitch_check.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
