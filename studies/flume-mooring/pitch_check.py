"""Why do the single buoy and the pinned buoys pitch so much near resonance? A check of the
pitch damping the FloatSim model actually has, and of how much the Morison drag does.

1. Single-buoy pitch free decay (calm water, unmoored) from 2 to 30 deg: natural period and
   damping ratio per amplitude (log decrement of successive peaks), with and without drag.
2. Forced single-buoy pitch in regular waves, H = 0.3 m, T = 1.6-3.5 s, unmoored, with the
   Morison drag on the velocity RELATIVE to the wave (the physical form used in the mooring
   studies), on the body velocity only (calm-water form), and with no drag.
3. The cluster's in-phase buoy tilt at its 2.86 s resonance (T = 2.8 s, H = 0.3 m), same three
   drag forms.
4. Sensitivity of the resonant single-buoy pitch to the (unmeasured) drag coefficients: every Cd
   of the spar and plate scaled x2 and x4, relative drag.

Writes pitch_check.json + pitch_check.png.  Run: python pitch_check.py
"""
# ruff: noqa: E402, E702, RUF001  -- sys.path bootstrap first; compact lines; typography
from __future__ import annotations

import dataclasses
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PLAT_ROT_DEG", "45")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent.parent, HERE.parent / "platform-12buoy" / "flume-wall-effect",
           HERE.parent / "osu-test-buoy", HERE):
    sys.path.insert(0, str(_p))

import drift_td as dtd

from floatsim.hydro.morison import make_morison_state_force
from floatsim.solver.newmark import integrate_cummins

OUT = HERE / "pitch_check.json"
T_SWEEP = (1.6, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.6, 2.8, 3.0, 3.5)
MODES = ("relative", "calm", "none")
LAB = {"relative": "Morison drag, relative to the wave (used in the studies)",
       "calm": "Morison drag, body velocity only (calm-water form)",
       "none": "no drag (radiation damping only)"}
COL = {"relative": "#0c8b96", "calm": "#6a3d9a", "none": "#b2432c"}


def decay(m: dict, theta0_deg: float, with_drag: bool) -> dict:
    n = m["n"]; xi0 = np.zeros(n); xi0[4] = np.radians(theta0_deg)
    drag = make_morison_state_force(m["elements"], n_dof=n,
                                    fluid_velocity_fn=lambda p, t: np.zeros(3), rho=dtd.RHO)

    def state(t, xi, xd):
        if not with_drag:
            return np.zeros(n)
        f = drag(t, xi, xd); f[5] = 0.0
        return f

    r = integrate_cummins(lhs=m["lhs"], kernel=m["kernel"], xi0=xi0, xi_dot0=np.zeros(n),
                          duration=30.0, dt=dtd.DT, rho_inf=0.8, external_force=None,
                          state_force=state)
    th = r.xi[:, 4]
    pk, _ = find_peaks(np.abs(th), distance=int(0.6 / dtd.DT))
    pk = pk[:7]
    amp = np.abs(th[pk])
    zeta = [float(np.log(amp[i] / amp[i + 2]) / (2 * np.pi)) for i in range(len(amp) - 2)]
    period = float(2 * np.mean(np.diff(r.t[pk][:6])))
    return {"theta0_deg": theta0_deg, "drag": with_drag, "T_s": period,
            "zeta_first_cycle": zeta[0] if zeta else None, "peaks_deg": np.degrees(amp).tolist()}


def forced_pitch(m: dict, T: float, drag_mode: str) -> float:
    r, wv = dtd.simulate(m, T, 0.3, n_win=6, moored=False, drag_mode=drag_mode)
    msk = r.t >= r.t[-1] - 6 * T + 0.5 * dtd.DT
    t = r.t[msk]; w = wv["omega"]
    idx = [4] if m["n"] == 6 else [6 * b + 4 for (b, _p, _r) in m["spars"]]
    amps = []
    for i in idx:
        D = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
        c, *_ = np.linalg.lstsq(D, r.xi[msk, i] - m["xi0"][i], rcond=None)
        amps.append(float(np.hypot(c[0], c[1])))
    return float(np.degrees(max(amps))), float(wv["A"]), float(wv["k"])


def scaled(m: dict, s: float) -> dict:
    """Copy of the article with every Morison Cd (spar Cd, plate Cd_n / Cd_t) scaled by s."""
    el = []
    for e in m["elements"]:
        if hasattr(e, "Cd_n"):
            el.append(dataclasses.replace(e, Cd_n=e.Cd_n * s, Cd_t=e.Cd_t * s))
        else:
            el.append(dataclasses.replace(e, Cd=e.Cd * s))
    return {**m, "elements": el}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore")
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    m = dtd.build("buoy")
    if "decay" not in res:
        res["decay"] = [decay(m, a, d) for d in (True, False) for a in (2, 5, 10, 20, 30)]
        for dd in res["decay"]:
            print(f"decay {dd['theta0_deg']:>2}° drag={dd['drag']}: T {dd['T_s']:.3f} s, "
                  f"zeta {dd['zeta_first_cycle']:.4f}", flush=True)
    if "buoy" not in res:
        res["buoy"] = {}
        for mode in MODES:
            rows = []
            for T in T_SWEEP:
                th, A, k = forced_pitch(m, T, mode)
                rows.append({"T": T, "pitch_deg": th, "wave_slope_deg": float(np.degrees(k * A))})
                print(f"buoy {mode:>8} T {T:.1f}: pitch {th:6.1f}°  (wave slope "
                      f"{np.degrees(k * A):.1f}°)", flush=True)
            res["buoy"][mode] = rows
        OUT.write_text(json.dumps(res, indent=1))
    if "cd_scale" not in res:
        res["cd_scale"] = {}
        for sc in (2.0, 4.0):
            ms_ = scaled(m, sc); rows = []
            for T in (1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.6):
                th, _A, _k = forced_pitch(ms_, T, "relative")
                rows.append({"T": T, "pitch_deg": th})
                print(f"buoy Cd x{sc:g} T {T:.1f}: pitch {th:6.1f}°", flush=True)
            res["cd_scale"][f"{sc:g}"] = rows
        OUT.write_text(json.dumps(res, indent=1))
    if "cluster" not in res:
        mc = dtd.build("cluster")
        res["cluster"] = {}
        for mode in MODES:
            th, A, k = forced_pitch(mc, 2.8, mode)
            res["cluster"][mode] = {"T": 2.8, "tilt_deg": th, "wave_slope_deg":
                                    float(np.degrees(k * A))}
            print(f"cluster {mode:>8} T 2.8: buoy tilt {th:6.1f}°", flush=True)
    OUT.write_text(json.dumps(res, indent=1))
    plot(res)


def plot(res: dict) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw=dict(width_ratios=[1, 1.35, 0.8]))
    dd = [d for d in res["decay"] if d["drag"]]; d0 = [d for d in res["decay"] if not d["drag"]]
    ax[0].plot([d["theta0_deg"] for d in dd], [100 * d["zeta_first_cycle"] for d in dd], "o-",
               color=COL["relative"], lw=2, label="with Morison drag")
    ax[0].plot([d["theta0_deg"] for d in d0], [100 * d["zeta_first_cycle"] for d in d0], "s--",
               color=COL["none"], lw=1.6, label="radiation only")
    ax[0].axhline(12, color="0.45", ls=":", lw=1.2)
    ax[0].text(1.5, 12.4, "heave damping (field decay) ≈ 12–13 %", fontsize=8, color="0.35")
    ax[0].set_xlabel("initial pitch (deg)"); ax[0].set_ylabel("pitch damping ratio ζ (%)")
    ax[0].set_title(f"(a) Single-buoy pitch free decay, T ≈ {dd[0]['T_s']:.2f} s",
                    fontsize=11, fontweight="bold")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8.5, loc="center right"); ax[0].set_ylim(0, 16)
    for mode in MODES:
        rr = res["buoy"][mode]
        ax[1].plot([r["T"] for r in rr], [r["pitch_deg"] for r in rr], "o-", color=COL[mode],
                   lw=2, label=LAB[mode])
    for sc, ls in (("2", "--"), ("4", "-.")):
        rr = res.get("cd_scale", {}).get(sc, [])
        if rr:
            ax[1].plot([r["T"] for r in rr], [r["pitch_deg"] for r in rr], ls,
                       color=COL["relative"], lw=1.4, label=f"relative drag, all Cd ×{sc}")
    rr = res["buoy"]["relative"]
    ax[1].plot([r["T"] for r in rr], [r["wave_slope_deg"] for r in rr], ":", color="0.4",
               lw=1.6, label="wave slope kA (a buoy that follows the surface)")
    ax[1].axvline(dd[0]["T_s"], color="0.6", lw=0.8)
    ax[1].set_yscale("log"); ax[1].set_xlabel("wave period T (s)")
    ax[1].set_ylabel("pitch amplitude (deg), H = 0.3 m")
    ax[1].set_title("(b) Single buoy in waves, unmoored", fontsize=11, fontweight="bold")
    ax[1].grid(alpha=0.3, which="both"); ax[1].legend(fontsize=8, loc="upper right")
    cl = res.get("cluster", {})
    if cl:
        ax[2].bar(range(3), [cl[mm]["tilt_deg"] for mm in MODES], color=[COL[mm] for mm in MODES])
        ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["relative\ndrag", "calm-water\ndrag",
                                                           "no drag"], fontsize=9)
        for i, mm in enumerate(MODES):
            ax[2].text(i, cl[mm]["tilt_deg"], f"{cl[mm]['tilt_deg']:.0f}°", ha="center",
                       va="bottom", fontsize=9)
        ax[2].set_yscale("log"); ax[2].set_ylabel("buoy tilt amplitude (deg)")
        ax[2].set_title("(c) Cluster tilt mode, T = 2.8 s", fontsize=11, fontweight="bold")
        ax[2].grid(alpha=0.3, axis="y", which="both")
    fig.suptitle("Pitch in the FloatSim model: Morison drag is included, but it damps pitch "
                 "weakly — the resonance is limited only at large angles", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(); fig.savefig(HERE / "pitch_check.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
