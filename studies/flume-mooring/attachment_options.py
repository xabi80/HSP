"""Flume mooring for the pinned articles (cluster, 4x4 platform): the STATIC comparison of the
attachment options, all in FloatSim (decks from floatsim_decks, FloatSim Catenary lines, the moored
equilibrium settled by FloatSim's constrained integrator and gated on the joint-projected residual).

Why: with the documented design (fairlead at the spar SWL, 0.72 m below the pins) the line
pretension alone tilts each moored pinned buoy about its pin (cluster 4.9 deg, platform 9.1 deg).

Options (``floatsim_decks.mooring_lines`` keywords):
  swl_spar   the documented design (baseline)
  pin_level  fairlead at the spar top = the pin (buoy body z = +1.624, absolute +0.717 m), anchors
             at the same height so the lines stay horizontal: the line acts through the pin
  balanced   each moored spar also gets the line to the anchor at the other end of the flume on
             its side (k and T0 halved, total Kx unchanged): the axial pretension cancels on the
             spar; the lateral components remain
  t0_H0.3    pretension designed for H = 0.3 m instead of 0.5 m (T0 x 0.43): lines go slack above
  t0_H0.2    pretension designed for H = 0.2 m (T0 x 0.23)

Also a drag-free tilt-mode free decay (every buoy pitched theta0 about its own pin, joints
consistent at t = 0) to re-check the "pin level stiffens the tilt mode ~9 %" claim, which came
from the superseded linear-spring model (mooring_verify.py). Dynamic tilts in waves WAIT for the
relative-velocity drag fix (tracker DRAG-WAVE-KINEMATICS-UNWIRED).

Usage:
    python attachment_options.py static <cluster|platform> <option>
    python attachment_options.py tilt <cluster|platform> <free|option>
    python attachment_options.py summary
Writes attachment_options.json (merged, locked).
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import floatsim_decks as fd

from floatsim.driver import build_system
from floatsim.solver.newmark import integrate_cummins

OUT = HERE / "attachment_options.json"
PIN_B = np.array([0.0, 0.0, 1.624])          # the yaw-locked pin, buoy (CoG) frame
Z_PIN = -0.907 + 1.624                        # absolute pin height (+0.717 m)
OPTIONS = {"swl_spar": {}, "pin_level": {"fairlead": PIN_B, "anchor_z": Z_PIN},
           "balanced": {"balanced": True}, "t0_H0.3": {"H_design": 0.3},
           "t0_H0.2": {"H_design": 0.2}}
# pin-level lines hang wholly in AIR: FloatSim's catenary needs their DRY weight (one uniform
# value, no air/water split). The dry weight of the real spring + rope is not known yet, so
# span a plausible range (the decks' 0.02 N/m is an in-water, near-neutral assumption).
for _w in (0.1, 0.3, 1.0):
    OPTIONS[f"pin_level_w{_w:g}"] = {"fairlead": PIN_B, "anchor_z": Z_PIN, "w_line": _w}
THETA0 = np.radians(2.0)
DECAY_S = 40.0


def _tag(article: str, option: str) -> str:
    return article if option == "swl_spar" else f"{article}:{option}"


def _save(key: str, rec: dict) -> None:
    lock = OUT.with_suffix(".lock")
    for _ in range(600):
        try:
            fdl = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        d = json.loads(OUT.read_text()) if OUT.exists() else {}
        d[key] = rec
        OUT.write_text(json.dumps(d, indent=1))
    finally:
        os.close(fdl)
        lock.unlink(missing_ok=True)


def _buoys(dk) -> list[int]:  # type: ignore[no-untyped-def]
    return [k for k, b in enumerate(dk.bodies) if b.hydro_body_label or b.hydro_database]


def static(article: str, option: str) -> None:
    opts = OPTIONS[option]
    xi = fd.moored_equilibrium(article, tag=_tag(article, option), **opts)
    dk0 = fd.deck(article)
    _, lines = fd.moored(dk0, article, **opts)
    eq = json.loads(fd.EQ_CACHE.read_text())[_tag(article, option)]
    b = _buoys(dk0)
    moored_b = sorted({ln["body"] for ln in lines})
    tilt = {k: float(np.degrees(np.hypot(xi[6 * k + 3], xi[6 * k + 4]))) for k in b}
    rest = [k for k, bb in enumerate(dk0.bodies) if k not in b]          # hub(s) / deck
    rec = {"article": article, "option": option, "n_lines": len(lines),
           "T0_per_line_N": sorted({round(ln["T0"], 3) for ln in lines}),
           "k_per_line_N_m": sorted({round(ln["k"], 3) for ln in lines}),
           "max_tilt_moored_deg": max(tilt[k] for k in moored_b),
           "max_tilt_all_deg": max(tilt.values()),
           "max_pitch_moored_deg": max(abs(float(np.degrees(xi[6 * k + 4]))) for k in moored_b),
           "max_roll_moored_deg": max(abs(float(np.degrees(xi[6 * k + 3]))) for k in moored_b),
           "hub_deck_heave_mm": [round(1000 * float(xi[6 * k + 2]), 2) for k in rest],
           "max_buoy_heave_mm": round(1000 * max(abs(float(xi[6 * k + 2])) for k in b), 2),
           "joint_residual_N": eq["joint_residual_N"]}
    _save(f"static:{article}:{option}", rec)
    print(f"{article:8s} {option:9s}: moored-buoy tilt {rec['max_tilt_moored_deg']:.2f} deg "
          f"(pitch {rec['max_pitch_moored_deg']:.2f}, roll {rec['max_roll_moored_deg']:.2f}); "
          f"residual {rec['joint_residual_N']:.3f} N", flush=True)


def lines(article: str, option: str) -> None:
    """Per-line FloatSim catenary forces at the settled equilibrium, the total vertical load
    the lines put on the article, and the surge stiffness Kx from a +-5 mm rigid surge of
    every body (FloatSim's own catenary force, central difference)."""
    opts = OPTIONS[option]
    xi = fd.moored_equilibrium(article, tag=_tag(article, option), **opts)
    dk0 = fd.deck(article, drag=False)
    dkm, lns = fd.moored(dk0, article, **opts)
    s = build_system(fd.with_positions(dkm, xi), dt=fd.DT, t_max_kernel=fd.T_KERNEL,
                     solve_equilibrium=False, **fd.hdbs(article))
    n = s.lhs.n_dof
    z = np.zeros(n)
    f0 = s.state_force(0.0, xi, z)
    fz = sum(float(f0[6 * k + 2]) for k in range(n // 6))
    d = 0.005
    sh = np.zeros(n)
    sh[0::6] = d
    fx = [sum(float(s.state_force(0.0, xi + sg * sh, z)[6 * k]) for k in range(n // 6))
          for sg in (1.0, -1.0)]
    kx = -(fx[0] - fx[1]) / (2 * d)
    b = _buoys(dk0)
    tiltmax = max(float(np.degrees(np.hypot(xi[6 * k + 3], xi[6 * k + 4]))) for k in b)
    rec = {"article": article, "option": option, "w_line": lns[0]["w"],
           "total_vertical_line_force_N": fz, "Kx_lines_N_per_m": kx,
           "Kx_design_N_per_m": fd.line_design(article, len(b))["Kx"],
           "max_buoy_tilt_deg": tiltmax}
    _save(f"lines:{article}:{option}", rec)
    print(f"{article:8s} {option:14s}: w {rec['w_line']:.2f} N/m  vertical line load "
          f"{fz:+.3f} N  Kx {kx:.2f} N/m (design {rec['Kx_design_N_per_m']:.2f})  "
          f"max buoy tilt {tiltmax:.3f} deg", flush=True)


def tilt(article: str, option: str) -> None:
    """Drag-free free decay of the in-phase buoy-tilt mode; period from the band-passed mean
    buoy pitch (1.7-5 s band)."""
    warnings.simplefilter("ignore")
    dk0 = fd.deck(article, drag=False)
    hd = fd.hdbs(article)
    if option == "free":
        setup = build_system(dk0, dt=fd.DT, t_max_kernel=fd.T_KERNEL, solve_equilibrium=True,
                             **hd)
        dk = dk0
    else:
        opts = OPTIONS[option]
        # the static equilibrium does not depend on drag, so the (drag) deck's settle applies
        xi_eq = fd.moored_equilibrium(article, tag=_tag(article, option), **opts)
        dk, _ = fd.moored(dk0, article, **opts)
        setup = build_system(fd.with_positions(dk, xi_eq), dt=fd.DT, t_max_kernel=fd.T_KERNEL,
                             solve_equilibrium=False, **hd)
    b = _buoys(dk)
    xi0 = setup.xi0.copy()
    for k in b:                                   # rotate about the pin: pin point stays put
        xi0[6 * k + 4] += THETA0
        xi0[6 * k + 0] -= PIN_B[2] * THETA0
    r = integrate_cummins(lhs=setup.lhs, kernel=setup.kernel, xi0=xi0, xi_dot0=setup.xi_dot0,
                          duration=DECAY_S, dt=fd.DT, rho_inf=0.8, constraints=setup.constraints,
                          state_force=setup.state_force, projection_interval=1)
    th = np.mean([r.xi[:, 6 * k + 4] - setup.xi0[6 * k + 4] for k in b], axis=0)
    bb, aa = butter(3, [1 / 5.0, 1 / 1.7], btype="band", fs=1 / fd.DT)
    f = filtfilt(bb, aa, th)
    m = (r.t > 2.0) & (r.t < DECAY_S - 2.0)
    pk, _ = find_peaks(f[m], distance=int(1.5 / fd.DT))
    T = float(np.mean(np.diff(r.t[m][pk])))
    rec = {"article": article, "option": option, "T_tilt_s": T, "n_peaks": int(pk.size),
           "theta0_deg": float(np.degrees(THETA0)), "decay_s": DECAY_S, "drag": False}
    _save(f"tilt:{article}:{option}", rec)
    print(f"{article:8s} {option:9s}: tilt-mode period {T:.4f} s ({pk.size} peaks)", flush=True)


def summary() -> None:
    d = json.loads(OUT.read_text())
    print("STATIC (FloatSim settled moored equilibrium, no waves)")
    for art in ("cluster", "platform"):
        for o in OPTIONS:
            r = d.get(f"static:{art}:{o}")
            if r:
                print(f"  {art:8s} {o:9s} lines {r['n_lines']:>2}  T0/line {r['T0_per_line_N']}  "
                      f"moored tilt {r['max_tilt_moored_deg']:.2f} deg (pitch "
                      f"{r['max_pitch_moored_deg']:.2f}, roll {r['max_roll_moored_deg']:.2f})  "
                      f"res {r['joint_residual_N']:.3f} N")
    print("TILT MODE (drag-free free decay)")
    for art in ("cluster", "platform"):
        f = d.get(f"tilt:{art}:free")
        for o in ("free", *OPTIONS):
            r = d.get(f"tilt:{art}:{o}")
            if r:
                sh = (f"{100 * (r['T_tilt_s'] / f['T_tilt_s'] - 1):+.2f} %"
                      if f and o != "free" else "")
                print(f"  {art:8s} {o:9s} T {r['T_tilt_s']:.4f} s  {sh}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore")
    step = sys.argv[1]
    if step == "static":
        static(sys.argv[2], sys.argv[3])
    elif step == "tilt":
        tilt(sys.argv[2], sys.argv[3])
    elif step == "lines":
        lines(sys.argv[2], sys.argv[3])
    else:
        summary()
