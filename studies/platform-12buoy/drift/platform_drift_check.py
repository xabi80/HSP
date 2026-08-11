"""Drift check for the 12-buoy platform simulations.

Two independent kinds of drift:
  (1) NUMERICAL / constraint drift -- do the velocity-level KKT joints hold, or
      do the bodies slowly separate?  Measured by max|phi(xi)| (joint-gap
      violation) over the whole run; the integrator projects every step, so this
      should sit at the projection floor.
  (2) PHYSICAL secular drift -- surge/sway/yaw have NO hydrostatic restoring and
      the platform is unmoored, so any net mean force (quadratic Morison drag
      rectification, cross-buoy wave-force asymmetry) makes them translate.
      Heave/roll/pitch are buoyancy-restored and should stay bounded about a
      fixed mean.  Measured by a linear fit of the platform reference DOF over
      the back half of the run + the per-DOF mean excursion.

Runs a fixed long duration per case (full trajectory captured), for the study's
own build (pff._hdb/_build + prp.run_case force setup).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_STUDY = Path(__file__).resolve().parent.parent  # studies/platform-12buoy (this file: drift/)
sys.path.insert(0, str(_STUDY / "fin_study"))
sys.path.insert(0, str(_STUDY))
sys.path.insert(0, str(_STUDY.parent / "cluster-3buoy-rigid"))

import platform_fin_fan as pff  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402
from floatsim.hydro.excitation import make_regular_wave_force  # noqa: E402
from floatsim.solver.newmark import integrate_cummins  # noqa: E402
from floatsim.solver.ramp import HalfCosineRamp  # noqa: E402
from floatsim.waves.regular import RegularWave  # noqa: E402

_PLAT = 16
_DOF = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
_UNIT = ["m", "m", "m", "rad", "rad", "rad"]
_DUR = 220.0
_DT = 0.01
_RAMP = 20.0
_CASES = [("0215", 3.141, 0.08), ("none", 2.500, 0.08)]


def _run(tag, T, H):  # type: ignore[no-untyped-def]
    plate_r = 0.215 if tag == "0215" else (0.15 if tag == "015" else pff._R_SPAR)
    hdb = pff._hdb(tag)
    hydro_dof = prp._hydro_dof(prp._deck_with_drag())
    setup = pff._build(plate_r, 5.0, hdb)
    omega = 2.0 * np.pi / T
    wave = RegularWave(amplitude=0.5 * H, omega=omega, heading_deg=0.0)
    f72 = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0),
                                  ramp=HalfCosineRamp(duration=_RAMP))

    def ext(t):  # type: ignore[no-untyped-def]
        f = np.zeros(102)
        f[hydro_dof] = f72(t)
        return f

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = integrate_cummins(
            lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0, xi_dot0=setup.xi_dot0,
            duration=_DUR, dt=_DT, rho_inf=0.8, constraints=setup.constraints,
            external_force=ext, state_force=setup.state_force, projection_interval=1)
    return r, setup, omega


def _analyse(tag, T, H, r, setup, omega):  # type: ignore[no-untyped-def]
    t = r.t
    print(f"\n=== fin {tag}  T={T} s  H={H} m   ({t.size} steps, {t[-1]:.0f} s) ===")

    # (1) constraint (joint) drift -- evaluate phi over the run
    phi_max = 0.0
    for n in range(0, t.size, 50):
        phi_max = max(phi_max, float(np.max(np.abs(setup.constraints.phi(r.xi[n])))))
    phi_end = float(np.max(np.abs(setup.constraints.phi(r.xi[-1]))))
    print(f"  [numerical] max|joint gap phi| over run = {phi_max:.2e} m   (at end {phi_end:.2e})")

    # (2) physical secular drift of the platform reference point
    seg = t >= 120.0
    tt = t[seg]
    A = np.column_stack([np.ones_like(tt), tt])
    period = 2 * np.pi / omega
    print(f"  [physical] platform reference DOF, linear fit over t=[120,{t[-1]:.0f}]s:")
    print(f"    {'DOF':6} {'slope':>13} {'drift/100s':>12} {'mean':>11} {'osc-amp':>10}")
    for j in range(6):
        y = r.xi[seg, 6 * _PLAT + j]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope = coef[1]
        resid = y - A @ coef
        osc = 0.5 * (resid.max() - resid.min())
        mean = float(np.mean(y))
        u = _UNIT[j]
        print(f"    {_DOF[j]:6} {slope:10.2e} {u}/s {slope*100:9.2e} {u} "
              f"{mean:8.2e} {u} {osc:7.2e} {u}")

    # per-period mean of surge (drift signature) -- first vs last 5 periods
    def pmean(dof, lo, hi):  # type: ignore[no-untyped-def]
        m = (t >= lo) & (t < hi)
        return float(np.mean(r.xi[m, 6 * _PLAT + dof]))
    tw = t[-1]
    for j in (0, 1, 5):  # surge, sway, yaw
        early = pmean(j, 60, 60 + 5 * period)
        late = pmean(j, tw - 5 * period, tw)
        print(f"    surge/sway/yaw check -- {_DOF[j]:5}: mean(early)={early:.2e} "
              f"mean(late)={late:.2e}  Delta={late - early:.2e} {_UNIT[j]}")

    # drift WITHIN the study's 6-period measurement window (does it bias the RAO fit?)
    win = t >= (tw - 6 * period)
    for j in (2,):  # heave (the measured channel)
        y = r.xi[win, 6 * _PLAT + j]
        A2 = np.column_stack([np.ones(win.sum()), t[win]])
        c2, *_ = np.linalg.lstsq(A2, y, rcond=None)
        print(f"  [measurement] heave trend inside 6-period RAO window: "
              f"slope={c2[1]:.2e} m/s (drift {c2[1] * 6 * period:.2e} m over the window)")


def main() -> None:
    for tag, T, H in _CASES:
        r, setup, omega = _run(tag, T, H)
        _analyse(tag, T, H, r, setup, omega)
    print("\nDone.")


if __name__ == "__main__":
    main()
