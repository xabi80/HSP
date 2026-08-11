"""Trace the SOURCE of the platform surge drift: ramp impulse vs steady drag
rectification.

The wave excitation is a pure single-frequency sinusoid, so it has zero mean by
construction (confirmed here) -- a wave-force mean is not the source. That leaves
two candidates:
  (A) RAMP IMPULSE  -- the half-cosine startup ramp imparts a one-time net surge
      impulse; afterwards nothing drives surge and it coasts. Signature: the
      terminal drift velocity SHRINKS as the ramp lengthens (adiabatic limit ->
      zero net impulse).
  (B) STEADY DRAG RECTIFICATION -- the calm-water Morison drag on the pitching/
      surging buoys rectifies to a persistent mean surge force during steady
      oscillation. Signature: terminal drift velocity is INDEPENDENT of ramp
      duration (set by the drag/force balance, not the startup).

Discriminator 1: ramp scan {20, 60, 120} s, same case -> compare drift velocity.
Discriminator 2: decompose the net system surge force into wave (ext) and drag
(state_force) parts over a steady window; confirm mean(wave)~0 and report mean(drag).

Case: fin 0.215, T=3.141 s (near platform heave resonance), H=0.08 m.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

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
_T = 3.141
_H = 0.08
_DT = 0.01
_SIDX = [6 * b for b in range(17)]  # every body's surge DOF -> system surge


def _build():  # type: ignore[no-untyped-def]
    hdb = pff._hdb("0215")
    hydro_dof = prp._hydro_dof(prp._deck_with_drag())
    setup = pff._build(0.215, 5.0, hdb)
    return hdb, hydro_dof, setup


def _run(hdb, hydro_dof, setup, ramp_s, dur):  # type: ignore[no-untyped-def]
    omega = 2.0 * np.pi / _T
    wave = RegularWave(amplitude=0.5 * _H, omega=omega, heading_deg=0.0)
    f72 = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0),
                                  ramp=HalfCosineRamp(duration=ramp_s))

    def ext(t):  # type: ignore[no-untyped-def]
        f = np.zeros(102)
        f[hydro_dof] = f72(t)
        return f

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = integrate_cummins(
            lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0, xi_dot0=setup.xi_dot0,
            duration=dur, dt=_DT, rho_inf=0.8, constraints=setup.constraints,
            external_force=ext, state_force=setup.state_force, projection_interval=1)
    return r, ext, omega


def _steady_window(t, omega, n_per=18):  # type: ignore[no-untyped-def]
    T = 2 * np.pi / omega
    return t >= (t[-1] - n_per * T)


def main() -> None:
    hdb, hydro_dof, setup = _build()
    omega = 2.0 * np.pi / _T
    print(f"Case fin 0.215  T={_T}s  H={_H}m   (surge has no restoring; drift source test)\n")
    print(f"{'ramp(s)':>8} {'dur(s)':>7} {'drift v0 (mm/s)':>16} {'excursion(mm)':>14} "
          f"{'osc-amp(mm)':>12}")

    v0 = {}
    r20 = None
    for ramp_s, dur in [(20.0, 180.0), (60.0, 220.0), (120.0, 280.0)]:
        r, ext, _ = _run(hdb, hydro_dof, setup, ramp_s, dur)
        m = _steady_window(r.t, omega)
        surge = r.xi[m, 6 * _PLAT]
        vsurge = r.xi_dot[m, 6 * _PLAT]
        drift = float(np.mean(vsurge)) * 1000  # mm/s (mean surge velocity in steady window)
        A = np.column_stack([np.ones(m.sum()), r.t[m]])
        cf, *_ = np.linalg.lstsq(A, surge, rcond=None)
        osc = 0.5 * (np.ptp(surge - A @ cf)) * 1000
        excursion = (surge[-1] - r.xi[0, 6 * _PLAT]) * 1000
        v0[ramp_s] = drift
        print(f"{ramp_s:8.0f} {dur:7.0f} {drift:16.4f} {excursion:14.1f} {osc:12.1f}")
        if ramp_s == 20.0:
            r20 = (r, ext)

    # Discriminator 2: force decomposition over the steady window (ramp=20 run)
    r, ext = r20
    m = _steady_window(r.t, omega)
    idx = np.where(m)[0]
    fw = np.array([ext(r.t[n])[_SIDX].sum() for n in idx])
    fd = np.array([setup.state_force(r.t[n], r.xi[n], r.xi_dot[n])[_SIDX].sum() for n in idx])
    print("\nNet SYSTEM surge force over steady window (ramp=20), sum over all 17 bodies:")
    print(f"  wave excitation : mean = {fw.mean():+.4e} N   amp = {0.5*np.ptp(fw):.3e} N")
    print(f"  Morison drag    : mean = {fd.mean():+.4e} N   amp = {0.5*np.ptp(fd):.3e} N")
    print(f"  wave+drag       : mean = {(fw+fd).mean():+.4e} N   (net steady surge force)")

    print("\nInterpretation:")
    print(f"  drift v0 vs ramp:  20s -> {v0[20.0]:.3f},  60s -> {v0[60.0]:.3f},  "
          f"120s -> {v0[120.0]:.3f} mm/s")
    ratio = abs(v0[120.0] / v0[20.0]) if v0[20.0] else float("nan")
    print(f"  v0(120s)/v0(20s) = {ratio:.2f}  "
          f"(-> ~1 = steady rectification [B];  << 1 = ramp impulse [A])")


if __name__ == "__main__":
    main()
