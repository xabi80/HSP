"""Wall effect on accelerations in ALL DOFs at the sensor points (platform centre +
4 cluster centres), articulated 17-body FloatSim. Extends articulated_wall.run_rao,
which only recorded vertical (heave) acceleration.

Head seas (heading 0 deg, +X): surge / heave / pitch are excited; sway / yaw are zero by
port-starboard symmetry; a small roll leaks in because each cluster's three buoys sit at
120 deg (a triangle), which is not perfectly mirror-symmetric about the wave axis. The wall
effect is reported as a percentage on the excited DOFs and as an absolute magnitude on the
(near-)zero DOFs, where a percentage would be ill-defined.

Reads coupled_osu_open_psd.nc / coupled_osu_walled_psd.nc (from coupled_bem_osu.py ->
psd_project.py). Writes accel_multidof.npz + prints the wall-effect table.

Usage: python accel_multidof.py
"""
# ruff: noqa: E402, E702  -- sys.path bootstrap precedes the floatsim/local imports (E402);
# compact multi-statement setup lines (E702) in a one-off cross-check script.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))

import articulated_wall as aw

from floatsim.hydro.excitation import make_regular_wave_force
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave

DOFS = ["surge", "sway", "heave", "roll", "pitch", "yaw"]  # body-local order
EXCITED = ["surge", "heave", "pitch"]                       # non-zero in head seas


def run(nc, periods, A=0.05):
    dk, setup, shared = aw.setup_for(nc)
    hd = aw.hydro_dof(dk)
    # acc[point][dof] = list over periods of the acceleration RAO (m/s^2 or rad/s^2 per m)
    acc = {pt: {d: [] for d in DOFS} for pt in aw.ACC_PTS}
    for T in periods:
        w = 2 * np.pi / T
        wave = RegularWave(amplitude=A, omega=w, heading_deg=0.0)
        f72 = make_regular_wave_force(hdb=shared, wave=wave, body_position=(0.0, 0.0, 0.0),
                                      ramp=HalfCosineRamp(duration=15.0))

        def ext(t, _f=f72, _h=hd):
            f = np.zeros(aw.N_DOF); f[_h] = _f(t); return f
        r = integrate_cummins(lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0,
                              xi_dot0=setup.xi_dot0, duration=90.0, dt=0.01, rho_inf=0.8,
                              constraints=setup.constraints, external_force=ext,
                              state_force=setup.state_force, projection_interval=1)
        t = r.t; m = t >= t[-1] - 30.0
        for pt, b in aw.ACC_PTS.items():
            for j, d in enumerate(DOFS):
                dof = 6 * b + j
                amp = aw.fit_amp(t[m], r.xi[m, dof] - setup.xi0[dof], w)
                acc[pt][d].append(w**2 * amp / A)
        print(f"  {nc.name}  T={T:.2f}s done", flush=True)
    return {pt: {d: np.array(v) for d, v in dd.items()} for pt, dd in acc.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ncs = {"open": SCR / "coupled_osu_open_psd.nc", "walled": SCR / "coupled_osu_walled_psd.nc"}
    P = np.array([2.19, 2.52, 3.0, 3.5])
    Ao = run(ncs["open"], P)
    Aw = run(ncs["walled"], P)

    save = {"P": P}
    for pt in aw.ACC_PTS:
        key = pt.replace(" ", "_")
        for d in DOFS:
            save[f"o_{key}_{d}"] = Ao[pt][d]
            save[f"w_{key}_{d}"] = Aw[pt][d]
    np.savez(SCR / "accel_multidof.npz", **save)

    units = {"surge": "m/s^2", "heave": "m/s^2", "pitch": "rad/s^2"}
    for d in EXCITED:
        print(f"\n=== {d.upper()} acceleration RAO ({units[d]} per m wave amp) — wall effect % ===")
        print(" T(s) " + "".join(f"{pt.split()[-1]:>11}" for pt in aw.ACC_PTS))
        for i, T in enumerate(P):
            row = []
            for pt in aw.ACC_PTS:
                o, wv = Ao[pt][d][i], Aw[pt][d][i]
                row.append(f"{100 * (wv / o - 1):+11.1f}" if abs(o) > 1e-9 else f"{'~0':>11}")
            print(f"{T:5.2f} " + "".join(row))

    print("\n=== symmetry-(near-)zero DOFs (sway/roll/yaw): max |accel| vs peak heave accel ===")
    ref = max(np.max(np.abs(Ao[pt]["heave"])) for pt in aw.ACC_PTS)
    for d in ["sway", "roll", "yaw"]:
        mo = max(np.max(np.abs(Ao[pt][d])) for pt in aw.ACC_PTS)
        mw = max(np.max(np.abs(Aw[pt][d])) for pt in aw.ACC_PTS)
        print(f"  {d:5}: open {mo:.2e}, walled {mw:.2e}  (peak heave accel {ref:.2e}) "
              f"-> {100 * mo / ref:.2f}% of heave")


if __name__ == "__main__":
    main()
