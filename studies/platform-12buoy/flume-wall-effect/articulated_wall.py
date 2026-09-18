"""Full articulated 17-body FloatSim run for the flume wall effect: OSU-plate 12-buoy
platform (gimbal-tilt buoys + platform + KKT joints + Morison drag), OPEN vs WALLED
coupled BEM. Free-decay (platform heave release) + regular-wave sweep, all free DOFs.

Usage: python articulated_wall.py <decay|rao>
Reads coupled_osu_open.nc / coupled_osu_walled.nc (from coupled_bem_osu.py).
"""
# ruff: noqa: E402, E702  -- sys.path bootstrap precedes the floatsim imports (E402);
# compact multi-statement setup lines (E702) in a one-off cross-check script.
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

REPO = Path("C:/Users/xlama/OneDrive/Documents/buoy/HSP_code")
SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
warnings.simplefilter("ignore")

from floatsim.driver import build_system
from floatsim.hydro.excitation import make_regular_wave_force
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    Output,
    PlateMember,
    Simulation,
    YawLockedJoint,
    distributed_cylinder_drag,
)
from floatsim.io.deck import RegularWave as DeckWave
from floatsim.solver.equilibrium import solve_static_equilibrium  # noqa: F401
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave

RHO, G = 998.0, 9.806
M_BUOY, IXX, IYY, IZZ = 21.52, 10.2, 10.2, 0.063
CoG_Z = -0.907
ZB, ZH, ZP = -0.907, 0.717, 0.90          # buoy(CoG)/hub(gimbal at spar top)/platform ref z
# OSU drag geometry, body frame (buoy ref at CoG z=-0.907): inertial z + 0.907
_WL_B = 0.0 - ZB                           # waterline body frame = +0.907
_SPAR_BOT_B = -0.967 - ZB                  # -0.060
_PLATE_B = -1.383 - ZB                     # -0.476
SPAR_D, SPAR_CD = 0.1593, 1.2
PLATE_R, PLATE_T, PLATE_CDN, PLATE_CDT = 0.1437, 0.0039, 5.0, 1.5
N_DOF = 17 * 6
OVR = "flume wall-effect: coarse coupled OSU BEM, small-body kernel"


def centers():
    s = 1.25 / 1.5
    out = []
    for pc in np.deg2rad([0, 90, 180, 270]):
        cx, cy = s * np.cos(pc), s * np.sin(pc)
        for tb in np.deg2rad([0, 120, 240]):
            out.append((cx + 0.5 * s * np.cos(tb), cy + 0.5 * s * np.sin(tb)))
    return out


def deck() -> Deck:
    spar = distributed_cylinder_drag(z_bottom=_SPAR_BOT_B, z_top=_WL_B, diameter=SPAR_D,
                                     cd=SPAR_CD, n_segments=10)
    plate = PlateMember(type="plate", center=[0.0, 0.0, _PLATE_B], normal=[0.0, 0.0, 1.0],
                        radius=PLATE_R, thickness=PLATE_T, Cd_n=PLATE_CDN, Cd_t=PLATE_CDT)
    cen = centers()
    s = 1.25 / 1.5
    bodies: list = []
    joints: list = []
    for c, pc in enumerate(np.deg2rad([0, 90, 180, 270])):
        cx, cy = 1.0 * s * np.cos(pc), 1.0 * s * np.sin(pc)
        for b, tb in enumerate(np.deg2rad([0, 120, 240])):
            k = 3 * c + b
            bx, by = cen[k]
            bodies.append(Body(
                name=f"buoy{k + 1}", reference_point=[bx, by, ZB], mass=M_BUOY,
                inertia=Inertia(Ixx=IXX, Iyy=IYY, Izz=IZZ), hydro_body_label=f"buoy{k + 1}",
                initial_conditions=InitialConditions(), drag_elements=[*spar, plate]))
            joints.append(YawLockedJoint(
                type="yaw_locked", body_a=f"buoy{k + 1}", body_b=f"hub{c + 1}",
                attach_a_body=[0.0, 0.0, ZH - ZB],
                attach_b_body=[0.5 * s * np.cos(tb), 0.5 * s * np.sin(tb), 0.0], axis=[0, 0, 1.0]))
        bodies.append(Body(name=f"hub{c + 1}", reference_point=[cx, cy, ZH], mass=2.0,
                           inertia=Inertia(Ixx=0.3, Iyy=0.3, Izz=0.6), structural=True))
        joints.append(YawLockedJoint(
            type="yaw_locked", body_a=f"hub{c + 1}", body_b="platform",
            attach_a_body=[0.0, 0.0, 0.0], attach_b_body=[cx, cy, ZH - ZP], axis=[0, 0, 1.0]))
    bodies.append(Body(name="platform", reference_point=[0.0, 0.0, ZP], mass=6.0,
                       inertia=Inertia(Ixx=6.0, Iyy=6.0, Izz=12.0), structural=True))
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=RHO, gravity=G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=bodies,
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path="placeholder.nc"),
        joints=joints, output=Output(file="o.h5", channels=["heave"], sample_rate=10.0))


def buoy_body_index(k0):
    return 4 * (k0 // 3) + (k0 % 3)


PLAT = 16  # platform body index


def hydro_dof(dk):
    idx = []
    for k, b in enumerate(dk.bodies):
        if b.hydro_body_label is not None:
            idx.extend(range(6 * k, 6 * k + 6))
    return np.asarray(idx, dtype=int)


def setup_for(nc):
    dk = deck()
    shared = read_capytaine(nc)
    setup = build_system(dk, bem_databases={}, dt=0.01, t_max_kernel=30.0,
                         solve_equilibrium=True, shared_hydro_database=shared,
                         asymptote_check_override=OVR, kernel_decay_floor_override=OVR)
    return dk, setup, shared


def fit_amp(t, x, w):
    D = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(D, x, rcond=None)
    return float(np.hypot(c[0], c[1]))


def run_decay(nc):
    _dk, setup, _shared = setup_for(nc)
    ph = 6 * PLAT + 2
    xi0 = setup.xi0.copy(); xi0[2::6] += 0.10   # rigid heave release: every body's heave +0.10 m
    r = integrate_cummins(lhs=setup.lhs, kernel=setup.kernel, xi0=xi0, xi_dot0=setup.xi_dot0,
                          duration=60.0, dt=0.01, rho_inf=0.8, constraints=setup.constraints,
                          state_force=setup.state_force, projection_interval=1)
    out = {}
    x = r.xi[:, ph] - setup.xi0[ph]
    pk, _ = find_peaks(x, height=1e-4)
    out["heave_T"] = float(np.mean(np.diff(r.t[pk][:8])))
    d = np.log(x[pk][:-1] / x[pk][1:]); d = d[np.isfinite(d) & (d > 0)]
    out["heave_zeta"] = float(np.mean(d[:2]) / (2 * np.pi))
    # buoy tilt amplitude (first buoy pitch) as an articulation-mode witness
    bt = 6 * buoy_body_index(0) + 4
    out["buoy1_pitch_max"] = float(np.max(np.abs(r.xi[:, bt] - setup.xi0[bt])))
    return out


# accelerometer points: platform centre (platform body) + 4 cluster centres (the hubs)
ACC_PTS = {"platform centre": 16, "cluster 1": 3, "cluster 2": 7, "cluster 3": 11, "cluster 4": 15}


def run_rao(nc, periods, A=0.05):
    dk, setup, shared = setup_for(nc)
    hd = hydro_dof(dk); ph = 6 * PLAT + 2
    raos = []
    acc = {k: [] for k in ACC_PTS}      # vertical-acceleration RAO (m/s^2 per m) at each point
    for T in periods:
        w = 2 * np.pi / T
        wave = RegularWave(amplitude=A, omega=w, heading_deg=0.0)
        f72 = make_regular_wave_force(hdb=shared, wave=wave, body_position=(0., 0., 0.),
                                      ramp=HalfCosineRamp(duration=15.0))

        def ext(t, _f=f72, _h=hd):
            f = np.zeros(N_DOF); f[_h] = _f(t); return f
        r = integrate_cummins(lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0,
                              xi_dot0=setup.xi_dot0, duration=90.0, dt=0.01, rho_inf=0.8,
                              constraints=setup.constraints, external_force=ext,
                              state_force=setup.state_force, projection_interval=1)
        t = r.t; m = t >= t[-1] - 30.0
        raos.append(fit_amp(t[m], r.xi[m, ph] - setup.xi0[ph], w) / A)
        for k, b in ACC_PTS.items():
            dz = 6 * b + 2
            acc[k].append(w**2 * fit_amp(t[m], r.xi[m, dz] - setup.xi0[dz], w) / A)
    return np.array(raos), {k: np.array(v) for k, v in acc.items()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    what = sys.argv[1] if len(sys.argv) > 1 else "decay"
    ncs = {"open": SCR / "coupled_osu_open_psd.nc", "walled": SCR / "coupled_osu_walled_psd.nc"}
    if what == "decay":
        res = {k: run_decay(v) for k, v in ncs.items()}
        o, w = res["open"], res["walled"]
        print(f"DECAY  heave T: open {o['heave_T']:.3f}s -> walled {w['heave_T']:.3f}s "
              f"({100 * (w['heave_T'] / o['heave_T'] - 1):+.2f}%)")
        print(f"       heave zeta: open {o['heave_zeta'] * 100:.1f}% -> "
              f"walled {w['heave_zeta'] * 100:.1f}%")
        print(f"       buoy1 pitch (articulation): open {o['buoy1_pitch_max']:.4f} -> "
              f"walled {w['buoy1_pitch_max']:.4f} rad")
    else:
        P = np.array([2.19, 2.52, 3.0, 3.5])
        Ro, Ao = run_rao(ncs["open"], P); Rw, Aw = run_rao(ncs["walled"], P)
        print(" T(s)  RAO open  RAO wall  wall%")
        for i, T in enumerate(P):
            print(f"{T:5.2f} {Ro[i]:9.4f} {Rw[i]:9.4f} {100 * (Rw[i] / Ro[i] - 1):+7.1f}")
        print("\nVERTICAL ACCELERATION RAO wall effect (%) at the sensor points:")
        print(" T(s) " + "".join(f"{k.split()[-1]:>9}" for k in ACC_PTS))
        for i, T in enumerate(P):
            print(f"{T:5.2f} "
                  + "".join(f"{100 * (Aw[k][i] / Ao[k][i] - 1):+9.1f}" for k in ACC_PTS))
        np.savez(SCR / "articulated_rao.npz", P=P, Ro=Ro, Rw=Rw,
                 **{f"Ao_{k.replace(' ', '_')}": Ao[k] for k in ACC_PTS},
                 **{f"Aw_{k.replace(' ', '_')}": Aw[k] for k in ACC_PTS})


if __name__ == "__main__":
    main()
