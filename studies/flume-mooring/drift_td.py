"""Relative-velocity mean wave drift on the three test articles, from drag-limited FloatSim
time-domain runs (refines the fixed-body upper bound of mooring_sizing.py).

The mean drift on these slender spars is splash-zone drag: only the intermittently wetted strip
between the spar's waterline and the free surface carries a non-zero cycle-mean drag. For a spar
held fixed that gives  F = (2/3 pi) rho Cd D A U^2  (the upper bound used so far). A spar that
moves with the wave sees the RELATIVE elevation eta_r = eta - z_wl and the RELATIVE velocity
u_r = u - x_dot_wl at its waterline point, so per spar

    F = 1/2 rho Cd D < eta_r |u_r| u_r >        (cycle mean; fixed body: eta_r = eta, u_r = u)

The body motion comes from the FloatSim model of each article, run in the time domain with:
  * the linear BEM excitation / radiation (as everywhere in these studies);
  * Morison drag on the spar + heave plate with the WAVE-RELATIVE velocity (the driver's drag is
    calm-water; here the same elements are rebuilt with deep-water Airy kinematics, ramped);
  * the design station-keeping mooring (mooring_verify.mooring_design at T_surge = 15 s, lines
    at the waterline of the recommended spars) as an anisotropic linear spring.
Deep-water kinematics match the deep-water BEM; the ratio R = F_rel / F_fixed is applied to the
flume-depth fixed-body drift in drift_refined.py. Wave height H (default 0.5 m) is capped at the
H/L = 1/15 steepness limit, as in mooring_sizing.py.

Usage: python drift_td.py <buoy|cluster|platform> <T1,T2,...> [--H 0.5]
Writes one JSON per case to drift_rows/.
"""
# ruff: noqa: E402, E702  -- env + sys.path bootstrap precede the study imports; compact lines
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PLAT_ROT_DEG", "45")

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent.parent, HERE.parent / "platform-12buoy" / "flume-wall-effect",
           HERE.parent / "osu-test-buoy", HERE):
    sys.path.insert(0, str(_p))

import articulated_wall as aw
import mooring_sizing as ms
import mooring_verify as mv
import osu_buoy_common as obc

from floatsim.driver import build_system
from floatsim.hydro.excitation import make_regular_wave_force
from floatsim.hydro.morison import MorisonElement, PlateDragElement, make_morison_state_force
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import PlateMember
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave

OUT = HERE / "drift_rows"
RAMP_S, DT = 15.0, 0.01
N_WIN = 10                                  # analysis window: last 10 wave periods
RHO, G = ms.RHO, ms.G


# ------------------------------------------------------------------ model assembly
def _deck_elements(dk) -> list:
    """The deck's drag elements as Morison elements (same conversion as the driver)."""
    el: list = []
    for k, body in enumerate(dk.bodies):
        for e in body.drag_elements:
            if isinstance(e, PlateMember):
                el.append(PlateDragElement(
                    body_index=k, center_body=np.asarray(e.center, float),
                    normal_body=np.asarray(e.normal, float), radius=e.radius,
                    thickness=e.thickness, Cd_n=e.Cd_n, Cd_t=e.Cd_t,
                    n_radial=e.n_radial, n_azimuthal=e.n_azimuthal))
            else:
                el.append(MorisonElement(
                    body_index=k, node_a_body=np.asarray(e.node_a, float),
                    node_b_body=np.asarray(e.node_b, float), diameter=e.diameter, Cd=e.Cd,
                    include_inertia=False))
    return el


def _buoy_elements() -> list:
    """Single buoy (waterline reference): osu_buoy_common.make_drag's elements."""
    el: list = []
    edges = np.linspace(obc._SPAR_BOT, obc._WL, 11)
    for i in range(10):
        el.append(MorisonElement(body_index=0, node_a_body=np.array([0.0, 0.0, edges[i]]),
                                 node_b_body=np.array([0.0, 0.0, edges[i + 1]]),
                                 diameter=obc._SPAR_D, Cd=obc._SPAR_CD))
    el.append(PlateDragElement(body_index=0, center_body=np.array([0.0, 0.0, obc._PLATE_Z]),
                               normal_body=np.array([0.0, 0.0, 1.0]), radius=obc._PLATE_R,
                               thickness=obc._PLATE_T, Cd_n=obc._PLATE_CD_N,
                               Cd_t=obc._PLATE_CD_T))
    return el


def build(article: str) -> dict:
    """lhs/kernel/xi0/constraints + drag elements + spars + mooring points for one article."""
    if article == "buoy":
        hdb = obc.load_hdb()
        return dict(name="1 buoy", n=6, hdb=hdb, lhs=obc.build_lhs(hdb),
                    kernel=obc.build_kernel(hdb), xi0=np.zeros(6), constraints=None,
                    hydro_dof=np.arange(6), elements=_buoy_elements(),
                    spars=[(0, np.zeros(3), np.zeros(3))], moor=[(0, (0.0, 0.0, 0.0))])
    if article == "cluster":
        dk, nc, name = mv._cluster_deck(), HERE / "cluster_osu_open_rot45_psd.nc", \
            "1 cluster (4 buoys)"
    else:
        dk, nc, name = aw.deck(), mv.FWE / f"coupled_osu_open{aw.SUF}_psd.nc", \
            "4x4 platform (45°)"
    hdb = read_capytaine(nc)
    setup = build_system(dk, bem_databases={}, dt=DT, t_max_kernel=30.0, solve_equilibrium=True,
                         shared_hydro_database=hdb, asymptote_check_override=aw.OVR,
                         kernel_decay_floor_override=aw.OVR)
    buoys = [k for k, b in enumerate(dk.bodies) if b.hydro_body_label is not None]
    wl = np.array([0.0, 0.0, -aw.ZB])            # spar waterline in the buoy (CoG) frame
    return dict(name=name, n=setup.lhs.M_plus_Ainf.shape[0], hdb=hdb, lhs=setup.lhs,
                kernel=setup.kernel, xi0=np.asarray(setup.xi0), constraints=setup.constraints,
                hydro_dof=aw.hydro_dof(dk), elements=_deck_elements(dk),
                spars=[(k, np.asarray(dk.bodies[k].reference_point, float), wl) for k in buoys],
                moor=[(k, tuple(wl)) for k in mv._outer_buoys(dk, "rows")])


# ------------------------------------------------------------------ one regular-wave case
def simulate(m: dict, T: float, H: float, n_win: int = N_WIN):  # type: ignore[no-untyped-def]
    """Drag-limited, moored time-domain run in a regular wave. Returns the integration result
    and the wave (H_used, A, omega, k); the last n_win periods are the settled window."""
    Hu = float(ms.drift_per_spar(H, T)[2]); A = 0.5 * Hu
    w = 2 * np.pi / T; k = w * w / G
    ramp = HalfCosineRamp(duration=RAMP_S)

    def fluid_u(p, t):
        e = np.exp(k * min(float(p[2]), 0.0)); th = w * t - k * float(p[0])
        return ramp.value(t) * A * w * e * np.array([np.cos(th), 0.0, -np.sin(th)])

    drag = make_morison_state_force(m["elements"], n_dof=m["n"], fluid_velocity_fn=fluid_u,
                                    rho=RHO)
    f_spar05 = max(sum(ms.drift_per_spar(0.5, Tw)[:2]) for Tw in ms.T_WAVE)
    kxyz, kyaw = mv.mooring_design({"name": m["name"], "nspar": len(m["spars"])}, mv.T_SURGE,
                                   f_spar05)
    K = mv.mooring_k(m["n"], m["moor"], kxyz, kyaw); xi0 = m["xi0"]
    # drag yaw moment of each buoy about its own axis is physically nil (axisymmetric spar +
    # disc), but the discretised plate's tangential drag on the tiny buoy yaw inertia
    # (0.063 kg m^2) makes the explicitly lagged state force numerically unstable -> zero it
    yaw = np.array([6 * b + 5 for (b, _p, _r) in m["spars"]])

    def state(t, xi, xd):
        f = drag(t, xi, xd); f[yaw] = 0.0
        return f - K @ (xi - xi0)

    f_wave = make_regular_wave_force(hdb=m["hdb"], wave=RegularWave(amplitude=A, omega=w,
                                                                    heading_deg=0.0),
                                     body_position=(0.0, 0.0, 0.0), ramp=ramp)
    hd = m["hydro_dof"]

    def ext(t):
        f = np.zeros(m["n"]); f[hd] = f_wave(t); return f

    dur = RAMP_S + 30.0 + n_win * T
    r = integrate_cummins(lhs=m["lhs"], kernel=m["kernel"], xi0=xi0, xi_dot0=np.zeros(m["n"]),
                          duration=dur, dt=DT, rho_inf=0.8, constraints=m["constraints"],
                          external_force=ext, state_force=state, projection_interval=1)
    return r, dict(H_used=Hu, A=A, omega=w, k=k)


def run_case(m: dict, T: float, H: float) -> dict:
    r, wv = simulate(m, T, H)
    Hu, A, w, k, xi0 = wv["H_used"], wv["A"], wv["omega"], wv["k"], m["xi0"]
    msk = r.t >= r.t[-1] - N_WIN * T + 0.5 * DT
    t = r.t[msk]; X = r.xi[msk] - xi0; V = r.xi_dot[msk]
    c = 0.5 * RHO * ms.CD_SPAR * ms.SPAR_D
    rel, fix, per, tilt = 0.0, 0.0, [], 0.0
    for (b, p0, roff) in m["spars"]:
        rot, om = X[:, 6 * b + 3:6 * b + 6], V[:, 6 * b + 3:6 * b + 6]
        d = X[:, 6 * b:6 * b + 3] + np.cross(rot, roff)          # waterline-point displacement
        v = V[:, 6 * b:6 * b + 3] + np.cross(om, roff)           # waterline-point velocity
        th = w * t - k * (p0[0] + d[:, 0]); th0 = w * t - k * p0[0]
        eta_r = A * np.cos(th) - d[:, 2]; u_r = A * w * np.cos(th) - v[:, 0]
        eta, u = A * np.cos(th0), A * w * np.cos(th0)
        f_rel = c * float(np.mean(eta_r * np.abs(u_r) * u_r))
        f_fix = c * float(np.mean(eta * np.abs(u) * u))
        rel += f_rel; fix += f_fix; per.append(f_rel / f_fix)
        tilt = max(tilt, float(np.max(np.hypot(rot[:, 0], rot[:, 1]))))
    return dict(article=m["name"], T=T, H_req=H, H_used=Hu, A=A, n_spar=len(m["spars"]),
                F_rel_N=rel, F_fixed_N=fix, R=rel / fix, R_per_spar_min=min(per),
                R_per_spar_max=max(per), max_tilt_rad=tilt,
                F_fixed_analytic_N=len(m["spars"]) * (2 / (3 * np.pi)) * RHO * ms.CD_SPAR
                * ms.SPAR_D * A * (A * w) ** 2)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("article", choices=["buoy", "cluster", "platform"])
    ap.add_argument("periods")
    ap.add_argument("--H", type=float, default=0.5)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    m = build(a.article)
    for T in [float(x) for x in a.periods.split(",")]:
        jp = OUT / (f"{a.article}_H{a.H:g}_T{T:g}".replace(".", "p") + ".json")
        if jp.exists():
            continue
        t0 = time.perf_counter()
        row = run_case(m, T, a.H); row["wall_min"] = (time.perf_counter() - t0) / 60
        jp.write_text(json.dumps(row))
        print(f"{a.article} T {T:.2f} H {row['H_used']:.3f}: F_rel {row['F_rel_N']:.2f} N, "
              f"F_fixed {row['F_fixed_N']:.2f} N (analytic {row['F_fixed_analytic_N']:.2f}), "
              f"R {row['R']:.3f} [{row['R_per_spar_min']:.2f}-{row['R_per_spar_max']:.2f}], "
              f"tilt {np.degrees(row['max_tilt_rad']):.1f}°, {row['wall_min']:.1f} min",
              flush=True)


if __name__ == "__main__":
    main()
