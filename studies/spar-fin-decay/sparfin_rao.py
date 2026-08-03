"""Single-buoy (spar-fin) heave RAO + vertical (Nz) acceleration wave sweep.

Part of the 1-vs-3-vs-12 cross-model comparison (single buoy / 3-buoy
articulated cluster / 12-buoy platform). Same grid as the platform fan but at
the corrected model-scale heights:

  heights 0.04..0.12 m (= 2..6 m full-scale at 1:50) x
  periods 2.0..3.3 s   (= 14.1..23.3 s full-scale)  x  plate Cd_n in {5, 1}.

Single 6-DOF rigid body. LHS + retardation kernel from the proven
``study_common`` hand-assembly (carries the small-body Item-25 asymptote
override the spar-fin BEM needs). Drag is IDENTICAL to the platform buoys --
distributed spar cylinder (D=0.1682, Cd=1.2) + heave plate (r=0.215,
Cd_n swept, Cd_t=1.5) -- so the three models differ only in coupling/draft,
not in the per-buoy drag model. Wave forcing via ``make_regular_wave_force``.
Adaptive settle (integrate-until-window-converges), same as the platform fan.

Frame: ``study_common`` references the assembly to the BEM/mesh origin (the
isolated waterline, z=0), CoG offset baked into the mass matrix. Drag element
positions are body-frame relative to that origin (``_build_drag_state_force``
uses ``center`` verbatim), so the plate sits at mesh z=-1.278 and the wetted
spar spans -1.278..0 (length 1.278 -- the isolated buoy floats higher than the
cluster/platform buoys, hence its shorter wetted spar and 2.97 s heave period).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import study_common as sc  # noqa: E402

from floatsim.driver import _build_drag_state_force  # noqa: E402
from floatsim.hydro.excitation import make_regular_wave_force  # noqa: E402
from floatsim.io.deck import (  # noqa: E402
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    Output,
    PlateMember,
    Simulation,
    distributed_cylinder_drag,
)
from floatsim.io.deck import RegularWave as DeckWave  # noqa: E402
from floatsim.solver.equilibrium import solve_static_equilibrium  # noqa: E402
from floatsim.solver.newmark import integrate_cummins  # noqa: E402
from floatsim.solver.ramp import HalfCosineRamp  # noqa: E402
from floatsim.waves.regular import RegularWave  # noqa: E402

_OUT = _HERE / "sparfin_rao_out"
_NC = sc._NC
_HEAVE_DOF = 2
_N_DOF = 6

# Drag geometry -- single-buoy mesh frame (origin at isolated waterline z=0).
_PLATE_Z = sc.PLATE_Z  # -1.278 (mesh); plate at the buoy bottom
_WL_Z = 0.0  # waterline = mesh origin
_SPAR_D = 0.1682  # 2*R_SPAR (cluster_common); identical to platform
_SPAR_CD = 1.2
_PLATE_R = sc.PLATE_RADIUS  # 0.215
_PLATE_T = 0.0039
_PLATE_CD_T = 1.5
_SETTLE_TOL = 0.02

_HEIGHTS = [0.04, 0.06, 0.08, 0.10, 0.12]
_PERIODS = [2.0, 2.5, 2.8, 3.0, 3.141, 3.257, 3.3]


def _drag_deck(cd_n: float) -> Deck:
    spar = distributed_cylinder_drag(
        z_bottom=_PLATE_Z, z_top=_WL_Z, diameter=_SPAR_D, cd=_SPAR_CD, n_segments=10
    )
    plate = PlateMember(
        type="plate",
        center=[0.0, 0.0, _PLATE_Z],
        normal=[0.0, 0.0, 1.0],
        radius=_PLATE_R,
        thickness=_PLATE_T,
        Cd_n=cd_n,
        Cd_t=_PLATE_CD_T,
    )
    body = Body(
        name="buoy",
        reference_point=[0.0, 0.0, 0.0],
        mass=sc.M_BODY,
        inertia=Inertia(Ixx=sc.I_XX, Iyy=sc.I_YY, Izz=sc.I_ZZ),
        hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC)),
        drag_elements=[*spar, plate],
    )
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=sc.RHO, gravity=sc.G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[body],
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def _fit_amplitude(t: NDArray, x: NDArray, omega: float) -> float:
    d = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(d, x, rcond=None)
    return float(np.hypot(c[0], c[1]))


def _win_amp(t: NDArray, x: NDArray, omega: float, t_hi: float, w_s: float) -> float:
    m = (t >= t_hi - w_s) & (t <= t_hi + 1e-9)
    return _fit_amplitude(t[m], x[m], omega)


def run_case(
    lhs,
    kernel,
    hdb,
    drag,
    xi_eq,
    *,
    height_m,
    period_s,
    ramp_s=20.0,
    cap_settle_s=450.0,
    window_periods=6.0,
    dt=0.01,
):  # type: ignore[no-untyped-def]
    amp = 0.5 * height_m
    omega = 2.0 * np.pi / period_s
    window_s = window_periods * period_s
    wave = RegularWave(amplitude=amp, omega=omega, heading_deg=0.0)
    f6 = make_regular_wave_force(
        hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0), ramp=HalfCosineRamp(duration=ramp_s)
    )

    def stop_check(tt, xx):  # type: ignore[no-untyped-def]
        t_now = float(tt[-1])
        if t_now < ramp_s + 2.0 * window_s:
            return False
        a1 = _win_amp(tt, xx[:, _HEAVE_DOF], omega, t_now, window_s)
        a0 = _win_amp(tt, xx[:, _HEAVE_DOF], omega, t_now - window_s, window_s)
        return a1 > 0.0 and abs(a1 - a0) / a1 < _SETTLE_TOL

    duration = ramp_s + cap_settle_s + 2.0 * window_s
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi_eq.copy(),
        xi_dot0=np.zeros(_N_DOF),
        duration=duration,
        dt=dt,
        rho_inf=0.8,
        external_force=f6,
        state_force=drag,
        stop_check=stop_check,
        stop_check_interval=max(1, round(window_s / dt)),
    )
    t = res.t
    dur_used = float(t[-1])
    converged = dur_used < duration - 0.5 * dt
    mask = t >= dur_used - window_s
    tw = t[mask]
    heave = res.xi[mask, _HEAVE_DOF] - xi_eq[_HEAVE_DOF]
    acc = res.xi_ddot[mask, _HEAVE_DOF]
    a1 = _win_amp(t, res.xi[:, _HEAVE_DOF], omega, dur_used, window_s)
    a0 = _win_amp(t, res.xi[:, _HEAVE_DOF], omega, dur_used - window_s, window_s)
    ratio = abs(a1 - a0) / a1 if a1 > 0 else float("nan")
    return {
        "height_m": height_m,
        "period_s": period_s,
        "omega": omega,
        "amp_m": amp,
        "rao_heave": _fit_amplitude(tw, heave, omega) / amp,
        "acc_heave_amp": 0.5 * (acc.max() - acc.min()),
        "acc_heave_peak": float(np.max(np.abs(acc))),
        "settle_ratio": ratio,
        "settled": bool(ratio < _SETTLE_TOL),
        "converged_early": bool(converged),
        "duration_used_s": dur_used,
        "t": tw,
        "heave_m": heave,
        "heave_acc_mps2": acc,
    }


def _write_case_csv(path: Path, case: dict) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "buoy_heave_m", "buoy_heave_acc_mps2"])
        w.writerows(np.column_stack([case["t"], case["heave_m"], case["heave_acc_mps2"]]).tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _OUT.mkdir(exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdb = sc.load_hdb()
        lhs = sc.build_lhs(hdb)
        kernel = sc.build_kernel(hdb)
    tn = 2.0 * np.pi * np.sqrt(lhs.M_plus_Ainf[2, 2] / lhs.C[2, 2])
    print(
        f"single buoy: assembled heave T_n = {tn:.3f} s "
        f"(M+A={lhs.M_plus_Ainf[2, 2]:.2f}, C={lhs.C[2, 2]:.2f}); BEM pre-check 2.966 s"
    )
    eq = solve_static_equilibrium(lhs=lhs, state_force=None)

    heights = [0.08] if args.smoke else _HEIGHTS
    periods = [3.0] if args.smoke else _PERIODS
    for cd in (5.0, 1.0):
        drag = _build_drag_state_force(_drag_deck(cd), n_dof=_N_DOF, rho=sc.RHO)
        rows: list[dict] = []
        for p in periods:
            for h in heights:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c = run_case(lhs, kernel, hdb, drag, eq.xi_eq, height_m=h, period_s=p)
                tag = f"Cd{cd:g}_H{h:g}_T{p:g}".replace(".", "p")
                _write_case_csv(_OUT / f"case_{tag}.csv", c)
                rows.append(
                    dict(
                        height_m=c["height_m"],
                        period_s=c["period_s"],
                        omega=c["omega"],
                        amp_m=c["amp_m"],
                        rao_center=c["rao_heave"],
                        acc_center_amp=c["acc_heave_amp"],
                        acc_center_peak=c["acc_heave_peak"],
                        rao_buoy=c["rao_heave"],
                        acc_buoy_amp=c["acc_heave_amp"],
                        acc_buoy_peak=c["acc_heave_peak"],
                        settled=c["settled"],
                        converged_early=c["converged_early"],
                        duration_used_s=c["duration_used_s"],
                    )
                )
                print(
                    f"  Cd={cd} H={h:.2f} T={p:.3f}: RAO={c['rao_heave']:.4f} "
                    f"acc_amp={c['acc_heave_amp']:.4f} settled={c['settled']} "
                    f"dur={c['duration_used_s']:.0f}s",
                    flush=True,
                )
        if not args.smoke:
            with (_OUT / f"rao_summary_Cd{cd:g}.csv").open("w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                wr.writeheader()
                wr.writerows(rows)
    if not args.smoke:
        with (_OUT / "manifest.json").open("w") as fh:
            json.dump(
                {
                    "model": "single spar-fin buoy (6-DOF)",
                    "T_n_s": tn,
                    "grid_heights_m": _HEIGHTS,
                    "grid_periods_s": _PERIODS,
                    "plate_Cd_n": [5.0, 1.0],
                    "scale": "1:50 model; H 0.04-0.12 m = 2-6 m full-scale; accel Froude-invariant",
                    "drag": "spar D=0.1682 Cd=1.2 (10 seg) + plate r=0.215 Cd_n swept Cd_t=1.5",
                },
                fh,
                indent=2,
            )
    print(f"\nDone. Outputs in {_OUT}")


if __name__ == "__main__":
    main()
