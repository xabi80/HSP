"""3-buoy ARTICULATED cluster heave RAO + vertical (Nz) acceleration wave sweep.

Part of the 1-vs-3-vs-12 cross-model comparison. The cluster is 3 spar-fin
buoys (0.5 m radius, 0/120/240 deg) articulated (``yaw_locked``) to a central
hub -- i.e. exactly one cluster of the 12-buoy platform, with the hub free
instead of joined to a platform. Coupled 18-DOF hydro
(``capytaine_multibody_18dof.nc``) + per-body single-buoy hydrostatic
(``reference_single_bem.nc``), the M10 PR2 / M11a PR4 build. Two contaminated
BEM frequencies (4.934, 20.909 rad/s) are dropped (M11a PR3 conditioning).

Grid (same as the platform / single-buoy fans): heights 0.04..0.12 m
(= 2..6 m full-scale at 1:50) x periods 2.0..3.3 s x plate Cd_n in {5, 1}.
Drag identical to the platform buoys: spar D=0.1682 Cd=1.2 (10 seg) + plate
r=0.215 Cd_n swept Cd_t=1.5, in the buoy body frame (plate at body z=-0.2617,
wetted spar -0.2617..+1.1957). Outputs at the hub centre and buoy 1.
Adaptive settle. Heave natural period ~3.09-3.11 s (BEM pre-check / M10).
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
from scipy.linalg import eigh, null_space

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from floatsim.driver import build_system  # noqa: E402
from floatsim.hydro.database import HydroDatabase  # noqa: E402
from floatsim.hydro.excitation import make_regular_wave_force  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402
from floatsim.io.deck import (  # noqa: E402
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
from floatsim.io.deck import RegularWave as DeckWave  # noqa: E402
from floatsim.solver.newmark import integrate_cummins  # noqa: E402
from floatsim.solver.ramp import HalfCosineRamp  # noqa: E402
from floatsim.waves.regular import RegularWave  # noqa: E402

_NC = _HERE / "capytaine_multibody_18dof.nc"
_REF = _HERE / "reference_single_bem.nc"
_OUT = _HERE / "cluster_rao_out"
_CONTAM = (4.934, 20.909)
_R, _ANG = 0.5, np.deg2rad([0.0, 120.0, 240.0])
_ZB, _ZA = -1.1956674320202696, 0.4933695679797303
_RHO, _G = 1025.0, 9.81

_ZPLATE_BODY = -0.2617  # plate body z (identical to platform buoys)
_ZWL_BODY = -_ZB  # waterline body z = +1.19567
_SPAR_D, _SPAR_CD = 0.1682, 1.2
_PLATE_R, _PLATE_T, _PLATE_CD_T = 0.215, 0.0039, 1.5
_ASYMPTOTE_OVR = "cluster small-body spar-fin hulls (ITEM25)"

_N_DOF = 24
_HUB_HEAVE = 6 * 3 + 2  # hub is body index 3 -> DOF 20
_BUOY1_HEAVE = 2
_SETTLE_TOL = 0.02
_HEIGHTS = [0.04, 0.06, 0.08, 0.10, 0.12]
_PERIODS = [2.0, 2.5, 2.8, 3.0, 3.141, 3.257, 3.3]


def _hdb18() -> HydroDatabase:
    h = read_capytaine(_NC)
    w = np.asarray(h.omega)
    drop = {int(np.argmin(np.abs(w - c))) for c in _CONTAM}
    keep = np.array([k for k in range(w.size) if k not in drop])
    return HydroDatabase(
        omega=h.omega[keep],
        heading_deg=h.heading_deg,
        A=h.A[:, :, keep],
        B=h.B[:, :, keep],
        A_inf=h.A_inf,
        C=h.C,
        RAO=h.RAO[:, keep, :],
        reference_point=h.reference_point,
        C_source=h.C_source,
        metadata=dict(h.metadata),
        body_labels=h.body_labels,
    )


def _deck(cd_n: float) -> Deck:
    spar = distributed_cylinder_drag(
        z_bottom=_ZPLATE_BODY, z_top=_ZWL_BODY, diameter=_SPAR_D, cd=_SPAR_CD, n_segments=10
    )
    plate = PlateMember(
        type="plate",
        center=[0.0, 0.0, _ZPLATE_BODY],
        normal=[0.0, 0.0, 1.0],
        radius=_PLATE_R,
        thickness=_PLATE_T,
        Cd_n=cd_n,
        Cd_t=_PLATE_CD_T,
    )
    buoys = [
        Body(
            name=f"buoy{i + 1}",
            reference_point=[_R * np.cos(a), _R * np.sin(a), _ZB],
            mass=28.67,
            inertia=Inertia(Ixx=24.0, Iyy=24.0, Izz=0.114),
            hydro_body_label=f"buoy{i + 1}",
            initial_conditions=InitialConditions(),
            drag_elements=[*spar, plate],
        )
        for i, a in enumerate(_ANG)
    ]
    hub = Body(
        name="hub",
        reference_point=[0.0, 0.0, _ZA],
        mass=12.0,
        inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0),
        structural=True,
    )
    joints = [
        YawLockedJoint(
            type="yaw_locked",
            body_a=f"buoy{i + 1}",
            body_b="hub",
            attach_a_body=[0.0, 0.0, _ZA - _ZB],
            attach_b_body=[_R * np.cos(a), _R * np.sin(a), 0.0],
            axis=[0.0, 0.0, 1.0],
        )
        for i, a in enumerate(_ANG)
    ]
    return Deck(
        simulation=Simulation(duration=50.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=_RHO, gravity=_G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[*buoys, hub],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC)),
        hydrostatic_database=HydroDatabaseRef(format="capytaine", path=str(_REF)),
        joints=joints,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def _fit_amplitude(t: NDArray, x: NDArray, omega: float) -> float:
    d = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(d, x, rcond=None)
    return float(np.hypot(c[0], c[1]))


def _win_amp(t, x, omega, t_hi, w_s):  # type: ignore[no-untyped-def]
    m = (t >= t_hi - w_s) & (t <= t_hi + 1e-9)
    return _fit_amplitude(t[m], x[m], omega)


def run_case(
    setup, hdb, *, height_m, period_s, ramp_s=20.0, cap_settle_s=450.0, window_periods=6.0, dt=0.01
):  # type: ignore[no-untyped-def]
    amp = 0.5 * height_m
    omega = 2.0 * np.pi / period_s
    window_s = window_periods * period_s
    wave = RegularWave(amplitude=amp, omega=omega, heading_deg=0.0)
    f18 = make_regular_wave_force(
        hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0), ramp=HalfCosineRamp(duration=ramp_s)
    )

    def ext(t):  # type: ignore[no-untyped-def]
        f = np.zeros(_N_DOF)
        f[:18] = f18(t)
        return f

    def stop_check(tt, xx):  # type: ignore[no-untyped-def]
        t_now = float(tt[-1])
        if t_now < ramp_s + 2.0 * window_s:
            return False
        a1 = _win_amp(tt, xx[:, _HUB_HEAVE], omega, t_now, window_s)
        a0 = _win_amp(tt, xx[:, _HUB_HEAVE], omega, t_now - window_s, window_s)
        return a1 > 0.0 and abs(a1 - a0) / a1 < _SETTLE_TOL

    duration = ramp_s + cap_settle_s + 2.0 * window_s
    res = integrate_cummins(
        lhs=setup.lhs,
        kernel=setup.kernel,
        xi0=np.zeros(_N_DOF),
        xi_dot0=np.zeros(_N_DOF),
        duration=duration,
        dt=dt,
        rho_inf=0.8,
        constraints=setup.constraints,
        external_force=ext,
        state_force=setup.state_force,
        projection_interval=1,
        stop_check=stop_check,
        stop_check_interval=max(1, round(window_s / dt)),
    )
    t = res.t
    dur_used = float(t[-1])
    converged = dur_used < duration - 0.5 * dt
    mask = t >= dur_used - window_s
    tw = t[mask]
    out = {
        "height_m": height_m,
        "period_s": period_s,
        "omega": omega,
        "amp_m": amp,
        "converged_early": bool(converged),
        "duration_used_s": dur_used,
        "t": tw,
    }
    a1 = _win_amp(t, res.xi[:, _HUB_HEAVE], omega, dur_used, window_s)
    a0 = _win_amp(t, res.xi[:, _HUB_HEAVE], omega, dur_used - window_s, window_s)
    out["settle_ratio"] = abs(a1 - a0) / a1 if a1 > 0 else float("nan")
    out["settled"] = bool(out["settle_ratio"] < _SETTLE_TOL)
    for name, dof in [("center", _HUB_HEAVE), ("buoy1", _BUOY1_HEAVE)]:
        hv = res.xi[mask, dof]
        ac = res.xi_ddot[mask, dof]
        out[f"rao_{name}_heave"] = _fit_amplitude(tw, hv, omega) / amp
        out[f"acc_{name}_amp"] = 0.5 * (ac.max() - ac.min())
        out[f"acc_{name}_peak"] = float(np.max(np.abs(ac)))
        out[f"{name}_heave_m"] = hv
        out[f"{name}_heave_acc_mps2"] = ac
    return out


def _write_case_csv(path: Path, case: dict) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "t_s",
                "center_heave_m",
                "center_heave_acc_mps2",
                "buoy1_heave_m",
                "buoy1_heave_acc_mps2",
            ]
        )
        w.writerows(
            np.column_stack(
                [
                    case["t"],
                    case["center_heave_m"],
                    case["center_heave_acc_mps2"],
                    case["buoy1_heave_m"],
                    case["buoy1_heave_acc_mps2"],
                ]
            ).tolist()
        )


def _build(cd_n, hdb):  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(
            _deck(cd_n),
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=hdb,
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override=_ASYMPTOTE_OVR,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _OUT.mkdir(exist_ok=True)
    hdb = _hdb18()

    setup5 = _build(5.0, hdb)
    # constrained natural periods: generalized eig on the joint-feasible subspace
    # null(G) (the hub is structural, so a raw hub-diagonal proxy is meaningless).
    gmat = setup5.constraints.jacobian(np.zeros(_N_DOF))
    zmat = null_space(gmat)
    w2, vq = eigh(zmat.T @ setup5.lhs.C @ zmat, zmat.T @ setup5.lhs.M_plus_Ainf @ zmat)
    modes = zmat @ vq
    per = 2.0 * np.pi / np.sqrt(np.abs(w2))
    inband = sorted(
        (float(per[k]), float(abs(modes[_HUB_HEAVE, k])))
        for k in range(len(w2))
        if 2.0 < per[k] < 4.5
    )
    tn = max(inband, key=lambda ph: ph[1])[0] if inband else float("nan")
    print(
        f"cluster: constrained modes in 2-4.5s = "
        f"{[f'{p:.3f}(hub-heave {h:.2f})' for p, h in inband]}; "
        f"heave T_n ~ {tn:.3f} s (BEM composite pre-check 3.089 s, M10 ~3.106 s)"
    )

    heights = [0.08] if args.smoke else _HEIGHTS
    periods = [3.0] if args.smoke else _PERIODS
    for cd, setup in ((5.0, setup5), (1.0, None)):
        s = setup if setup is not None else _build(1.0, hdb)
        rows: list[dict] = []
        for p in periods:
            for h in heights:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c = run_case(s, hdb, height_m=h, period_s=p)
                tag = f"Cd{cd:g}_H{h:g}_T{p:g}".replace(".", "p")
                _write_case_csv(_OUT / f"case_{tag}.csv", c)
                rows.append(
                    dict(
                        height_m=c["height_m"],
                        period_s=c["period_s"],
                        omega=c["omega"],
                        amp_m=c["amp_m"],
                        rao_center=c["rao_center_heave"],
                        acc_center_amp=c["acc_center_amp"],
                        acc_center_peak=c["acc_center_peak"],
                        rao_buoy=c["rao_buoy1_heave"],
                        acc_buoy_amp=c["acc_buoy1_amp"],
                        acc_buoy_peak=c["acc_buoy1_peak"],
                        settled=c["settled"],
                        converged_early=c["converged_early"],
                        duration_used_s=c["duration_used_s"],
                    )
                )
                print(
                    f"  Cd={cd} H={h:.2f} T={p:.3f}: RAO_ctr={c['rao_center_heave']:.4f} "
                    f"RAO_b1={c['rao_buoy1_heave']:.4f} settled={c['settled']} "
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
                    "model": "3-buoy articulated cluster (24-DOF: 3x6 buoy + hub; 12 constraints)",
                    "T_n_s_approx": tn,
                    "grid_heights_m": _HEIGHTS,
                    "grid_periods_s": _PERIODS,
                    "plate_Cd_n": [5.0, 1.0],
                    "dropped_bem_omega": list(_CONTAM),
                    "scale": "1:50 model; H 0.04-0.12 m = 2-6 m full-scale; accel Froude-invariant",
                    "outputs": "hub centre + buoy1 heave RAO & Nz-accel",
                },
                fh,
                indent=2,
            )
    print(f"\nDone. Outputs in {_OUT}")


if __name__ == "__main__":
    main()
