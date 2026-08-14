"""16-buoy platform RAO + buoy-acceleration driver. 16-buoy variant of
``studies/platform-12buoy/platform_rao_pilot.py``: same per-buoy drag (spar
cylinder + heave plate) and same ``run_case`` machinery, but the deck geometry
and DOF indexing come from ``platform16_common`` (4 buoys/cluster, 21 bodies,
126 DOF). Settling helpers are reused from the 12-buoy pilot (geometry-agnostic).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "platform-12buoy"))
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))

import cluster_common as cc  # noqa: E402
import platform16_common as pc16  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402  (reuse _fit_amplitude/_window_amp/_SETTLE_TOL)

from floatsim.hydro.excitation import make_regular_wave_force  # noqa: E402
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

_PLAT_NC = _HERE / "platform16_bem.nc"  # unused by the fan (passes DB objects); placeholder
_N_DOF = pc16.N_DOF  # 126

_ZB, _ZA, _ZP = pc16.Z_BUOY_REF, pc16.Z_HUB_REF, pc16.Z_PLATFORM_REF
_ZPLATE_BODY = -0.2617
_ZWL_BODY = 0.0 - _ZB

# Drag (identical single-buoy geometry to the 12-buoy; module globals the fan
# overrides via _build, exactly as platform_fin_fan does with platform_rao_pilot).
_SPAR_D = 2.0 * cc.R_SPAR
_SPAR_CD = 1.2
_PLATE_R = 0.215
_PLATE_T = 0.0039
_PLATE_CD_N = 5.0
_PLATE_CD_T = 1.5

_ASYMPTOTE_OVR = "platform16 parametric spar-fin small-body hulls (L~1.85 m)"
_KERNEL_EXEMPT = "platform16: rigid-yaw radiation noise floor on the coarse omega grid"


def _buoy_body_index(buoy_k0: int) -> int:
    return pc16.buoy_body_index(buoy_k0)


def _buoy_body_index_platform() -> int:
    return pc16.platform_body_index()


def _deck_with_drag() -> Deck:
    """21-body / 20-joint 16-buoy deck with spar + plate drag on every buoy."""
    spar = distributed_cylinder_drag(
        z_bottom=_ZPLATE_BODY, z_top=_ZWL_BODY, diameter=_SPAR_D, cd=_SPAR_CD, n_segments=10
    )
    plate = PlateMember(
        type="plate", center=[0.0, 0.0, _ZPLATE_BODY], normal=[0.0, 0.0, 1.0],
        radius=_PLATE_R, thickness=_PLATE_T, Cd_n=_PLATE_CD_N, Cd_t=_PLATE_CD_T,
    )
    bodies: list = []
    joints: list = []
    for c, pcang in enumerate(np.deg2rad(pc16.CLUSTER_ANGLES_DEG)):
        cx, cy = pc16.CLUSTER_ARM_RADIUS * np.cos(pcang), pc16.CLUSTER_ARM_RADIUS * np.sin(pcang)
        for b, tb in enumerate(np.deg2rad(pc16.BUOY_ANGLES_DEG)):
            k = pc16.N_PER_CLUSTER * c + b
            bx = cx + pc16.BUOY_RADIUS * np.cos(tb)
            by = cy + pc16.BUOY_RADIUS * np.sin(tb)
            bodies.append(
                Body(
                    name=f"buoy{k + 1}",
                    reference_point=[bx, by, _ZB],
                    mass=cc.M_BUOY,
                    inertia=Inertia(Ixx=cc.I_XX_BUOY, Iyy=cc.I_YY_BUOY, Izz=cc.I_ZZ_BUOY),
                    hydro_body_label=f"buoy{k + 1}",
                    initial_conditions=InitialConditions(),
                    drag_elements=[*spar, plate],
                )
            )
            joints.append(
                YawLockedJoint(
                    type="yaw_locked", body_a=f"buoy{k + 1}", body_b=f"hub{c + 1}",
                    attach_a_body=[0.0, 0.0, _ZA - _ZB],
                    attach_b_body=[pc16.BUOY_RADIUS * np.cos(tb), pc16.BUOY_RADIUS * np.sin(tb), 0.0],
                    axis=[0.0, 0.0, 1.0],
                )
            )
        bodies.append(
            Body(
                name=f"hub{c + 1}", reference_point=[cx, cy, _ZA],
                mass=pc16.ARM_MASS_PER_CLUSTER, inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0),
                structural=True,
            )
        )
        joints.append(
            YawLockedJoint(
                type="yaw_locked", body_a=f"hub{c + 1}", body_b="platform",
                attach_a_body=[0.0, 0.0, 0.0],
                attach_b_body=[cx, cy, _ZA - _ZP], axis=[0.0, 0.0, 1.0],
            )
        )
    bodies.append(
        Body(
            name="platform", reference_point=[0.0, 0.0, _ZP],
            mass=pc16.PLATFORM_MASS, inertia=Inertia(Ixx=10.0, Iyy=10.0, Izz=20.0),
            structural=True,
        )
    )
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=cc.RHO, gravity=cc.G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=bodies,
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_PLAT_NC)),
        joints=joints,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def _hydro_dof(deck: Deck) -> NDArray[np.int_]:
    idx = []
    for k, body in enumerate(deck.bodies):
        if body.hydro_body_label is not None:
            idx.extend(range(6 * k, 6 * k + 6))
    return np.asarray(idx, dtype=np.int_)


def run_case(
    setup, hdb, hydro_dof: NDArray[np.int_], *,
    height_m: float, period_s: float, ramp_s: float, cap_settle_s: float,
    window_periods: float, dt: float,
) -> dict:
    """Integrate one (H, T) case with adaptive settle -- 16-buoy indexing."""
    amp = 0.5 * height_m
    omega = 2.0 * np.pi / period_s
    window_s = window_periods * period_s
    plat_heave_dof = 6 * _buoy_body_index_platform() + 2
    wave = RegularWave(amplitude=amp, omega=omega, heading_deg=0.0)
    f_hydro = make_regular_wave_force(
        hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0), ramp=HalfCosineRamp(duration=ramp_s)
    )

    def ext(t: float) -> NDArray[np.float64]:
        f = np.zeros(_N_DOF, dtype=np.float64)
        f[hydro_dof] = f_hydro(t)
        return f

    def stop_check(tt: NDArray[np.float64], xx: NDArray[np.float64]) -> bool:
        t_now = float(tt[-1])
        if t_now < ramp_s + 2.0 * window_s:
            return False
        a_last = prp._window_amp(tt, xx[:, plat_heave_dof], omega, t_now, window_s)
        a_prev = prp._window_amp(tt, xx[:, plat_heave_dof], omega, t_now - window_s, window_s)
        return a_last > 0.0 and abs(a_last - a_prev) / a_last < prp._SETTLE_TOL

    duration = ramp_s + cap_settle_s + 2.0 * window_s
    res = integrate_cummins(
        lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0, xi_dot0=setup.xi_dot0,
        duration=duration, dt=dt, rho_inf=0.8, constraints=setup.constraints,
        external_force=ext, state_force=setup.state_force, projection_interval=1,
        stop_check=stop_check, stop_check_interval=max(1, round(window_s / dt)),
    )
    t = res.t
    duration_used = float(t[-1])
    converged = duration_used < duration - 0.5 * dt
    mask = t >= duration_used - window_s
    tw, xi_w, acc_w = t[mask], res.xi[mask], res.xi_ddot[mask]
    a_last = prp._window_amp(t, res.xi[:, plat_heave_dof], omega, duration_used, window_s)
    a_prev = prp._window_amp(t, res.xi[:, plat_heave_dof], omega, duration_used - window_s, window_s)
    settle_ratio = float(abs(a_last - a_prev) / a_last) if a_last > 0 else float("nan")

    rao = {"platform_heave": prp._fit_amplitude(tw, xi_w[:, plat_heave_dof], omega) / amp}
    for k0 in range(pc16.N_BUOY):
        dof = 6 * _buoy_body_index(k0) + 2
        rao[f"buoy{k0 + 1}_heave"] = prp._fit_amplitude(tw, xi_w[:, dof], omega) / amp

    return {
        "height_m": height_m, "period_s": period_s, "omega": omega, "amp_m": amp,
        "duration_s": duration_used, "converged_early": bool(converged),
        "n_steps": int(t.size - 1), "settle_ratio": settle_ratio,
        "settled": bool(settle_ratio < prp._SETTLE_TOL), "rao": rao,
        "t": tw, "xi": xi_w, "acc": acc_w,
    }


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deck = _deck_with_drag()
        hd = _hydro_dof(deck)
    print(f"16-buoy deck: {len(deck.bodies)} bodies, {len(deck.joints)} joints, "
          f"{len(hd)} hydro DOFs (= {pc16.N_BUOY} buoys x 6); N_DOF={_N_DOF}")
    print(f"platform heave DOF = {6 * _buoy_body_index_platform() + 2}; "
          f"buoy1/buoy16 heave DOF = {6 * _buoy_body_index(0) + 2}/{6 * _buoy_body_index(15) + 2}")
