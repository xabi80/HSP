"""M11b PR8 -- RAO + buoy-acceleration pilot for the OrcaFlex comparison.

SCOPE (Xabier, PR8 disposition). This script PRODUCES outputs in a form
comparable to OrcaFlex; it does NOT judge agreement (no pass/fail tolerance).
The 5-10% BEM screening threshold does not apply here.

What it does, per (wave height H, period T):
  * build the 12-buoy platform (shared BEM = platform12_bem.nc, whose stored
    hydrostatic C was corrected to the single-body-tiled block in M11b PR8
    STEP 4; build_system uses it directly + gravity_restoring, no injection)
    ONCE, with the M11b PR8 kernel Check-3 noise-floor exemption (rigid-yaw
    radiation on the coarse 13-omega grid) and the M10 PR0 asymptote override
    (small-body hulls);
  * force it with a regular Airy wave (amplitude A = H/2), the coupled
    diffraction RAO scattered into the 102-DOF global vector at the buoy slots;
  * integrate (generalized-alpha + velocity-level KKT joints + Morison drag)
    long enough to reach steady state, VERIFYING settling per case;
  * extract RAO = |response amplitude| / A for platform heave (ref point) and
    every buoy heave, and the translational acceleration time history for THREE
    buoys (cluster A / a side cluster / the opposite cluster);
  * write per-case CSV time histories, a summary CSV of all RAOs, and a JSON
    manifest recording every convention + provenance.

HEADING CONVENTION (documented, flagged for Xabier). The coupled RAO database
platform12_bem.nc was solved at heading 0 ONLY (single heading). FloatSim
heading 0 => wave propagates +x (eta = A cos(wt - k x); crest reaches larger x
LATER), so the wave ARRIVES first at the -x side. Therefore under the only
available heading, cluster C at (-1,0) is UPWAVE and cluster A at (+1,0) is
DOWNWAVE -- opposite to the pinned "cluster A upwave" note. Making cluster A
upwave requires re-solving the BEM at heading 180 (~190 min). The RAO
magnitudes are heading-label-independent; only the up/down-wave ROLE labels of
the three acceleration buoys are affected. Every buoy's (x, y) and incident
phase is recorded so the mapping is unambiguous.

Drag provenance (manifest-recorded):
  * heave plate: PlateMember, Cd_n = 5.0 (KNOWN heave-plate broadside,
    cluster_common.PLATE_CD; M11a PR4), Cd_t = 1.5 (mid of the [1,2]
    tank-pending sensitivity; M11a PR4), radius 0.215 m, thickness 0.0039 m,
    body-frame z = -0.2617 (M11a PR4 committed depth).
  * spar column: distributed cylinder, D = 0.1682 m (2*R_SPAR,
    cluster_common.R_SPAR = 0.0841), Cd = 1.2 (standard smooth cylinder,
    transverse), 10 segments from the plate (z=-0.2617) up to the waterline
    (z=+1.1957, body frame).
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
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))

import cluster_common as cc  # noqa: E402
import platform_common as pc  # noqa: E402

from floatsim.driver import build_system  # noqa: E402
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

# --- paths + fixed model constants ------------------------------------------
_PLAT_NC = _HERE / "platform12_bem.nc"
_OUT = _HERE / "pr8_pilot_out"

_ZB = pc.Z_BUOY_REF  # -1.19567 buoy reference z (inertial, equilibrium)
_ZA = pc.Z_HUB_REF  # +0.49337 hub reference z
_ZP = pc.Z_PLATFORM_REF  # +0.70 platform reference z
_ZPLATE_BODY = -0.2617  # plate depth, body frame (M11a PR4 committed record)
_ZWL_BODY = 0.0 - _ZB  # waterline in body frame = +1.19567

# Drag coefficients (provenance in the module docstring). Plate values are the
# M11a PR4 authoritative constants (tests/validation/test_m11a_pr4_plate_drag.py:
# _A_PLATE, _T_PLATE, _CD_N, _CD_T); the spar diameter is 2*R_SPAR (cluster_common).
_SPAR_D = 2.0 * cc.R_SPAR  # 0.1682 (R_SPAR = 0.0841)
_SPAR_CD = 1.2
_PLATE_R = 0.215
_PLATE_T = 0.0039
_PLATE_CD_N = 5.0  # KNOWN heave-plate broadside (M11a PR4)
_PLATE_CD_T = 1.5  # mid of the [1,2] tank-pending sensitivity (M11a PR4)

_N_DOF = 102  # 17 bodies x 6
_KERNEL_EXEMPT = (
    "M11b PR8: rigid-yaw radiation (peak|K|/dominant ~4e-15) is numerical noise "
    "on the coarse 13-omega grid; exempt the 12 buoy-yaw DOFs from Check 3"
)
_ASYMPTOTE_OVR = (
    "M11b PR8: small-body spar-fin hulls (L~1.85 m) do not reach 1/omega^4 by omega_max"
)


def _buoy_body_index(buoy_k0: int) -> int:
    """Global deck-body index of buoy k (0-based). Deck order per cluster is
    [3 buoys, 1 hub], so cluster c occupies 4 body slots: buoy (c, b) -> 4c+b."""
    c, b = divmod(buoy_k0, 3)
    return 4 * c + b


def _deck_with_drag() -> Deck:
    """The 17-body / 16-joint platform deck (build_platform_deck geometry +
    validated cc masses) with spar + plate drag on every buoy."""
    spar = distributed_cylinder_drag(
        z_bottom=_ZPLATE_BODY, z_top=_ZWL_BODY, diameter=_SPAR_D, cd=_SPAR_CD, n_segments=10
    )
    plate = PlateMember(
        type="plate",
        center=[0.0, 0.0, _ZPLATE_BODY],
        normal=[0.0, 0.0, 1.0],
        radius=_PLATE_R,
        thickness=_PLATE_T,
        Cd_n=_PLATE_CD_N,
        Cd_t=_PLATE_CD_T,
    )
    bodies: list = []
    joints: list = []
    for c, pcang in enumerate(np.deg2rad(pc.CLUSTER_ANGLES_DEG)):
        cx, cy = pc.CLUSTER_ARM_RADIUS * np.cos(pcang), pc.CLUSTER_ARM_RADIUS * np.sin(pcang)
        for b, tb in enumerate(np.deg2rad(pc.BUOY_ANGLES_DEG)):
            k = 3 * c + b
            bx = cx + pc.BUOY_RADIUS * np.cos(tb)
            by = cy + pc.BUOY_RADIUS * np.sin(tb)
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
                    type="yaw_locked",
                    body_a=f"buoy{k + 1}",
                    body_b=f"hub{c + 1}",
                    attach_a_body=[0.0, 0.0, _ZA - _ZB],
                    attach_b_body=[pc.BUOY_RADIUS * np.cos(tb), pc.BUOY_RADIUS * np.sin(tb), 0.0],
                    axis=[0.0, 0.0, 1.0],
                )
            )
        bodies.append(
            Body(
                name=f"hub{c + 1}",
                reference_point=[cx, cy, _ZA],
                mass=cc.ARM_MASS_TOTAL,
                inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0),
                structural=True,
            )
        )
        joints.append(
            YawLockedJoint(
                type="yaw_locked",
                body_a=f"hub{c + 1}",
                body_b="platform",
                attach_a_body=[0.0, 0.0, 0.0],
                attach_b_body=[cx, cy, _ZA - _ZP],
                axis=[0.0, 0.0, 1.0],
            )
        )
    bodies.append(
        Body(
            name="platform",
            reference_point=[0.0, 0.0, _ZP],
            mass=pc.PLATFORM_MASS,
            inertia=Inertia(Ixx=10.0, Iyy=10.0, Izz=20.0),
            structural=True,
        )
    )
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=cc.RHO, gravity=cc.G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=bodies,
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_PLAT_NC)),
        # No hydrostatic_database: platform12_bem.nc now stores the CORRECT
        # single-body-tiled hydrostatic C (M11b PR8 STEP 4). build_system uses
        # shared_db.C + gravity_restoring; injecting reference_single too would
        # double-count buoyancy.
        joints=joints,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def _hydro_dof(deck: Deck) -> NDArray[np.int_]:
    """The global DOF slots of the hydro (buoy) bodies, in deck order -- the
    scatter target for the 72-DOF coupled wave force (same order as the DB
    body_labels buoy1..buoy12)."""
    idx = []
    for k, body in enumerate(deck.bodies):
        if body.hydro_body_label is not None:
            idx.extend(range(6 * k, 6 * k + 6))
    return np.asarray(idx, dtype=np.int_)


def _fit_amplitude(t: NDArray, x: NDArray, omega: float) -> float:
    """Least-squares amplitude of x(t) ~ a cos(wt) + b sin(wt) + c over the
    given window. Returns sqrt(a^2 + b^2)."""
    design = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    coeffs, *_ = np.linalg.lstsq(design, x, rcond=None)
    return float(np.hypot(coeffs[0], coeffs[1]))


_SETTLE_TOL = 0.02  # 2% window-to-window amplitude agreement (settling criterion)


def _window_amp(t: NDArray, x: NDArray, omega: float, t_hi: float, w_s: float) -> float:
    """Sinusoid amplitude of ``x`` over the window ``[t_hi - w_s, t_hi]``."""
    m = (t >= t_hi - w_s) & (t <= t_hi + 1e-9)
    return _fit_amplitude(t[m], x[m], omega)


def run_case(
    setup,  # type: ignore[no-untyped-def]
    hdb,  # type: ignore[no-untyped-def]
    hydro_dof: NDArray[np.int_],
    *,
    height_m: float,
    period_s: float,
    ramp_s: float,
    cap_settle_s: float,
    window_periods: float,
    dt: float,
) -> dict:
    """Integrate one (H, T) case with ADAPTIVE settle (M11b PR8 STEP 1):
    integrate-until-window-converges, capped at ``cap_settle_s``.

    The integrator stops as soon as the platform-heave amplitude over two
    consecutive ``window_periods``-long windows agrees to within
    ``_SETTLE_TOL``; the RAO/accel window is the final converged window. A run
    that reaches the cap without converging is reported ``settled=False``
    (STOP condition -- a finding about the mode's Q, not a cap to raise)."""
    amp = 0.5 * height_m  # A = H/2
    omega = 2.0 * np.pi / period_s
    window_s = window_periods * period_s
    plat_heave_dof = 6 * _buoy_body_index_platform() + 2
    wave = RegularWave(amplitude=amp, omega=omega, heading_deg=0.0)
    f72 = make_regular_wave_force(
        hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0), ramp=HalfCosineRamp(duration=ramp_s)
    )

    def ext(t: float) -> NDArray[np.float64]:
        f = np.zeros(_N_DOF, dtype=np.float64)
        f[hydro_dof] = f72(t)
        return f

    def stop_check(tt: NDArray[np.float64], xx: NDArray[np.float64]) -> bool:
        t_now = float(tt[-1])
        # Need the ramp done plus two full comparison windows of steady data.
        if t_now < ramp_s + 2.0 * window_s:
            return False
        a_last = _window_amp(tt, xx[:, plat_heave_dof], omega, t_now, window_s)
        a_prev = _window_amp(tt, xx[:, plat_heave_dof], omega, t_now - window_s, window_s)
        if a_last <= 0.0:
            return False
        return abs(a_last - a_prev) / a_last < _SETTLE_TOL

    duration = ramp_s + cap_settle_s + 2.0 * window_s
    res = integrate_cummins(
        lhs=setup.lhs,
        kernel=setup.kernel,
        xi0=setup.xi0,
        xi_dot0=setup.xi_dot0,
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
    duration_used = float(t[-1])
    converged = duration_used < duration - 0.5 * dt  # stopped early via stop_check

    # RAO / accel window = the final converged window.
    mask = t >= duration_used - window_s
    tw, xi_w, acc_w = t[mask], res.xi[mask], res.xi_ddot[mask]

    # Settling verification: last two windows must agree (the stop criterion).
    a_last = _window_amp(t, res.xi[:, plat_heave_dof], omega, duration_used, window_s)
    a_prev = _window_amp(t, res.xi[:, plat_heave_dof], omega, duration_used - window_s, window_s)
    settle_ratio = float(abs(a_last - a_prev) / a_last) if a_last > 0 else float("nan")
    settled = settle_ratio < _SETTLE_TOL

    rao = {"platform_heave": _fit_amplitude(tw, xi_w[:, plat_heave_dof], omega) / amp}
    for k0 in range(12):
        dof = 6 * _buoy_body_index(k0) + 2
        rao[f"buoy{k0 + 1}_heave"] = _fit_amplitude(tw, xi_w[:, dof], omega) / amp

    return {
        "height_m": height_m,
        "period_s": period_s,
        "omega": omega,
        "amp_m": amp,
        "duration_s": duration_used,
        "duration_cap_s": duration,
        "converged_early": bool(converged),
        "n_steps": int(t.size - 1),
        "settle_ratio": settle_ratio,
        "settled": bool(settled),
        "rao": rao,
        "t": tw,
        "xi": xi_w,
        "acc": acc_w,
    }


def _buoy_body_index_platform() -> int:
    """Platform is the last deck body: 4 clusters x 4 slots = 16 -> index 16."""
    return 16


def _accel_channels() -> list[tuple[str, int, str]]:
    """The three acceleration buoys: (label, buoy_k0, role-under-heading-0).
    buoy1 cluster A (+x extremum, on wave axis); buoy4 side cluster B;
    buoy7 cluster C (on wave axis). Roles are under the ONLY available
    heading (0): +x is downwave, -x is upwave."""
    return [
        ("buoy1_clusterA", 0, "cluster A (1.5, 0) -- +x extremum, DOWNWAVE under heading 0"),
        ("buoy4_clusterB", 3, "cluster B side (0.5, 1)"),
        (
            "buoy7_clusterC",
            6,
            "cluster C (-0.5, 0) -- on wave axis, UPWAVE-side cluster under heading 0",
        ),
    ]


def _write_case_csv(path: Path, case: dict) -> None:
    """Per-case time history: t + platform heave + the 3 buoys' surge/sway/heave
    displacement and acceleration."""
    t = case["t"]
    cols: dict[str, NDArray] = {"t_s": t}
    ph = 6 * _buoy_body_index_platform() + 2
    cols["platform_heave_m"] = case["xi"][:, ph]
    cols["platform_heave_acc_mps2"] = case["acc"][:, ph]
    for label, k0, _role in _accel_channels():
        base = 6 * _buoy_body_index(k0)
        for j, comp in enumerate(("surge", "sway", "heave")):
            cols[f"{label}_{comp}_m"] = case["xi"][:, base + j]
            cols[f"{label}_{comp}_acc_mps2"] = case["acc"][:, base + j]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(cols.keys()))
        rows = np.column_stack(list(cols.values()))
        w.writerows(rows.tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 short case to validate the pipeline")
    args = ap.parse_args()

    _OUT.mkdir(exist_ok=True)
    deck = _deck_with_drag()
    hydro_dof = _hydro_dof(deck)
    shared = read_capytaine(_PLAT_NC)
    hdb_force = read_capytaine(_PLAT_NC)

    dt = 0.01
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setup = build_system(
            deck,
            bem_databases={},
            dt=dt,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=shared,
            asymptote_check_override=_ASYMPTOTE_OVR,
            kernel_decay_floor_override=_KERNEL_EXEMPT,
        )

    # Pilot matrix: 2 resonances (T=3.141, 3.257 s) + 1 off-resonance anchor
    # (T=2.0 s), each at 3 heights spanning the requested 0.03-1.2 m band.
    # Adaptive settle (M11b PR8 STEP 1): the integrator stops when converged;
    # cap_settle is the hard ceiling only, sized for the slowest (small-H) case.
    if args.smoke:
        matrix = [(0.30, 3.257, 20.0, 450.0, 6.0)]  # (H, T, ramp, cap_settle, window_periods)
    else:
        heights = [0.05, 0.30, 1.00]
        # (period, ramp_s, cap_settle_s, window_periods)
        periods = [
            (3.257, 20.0, 450.0, 6.0),
            (3.141, 20.0, 450.0, 6.0),
            (2.000, 20.0, 250.0, 8.0),
        ]
        matrix = [(h, p, r, s, wp) for (p, r, s, wp) in periods for h in heights]

    summary_rows: list[dict] = []
    for i, (h, p, r, s, wp) in enumerate(matrix):
        print(f"[{i + 1}/{len(matrix)}] H={h} m  T={p} s  (cap {s}s)...", flush=True)
        case = run_case(
            setup,
            hdb_force,
            hydro_dof,
            height_m=h,
            period_s=p,
            ramp_s=r,
            cap_settle_s=s,
            window_periods=wp,
            dt=dt,
        )
        tag = f"H{h:g}_T{p:g}".replace(".", "p")
        _write_case_csv(_OUT / f"case_{tag}.csv", case)
        row = {
            "height_m": h,
            "period_s": p,
            "omega": case["omega"],
            "amp_m": case["amp_m"],
            "settle_ratio": case["settle_ratio"],
            "settled": case["settled"],
            "converged_early": case["converged_early"],
            "duration_used_s": case["duration_s"],
            "duration_cap_s": case["duration_cap_s"],
            "n_steps": case["n_steps"],
        }
        row.update({f"rao_{k}": v for k, v in case["rao"].items()})
        summary_rows.append(row)
        print(
            f"    platform_heave RAO={case['rao']['platform_heave']:.4f}  "
            f"buoy1 RAO={case['rao']['buoy1_heave']:.4f}  "
            f"settled={case['settled']} (ratio {case['settle_ratio']:.2e})  "
            f"dur_used={case['duration_s']:.0f}s/cap{case['duration_cap_s']:.0f}s "
            f"converged_early={case['converged_early']}",
            flush=True,
        )

    # Summary CSV
    with (_OUT / "rao_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Manifest
    centers = pc.buoy_centers()
    manifest = {
        "study": "M11b PR8 -- 12-buoy platform RAO + acceleration for OrcaFlex comparison",
        "scope": "PRODUCES comparable outputs; does NOT judge agreement (no tolerance).",
        "rao_normalization": "per wave amplitude A = H/2 (RAO = |response amplitude| / A)",
        "heading": {
            "value_deg": 0.0,
            "database_headings_available": [0.0],
            "propagation": "+x (eta = A cos(wt - k x))",
            "upwave_side": "-x (wave arrives there first); cluster C (-1,0) is UPWAVE",
            "downwave_side": "+x; cluster A (+1,0) is DOWNWAVE",
            "flag": (
                "CONTRADICTS the pinned 'cluster A upwave' note: the coupled RAO DB "
                "has only heading 0, so cluster A is DOWNWAVE. Making cluster A upwave "
                "needs a heading-180 BEM re-solve (~190 min). RAO magnitudes are "
                "heading-label-independent; only accel role labels are affected."
            ),
        },
        "wave_height_range_m": [0.03, 1.2],
        "wave_period_range_s": [1.2, 3.3],
        "steepness_caveat": "H/lambda > 0.04 flagged per case (see rao_summary + below)",
        "drag": {
            "plate": {
                "type": "PlateMember (anisotropic)",
                "Cd_n": _PLATE_CD_N,
                "Cd_t": _PLATE_CD_T,
                "radius_m": _PLATE_R,
                "thickness_m": _PLATE_T,
                "z_body_m": _ZPLATE_BODY,
                "provenance": (
                    "Cd_n = 5.0 KNOWN heave-plate broadside; "
                    "Cd_t = 1.5 mid of [1,2] tank-pending; "
                    "M11a PR4 (test_m11a_pr4_plate_drag.py)"
                ),
            },
            "spar": {
                "type": "distributed cylinder (10 seg)",
                "diameter_m": _SPAR_D,
                "Cd": _SPAR_CD,
                "z_bottom_body_m": _ZPLATE_BODY,
                "z_top_body_m": _ZWL_BODY,
                "provenance": (
                    "D=2*R_SPAR (cluster_common.R_SPAR=0.0841); "
                    "Cd=1.2 standard cylinder transverse"
                ),
            },
        },
        "rao_dofs": ["platform heave (ref point)", "buoy heave (all 12)"],
        "acceleration_channels": [
            {"label": lbl, "buoy": k0 + 1, "xy": centers[k0].tolist(), "role": role}
            for (lbl, k0, role) in _accel_channels()
        ],
        "buoy_centers_xy": {f"buoy{k + 1}": centers[k].tolist() for k in range(12)},
        "kernel_decay_floor_override": _KERNEL_EXEMPT,
        "asymptote_check_override": _ASYMPTOTE_OVR,
        "screening_verdict": (
            "platform12_bem.nc passed PR7 conditioning screen "
            "(cond(K) 4.934 flat; B-min-eig clean)"
        ),
        "bem_grid_omega": read_capytaine(_PLAT_NC).omega.tolist(),
        "cases": summary_rows,
    }
    # Steepness caveat per case (deep-water lambda = g T^2 / 2pi).
    for row in summary_rows:
        lam = cc.G * row["period_s"] ** 2 / (2.0 * np.pi)
        row["wavelength_m"] = float(lam)
        row["steepness_H_over_lambda"] = float(row["height_m"] / lam)
        row["steepness_flag"] = bool(row["height_m"] / lam > 0.04)
    with (_OUT / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nDone. Outputs in {_OUT}")
    n_unsettled = sum(1 for r in summary_rows if not r["settled"])
    if n_unsettled:
        print(
            f"WARNING: {n_unsettled}/{len(summary_rows)} cases did not settle (<2% window ratio)."
        )


if __name__ == "__main__":
    main()
