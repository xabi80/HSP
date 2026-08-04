"""Fin-sensitivity RAO + Nz-accel fan for the 3-buoy ARTICULATED cluster.

Uses the parametric coupled cluster BEMs (cluster_fin_bem.py) per fin radius
{0.215, 0.15, none}; the plate drag radius is matched to the fin (no-fin: the
spar bottom-cap, r=R_spar). Refined period grid {2.0..3.3 with 2.2/2.3/2.4}.
Reuses cluster_rao.run_case (hub centre + buoy1 heave RAO & Nz-accel). Outputs
in the uniform cross-model schema so the fin plotter reads it like the single buoy.
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import cluster_rao as cr  # noqa: E402

from floatsim.driver import build_system  # noqa: E402
from floatsim.hydro.database import HydroDatabase  # noqa: E402
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

_STUDY = _HERE / "fin_study"
_HEIGHTS = [0.04, 0.06, 0.08, 0.10, 0.12]
# cluster heave resonances (fin 0.215/0.15/none) sit at ~3.12/2.63/2.47 s, so
# the grid adds 2.6/2.65 to sample the 0.15 cluster's 2.63 s peak (it would
# otherwise fall in the 2.5-2.8 gap).
_PERIODS = [2.0, 2.2, 2.3, 2.4, 2.5, 2.6, 2.65, 2.8, 3.0, 3.141, 3.257, 3.3]
_CAP = 350.0
_R_SPAR = 0.0841
# per-parametric-BEM contaminated (irregular-frequency) omegas to drop; filled
# in after inspecting the generated BEMs (empty = none dropped).
_CONTAM: dict[str, tuple[float, ...]] = {"0215": (4.934,), "015": (), "none": (27.910,)}
# (bem_tag, fin_radius, [(Cd_n, plate_drag_radius)])
_CONFIGS = [
    ("0215", 0.215, [(5.0, 0.215), (1.0, 0.215)]),
    ("015", 0.15, [(5.0, 0.15), (1.0, 0.15)]),
    ("none", None, [(5.0, _R_SPAR)]),  # no-fin BEM + spar bottom-cap drag
]


def _hdb(tag: str) -> HydroDatabase:
    h = read_capytaine(_STUDY / f"capytaine_cluster_fin{tag}.nc")
    drop = _CONTAM.get(tag, ())
    if not drop:
        return h
    w = np.asarray(h.omega)
    di = {int(np.argmin(np.abs(w - c))) for c in drop}
    keep = np.array([k for k in range(w.size) if k not in di])
    return HydroDatabase(
        omega=h.omega[keep], heading_deg=h.heading_deg, A=h.A[:, :, keep], B=h.B[:, :, keep],
        A_inf=h.A_inf, C=h.C, RAO=h.RAO[:, keep, :], reference_point=h.reference_point,
        C_source=h.C_source, metadata=dict(h.metadata), body_labels=h.body_labels,
    )


def _deck(plate_r: float, cd_n: float) -> Deck:
    spar = distributed_cylinder_drag(
        z_bottom=cr._ZPLATE_BODY, z_top=cr._ZWL_BODY, diameter=cr._SPAR_D,
        cd=cr._SPAR_CD, n_segments=10)
    plate = PlateMember(
        type="plate", center=[0.0, 0.0, cr._ZPLATE_BODY], normal=[0.0, 0.0, 1.0],
        radius=plate_r, thickness=cr._PLATE_T, Cd_n=cd_n, Cd_t=cr._PLATE_CD_T)
    buoys = [
        Body(name=f"buoy{i + 1}", reference_point=[cr._R * np.cos(a), cr._R * np.sin(a), cr._ZB],
             mass=28.67, inertia=Inertia(Ixx=24.0, Iyy=24.0, Izz=0.114),
             hydro_body_label=f"buoy{i + 1}", initial_conditions=InitialConditions(),
             drag_elements=[*spar, plate])
        for i, a in enumerate(cr._ANG)
    ]
    hub = Body(name="hub", reference_point=[0.0, 0.0, cr._ZA], mass=12.0,
               inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0), structural=True)
    joints = [
        YawLockedJoint(
            type="yaw_locked", body_a=f"buoy{i + 1}", body_b="hub",
            attach_a_body=[0.0, 0.0, cr._ZA - cr._ZB],
            attach_b_body=[cr._R * np.cos(a), cr._R * np.sin(a), 0.0], axis=[0.0, 0.0, 1.0])
        for i, a in enumerate(cr._ANG)
    ]
    return Deck(
        simulation=Simulation(duration=50.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=cr._RHO, gravity=cr._G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[*buoys, hub],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path="x.nc"),
        hydrostatic_database=HydroDatabaseRef(format="capytaine", path=str(cr._REF)),
        joints=joints, output=Output(file="o.h5", channels=["heave"], sample_rate=10.0))


def _build(plate_r: float, cd_n: float, hdb: HydroDatabase):  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(
            _deck(plate_r, cd_n), bem_databases={}, dt=0.01, t_max_kernel=60.0,
            solve_equilibrium=False, shared_hydro_database=hdb,
            hydrostatic_database=read_capytaine(cr._REF),
            asymptote_check_override=cr._ASYMPTOTE_OVR,
            kernel_decay_floor_override="cluster parametric spar-fin small-body")


def _fan(tag, label, setup, hdb):  # type: ignore[no-untyped-def]
    rows = []
    for p in _PERIODS:
        for h in _HEIGHTS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = cr.run_case(setup, hdb, height_m=h, period_s=p, cap_settle_s=_CAP)
            fname = f"case_cluster_fin{tag}_{label}_H{h:g}_T{p:g}".replace(".", "p") + ".csv"
            cr._write_case_csv(_STUDY / fname, c)
            rows.append(dict(
                height_m=h, period_s=p, omega=c["omega"], amp_m=c["amp_m"],
                rao_center=c["rao_center_heave"], acc_center_amp=c["acc_center_amp"],
                acc_center_peak=c["acc_center_peak"], rao_buoy=c["rao_buoy1_heave"],
                acc_buoy_amp=c["acc_buoy1_amp"], acc_buoy_peak=c["acc_buoy1_peak"],
                settled=c["settled"], converged_early=c["converged_early"],
                duration_used_s=c["duration_used_s"]))
            print(f"  cluster fin={tag} {label} H={h:.2f} T={p:.3f}: "
                  f"RAO_ctr={c['rao_center_heave']:.3f} RAO_b1={c['rao_buoy1_heave']:.3f} "
                  f"settled={c['settled']}", flush=True)
    with (_STUDY / f"rao_summary_cluster_fin{tag}_{label}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return sum(1 for r in rows if not r["settled"])


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    meta: dict = {"grid_heights_m": _HEIGHTS, "grid_periods_s": _PERIODS, "fins": {}}
    for tag, fin_r, cds in _CONFIGS:
        if only and tag != only:
            continue
        hdb = _hdb(tag)
        for cd, plate_r in cds:
            setup = _build(plate_r, cd, hdb)
            label = "cap" if fin_r is None else f"Cd{cd:g}"
            print(f"\n=== cluster fin {tag} {label} (plate r={plate_r}) ===", flush=True)
            nun = _fan(tag, label, setup, hdb)
            meta["fins"].setdefault(tag, {"radius_m": fin_r})
            if nun:
                print(f"  [{tag} {label}] {nun}/50 did not settle")
    with (_STUDY / "cluster_fin_manifest.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
