"""Fin-sensitivity RAO + Nz-accel fan for the single buoy: fin radius in
{0.215, 0.15, none} x the corrected heights 0.04-0.12 m x plate Cd_n in {5,1}
(no-fin has no plate -> one spar-only config). Each fin uses its OWN parametric
BEM (sparfin_fin_bem.py) for the correct added mass + excitation; the plate drag
radius is matched to the fin. Reuses sparfin_rao.run_case (adaptive settle, but a
larger cap: the small/no-fin cases are lightly damped and ring longer)."""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sparfin_rao as srp  # noqa: E402
import study_common as sc  # noqa: E402

from floatsim.driver import _build_drag_state_force  # noqa: E402
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402
from floatsim.hydro.retardation import compute_retardation_kernel  # noqa: E402
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

_STUDY = _HERE / "fin_study"
_HEIGHTS = [0.04, 0.06, 0.08, 0.10, 0.12]
_PERIODS = [2.0, 2.5, 2.8, 3.0, 3.141, 3.257, 3.3]
_CAP = 350.0  # fin cases settle in <100s; no-fin near-resonance won't (the finding)
# (tag, fin_radius, [Cd_n configs])  -- no-fin: no plate, spar-only single config
_FINS = [("0215", 0.215, [5.0, 1.0]), ("015", 0.15, [5.0, 1.0]), ("none", None, [None])]


def _drag(fin_r, cd_n):  # type: ignore[no-untyped-def]
    spar = distributed_cylinder_drag(
        z_bottom=srp._PLATE_Z, z_top=srp._WL_Z, diameter=srp._SPAR_D, cd=srp._SPAR_CD, n_segments=10
    )
    elems = list(spar)
    if fin_r is not None:
        elems.append(PlateMember(
            type="plate", center=[0.0, 0.0, srp._PLATE_Z], normal=[0.0, 0.0, 1.0],
            radius=fin_r, thickness=srp._PLATE_T, Cd_n=cd_n, Cd_t=srp._PLATE_CD_T))
    deck = Deck(
        simulation=Simulation(duration=10.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=sc.RHO, gravity=sc.G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[Body(
            name="buoy", reference_point=[0.0, 0.0, 0.0], mass=sc.M_BODY,
            inertia=Inertia(Ixx=sc.I_XX, Iyy=sc.I_YY, Izz=sc.I_ZZ),
            hydro_database=HydroDatabaseRef(format="capytaine", path="x.nc"),
            drag_elements=elems)],
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )
    return _build_drag_state_force(deck, n_dof=6, rho=sc.RHO)


def _fan(tag, fin_r, cd_n, lhs, kernel, hdb, eq, label):  # type: ignore[no-untyped-def]
    drag = _drag(fin_r, cd_n)
    rows: list[dict] = []
    for p in _PERIODS:
        for h in _HEIGHTS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = srp.run_case(lhs, kernel, hdb, drag, eq.xi_eq, height_m=h, period_s=p,
                                 cap_settle_s=_CAP)
            fname = f"case_fin{tag}_{label}_H{h:g}_T{p:g}".replace(".", "p") + ".csv"
            srp._write_case_csv(_STUDY / fname, c)
            rows.append(dict(
                height_m=h, period_s=p, omega=c["omega"], amp_m=c["amp_m"],
                rao_center=c["rao_heave"], acc_center_amp=c["acc_heave_amp"],
                acc_center_peak=c["acc_heave_peak"], rao_buoy=c["rao_heave"],
                acc_buoy_amp=c["acc_heave_amp"], acc_buoy_peak=c["acc_heave_peak"],
                settled=c["settled"], converged_early=c["converged_early"],
                duration_used_s=c["duration_used_s"]))
            print(f"  fin={tag} {label} H={h:.2f} T={p:.3f}: RAO={c['rao_heave']:.3f} "
                  f"acc={c['acc_heave_amp']:.3f} settled={c['settled']} "
                  f"dur={c['duration_used_s']:.0f}s", flush=True)
    with (_STUDY / f"rao_summary_fin{tag}_{label}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return sum(1 for r in rows if not r["settled"])


def main() -> None:
    _STUDY.mkdir(exist_ok=True)
    meta: dict = {"grid_heights_m": _HEIGHTS, "grid_periods_s": _PERIODS, "cap_settle_s": _CAP,
                  "fins": {}, "note": "each fin uses its own parametric BEM; plate drag "
                  "radius matched to the fin; no-fin has no plate (spar drag only)."}
    for tag, fin_r, cds in _FINS:
        nc = _STUDY / f"capytaine_fin{tag}.nc"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hdb = read_capytaine(nc)
            lhs = sc.build_lhs(hdb)
            # t_max=60 (the standard buffer): the parametric-mesh heave kernel needs
            # the full 60 s to decay <0.10% (Check 3); study_common hardcodes 30 s.
            kernel = compute_retardation_kernel(
                hdb, t_max=60.0, dt=sc.DT, asymptote_check_override=sc._OVERRIDE)
            eq = solve_static_equilibrium(lhs=lhs, state_force=None)
        a33 = float(lhs.M_plus_Ainf[2, 2] - 28.67)
        tn = 2 * np.pi * np.sqrt(lhs.M_plus_Ainf[2, 2] / lhs.C[2, 2])
        print(f"\n=== fin {tag} (R={fin_r}): T_n={tn:.3f} s, A33={a33:.2f} "
              f"C33={lhs.C[2, 2]:.2f} ===", flush=True)
        meta["fins"][tag] = {"radius_m": fin_r, "T_n_s": tn, "A33_kg": a33}
        for cd in cds:
            label = "spar" if cd is None else f"Cd{cd:g}"
            nun = _fan(tag, fin_r, cd, lhs, kernel, hdb, eq, label)
            if nun:
                print(f"  [{tag} {label}] {nun}/35 did not settle within {_CAP:.0f}s "
                      f"(lightly damped -- expected for small/no fin near resonance)")
    with (_STUDY / "manifest.json").open("w") as fh:
        json.dump(meta, fh, indent=2)
    print("\nDone. Outputs in", _STUDY)


if __name__ == "__main__":
    main()
