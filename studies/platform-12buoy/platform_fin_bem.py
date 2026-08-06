"""Fin-sensitivity coupled 12-buoy platform BEM, MEMORY-BOUNDED. Parametric spar+fin
(n_theta=16 coarse mesh -- single-buoy A33 within 1.3% of the fine mesh) replicated
at the 12 platform positions + platform draft -> 72-DOF coupled radiation +
diffraction per fin {0.215, 0.15, none}.

Solves ONE frequency at a time with a FRESH BEMSolver each, so Capytaine's
per-frequency influence-matrix cache is freed between frequencies instead of
accumulating (the all-at-once solve_all blew up to 47 GB and paged the machine).
~80 s/frequency, ~107 min/fin, RAM bounded. Modes:
  test  -- 2 frequencies + inf; save, read back, verify the assembled NetCDF
  all   -- full sweep per fin
"""

from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import capytaine as cpt
import numpy as np
import xarray as xr
from capytaine.io.xarray import separate_complex_values

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))
sys.path.insert(0, str(_HERE.parent / "spar-fin-decay"))

import cluster_common as cc  # noqa: E402
import cluster_fin_bem as cfb  # noqa: E402
import platform_common as pc  # noqa: E402

from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402

_OUT = _HERE / "fin_study"
_OMEGA = np.geomspace(0.1, 30.0, 80)
_NTHETA = 16
_COG_Z = cc.CoG_Z_SINGLE - pc.PLATFORM_DZ
_FINS = {"0215": 0.215, "015": 0.15, "none": None}


def make_combined(r_fin: float | None):  # type: ignore[no-untyped-def]
    cfb._NTHETA = _NTHETA
    base = cfb.build_full_mesh(r_fin).translated([0.0, 0.0, -pc.PLATFORM_DZ])
    bodies = []
    for i, (dx, dy) in enumerate(pc.buoy_centers()):
        cog = [float(dx), float(dy), _COG_Z]
        b = cpt.FloatingBody(
            mesh=base.translated([float(dx), float(dy), 0.0]),
            center_of_mass=cog,
            name=f"buoy{i + 1}",
        )
        b.rotation_center = np.asarray(cog)
        b.add_all_rigid_body_dofs()
        bodies.append(b)
    allb = bodies[0]
    for b in bodies[1:]:
        allb = allb + b
    return allb


def _solve_omega(allb, dofs, w: float, with_diff: bool):  # type: ignore[no-untyped-def]
    solver = cpt.BEMSolver()  # FRESH each frequency -> cache freed on GC
    probs = [
        cpt.RadiationProblem(
            body=allb, omega=w, radiating_dof=d, water_depth=float("inf"), rho=cc.RHO, g=cc.G
        )
        for d in dofs
    ]
    if with_diff:
        probs.append(
            cpt.DiffractionProblem(
                body=allb, omega=w, wave_direction=0.0, water_depth=float("inf"), rho=cc.RHO, g=cc.G
            )
        )
    ds = cpt.assemble_dataset(solver.solve_all(probs, progress_bar=False))
    del solver
    gc.collect()
    return ds


def run(tag: str, r_fin: float | None, test: bool = False) -> None:
    allb = make_combined(r_fin)
    dofs = list(allb.dofs)
    print(
        f"  fin={tag}: {allb.immersed_part().mesh.nb_faces} wet panels, {len(dofs)} DOF", flush=True
    )
    finite = _OMEGA[:2] if test else _OMEGA
    t0 = time.perf_counter()
    dss = []
    for k, w in enumerate(finite):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dss.append(_solve_omega(allb, dofs, float(w), with_diff=True))
        if not test and (k + 1) % 10 == 0:
            print(
                f"    {k + 1}/{len(finite)} freqs, {(time.perf_counter() - t0) / 60:.0f} min",
                flush=True,
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dss.append(_solve_omega(allb, dofs, float("inf"), with_diff=False))  # A_inf
    ds = xr.concat(dss, dim="omega", coords="minimal", compat="override")
    for c in ("radiating_dof", "influenced_dof"):
        ds = ds.assign_coords({c: [str(x) for x in ds[c].values]})
    out = _OUT / ("_test_platform.nc" if test else f"capytaine_platform_fin{tag}.nc")
    separate_complex_values(ds).to_netcdf(str(out))

    heave = [f"buoy{i + 1}__Heave" for i in range(12)]
    ainf = ds["added_mass"].sel(omega=np.inf)
    hb = sum(float(ainf.sel(radiating_dof=hi, influenced_dof=hj)) for hi in heave for hj in heave)
    print(
        f"    solved {(time.perf_counter() - t0) / 60:.0f} min; heave-block A33 sum(inf) = "
        f"{hb:.2f} kg -> {out.name}",
        flush=True,
    )
    if test:
        h = read_capytaine(out)
        print(
            f"    read_capytaine OK: omega {np.asarray(h.omega).shape}, "
            f"A {np.asarray(h.A).shape}, RAO {None if h.RAO is None else np.asarray(h.RAO).shape}, "
            f"A_inf[2,2]={float(np.asarray(h.A_inf)[2, 2]):.2f}",
            flush=True,
        )


def main() -> None:
    _OUT.mkdir(exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "test":
        run("0215", 0.215, test=True)
    else:
        for tag, rf in _FINS.items():
            if mode != "all" and tag != mode:
                continue
            run(tag, rf)
    print("Done.")


if __name__ == "__main__":
    main()
