"""Coupled Capytaine BEM of each payload case at its own static equilibrium (tilted_cases.json).

The records' coupled databases (flume-mooring bem_cluster.py, platform-12buoy/flume-wall-effect
coupled_bem_osu.py) place every buoy at the design draft and tile one buoy's hydrostatic matrix.
Here each buoy is meshed where the large-angle statics put it:
- at its own horizontal position;
- lowered by its own sinkage. The payload tilts the deck or hub; the buoys still hang vertically.
Each buoy's hydrostatic matrix is computed at that draft, about its sunk CoG (the rotation centre,
as in the records).

Everything else follows the records:
- the mesh is coupled_bem_osu.buoy_mesh, 12 panels round;
- deep water, waves along +x (heading 0);
- 48 frequencies, 0.1-30 rad/s, plus infinity;
- the FloatSim capytaine schema;
- the symmetrised, PSD-projected radiation matrices (psd_project).

Writes tilted_bem/<article>_<case>.nc and _psd.nc.
Run: python tilted_bem.py <article> <case> [<case> ...]   (e.g. platform tilt10)
"""

# ruff: noqa: E402, E702  -- sys.path bootstrap precedes the record imports; compact assembly
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
FWE = HERE.parent / "platform-12buoy" / "flume-wall-effect"
sys.path.insert(0, str(FWE))
warnings.simplefilter("ignore")

import capytaine as cpt
import coupled_bem_osu as cb
import draft_hydrostatics as dh
import psd_project
from capytaine.bem.airy_waves import froude_krylov_force

OUT = HERE / "tilted_bem"
OMEGA = np.geomspace(0.1, 30.0, 48)          # bem_cluster.py / coupled_bem_osu.py grid


def sunk_mesh(x: float, y: float, s: float):  # type: ignore[no-untyped-def]
    """coupled_bem_osu.buoy_mesh at (x, y), lowered by s, wetted part."""
    spar = cpt.mesh_vertical_cylinder(length=cb.Z_TOP - cb.Z_BOT, radius=cb.SPAR_R,
                                      center=(x, y, (cb.Z_BOT + cb.Z_TOP) / 2 - s),
                                      resolution=(2, cb.NT, 24))
    plate = cpt.mesh_vertical_cylinder(length=0.02, radius=cb.PLATE_R,
                                       center=(x, y, cb.PLATE_Z - s), resolution=(4, cb.NT, 2))
    return cpt.FloatingBody(mesh=spar.join_meshes(plate)).immersed_part().mesh


def run(case: dict) -> Path:
    buoys = case["buoys"]; nb = len(buoys); ndof = 6 * nb
    bodies = []
    for i, r in enumerate(buoys):
        cog = (r["x"], r["y"], cb.COG_Z - r["sink_m"])
        bd = cpt.FloatingBody(mesh=sunk_mesh(r["x"], r["y"], r["sink_m"]), center_of_mass=cog,
                              name=f"buoy{i + 1}")
        bd.rotation_center = np.asarray(cog); bd.add_all_rigid_body_dofs()
        bodies.append(bd)
    body = bodies[0]
    for bd in bodies[1:]:
        body = body + bd
    dofs = list(body.dofs)
    print(f"{case['article']} {case['case']}: {body.mesh.nb_faces} panels, {ndof} DOF", flush=True)
    t0 = time.perf_counter()
    A = np.zeros((len(OMEGA) + 1, ndof, ndof)); B = np.zeros_like(A)
    F = np.zeros((len(OMEGA) + 1, ndof), complex)
    for k, w in enumerate([*list(OMEGA), np.inf]):
        res = cb.solve_freq(body, dofs, dofs, float(w), with_diff=np.isfinite(w))
        for a, rr in enumerate(res[:ndof]):
            A[k, a, :] = [float(np.real(rr.added_masses[d])) for d in dofs]
            if np.isfinite(w):
                B[k, a, :] = [float(np.real(rr.radiation_dampings[d])) for d in dofs]
        if np.isfinite(w):
            dif = res[ndof]; fk = froude_krylov_force(dif.problem)
            F[k, :] = [complex(dif.forces[d] + fk[d]) for d in dofs]
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(OMEGA) + 1} {(time.perf_counter() - t0) / 60:.1f} min",
                  flush=True)
    C = np.zeros((ndof, ndof))
    for i, r in enumerate(buoys):                     # each buoy at its own draft
        C[6 * i:6 * i + 6, 6 * i:6 * i + 6] = np.asarray(dh.hydrostatics(r["sink_m"], cb.NT)["C6"])
    ds = xr.Dataset(
        data_vars=dict(
            added_mass=(("omega", "radiating_dof", "influenced_dof"), A),
            radiation_damping=(("omega", "radiating_dof", "influenced_dof"), B),
            hydrostatic_stiffness=(("radiating_dof", "influenced_dof"), C),
            excitation_force=(("complex", "omega", "wave_direction", "influenced_dof"),
                              np.stack([F.real, F.imag], 0)[:, :, None, :]),
        ),
        coords=dict(omega=("omega", np.append(OMEGA, np.inf)),
                    wave_direction=("wave_direction", [0.0]),
                    radiating_dof=("radiating_dof", dofs), influenced_dof=("influenced_dof", dofs),
                    complex=("complex", ["re", "im"])),
        attrs=dict(rho=cb.RHO, g=cb.G, water_depth="inf",
                   body_name=f"osu_{case['article']}_{case['case']}"),
    )
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{case['article']}_{case['case']}.nc"
    ds.to_netcdf(out)
    psd_project.project(str(out))
    print(f"wrote {out.name} + _psd ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    article, names = sys.argv[1], sys.argv[2:]
    cases = json.loads((HERE / "tilted_cases.json").read_text(encoding="utf-8"))["cases"]
    for c in cases:
        if c["article"] == article and (not names or c["case"] in names):
            run(c)


if __name__ == "__main__":
    main()
