"""Coupled BEM for the Phase-2 test cluster: 4 OSU-plate buoys (square, 0/90/180/270 deg) on the
Phase-3 intra-cluster radius (0.5 m x 1.25/1.5 = 0.417 m), centred at the origin, deep water, no
walls. Same hull, mesh resolution and FloatSim capytaine schema as the platform's
coupled_bem_osu.py (reuses its mesh / hydrostatics helpers), so the cluster, platform and single
buoy share one hydrodynamic model. Writes cluster_osu_open.nc (+ _psd.nc).

Run: python bem_cluster.py
"""
# ruff: noqa: E402, E702  -- sys.path bootstrap precedes the study imports; compact assembly
from __future__ import annotations

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
import psd_project
from capytaine.bem.airy_waves import froude_krylov_force

R_INTRA = 0.5 * 1.25 / 1.5
ANG = np.deg2rad([0.0, 90.0, 180.0, 270.0])
CEN = [(R_INTRA * np.cos(a), R_INTRA * np.sin(a)) for a in ANG]
NB = len(CEN); NDOF = 6 * NB


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    om = np.geomspace(0.1, 30.0, 48)
    bodies = []
    for i, (cx, cy) in enumerate(CEN):
        cog = (cx, cy, cb.COG_Z)
        b = cpt.FloatingBody(mesh=cb.buoy_mesh(cx, cy), center_of_mass=cog, name=f"buoy{i + 1}")
        b.rotation_center = np.asarray(cog); b.add_all_rigid_body_dofs()
        bodies.append(b)
    body = bodies[0]
    for b in bodies[1:]:
        body = body + b
    labels = list(body.dofs)
    print(f"cluster: {body.mesh.nb_faces} panels, {len(labels)} DOF", flush=True)
    t0 = time.perf_counter()
    A = np.zeros((len(om) + 1, NDOF, NDOF)); B = np.zeros_like(A)
    F = np.zeros((len(om) + 1, NDOF), complex)
    for k, w in enumerate([*list(om), np.inf]):
        res = cb.solve_freq(body, labels, labels, float(w), with_diff=np.isfinite(w))
        for a, rr in enumerate(res[:NDOF]):
            A[k, a, :] = [float(np.real(rr.added_masses[lab])) for lab in labels]
            if np.isfinite(w):
                B[k, a, :] = [float(np.real(rr.radiation_dampings[lab])) for lab in labels]
        if np.isfinite(w):
            dif = res[NDOF]; fk = froude_krylov_force(dif.problem)
            F[k, :] = [complex(dif.forces[lab] + fk[lab]) for lab in labels]
    C6 = cb.single_buoy_c(); C = np.zeros((NDOF, NDOF))
    for i in range(NB):
        C[6 * i:6 * i + 6, 6 * i:6 * i + 6] = C6
    ds = xr.Dataset(
        data_vars=dict(
            added_mass=(("omega", "radiating_dof", "influenced_dof"), A),
            radiation_damping=(("omega", "radiating_dof", "influenced_dof"), B),
            hydrostatic_stiffness=(("radiating_dof", "influenced_dof"), C),
            excitation_force=(("complex", "omega", "wave_direction", "influenced_dof"),
                              np.stack([F.real, F.imag], 0)[:, :, None, :]),
        ),
        coords=dict(omega=("omega", np.append(om, np.inf)),
                    wave_direction=("wave_direction", [0.0]),
                    radiating_dof=("radiating_dof", labels),
                    influenced_dof=("influenced_dof", labels),
                    complex=("complex", ["re", "im"])),
        attrs=dict(rho=cb.RHO, g=cb.G, water_depth="inf", body_name="osu_cluster4_open"),
    )
    out = HERE / "cluster_osu_open.nc"
    ds.to_netcdf(out)
    print(f"wrote {out.name} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    psd_project.project(str(out))


if __name__ == "__main__":
    main()
