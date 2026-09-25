"""Coupled BEM for the Phase-2 test cluster: 4 OSU-plate buoys on the Phase-3 intra-cluster radius
(0.5 m x 1.25/1.5 = 0.417 m) at 45/135/225/315 deg -- the 45 deg test orientation, a square with
two buoys facing the waves (CLUSTER_ROT_DEG = 0 gives the 0/90/180/270 diamond) -- centred at the
origin, deep water, no walls. Same hull, mesh resolution and FloatSim capytaine schema as the
platform's coupled_bem_osu.py (reuses its mesh / hydrostatics helpers), so the cluster, platform
and single buoy share one hydrodynamic model. Writes cluster_osu_open_rot45.nc (+ _psd.nc).

``python bem_cluster.py single`` builds the same coupled-schema database for ONE buoy at the
origin (single_osu_open.nc, label buoy1): the single-buoy article then uses the same hull, mesh,
omega grid and FloatSim coupled path as the cluster and platform.

Run: python bem_cluster.py [single]

Waterline resolution (flume-mooring decision 5): ``BEM_NT=<n>`` meshes the spar and plate with n
panels round instead of coupled_bem_osu.NT = 12 (whose 12-sided waterline holds 0.9549 of the
circle's area) and writes ``*_nt<n>.nc`` (+ _psd.nc), never the committed NT = 12 files.
``BEM_TIMING_ONLY=<k>`` solves only the first k finite frequencies and prints the time per
frequency (nothing written): the regeneration cost model.
"""
# ruff: noqa: E402, E702  -- sys.path bootstrap precedes the study imports; compact assembly
from __future__ import annotations

import os
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
CLUSTER_ROT_DEG = 45.0
SUF = "" if CLUSTER_ROT_DEG == 0 else f"_rot{int(CLUSTER_ROT_DEG)}"
ANG = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]) + CLUSTER_ROT_DEG)
CEN = [(R_INTRA * np.cos(a), R_INTRA * np.sin(a)) for a in ANG]
SINGLE = len(sys.argv) > 1 and sys.argv[1] == "single"
if SINGLE:
    CEN = [(0.0, 0.0)]
NB = len(CEN); NDOF = 6 * NB
BEM_NT = int(os.environ.get("BEM_NT", cb.NT))
cb.NT = BEM_NT
NT_SUF = "" if BEM_NT == 12 else f"_nt{BEM_NT}"
TIMING_ONLY = int(os.environ.get("BEM_TIMING_ONLY", "0"))


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
    dofs = list(body.dofs)          # capytaine's DOF names (results are keyed by these)
    # a lone FloatingBody names its DOFs without the body prefix -> coupled-schema labels
    labels = [d if "__" in d else f"buoy1__{d}" for d in dofs]
    print(f"{'single buoy' if SINGLE else 'cluster'}: {body.mesh.nb_faces} panels, "
          f"{len(labels)} DOF", flush=True)
    t0 = time.perf_counter()
    A = np.zeros((len(om) + 1, NDOF, NDOF)); B = np.zeros_like(A)
    F = np.zeros((len(om) + 1, NDOF), complex)
    freqs = [*list(om), np.inf]
    if TIMING_ONLY:                 # spread the timed frequencies over the grid
        freqs = [float(om[i]) for i in np.linspace(0, len(om) - 1, TIMING_ONLY).astype(int)]
    for k, w in enumerate(freqs):
        tf = time.perf_counter()
        res = cb.solve_freq(body, dofs, dofs, float(w), with_diff=np.isfinite(w))
        if TIMING_ONLY:
            print(f"NT {BEM_NT}: {body.mesh.nb_faces} panels, omega {w:.3f}: "
                  f"{time.perf_counter() - tf:.1f} s", flush=True)
            continue
        for a, rr in enumerate(res[:NDOF]):
            A[k, a, :] = [float(np.real(rr.added_masses[d])) for d in dofs]
            if np.isfinite(w):
                B[k, a, :] = [float(np.real(rr.radiation_dampings[d])) for d in dofs]
        if np.isfinite(w):
            dif = res[NDOF]; fk = froude_krylov_force(dif.problem)
            F[k, :] = [complex(dif.forces[d] + fk[d]) for d in dofs]
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
        attrs=dict(rho=cb.RHO, g=cb.G, water_depth="inf",
                   body_name="osu_single_open" if SINGLE else "osu_cluster4_open"),
    )
    if TIMING_ONLY:
        return
    out = HERE / (("single_osu_open" if SINGLE else f"cluster_osu_open{SUF}") + f"{NT_SUF}.nc")
    ds.to_netcdf(out)
    print(f"wrote {out.name} ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    psd_project.project(str(out))


if __name__ == "__main__":
    main()
