"""Step 3: two BEM runs (reference single-at-cluster-draft + composite).

Usage:
  python cluster_bem.py reference   # single hull at cluster draft -> reference NetCDF
  python cluster_bem.py probe       # time 2 omegas on the composite, extrapolate
  python cluster_bem.py composite   # full composite BEM -> composite NetCDF

80 log-spaced omegas [0.1, 30] rad/s + inf; rho=1025, g=9.81; full
6-DOF radiation + diffraction. Reference rotation_center = single CoG
at cluster draft; composite rotation_center = composite CoG (Step 2).

Grid history: originally 40 points; widened to 80 at M8 PR4 to match
the 18-DOF fixture's production grid EXACTLY (capytaine_multibody_
diagnostic.py, PR3) so the excitation condensation gate compares both
models on identical grids BY CONSTRUCTION (plan Q4 lock — no
interpolation path). The 80-point grid includes omega=4.934 (the
contaminated frequency slice, tracker BEM-CONTAMINATED-FREQUENCY-
SLICE-CLUSTER-DRAFT) — deliberately: both models share the influence
matrix there, so the condensation identity must hold even on
contaminated data (the closure doc's worked example).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import capytaine as cpt
import cluster_common as cc
import numpy as np
from build_cluster_mesh import build as build_composite

from floatsim.hydro.mesh_hygiene import GdfMesh, load_gdf_panels, write_gdf_panels

_HERE = Path(__file__).resolve().parent
_REF_MESH = _HERE / "mesh" / "single_at_cluster_draft.gdf"
_REF_NC = _HERE / "reference_single_bem.nc"
_COMP_NC = _HERE / "composite_bem.nc"

_OMEGA_MIN, _OMEGA_MAX, _N_OMEGA = 0.1, 30.0, 80


def _make_reference_mesh() -> Path:
    """Single fullfix hull translated DOWN to the cluster draft."""
    single = load_gdf_panels(cc.SINGLE_EQDRAFT_MESH)
    p = single.panels.copy()
    p[..., 2] -= cc.DZ2
    write_gdf_panels(GdfMesh(header_lines=single.header_lines, panels=p), _REF_MESH)
    return _REF_MESH


def _run_bem(mesh_path: Path, cog, out_nc: Path, label: str, omegas) -> None:
    mesh = cpt.load_mesh(str(mesh_path), file_format="gdf")
    print(f"  [{label}] mesh faces: {mesh.nb_faces}")
    body = cpt.FloatingBody(mesh=mesh, center_of_mass=np.asarray(cog), name=label)
    body.rotation_center = np.asarray(cog)
    body.add_all_rigid_body_dofs()
    omegas_rad = [*omegas, float("inf")]
    problems = [
        cpt.RadiationProblem(
            body=body, omega=float(w), radiating_dof=d, water_depth=float("inf"), rho=cc.RHO, g=cc.G
        )
        for w in omegas_rad
        for d in body.dofs
    ] + [
        cpt.DiffractionProblem(
            body=body,
            omega=float(w),
            wave_direction=0.0,
            water_depth=float("inf"),
            rho=cc.RHO,
            g=cc.G,
        )
        for w in omegas
    ]
    print(f"  [{label}] problems: {len(problems)}")
    t0 = time.perf_counter()
    results = cpt.BEMSolver().solve_all(problems, n_jobs=1)
    dt = time.perf_counter() - t0
    print(f"  [{label}] solved {len(results)} problems in {dt:.1f} s")
    dataset = cpt.assemble_dataset(results)
    immersed = body.immersed_part()
    immersed.center_of_mass = np.asarray(cog)
    immersed.rotation_center = np.asarray(cog)
    dataset["hydrostatic_stiffness"] = immersed.compute_hydrostatic_stiffness(rho=cc.RHO, g=cc.G)
    from capytaine.io.xarray import separate_complex_values

    separate_complex_values(dataset).to_netcdf(str(out_nc))
    print(f"  [{label}] wrote {out_nc.name}")
    # Sanity.
    A = dataset["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave")
    A_inf = float(A.isel(omega=-1))
    K = dataset["hydrostatic_stiffness"]
    C33 = float(K.sel(radiating_dof="Heave", influenced_dof="Heave"))
    print(f"  [{label}] A_inf(heave) = {A_inf:.4f} kg;  C33 = {C33:.4f} N/m")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "reference"
    omegas = np.geomspace(_OMEGA_MIN, _OMEGA_MAX, _N_OMEGA)
    cog_single = (0.0, 0.0, cc.CoG_Z_SINGLE - cc.DZ2)
    import json

    props = json.loads((_HERE / "results" / "mass_properties.json").read_text())
    cog_comp = tuple(props["cog_m"])

    if mode == "reference":
        print("=== Step 3(a): REFERENCE single hull at cluster draft ===")
        print(f"  single CoG at cluster draft: {cog_single}")
        _make_reference_mesh()
        _run_bem(_REF_MESH, cog_single, _REF_NC, "reference", omegas)
        print("  PINS: A_inf(heave) in [21.0, 22.5]; C33 = 221 +/- 2%.")

    elif mode == "probe":
        print("=== Step 3(b) probe: composite runtime estimate ===")
        mesh = build_composite(cc.DZ2)
        write_gdf_panels(mesh, cc.CLUSTER_MESH)
        m = cpt.load_mesh(str(cc.CLUSTER_MESH), file_format="gdf")
        body = cpt.FloatingBody(mesh=m, center_of_mass=np.asarray(cog_comp), name="probe")
        body.rotation_center = np.asarray(cog_comp)
        body.add_all_rigid_body_dofs()
        probe_omegas = [float(omegas[0]), float(omegas[len(omegas) // 2])]
        probes = [
            cpt.RadiationProblem(
                body=body, omega=w, radiating_dof=d, water_depth=float("inf"), rho=cc.RHO, g=cc.G
            )
            for w in probe_omegas
            for d in body.dofs
        ]
        t0 = time.perf_counter()
        cpt.BEMSolver().solve_all(probes, n_jobs=1)
        dt = time.perf_counter() - t0
        per_problem = dt / len(probes)
        n_full = (_N_OMEGA + 1) * 6 + _N_OMEGA  # rad (incl inf) + diff
        est = per_problem * n_full
        print(f"  {len(probes)} probe problems in {dt:.1f} s " f"({per_problem:.2f} s/problem)")
        print(f"  full composite ~ {n_full} problems -> est {est:.0f} s " f"= {est/60:.1f} min")
        if est > 3 * 3600:
            raise SystemExit(f"STOP: composite projection {est/3600:.1f} h > 3 h.")
        print("  PROCEED (< 3 h).")

    elif mode == "composite":
        print("=== Step 3(b): COMPOSITE cluster BEM ===")
        print(f"  composite CoG: {cog_comp}")
        _run_bem(cc.CLUSTER_MESH, cog_comp, _COMP_NC, "composite", omegas)
        print("  PINS: C33_composite = 3 x C33_single +/- 2%; A33 raw.")

    else:
        raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
