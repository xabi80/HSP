"""Step A -- Capytaine BEM run for the spar+fin buoy heave-decay study.

Per studies/spar-fin-decay/README.md (this study is Tier 3 prep,
NOT a milestone). Runs radiation + diffraction over a frequency
sweep bracketing the expected heave natural period (T_n ~ 3-4 s ->
omega_n ~ 1.5-2 rad/s; sweep 0.1 - 30 rad/s).

The Capytaine 2.3.1 API used here:
  cpt.load_mesh(path, file_format="gdf")
    -> Mesh; the GDF parser handles WAMIT format directly.
  cpt.FloatingBody(mesh, center_of_mass=..., name=...)
    -> FloatingBody; mass/inertia are NOT needed for BEM (they
       only enter the time-domain solve, which FloatSim handles).
  body.add_all_rigid_body_dofs()
    -> Adds Surge, Sway, Heave, Roll, Pitch, Yaw DOFs about the
       body's center_of_mass.
  cpt.RadiationProblem / cpt.DiffractionProblem
    -> One problem per (omega, dof) for radiation, one per
       (omega, heading) for diffraction.
  solver.solve_all(problems) -> list of Result objects.
  cpt.assemble_dataset(results) -> xarray.Dataset matching the
    schema floatsim.hydro.readers.capytaine expects (added_mass,
    radiation_damping, excitation_force or {FK, diffraction}).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import capytaine as cpt

# ---------------------------------------------------------------------------
# Locked inputs (per Xabier's spar-fin study spec)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_MESH_PATH = _HERE / "mesh" / "test2_spar_fin_fullfix_eqdraft.gdf"
_OUTPUT_NC = _HERE / "capytaine_bem.nc"

_RHO_KG_M3 = 1025.0      # salt water
_G_M_S2 = 9.81           # matches GDF header (line 2: "1.000000  9.810000")
# CoG in the eqdraft-mesh frame. The mesh was translated DOWN by
# dz = 0.1846 m (waterline_balance.py / prepare_mesh.py) to place the
# free-floating equilibrium waterline at z=0, so the CoG shifts with
# it: z_cog = -0.8317 - 0.1846 = -1.0163 m.
_COG_BODY_FRAME = (0.0, 0.0, -1.0163)

# Frequency sweep: 80 points geomspace 0.1 to 30 rad/s.
# Brackets T_n ~ 3-4 s (omega_n ~ 1.5-2 rad/s) with at least 2 decades on
# either side so A_inf high-freq asymptote is well-resolved.
_OMEGA_MIN = 0.1
_OMEGA_MAX = 30.0
_N_OMEGA = 80


def main() -> None:
    if not _MESH_PATH.is_file():
        sys.exit(f"FATAL: mesh not found at {_MESH_PATH}")

    print("=" * 70)
    print("Step A -- Capytaine BEM run, spar+fin buoy decay study")
    print("=" * 70)
    print(f"Mesh:     {_MESH_PATH}")
    print(f"Output:   {_OUTPUT_NC}")
    print(f"rho:      {_RHO_KG_M3} kg/m^3 (salt water)")
    print(f"g:        {_G_M_S2} m/s^2 (matches GDF header)")
    print(f"CoG:      {_COG_BODY_FRAME} m (body frame)")
    print(f"omegas:   {_N_OMEGA} pts geomspace {_OMEGA_MIN} - {_OMEGA_MAX} rad/s")
    print()

    # --- load mesh ---
    print("Loading mesh ...")
    mesh = cpt.load_mesh(str(_MESH_PATH), file_format="gdf")
    n_panels = mesh.nb_faces
    print(f"  Loaded mesh with {n_panels} faces.")
    if n_panels != 1488:
        print(f"  WARNING: expected 1488 panels per spec; got {n_panels}.")

    # --- floating body + rigid DOFs about CoG ---
    cog = np.array(_COG_BODY_FRAME, dtype=np.float64)
    body = cpt.FloatingBody(
        mesh=mesh,
        center_of_mass=cog,
        name="spar_fin",
    )
    # Capytaine 2.x requires rotation_center explicitly for hydrostatic
    # stiffness computation; default to CoG for rigid-body dynamics.
    body.rotation_center = cog
    body.add_all_rigid_body_dofs()
    print(f"  Body DOFs: {list(body.dofs)}")

    # --- frequency sweep ---
    omegas = np.geomspace(_OMEGA_MIN, _OMEGA_MAX, _N_OMEGA)
    print(f"  omega grid: [{omegas[0]:.3f}, ..., {omegas[-1]:.3f}] rad/s")

    # --- build problems: radiation (one per dof per omega) + diffraction (one per omega) ---
    # Include omega = inf for radiation as the canonical A_inf case --
    # FloatSim's Capytaine reader needs this populated to fill A_inf
    # without requiring a caller-supplied kwarg.
    omegas_rad = list(omegas) + [float("inf")]
    radiation_problems = [
        cpt.RadiationProblem(
            body=body,
            omega=float(omega),
            radiating_dof=dof,
            water_depth=float("inf"),
            rho=_RHO_KG_M3,
            g=_G_M_S2,
        )
        for omega in omegas_rad
        for dof in body.dofs
    ]
    diffraction_problems = [
        cpt.DiffractionProblem(
            body=body,
            omega=float(omega),
            wave_direction=0.0,
            water_depth=float("inf"),
            rho=_RHO_KG_M3,
            g=_G_M_S2,
        )
        for omega in omegas  # diffraction at finite omegas only
    ]
    problems = radiation_problems + diffraction_problems
    print(f"  Total problems: {len(problems)} "
          f"({len(radiation_problems)} rad + {len(diffraction_problems)} diff)")

    # --- solve ---
    print("Solving (this can take 5-30 min depending on hardware) ...")
    solver = cpt.BEMSolver()
    results = solver.solve_all(problems, n_jobs=1)
    print(f"  Solved {len(results)} problems.")

    # --- assemble dataset ---
    print("Assembling xarray dataset ...")
    dataset = cpt.assemble_dataset(results)
    print(f"  Dataset variables: {list(dataset.data_vars)}")
    print(f"  Dataset dims: {dict(dataset.sizes)}")

    # --- hydrostatic stiffness on the immersed-part body ---
    # Capytaine 2.x requires explicit immersed_part() + rotation_center
    # for hydrostatic stiffness; the BEM solver auto-clips but the
    # hydrostatic path doesn't.
    print("Computing hydrostatic stiffness on immersed_part() with rotation_center ...")
    immersed = body.immersed_part()
    immersed.center_of_mass = cog
    immersed.rotation_center = cog
    K_hs = immersed.compute_hydrostatic_stiffness(rho=_RHO_KG_M3, g=_G_M_S2)
    dataset["hydrostatic_stiffness"] = K_hs
    print(f"  hydrostatic_stiffness shape: {K_hs.shape}")

    # Symmetrization retired 2026-07-03: handled at
    # HydroDatabase.__post_init__ since M7.5 PR2 (dafed8c); NetCDF now
    # saved raw. See m7.5-reader-hygiene-closure.md.

    # --- save (split complex variables into re/im per FloatSim reader format) ---
    print(f"Writing {_OUTPUT_NC} ...")
    from capytaine.io.xarray import separate_complex_values
    dataset_for_save = separate_complex_values(dataset)
    dataset_for_save.to_netcdf(str(_OUTPUT_NC))
    print("Done.")

    # ----- Step B: inline sanity checks -----
    print()
    print("=" * 70)
    print("Step B -- output sanity checks")
    print("=" * 70)

    # Check 1: A_inf for heave-heave should be O(30-70 kg)
    A_h = dataset["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave")
    A_inf_heave = float(A_h.isel(omega=-1))
    print(f"  A_inf(heave) = {A_inf_heave:.3f} kg "
          f"(expected O(30-70) kg)")
    if not (20.0 < A_inf_heave < 200.0):
        print(f"  CHECK 1 FAIL: A_inf(heave) = {A_inf_heave:.3f} outside expected band.")
        sys.exit(1)
    print("  Check 1: A_inf(heave) magnitude OK.")

    # Check 2: C33 hydrostatic restoring should equal 223.43 N/m (per locked input)
    expected_C33 = 223.43  # N/m, from rho*g*pi*r_spar^2
    if "hydrostatic_stiffness" in dataset.data_vars:
        K = dataset["hydrostatic_stiffness"]
        C33_cap = float(K.sel(radiating_dof="Heave", influenced_dof="Heave"))
        rel_err = abs(C33_cap - expected_C33) / expected_C33
        print(f"  C33 (Capytaine waterplane): {C33_cap:.3f} N/m "
              f"(expected {expected_C33:.3f}, rel-err {rel_err*100:.2f}%)")
        if rel_err > 0.02:
            print(f"  CHECK 2 WARNING: C33 deviates >2% from rho*g*pi*r_spar^2.")
            print(f"  This may reflect spar radius assumption vs actual mesh waterline.")
    else:
        print(f"  Check 2: hydrostatic_stiffness not in dataset; skipped.")

    # Check 3: B_heave(omega) non-negative everywhere
    B_h = dataset["radiation_damping"].sel(
        radiating_dof="Heave", influenced_dof="Heave"
    )
    if (B_h.values < -1e-6).any():
        n_neg = int((B_h.values < -1e-6).sum())
        print(f"  CHECK 3 FAIL: B_heave(omega) has {n_neg} negative values.")
        sys.exit(1)
    print(f"  Check 3: B_heave(omega) non-negative everywhere "
          f"(min = {float(B_h.min()):+.3e}, max = {float(B_h.max()):.3e}).")

    # Bonus: predicted natural period from M + A_inf and C33
    M_body = 28.67  # kg, locked input
    if "hydrostatic_stiffness" in dataset.data_vars:
        T_n_pred = 2.0 * np.pi * np.sqrt((M_body + A_inf_heave) / C33_cap)
        print(f"\n  Predicted T_n_heave = 2*pi*sqrt((M+A_inf)/C33) = {T_n_pred:.3f} s")
        if not (3.0 < T_n_pred < 4.0):
            print(f"  NOTE: predicted T_n outside expected 3.0-4.0 s band.")

    print()
    print("Step A + B complete.")


if __name__ == "__main__":
    main()
