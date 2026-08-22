"""Build a FloatSim-compatible Capytaine BEM database (.nc) for the OSU Test Buoy.

Geometry: the 6" spar (Ø0.1593, immersed to the 967 mm waterline) + a PLACEHOLDER solid
equal-area disc for the heave plate (the real perforated/webbed frame's added mass and
damping must come from the tank test — this disc is an upper-bound stand-in). Frame: still
water = z=0, z up.

Writes the standard capytaine on-disk schema (added_mass, radiation_damping, excitation_force
split-complex, hydrostatic_stiffness, an omega=inf sample for A_inf) that
`floatsim.hydro.readers.capytaine.read_capytaine` consumes. Assembled manually to avoid the
capytaine `fill_dataset` CategoricalDtype bug.

Requires: capytaine.  Usage: `python bem_database.py [out.nc]`
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")
import capytaine as cpt  # noqa: E402
from capytaine.bem.airy_waves import froude_krylov_force  # noqa: E402

cpt.set_logging("ERROR")

RHO, G = 998.0, 9.806                 # fresh (OSU Hinsdale lab)
M, ZG = 21.52, -0.907
R, Z_BOT, Z_TOP = 0.07965, -0.967, 0.717
A_PLATE = float(np.sqrt(0.328 * 0.198 / np.pi))   # equal-area disc radius (rect 328x198)
Z_PLATE = -1.383
DOFS = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
_OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "capytaine_osu_buoy.nc"


def _body():  # type: ignore[no-untyped-def]
    spar = cpt.mesh_vertical_cylinder(length=Z_TOP - Z_BOT, radius=R,
                                      center=(0, 0, (Z_BOT + Z_TOP) / 2), resolution=(4, 48, 70))
    plate = cpt.mesh_vertical_cylinder(length=0.02, radius=A_PLATE, center=(0, 0, Z_PLATE),
                                       resolution=(12, 48, 2))
    b = cpt.FloatingBody(mesh=spar.join_meshes(plate), mass=M, center_of_mass=(0, 0, ZG))
    b.rotation_center = np.array([0.0, 0.0, ZG])
    b.add_all_rigid_body_dofs()
    b = b.immersed_part()
    b.rotation_center = np.array([0.0, 0.0, ZG])
    return b


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    body = _body()
    print(f"immersed panels: {body.mesh.nb_faces}  plate Ø{2 * A_PLATE:.3f} m (placeholder)")
    hs = body.compute_hydrostatics(rho=RHO, g=G)
    C = np.array([[float(hs["hydrostatic_stiffness"].sel(radiating_dof=a, influenced_dof=b_))
                   for b_ in DOFS] for a in DOFS])
    print(f"C33={C[2, 2]:.1f}  C55={C[4, 4]:.2f}")

    # Reuse the omega grid of the validated single-buoy BEM (fine at low ω, out to 30 rad/s):
    # this is what makes the slender-spar surge/roll radiation kernel decay below FloatSim's
    # Check-3 gate. A uniform coarse grid does not.
    _ref = xr.open_dataset(Path(__file__).resolve().parents[2] / "studies/spar-fin-decay/capytaine_bem.nc")
    om = np.sort(np.asarray(_ref["omega"].values))
    om = np.ascontiguousarray(om[np.isfinite(om)])
    _ref.close()
    om_all = np.append(om, np.inf)
    solver = cpt.BEMSolver()
    nA = len(om_all)
    A = np.zeros((nA, 6, 6)); B = np.zeros((nA, 6, 6))
    for i, w in enumerate(om_all):
        for j, dof in enumerate(DOFS):
            res = solver.solve(cpt.RadiationProblem(body=body, omega=w, radiating_dof=dof,
                                                    rho=RHO, g=G, water_depth=np.inf), keep_details=False)
            A[i, j, :] = [res.added_masses[d] for d in DOFS]
            if np.isfinite(w):
                B[i, j, :] = [res.radiation_dampings[d] for d in DOFS]
        if i % 8 == 0:
            print(f"  radiation {i + 1}/{nA} (ω={w:.2f})", flush=True)
    # excitation (FK + diffraction) at finite ω, heading 0; on the SAME omega axis as
    # added_mass (the ω=inf entry stays 0 and is stripped by the reader's finite mask).
    Fexc = np.zeros((nA, 1, 6), complex)
    for i, w in enumerate(om):
        dp = cpt.DiffractionProblem(body=body, omega=w, wave_direction=0.0, rho=RHO, g=G, water_depth=np.inf)
        dres = solver.solve(dp, keep_details=False)
        fk = froude_krylov_force(dp)
        Fexc[i, 0, :] = [dres.forces[d] + fk[d] for d in DOFS]

    ds = xr.Dataset(
        data_vars=dict(
            added_mass=(("omega", "radiating_dof", "influenced_dof"), A),
            radiation_damping=(("omega", "radiating_dof", "influenced_dof"), B),
            hydrostatic_stiffness=(("radiating_dof", "influenced_dof"), C),
            excitation_force=(("complex", "omega", "wave_direction", "influenced_dof"),
                              np.stack([Fexc.real, Fexc.imag], 0)),
        ),
        coords=dict(omega=("omega", om_all), wave_direction=("wave_direction", [0.0]),
                    radiating_dof=("radiating_dof", DOFS), influenced_dof=("influenced_dof", DOFS),
                    complex=("complex", ["re", "im"])),
        attrs=dict(rho=RHO, g=G, water_depth="inf", body_name="osu_test_buoy_placeholder"),
    )
    ds.to_netcdf(_OUT)
    print(f"wrote {_OUT}")

    # verify FloatSim can read it
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from floatsim.hydro.readers.capytaine import read_capytaine
    hdb = read_capytaine(_OUT)
    print(f"read_capytaine OK: {hdb.omega.size} ω, A_inf[heave]={hdb.A_inf[2, 2]:.2f}, "
          f"C[2,2]={hdb.C[2, 2]:.1f}, |RAO_heave| range {np.abs(hdb.RAO[2]).min():.2f}..{np.abs(hdb.RAO[2]).max():.2f}")


if __name__ == "__main__":
    main()
