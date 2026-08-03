"""Fin-sensitivity study: parametric spar-fin BEM at fin radii {0.215, 0.15,
none}. Builds an axisymmetric spar+annular-fin mesh (validated to reproduce the
GDF-mesh baseline A33=21.1 kg / C33=221 N/m to ~2% at R=0.215) and runs the full
Capytaine radiation+diffraction sweep, saving one NetCDF per fin size.

Geometry from the GDF mesh: spar R=0.0841 (z +0.57..-1.279), thin fin (~4 mm) at
z=-1.141, waterline z=0, CoG z=-1.0163. The fin is a ~4 mm plate (negligible
buoyancy) so draft / C33 / equilibrium are fin-independent; only A33, B33(~0),
and F_exc change with fin size.
"""
from __future__ import annotations

import itertools
import sys
import warnings
from pathlib import Path

import capytaine as cpt
import numpy as np
from capytaine.io.xarray import separate_complex_values

_HERE = Path(__file__).resolve().parent
_OUT = _HERE / "fin_study"

RHO, G = 1025.0, 9.81
R_SPAR, Z_BOTTOM, Z_FIN, T_FIN = 0.0841, -1.279, -1.141, 0.008
COG = np.array([0.0, 0.0, -1.0163])
_OMEGA = np.geomspace(0.1, 30.0, 80)
_FINS = {"0215": 0.215, "015": 0.15, "none": None}


def _revolve(profile, n_theta=48):
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    verts, ring = [], []
    for r, z in profile:
        base = len(verts)
        if r < 1e-9:
            verts.append([0.0, 0.0, z])
            ring.append(("axis", base))
        else:
            verts.extend([r * ct[j], r * st[j], z] for j in range(n_theta))
            ring.append(("ring", base))
    faces = []
    for i in range(len(profile) - 1):
        (k0, b0), (k1, b1) = ring[i], ring[i + 1]
        for j in range(n_theta):
            jn = (j + 1) % n_theta
            if k0 == "ring" and k1 == "ring":
                faces.append([b0 + j, b1 + j, b1 + jn, b0 + jn])
            elif k1 == "axis":
                faces.append([b0 + j, b1, b1, b0 + jn])
            else:
                faces.append([b0, b1 + j, b1 + jn, b0])
    return np.array(verts, float), np.array(faces, int)


def _subdiv(pts, ds=0.035):
    out = [pts[0]]
    for a, b in itertools.pairwise(pts):
        a, b = np.array(a), np.array(b)
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / ds)))
        out.extend(tuple(a + (b - a) * k / n) for k in range(1, n + 1))
    return out


def build_mesh(r_fin):
    if r_fin is None:
        prof = [(R_SPAR, 0.0), (R_SPAR, Z_BOTTOM), (0.0, Z_BOTTOM)]
    else:
        prof = [
            (R_SPAR, 0.0),
            (R_SPAR, Z_FIN + T_FIN / 2), (r_fin, Z_FIN + T_FIN / 2),
            (r_fin, Z_FIN - T_FIN / 2), (R_SPAR, Z_FIN - T_FIN / 2),
            (R_SPAR, Z_BOTTOM), (0.0, Z_BOTTOM),
        ]
    v, f = _revolve(_subdiv(prof))
    m = cpt.Mesh(v, f)
    m.heal_mesh()
    return m


def run_bem(tag, r_fin):
    body = cpt.FloatingBody(mesh=build_mesh(r_fin), center_of_mass=COG, name=f"sf_{tag}")
    body.rotation_center = COG
    body.add_all_rigid_body_dofs()
    imm = body.immersed_part()
    imm.center_of_mass = COG
    imm.rotation_center = COG
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rad = [cpt.RadiationProblem(body=imm, omega=float(w), radiating_dof=d,
                                    water_depth=float("inf"), rho=RHO, g=G)
               for w in [*_OMEGA, float("inf")] for d in imm.dofs]
        dif = [cpt.DiffractionProblem(body=imm, omega=float(w), wave_direction=0.0,
                                      water_depth=float("inf"), rho=RHO, g=G)
               for w in _OMEGA]
        ds = cpt.assemble_dataset(cpt.BEMSolver().solve_all(rad + dif, progress_bar=False))
        ds["hydrostatic_stiffness"] = imm.compute_hydrostatic_stiffness(rho=RHO, g=G)
    out = _OUT / f"capytaine_fin{tag}.nc"
    separate_complex_values(ds).to_netcdf(str(out))
    A = ds["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave")
    C33 = float(ds["hydrostatic_stiffness"].sel(radiating_dof="Heave", influenced_dof="Heave"))
    a_wn = float(A.interp(omega=2.106))
    tn = 2 * np.pi * np.sqrt((28.67 + a_wn) / C33)
    print(f"  fin={tag:>5} ({r_fin}): panels={imm.mesh.nb_faces} C33={C33:.2f} "
          f"A33(wn)={a_wn:.2f} A_inf={float(A.isel(omega=-1)):.2f} T_n={tn:.3f} -> {out.name}",
          flush=True)


def main() -> None:
    _OUT.mkdir(exist_ok=True)
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for tag, rf in _FINS.items():
        if only and tag != only:
            continue
        run_bem(tag, rf)
    print("Done.")


if __name__ == "__main__":
    main()
