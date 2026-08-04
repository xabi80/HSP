"""Fin-sensitivity coupled 3-buoy cluster BEM. The parametric spar+fin axisymmetric
mesh (validated single-buoy generator, sparfin_fin_bem) is built as the FULL buoy
(spar above + below water), sunk to the cluster draft (-DZ2), and replicated at the
three cluster positions (0.5 m, 0/120/240 deg) as 3 FloatingBodies -> 18-DOF coupled
radiation + diffraction, matching capytaine_multibody_18dof.nc's format.

Fin radii {0.215, 0.15, none}. Validate the 0.215 case against the existing cluster
BEM (heave-block A33 sum ~64 kg / heave mode ~3.09-3.11 s). Modes:
  probe  -- time 2 omegas x 18 DOF, extrapolate the full cost
  all    -- generate all three capytaine_cluster_fin{tag}.nc
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import capytaine as cpt
import numpy as np
from capytaine.io.xarray import separate_complex_values

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "spar-fin-decay"))

import cluster_common as cc  # noqa: E402
from sparfin_fin_bem import R_SPAR, T_FIN, Z_BOTTOM, Z_FIN, _revolve, _subdiv  # noqa: E402

_OUT = _HERE / "fin_study"
_OMEGA = np.geomspace(0.1, 30.0, 80)
_Z_TOP = 0.573          # spar top, buoy frame (mesh z-max)
_NTHETA = 32
_COG_Z = cc.CoG_Z_SINGLE - cc.DZ2  # -1.19567 (cluster draft)
_FINS = {"0215": 0.215, "015": 0.15, "none": None}


def build_full_mesh(r_fin: float | None) -> cpt.Mesh:
    """FULL spar-fin buoy (spar +0.573 .. -1.279), so it clips correctly once
    sunk to the deeper cluster draft."""
    if r_fin is None:
        prof = [(R_SPAR, _Z_TOP), (R_SPAR, Z_BOTTOM), (0.0, Z_BOTTOM)]
    else:
        prof = [
            (R_SPAR, _Z_TOP),
            (R_SPAR, Z_FIN + T_FIN / 2), (r_fin, Z_FIN + T_FIN / 2),
            (r_fin, Z_FIN - T_FIN / 2), (R_SPAR, Z_FIN - T_FIN / 2),
            (R_SPAR, Z_BOTTOM), (0.0, Z_BOTTOM),
        ]
    v, f = _revolve(_subdiv(prof), _NTHETA)
    m = cpt.Mesh(v, f)
    m.heal_mesh()
    return m


def make_combined(r_fin: float | None):  # type: ignore[no-untyped-def]
    base = build_full_mesh(r_fin).translated([0.0, 0.0, -cc.DZ2])
    bodies = []
    for i, (dx, dy) in enumerate(cc.buoy_offsets()):
        cog = [float(dx), float(dy), _COG_Z]
        b = cpt.FloatingBody(mesh=base.translated([float(dx), float(dy), 0.0]),
                             center_of_mass=cog, name=f"buoy{i + 1}")
        b.rotation_center = np.asarray(cog)
        b.add_all_rigid_body_dofs()
        bodies.append(b)
    return bodies[0] + bodies[1] + bodies[2]


def run(tag: str, r_fin: float | None, probe: bool = False) -> None:
    allb = make_combined(r_fin)
    dofs = list(allb.dofs)
    print(f"  fin={tag}: combined wet panels = {allb.immersed_part().mesh.nb_faces}", flush=True)
    omegas = [float(_OMEGA[0]), float(_OMEGA[40])] if probe else [*_OMEGA, float("inf")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rad = [cpt.RadiationProblem(body=allb, omega=float(w), radiating_dof=d,
                                    water_depth=float("inf"), rho=cc.RHO, g=cc.G)
               for w in omegas for d in dofs]
        t0 = time.perf_counter()
        if probe:
            cpt.BEMSolver().solve_all(rad, progress_bar=False)
            per = (time.perf_counter() - t0) / len(rad)
            nfull = 81 * 18 + 80
            print(f"    probe {len(rad)} problems in {time.perf_counter()-t0:.1f}s "
                  f"({per:.2f}s/problem) -> full ~{per*nfull/60:.1f} min")
            return
        dif = [cpt.DiffractionProblem(body=allb, omega=float(w), wave_direction=0.0,
                                      water_depth=float("inf"), rho=cc.RHO, g=cc.G) for w in _OMEGA]
        ds = cpt.assemble_dataset(cpt.BEMSolver().solve_all(rad + dif, progress_bar=False))
    for c in ("radiating_dof", "influenced_dof"):
        ds = ds.assign_coords({c: [str(x) for x in ds[c].values]})
    separate_complex_values(ds).to_netcdf(str(_OUT / f"capytaine_cluster_fin{tag}.nc"))
    heave = [f"buoy{i + 1}__Heave" for i in range(3)]
    ainf = ds["added_mass"].sel(omega=np.inf)
    hb = sum(float(ainf.sel(radiating_dof=hi, influenced_dof=hj)) for hi in heave for hj in heave)
    print(f"    solved in {time.perf_counter()-t0:.1f}s; A33 heave-block sum(inf) = {hb:.2f} kg "
          f"-> capytaine_cluster_fin{tag}.nc", flush=True)


def main() -> None:
    _OUT.mkdir(exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "probe":
        run("0215", 0.215, probe=True)
    else:
        for tag, rf in _FINS.items():
            run(tag, rf)
    print("Done.")


if __name__ == "__main__":
    main()
