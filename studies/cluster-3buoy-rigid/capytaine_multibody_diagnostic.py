"""Pre-skeleton measurement A (Tier 3): true multi-body Capytaine.

The SAME three hulls at the SAME cluster draft, but defined as THREE
SEPARATE cpt.FloatingBody objects (buoy1/buoy2/buoy3) with all rigid
DOFs each -> 18 radiating DOFs and the full 18x18 coupled added-mass /
radiation-damping matrices. This is the exact input format the future
B4 multi-body BEM reader must parse. Radiation only; reduced grid.
"""

from __future__ import annotations

import time
from pathlib import Path

import capytaine as cpt
import numpy as np

import cluster_common as cc

_HERE = Path(__file__).resolve().parent
_OUT_NC = _HERE / "capytaine_multibody_18dof.nc"
_SINGLE = _HERE.parent / "spar-fin-decay" / "mesh" / "test2_spar_fin_fullfix_eqdraft.gdf"

_OMEGAS = np.geomspace(0.5, 8.0, 12)   # brackets omega_n ~ 1.97-2.02 rad/s
_COG_Z = cc.CoG_Z_SINGLE - cc.DZ2      # -1.19567 (cluster draft)


def main() -> None:
    print("=" * 70)
    print("Pre-skeleton A: multi-body (3 x 6-DOF = 18-DOF) Capytaine")
    print("=" * 70)
    base = cpt.load_mesh(str(_SINGLE), file_format="gdf")
    # Sink to cluster draft.
    base = base.translated([0.0, 0.0, -cc.DZ2])
    offsets = cc.buoy_offsets()
    bodies = []
    for i, (dx, dy) in enumerate(offsets):
        m = base.translated([float(dx), float(dy), 0.0])
        cog = [float(dx), float(dy), _COG_Z]
        b = cpt.FloatingBody(mesh=m, center_of_mass=cog, name=f"buoy{i+1}")
        b.rotation_center = np.asarray(cog)
        b.add_all_rigid_body_dofs()
        bodies.append(b)
    allb = bodies[0] + bodies[1] + bodies[2]
    dofs = list(allb.dofs)
    print(f"  Combined body '{allb.name}' with {len(dofs)} DOFs:")
    print(f"    {dofs}")

    omegas = list(_OMEGAS) + [float("inf")]
    problems = [
        cpt.RadiationProblem(body=allb, omega=float(w), radiating_dof=d,
                             water_depth=float("inf"), rho=cc.RHO, g=cc.G)
        for w in omegas for d in dofs
    ]
    print(f"  radiation problems: {len(problems)} "
          f"({len(dofs)} DOF x {len(omegas)} omega)")
    t0 = time.perf_counter()
    results = cpt.BEMSolver().solve_all(problems, n_jobs=1)
    dt = time.perf_counter() - t0
    print(f"  solved in {dt:.1f} s")
    ds = cpt.assemble_dataset(results)
    # Capytaine labels dof coords as pandas Categorical, which the netCDF
    # writer cannot serialize -- stringify them (B4's reader will parse
    # these exact labels).
    for c in ("radiating_dof", "influenced_dof"):
        ds = ds.assign_coords({c: [str(x) for x in ds[c].values]})

    # (a) dataset structure
    print("\n(a) DATASET STRUCTURE (B4 reader input format):")
    print(f"  dims: {dict(ds.sizes)}")
    print(f"  radiating_dof coords: {list(ds['radiating_dof'].values)}")
    A = ds["added_mass"]   # (omega, radiating_dof, influenced_dof)
    print(f"  added_mass dims: {A.dims}, shape {A.shape}")

    # (b) A_inf heave-block 3x3
    heave = [f"buoy{i+1}__Heave" for i in range(3)]
    Ainf = A.sel(omega=np.inf)
    hb = np.array([[float(Ainf.sel(radiating_dof=hi, influenced_dof=hj))
                    for hj in heave] for hi in heave])
    print("\n(b) A_inf HEAVE-BLOCK 3x3 (kg):")
    for row in hb:
        print("   " + "  ".join(f"{v:+.4f}" for v in row))
    total = float(hb.sum())
    print(f"  sum of 9 entries = {total:.4f} kg")
    # Consistency vs the single-body composite A33 (measured earlier).
    import json
    inter = json.loads((_HERE / "results" / "interaction.json").read_text())
    comp = inter["A33_composite_inf"]
    rel = (total - comp) / comp
    print(f"  single-body composite A33_inf = {comp:.4f} kg")
    print(f"  multi-body sum / composite - 1 = {rel:+.3%}  "
          f"(HARD STOP > 5%: {'FAIL' if abs(rel) > 0.05 else 'OK'})")

    # (c) off-diagonal magnitude
    diag = np.diag(hb).mean()
    offs = hb[~np.eye(3, dtype=bool)]
    print("\n(c) OFF-DIAGONAL heave coupling (fraction of diagonal):")
    print(f"  mean diagonal A33_ii = {diag:.4f} kg")
    print(f"  off-diagonal A33_ij: mean {offs.mean():+.4f}, "
          f"range [{offs.min():+.4f}, {offs.max():+.4f}]")
    print(f"  |A33_ij| / A33_ii = {np.abs(offs).mean()/diag:.3%} "
          f"(this is what block-diagonal zeroing discards)")

    # (d) symmetry of the full 18x18
    A18 = Ainf.values  # (18, 18)
    asym = np.abs(A18 - A18.T)
    scale = np.abs(A18).max()
    print("\n(d) SYMMETRY of the full 18x18 A_inf:")
    print(f"  max |A_ij - A_ji| = {asym.max():.4e}; "
          f"relative to max|A| ({scale:.3f}) = {asym.max()/scale:.3e}")

    # Save the fixture last (after all analysis is printed).
    from capytaine.io.xarray import separate_complex_values
    separate_complex_values(ds).to_netcdf(str(_OUT_NC))
    print(f"\n  wrote fixture {_OUT_NC.name}")


if __name__ == "__main__":
    main()
