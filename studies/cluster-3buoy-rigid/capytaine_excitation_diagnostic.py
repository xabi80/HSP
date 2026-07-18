"""M8 Phase-1 Measurement A: multi-body EXCITATION structure + condensation.

(a) 18-DOF (3 bodies) diffraction at 6 omegas, heading beta=0 -> report
    the excitation dataset structure (M8 reader input format).
(b) composite single body, same grid -> excitation on 6 DOF.
(c) condensation cross-check: T^T . F_exc_18(omega) vs F_exc_composite,
    T the 18x6 rigid-body kinematic map. Heave component must agree
    within 2% (5% HARD STOP).
"""

from __future__ import annotations

from pathlib import Path

import capytaine as cpt
import numpy as np

import cluster_common as cc

_HERE = Path(__file__).resolve().parent
_SINGLE = _HERE.parent / "spar-fin-decay" / "mesh" / "test2_spar_fin_fullfix_eqdraft.gdf"
_COMPOSITE = cc.CLUSTER_MESH

_OMEGAS = np.geomspace(0.5, 8.0, 6)
_COG_Z = cc.CoG_Z_SINGLE - cc.DZ2                 # -1.19567 (hull ref, cluster draft)
_CLUSTER_COG = np.array([0.0, 0.0, -0.9889])      # composite ref (mass_properties)


def _skew(r):
    return np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])


def _exc(ds, dof_labels):
    """Return complex excitation (n_omega, n_dof) ordered by dof_labels."""
    if "excitation_force" in ds:
        E = ds["excitation_force"]
    else:
        E = ds["Froude_Krylov_force"] + ds["diffraction_force"]
    E = E.sel(wave_direction=0.0)
    return np.array([[complex(E.sel(omega=w, influenced_dof=d).values)
                      for d in dof_labels] for w in _OMEGAS])


def main() -> None:
    print("=" * 70)
    print("M8 Measurement A: multi-body excitation + condensation")
    print("=" * 70)

    # --- (a) 18-DOF multi-body diffraction ---
    base = cpt.load_mesh(str(_SINGLE), file_format="gdf").translated([0, 0, -cc.DZ2])
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
    dofs18 = list(allb.dofs)
    probs = [cpt.DiffractionProblem(body=allb, omega=float(w), wave_direction=0.0,
                                    water_depth=float("inf"), rho=cc.RHO, g=cc.G)
             for w in _OMEGAS]
    ds18 = cpt.assemble_dataset(cpt.BEMSolver().solve_all(probs, n_jobs=1))
    for c in ("influenced_dof",):
        ds18 = ds18.assign_coords({c: [str(x) for x in ds18[c].values]})
    print("\n(a) 18-DOF EXCITATION dataset structure (M8 reader input):")
    print(f"  data_vars: {sorted(ds18.data_vars)}")
    print(f"  dims: {dict(ds18.sizes)}")
    ev = "excitation_force" if "excitation_force" in ds18 else "Froude_Krylov_force"
    print(f"  {ev} dims: {ds18[ev].dims}")
    print(f"  influenced_dof (18): {[str(x) for x in ds18['influenced_dof'].values]}")

    # --- (b) composite single body ---
    mc = cpt.load_mesh(str(_COMPOSITE), file_format="gdf")
    bc = cpt.FloatingBody(mesh=mc, center_of_mass=_CLUSTER_COG, name="composite")
    bc.rotation_center = _CLUSTER_COG
    bc.add_all_rigid_body_dofs()
    dofs6 = list(bc.dofs)
    probc = [cpt.DiffractionProblem(body=bc, omega=float(w), wave_direction=0.0,
                                    water_depth=float("inf"), rho=cc.RHO, g=cc.G)
             for w in _OMEGAS]
    dsc = cpt.assemble_dataset(cpt.BEMSolver().solve_all(probc, n_jobs=1))
    dsc = dsc.assign_coords({"influenced_dof": [str(x) for x in dsc["influenced_dof"].values]})

    F18 = _exc(ds18, dofs18)        # (6, 18)
    Fc = _exc(dsc, dofs6)           # (6, 6)

    # --- (c) build T (18x6) and condense ---
    T = np.zeros((18, 6))
    for i, (dx, dy) in enumerate(offsets):
        r = np.array([dx, dy, _COG_Z]) - _CLUSTER_COG
        Ti = np.zeros((6, 6))
        Ti[:3, :3] = np.eye(3)
        Ti[:3, 3:] = -_skew(r)
        Ti[3:, 3:] = np.eye(3)
        T[6 * i:6 * i + 6, :] = Ti
    Fcond = F18 @ T                 # (6, 6): T^T F_18 per omega  (F18 row . T)

    # NOTE: both models were solved on the SAME _OMEGAS array (lines
    # 'for w in _OMEGAS' in both problem lists) and extracted with exact
    # .sel(omega=w) -- matched grids by construction, no interpolation.
    names = ["surge", "sway", "heave", "roll", "pitch", "yaw"]
    # Relative error is only meaningful where the excitation is actually
    # non-zero. At heading beta=0 the cluster is y-mirror symmetric, so
    # sway/roll/yaw are symmetry-FORBIDDEN and sit at the numerical noise
    # floor in BOTH models -- a noise/noise ratio there is meaningless.
    # Floor: 1e-6 * max|F_comp| over all DOF at that omega.
    print("\n(c) CONDENSATION cross-check -- ALL 6 DOF (matched grids), per omega:")
    worst_per_dof = {}
    excited = {}
    for j, nm in enumerate(names):
        print(f"\n  --- {nm} (DOF {j}) ---")
        print(f"  {'omega':>8} {'|F_cond|':>14} {'|F_comp|':>14} "
              f"{'mag rel-diff':>14} {'phase diff (deg)':>17}")
        worst = 0.0
        any_excited = False
        for k, w in enumerate(_OMEGAS):
            fc, fk = Fcond[k, j], Fc[k, j]
            floor = 1e-6 * np.max(np.abs(Fc[k, :]))
            if abs(fk) <= floor:
                print(f"  {w:8.4f} {abs(fc):14.5e} {abs(fk):14.5e} "
                      f"{'below floor':>14} {'n/a':>17}")
                continue
            any_excited = True
            rel = abs(fc - fk) / abs(fk)
            dphase = (np.degrees(np.angle(fc) - np.angle(fk)) + 180) % 360 - 180
            worst = max(worst, rel)
            print(f"  {w:8.4f} {abs(fc):14.5f} {abs(fk):14.5f} "
                  f"{rel:14.4%} {dphase:17.4f}")
        worst_per_dof[nm] = worst if any_excited else float("nan")
        excited[nm] = any_excited
        print(f"  worst {nm} rel-diff = "
              f"{worst:.4%}" if any_excited else
              f"  {nm}: symmetry-FORBIDDEN at beta=0 -- below noise floor at all omega")

    print("\n  === per-DOF worst magnitude rel-diff (matched grids) ===")
    for nm, v in worst_per_dof.items():
        if not excited[nm]:
            print(f"    {nm:6s}: forbidden at beta=0 (noise floor) -- not compared")
        else:
            print(f"    {nm:6s}: {v:10.4%}   {'OK' if v < 0.05 else 'OVER 5%'}")
    h = worst_per_dof["heave"]
    print(f"\n  HEAVE gate: worst {h:.4%} "
          f"(pin 2%: {'OK' if h < 0.02 else 'over'}; "
          f"HARD STOP 5%: {'FAIL' if h > 0.05 else 'OK'})")

    # Surge / pitch response curves from BOTH models (feature visibility).
    print("\n  === surge & pitch excitation curves, BOTH models ===")
    print(f"  {'omega':>8} {'|surge|cond':>13} {'|surge|comp':>13} "
          f"{'|pitch|cond':>13} {'|pitch|comp':>13}")
    for k, w in enumerate(_OMEGAS):
        print(f"  {w:8.4f} {abs(Fcond[k,0]):13.5f} {abs(Fc[k,0]):13.5f} "
              f"{abs(Fcond[k,4]):13.5f} {abs(Fc[k,4]):13.5f}")


if __name__ == "__main__":
    main()
