"""M8 PR3 defect diagnostic (Steps 1-2 of the PR3 Step-C redirect).

Characterizes the spurious negative heave-damping feature at omega ~ 4.934
that the PSD gate flagged on the production-grid 18-DOF fixture.

Diagnostic A -- is it a sharp FEATURE (finite width, shoulders, neighbours
perturbed) or an ISOLATED single-point failure (neighbours clean, one omega
anomalous)? Cheapest reproducer: SINGLE hull at CLUSTER draft, heave only.
Plus the full 18x18 condition at omega=4.934 on the committed fixture.

Diagnostic B -- lid efficacy on the frequency Capytaine actually flags as
irregular (omega ~ 20.909), proper workflow (immersed_part -> generate_lid
-> lid_mesh=), no-lid vs lid, on the 3-hull cluster mesh.

Reports numbers verbatim; draws no conclusion the caller must act on.
"""

from __future__ import annotations

from pathlib import Path

import capytaine as cpt
import cluster_common as cc
import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
_GDF = _HERE.parent / "spar-fin-decay" / "mesh" / "test2_spar_fin_fullfix_eqdraft.gdf"
_FIXTURE = _HERE / "capytaine_multibody_18dof.nc"

_COG_Z = cc.CoG_Z_SINGLE - cc.DZ2  # cluster-draft CoG z


def _base_cluster_draft() -> cpt.Mesh:
    return cpt.load_mesh(str(_GDF), file_format="gdf").translated([0.0, 0.0, -cc.DZ2])


def diagnostic_a() -> None:
    print("=" * 70)
    print("DIAGNOSTIC A: omega~4.934 feature -- single hull @ cluster draft, heave")
    print("=" * 70)
    base = _base_cluster_draft()
    dx, dy = (float(v) for v in cc.buoy_offsets()[0])
    cog = [dx, dy, _COG_Z]
    m = base.translated([dx, dy, 0.0]).immersed_part()
    sb = cpt.FloatingBody(mesh=m, center_of_mass=cog, name="buoy1")
    sb.rotation_center = np.asarray(cog)
    sb.add_all_rigid_body_dofs()

    # Waterplane radius (for the Step-5 ka re-derivation): panel centroids
    # of the lid at z=0 -> max horizontal distance from the hull axis.
    lid = m.generate_lid(z=0.0)
    if lid is not None and lid.nb_faces > 0:
        c = lid.faces_centers
        r = np.hypot(c[:, 0] - dx, c[:, 1] - dy)
        print(f"  waterplane lid: {lid.nb_faces} faces; max radius ~ {r.max():.4f} m")

    omegas = [4.70, 4.80, 4.90, 4.92, 4.93, 4.934, 4.94, 4.95, 5.00, 5.10, 5.20]
    probs = [
        cpt.RadiationProblem(
            body=sb,
            omega=w,
            radiating_dof="Heave",
            water_depth=float("inf"),
            rho=cc.RHO,
            g=cc.G,
        )
        for w in omegas
    ]
    ds = cpt.assemble_dataset(cpt.BEMSolver().solve_all(probs, n_jobs=1))
    print("\n  B_heave(omega) verbatim:")
    for w in omegas:
        v = float(
            ds["radiation_damping"].sel(omega=w, radiating_dof="Heave", influenced_dof="Heave")
        )
        flag = "   <== NEGATIVE" if v < 0 else ""
        print(f"    w={w:6.3f}  B_heave={v:+.6e}{flag}")


def diagnostic_a_fullmatrix() -> None:
    print("\n" + "=" * 70)
    print("DIAGNOSTIC A (full 18x18 condition at omega=4.934, committed fixture)")
    print("=" * 70)
    with xr.open_dataset(_FIXTURE) as ds:
        w = ds["omega"].values
        B = ds["radiation_damping"].values  # (n_omega, 18, 18)
    fin = np.isfinite(w)
    w = w[fin]
    B = B[fin]
    order = np.argsort(w)
    w = w[order]
    B = B[order]
    k = int(np.argmin(np.abs(w - 4.934)))
    kL, kR = k - 1, k + 1
    Bk = B[k]
    ev = np.linalg.eigvalsh(0.5 * (Bk + Bk.T))
    print(f"  omega[{k}] = {w[k]:.4f} (neighbours {w[kL]:.4f}, {w[kR]:.4f})")
    print(f"  full 18x18 min eig = {ev[0]:+.4e};  max eig = {ev[-1]:+.4e}")
    print(
        f"  global max|B| = {np.abs(B).max():.4f};  min-eig / max|B| = {ev[0]/np.abs(B).max():+.3%}"
    )
    print(
        "\n  Do the LARGE (surge/roll/pitch) diagonals sit on their neighbours' "
        "trend at omega=4.934?"
    )
    labels = {0: "Surge", 3: "Roll", 4: "Pitch", 2: "Heave"}
    for i, name in labels.items():
        lo, mid, hi = B[kL, i, i], B[k, i, i], B[kR, i, i]
        interp = 0.5 * (lo + hi)
        dev = (mid - interp) / (abs(interp) + 1e-30)
        print(
            f"    DOF {i:2d} {name:6s}: B_ii(neighbours {lo:+.4e}/{hi:+.4e}) "
            f"mid={mid:+.4e}  dev-from-midpoint={dev:+.2%}"
        )


def diagnostic_b() -> None:
    print("\n" + "=" * 70)
    print("DIAGNOSTIC B: lid efficacy on Capytaine's FLAGGED irregular freq (~20.909)")
    print("=" * 70)
    base = _base_cluster_draft()
    offsets = cc.buoy_offsets()
    ws = (18.098, 20.909, 22.475)  # around the flagged band; 20.909 is a grid point

    def build(with_lid: bool) -> cpt.FloatingBody:
        bodies = []
        for i, (dx, dy) in enumerate(offsets):
            dx, dy = float(dx), float(dy)
            cog = [dx, dy, _COG_Z]
            mi = base.translated([dx, dy, 0.0]).immersed_part()
            lid = mi.generate_lid(z=0.0) if with_lid else None
            b = cpt.FloatingBody(mesh=mi, lid_mesh=lid, center_of_mass=cog, name=f"buoy{i+1}")
            b.rotation_center = np.asarray(cog)
            b.add_all_rigid_body_dofs()
            bodies.append(b)
        return bodies[0] + bodies[1] + bodies[2]

    for with_lid in (False, True):
        allb = build(with_lid)
        lf = allb.lid_mesh.nb_faces if allb.lid_mesh is not None else 0
        probs = [
            cpt.RadiationProblem(
                body=allb,
                omega=w,
                radiating_dof="buoy1__Heave",
                water_depth=float("inf"),
                rho=cc.RHO,
                g=cc.G,
            )
            for w in ws
        ]
        ds = cpt.assemble_dataset(cpt.BEMSolver().solve_all(probs, n_jobs=1))
        print(f"\n  with_lid={with_lid} (combined lid faces={lf}):")
        for w in ws:
            v = float(
                ds["radiation_damping"].sel(
                    omega=w, radiating_dof="buoy1__Heave", influenced_dof="buoy1__Heave"
                )
            )
            print(f"    w={w:7.3f}: B_heave={v:+.6e}")
        # also the 4.934 heave point, both cases, for the side-by-side
        p2 = [
            cpt.RadiationProblem(
                body=allb,
                omega=4.934,
                radiating_dof="buoy1__Heave",
                water_depth=float("inf"),
                rho=cc.RHO,
                g=cc.G,
            )
        ]
        d2 = cpt.assemble_dataset(cpt.BEMSolver().solve_all(p2, n_jobs=1))
        v2 = float(
            d2["radiation_damping"].sel(
                omega=4.934, radiating_dof="buoy1__Heave", influenced_dof="buoy1__Heave"
            )
        )
        print(f"    w=  4.934: B_heave={v2:+.6e}  " "(flagged-by-PSD, NOT-flagged-by-Capytaine)")


if __name__ == "__main__":
    diagnostic_a()
    diagnostic_a_fullmatrix()
    diagnostic_b()
