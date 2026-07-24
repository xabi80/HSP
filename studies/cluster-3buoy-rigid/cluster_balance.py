"""Step 2: cluster waterline balance (refine dz2) + composite mass properties.

Refines the additional sink dz2 so the composite displaces the cluster
mass (98.01 kg) at z=0, rebuilds the mesh, runs a tier-2 sanity check,
and computes the composite mass/CoG/inertia (about the composite CoG)
in-script.
"""

from __future__ import annotations

import json

import cluster_common as cc
import numpy as np
from build_cluster_mesh import build

from floatsim.hydro.mesh_hygiene import check_hydrostatic_volume, write_gdf_panels


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _parallel_axis(I_cm, mass, d):
    d = np.asarray(d, dtype=np.float64)
    return I_cm + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def main() -> None:
    print("=" * 70)
    print("Step 2: cluster waterline balance + mass properties")
    print("=" * 70)
    target = cc.M_CLUSTER
    print(f"Target displaced mass = {target:.4f} kg "
          f"(3 x {cc.M_BUOY} + {cc.ARM_MASS_TOTAL} arm)")
    print(f"Cluster waterplane A_wp = {cc.A_WP_CLUSTER:.6f} m^2")

    # --- Refine dz2: one Newton step (constant waterplane) is exact ---
    dz2 = cc.DZ2
    for it in range(4):
        mesh = build(dz2)
        m_disp = cc.RHO * cc.displaced_volume_below_waterline(mesh.panels)
        err = m_disp - target
        print(f"  iter {it}: dz2 = {dz2:.5f} m -> displaced {m_disp:.4f} kg "
              f"(err {err:+.4f} kg, {err/target:+.3%})")
        if abs(err / target) < 0.001:
            break
        dz2 -= err / (cc.RHO * cc.A_WP_CLUSTER)  # Newton on the waterplane

    print(f"\nFinal dz2 = {dz2:.5f} m  (predicted band [0.16, 0.19])")
    if not (0.16 <= dz2 <= 0.19):
        raise SystemExit(f"STOP: dz2 = {dz2:.5f} m outside [0.16, 0.19].")
    rel = (m_disp - target) / target
    if abs(rel) > 0.01:
        raise SystemExit(f"STOP: displaced mass off by {rel:+.3%} (> 1%).")
    print(f"Displaced mass = {m_disp:.4f} kg ({rel:+.3%} of target) -- PASS")

    # Rewrite the mesh at the final dz2.
    mesh = build(dz2)
    write_gdf_panels(mesh, cc.CLUSTER_MESH)
    z_top = float(mesh.panels[..., 2].max())
    z_bot = float(mesh.panels[..., 2].min())
    print(f"Rewrote {cc.CLUSTER_MESH.name}; z-extent [{z_bot:.4f}, {z_top:.4f}]")

    # --- Tier-2 sanity (full closed-mesh reserve buoyancy = 3x single) ---
    vr = check_hydrostatic_volume(mesh, rho=cc.RHO, mass=target)
    print(f"\nTier-2 (composite full-mesh): displaced_if_submerged = "
          f"{vr.displaced_mass:.2f} kg (expect ~3 x 40.9 = 122.7)")

    # --- Composite mass properties (about the composite CoG) ---
    offsets = cc.buoy_offsets()
    z_buoy_cog = cc.CoG_Z_SINGLE - dz2               # buoy CoM z, cluster frame
    z_arm = z_top + 0.1                              # arm structure z
    # Buoy CoMs (3) and arm CoMs (3 rods, mid-radius).
    buoy_coms = np.column_stack([offsets[:, 0], offsets[:, 1],
                                 np.full(3, z_buoy_cog)])
    ang = np.deg2rad(cc.BUOY_ANGLES_DEG)
    arm_coms = np.column_stack([0.5 * cc.ARM_CENTER_TO_TIP * np.cos(ang),
                                0.5 * cc.ARM_CENTER_TO_TIP * np.sin(ang),
                                np.full(3, z_arm)])
    # Composite CoG.
    total_m = 3 * cc.M_BUOY + cc.ARM_MASS_TOTAL
    cog = (cc.M_BUOY * buoy_coms.sum(axis=0)
           + cc.ARM_MASS_EACH * arm_coms.sum(axis=0)) / total_m
    print(f"\nComposite CoG = ({cog[0]:+.4e}, {cog[1]:+.4e}, {cog[2]:+.4f}) m "
          f"(predicted z ~ -0.98 +/- 0.03)")

    # Inertia tensor about the composite CoG.
    I = np.zeros((3, 3))
    I_buoy_cm = np.diag([cc.I_XX_BUOY, cc.I_YY_BUOY, cc.I_ZZ_BUOY]).astype(float)
    for com in buoy_coms:
        I += _parallel_axis(I_buoy_cm, cc.M_BUOY, com - cog)
    L = cc.ARM_CENTER_TO_TIP
    for a, com in zip(ang, arm_coms, strict=False):
        u = np.array([np.cos(a), np.sin(a), 0.0])         # rod axis
        I_rod_cm = (cc.ARM_MASS_EACH * L**2 / 12.0) * (np.eye(3) - np.outer(u, u))
        I += _parallel_axis(I_rod_cm, cc.ARM_MASS_EACH, com - cog)

    print("Composite inertia about CoG (kg*m^2):")
    for row in I:
        print("   " + "  ".join(f"{v:+.4f}" for v in row))

    props = {
        "dz2_m": dz2,
        "displaced_mass_kg": m_disp,
        "mass_kg": total_m,
        "cog_m": cog.tolist(),
        "z_buoy_cog": z_buoy_cog,
        "z_arm": z_arm,
        "inertia_about_cog": I.tolist(),
        "z_extent": [z_bot, z_top],
    }
    (cc._HERE / "results" / "mass_properties.json").write_text(
        json.dumps(props, indent=2)
    )
    print("\nWrote results/mass_properties.json")


if __name__ == "__main__":
    main()
