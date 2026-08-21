"""OSU Test Buoy — mass/CoG/draft reconciliation + spar BEM, from the mass spreadsheet
(`OSU Spar Buoy Platform Metric.xlsx`) and the measured unloaded waterline (967 mm from
the cylinder bottom = 0.574·L). See OSU-TEST-BUOY-GEOMETRY.md.

Requires: capytaine (already a FloatSim BEM dep) + gmsh-measured geometry. Frame: the
FloatSim convention (still water = z=0, z up).
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.simplefilter("ignore")

# --- measured / spreadsheet inputs (SI) ---
D = 0.1593              # spar OD (gmsh; spreadsheet B2)
R = D / 2
L = 1.683801           # float body length (gmsh; spreadsheet B51)
WL_FRAC = 0.574        # unloaded waterline as fraction of L (spreadsheet B3 = 967 mm)
WL = WL_FRAC * L       # 0.967 m from cylinder bottom
M_STRUCT = 8.16        # all parts minus ballast box (Xabier)
RHO_F, RHO_S, G = 998.0, 1025.0, 9.806
VOL_E = 0.02156562770346663   # displaced volume at the unloaded WL (spreadsheet B49, m^3)
COG_FROM_BOTTOM = 0.05929931334293206  # CoG above cyl bottom, fresh (spreadsheet B15)

A_WP = np.pi * R**2
M_TOT = RHO_F * VOL_E                 # total floating mass (fresh)
Z_BOT, Z_TOP = -WL, L - WL           # cylinder ends in the waterline frame
ZG = COG_FROM_BOTTOM - WL            # CoG in the waterline frame


def reconcile() -> None:
    print("=== mass / floating condition (unloaded, WL 967 mm = 0.574 L) ===")
    print(f"  waterplane A={A_WP:.5f} m^2 | heave stiffness rho*g*A = {RHO_F * G * A_WP:.1f} N/m")
    print(f"  total floating mass = {M_TOT:.2f} kg (fresh) / {RHO_S * VOL_E:.2f} kg (salt)")
    print(f"  structure {M_STRUCT} kg + ballast (lead) ~{M_TOT - M_STRUCT:.1f} kg")
    print(f"  CoG z = {ZG:+.3f} m (waterline frame; {COG_FROM_BOTTOM:.3f} above cyl bottom)")
    print(f"  cylinder z [{Z_BOT:+.3f},{Z_TOP:+.3f}] | submerged {WL:.3f}, freeboard {Z_TOP:.3f}; draft {WL + 0.458:.3f} m")


def spar_bem() -> None:
    import capytaine as cpt
    cpt.set_logging("ERROR")
    mesh = cpt.mesh_vertical_cylinder(length=L, radius=R, center=(0, 0, (Z_BOT + Z_TOP) / 2),
                                      resolution=(4, 44, 64))
    body = cpt.FloatingBody(mesh=mesh, mass=M_TOT, center_of_mass=(0, 0, ZG))
    body.rotation_center = np.array([0.0, 0.0, ZG])
    body.add_all_rigid_body_dofs()
    body = body.immersed_part()
    body.rotation_center = np.array([0.0, 0.0, ZG])
    hs = body.compute_hydrostatics(rho=RHO_F, g=G)
    C33 = float(hs["hydrostatic_stiffness"].sel(influenced_dof="Heave", radiating_dof="Heave"))
    om = np.linspace(0.8, 6.0, 13)
    sol = cpt.BEMSolver()
    A33 = np.array([sol.solve(cpt.RadiationProblem(body=body, omega=w, radiating_dof="Heave",
                    rho=RHO_F, water_depth=np.inf)).added_masses["Heave"] for w in om])
    w = 2.0
    for _ in range(80):
        w = np.sqrt(C33 / (M_TOT + np.interp(w, om, A33)))
    print("\n=== spar-cylinder BEM (capytaine) ===")
    print(f"  C33={C33:.1f} N/m (spreadsheet ~195)  spar heave added mass ~{np.interp(w, om, A33):.1f} kg")
    print(f"  spar-only heave period = {2 * np.pi / w:.2f} s")
    # heave-plate bracket: solid equal-area disc, A_plate = (8/3) rho a^3
    a = np.sqrt(0.328 * 0.198 / np.pi)                 # equal-area disc radius (rect 328x198)
    A_plate = (8.0 / 3.0) * RHO_F * a**3
    Tp = 2 * np.pi * np.sqrt((M_TOT + np.interp(w, om, A33) + A_plate) / C33)
    print(f"  + solid equal-area disc (Ø{2 * a:.2f} m) adds ~{A_plate:.1f} kg -> heave period {Tp:.2f} s (UPPER bound)")
    print("  real perforated/webbed plate: added mass < solid -> heave period ~2.2-2.4 s; pin by tank heave-decay.")


if __name__ == "__main__":
    reconcile()
    spar_bem()
