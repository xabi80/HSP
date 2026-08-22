"""Buoy pitch/roll/yaw inertia about the CoG, from the STEP per-part geometry (gmsh) +
the mass model (structure 8.16 kg over the solids at a uniform effective density; lead
13.36 kg as a compact mass at the ballast). Feeds osu_buoy_common.I_XX/I_YY/I_ZZ.

gmsh `getMatrixOfInertia` is about each solid's CENTROID at unit density (verified here
against the analytical pipe). Frame: STEP (cyl bottom z=0) → waterline (z −= 0.967 m).
Requires: `pip install gmsh`.  Usage: `python inertia_from_step.py ["…/OSU Test Buoy.stp"]`
"""
from __future__ import annotations

import sys

import gmsh
import numpy as np

_STEP = sys.argv[1] if len(sys.argv) > 1 else r"C:/Users/xlama/OneDrive/Documents/buoy/OSU Test Buoy.stp"
WL, ZG = 0.967, -0.907           # waterline height above the STEP datum; buoy CoG (waterline frame)
M_STRUCT, M_LEAD, Z_LEAD = 8.16, 13.36, -1.383


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(_STEP)
    gmsh.model.occ.synchronize()
    parts = []
    for dim, tag in gmsh.model.getEntities(3):
        V = gmsh.model.occ.getMass(dim, tag) / 1e9                       # m^3
        c = np.array(gmsh.model.occ.getCenterOfMass(dim, tag)) / 1000.0  # m
        Ig = np.array(gmsh.model.occ.getMatrixOfInertia(dim, tag)).reshape(3, 3) / 1e15  # /ρ, about centroid, m^5
        parts.append((V, c, Ig))
    gmsh.finalize()

    rho_eff = M_STRUCT / sum(p[0] for p in parts)
    Itot = np.zeros((3, 3))
    for V, c, Ig in parts:                       # structure, uniform effective density
        m = rho_eff * V
        cw = np.array([c[0], c[1], c[2] - WL])   # centroid, waterline frame
        d = cw - np.array([0.0, 0.0, ZG])        # relative to buoy CoG
        Itot += rho_eff * Ig + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    d = np.array([0.0, 0.0, Z_LEAD - ZG])        # lead as a compact mass at the ballast
    Itot += M_LEAD * ((d @ d) * np.eye(3) - np.outer(d, d))

    print(f"structure eff. density = {rho_eff:.0f} kg/m^3 (to match {M_STRUCT} kg)")
    print(f"buoy inertia about CoG (kg·m^2): I_xx={Itot[0, 0]:.2f}  I_yy={Itot[1, 1]:.2f}  I_zz={Itot[2, 2]:.3f}")


if __name__ == "__main__":
    main()
