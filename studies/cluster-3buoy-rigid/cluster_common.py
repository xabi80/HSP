"""Shared constants + helpers for the 3-buoy rigid-cluster study.

Composite single-body: three spar-fin fullfix hulls on a 0.5 m-radius
circle (120 deg apart, one on +x) rigidly joined by a 12 kg dry arm
structure. Built on the M7.5-hardened stack + the spar-fin study's
eqdraft mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
_SPARFIN = _HERE.parent / "spar-fin-decay"

# Source single-hull mesh: spar-fin eqdraft (z=0 at the ISOLATED
# free-floating waterline; hull already normal-corrected, 1488 panels).
SINGLE_EQDRAFT_MESH = _SPARFIN / "mesh" / "test2_spar_fin_fullfix_eqdraft.gdf"
CLUSTER_MESH = _HERE / "mesh" / "cluster3_fullfix.gdf"

# --- Physical constants ---
RHO = 1025.0
G = 9.81
R_SPAR = 0.0841
A_WP_SINGLE = np.pi * R_SPAR**2       # 0.022220 m^2
A_WP_CLUSTER = 3.0 * A_WP_SINGLE       # 0.066660 m^2

# --- Single-buoy properties (from the spar-fin study) ---
M_BUOY = 28.67                          # kg per hull
CoG_Z_SINGLE = -1.0163                  # eqdraft-frame CoG z (isolated)
I_XX_BUOY = I_YY_BUOY = 24.0            # kg*m^2 at single-buoy CoM
I_ZZ_BUOY = 0.114

# --- Arm structure (dry) ---
N_ARMS = 3
ARM_MASS_EACH = 4.0                     # kg (3 x 4 = 12 kg total)
ARM_MASS_TOTAL = N_ARMS * ARM_MASS_EACH
ARM_CENTER_TO_TIP = 0.5                 # m (radial rod, hub to buoy center)

# --- Cluster geometry ---
CLUSTER_RADIUS = 0.5                    # m
# Buoy placement: one on +x, 120 deg apart (CCW). Convention reported
# in build_cluster_mesh.py output.
BUOY_ANGLES_DEG = np.array([0.0, 120.0, 240.0])

M_CLUSTER = 3.0 * M_BUOY + ARM_MASS_TOTAL  # 98.01 kg

# Additional sink of the cluster below the single-buoy waterline, to
# carry the extra 12 kg arm mass. First-pass 0.1757; REFINED by
# cluster_balance.py to displace 98.01 kg at z=0 (measured 2026-07-04).
DZ2 = 0.17937


def buoy_offsets() -> NDArray[np.float64]:
    """(3, 2) horizontal (x, y) centres of the three hulls."""
    ang = np.deg2rad(BUOY_ANGLES_DEG)
    return np.column_stack([CLUSTER_RADIUS * np.cos(ang),
                            CLUSTER_RADIUS * np.sin(ang)])


def _clip_triangle_below(tri: NDArray[np.float64]) -> list[NDArray[np.float64]]:
    """Sutherland-Hodgman clip of a triangle to z<0; fan-triangulate."""
    z = tri[:, 2]
    if np.all(z < 0.0):
        return [tri]
    if np.all(z >= 0.0):
        return []

    def isect(p_in, p_out):
        t = p_in[2] / (p_in[2] - p_out[2])
        return p_in + t * (p_out - p_in)

    poly: list[NDArray[np.float64]] = []
    for i in range(3):
        c, n = tri[i], tri[(i + 1) % 3]
        ci, ni = c[2] < 0.0, n[2] < 0.0
        if ci:
            poly.append(c)
            if not ni:
                poly.append(isect(c, n))
        elif ni:
            poly.append(isect(c, n))
    if len(poly) < 3:
        return []
    return [np.stack([poly[0], poly[k], poly[k + 1]]) for k in range(1, len(poly) - 1)]


def displaced_volume_below_waterline(panels: NDArray[np.float64]) -> float:
    """Displaced volume (z<0) of an outward-oriented quad mesh.

    V = (1/6) sum over clipped-below triangles of v0.(v1 x v2); the z=0
    lid contributes 0 so the wetted-surface sum equals the enclosed
    below-water volume.
    """
    total = 0.0
    for p in panels:
        for tri in ((p[0], p[1], p[2]), (p[0], p[2], p[3])):
            a, b, c = (np.asarray(v, dtype=np.float64) for v in tri)
            if np.linalg.norm(np.cross(b - a, c - a)) < 1.0e-15:
                continue
            for t in _clip_triangle_below(np.stack([a, b, c])):
                total += float(np.dot(t[0], np.cross(t[1], t[2])))
    return total / 6.0
