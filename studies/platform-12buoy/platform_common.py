"""Shared constants + helpers for the 12-buoy platform (M11b PR6).

The platform is 4 clusters (C4-a: 90 deg, 1 m arm radius) x 3 buoys (0.5 m,
0/120/240, identical orientation) = 12 spar-fin hulls, articulated
(``yaw_locked``) buoy->hub and hub->platform. Geometry cited from
``docs/platform-geometry.md``; the single hull + intra-cluster constants are
reused from the M10 cluster study (``cluster_common``).

Draft: the buoys carry the WHOLE floating mass (arms + platform are dry, S8),
so each buoy displaces ``M_TOTAL / 12`` at equilibrium -- deeper than the
cluster because per-buoy support rises 32.67 -> 33.50 kg (geometry §3.3).
``PLATFORM_DZ`` is the additional sink below the isolated single-buoy
waterline, re-derived on the mesh by :func:`derive_draft` (the
``cluster_balance.py`` Newton method), NOT copied from the cluster ``DZ2``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))

import cluster_common as cc  # noqa: E402

PLATFORM_MESH = _HERE / "mesh" / "platform12_fullfix.gdf"

# --- Platform geometry (platform-geometry.md §1 supplied, C4-a assumption) ---
CLUSTER_ARM_RADIUS = 1.0  # m, cluster centre from platform centre (S2)
CLUSTER_ANGLES_DEG = np.array([0.0, 90.0, 180.0, 270.0])  # C4-a (§4)
BUOY_ANGLES_DEG = cc.BUOY_ANGLES_DEG  # [0,120,240], identical per cluster
BUOY_RADIUS = cc.CLUSTER_RADIUS  # 0.5 m intra-cluster (R1)

# --- Mass balance (platform-geometry.md §3.3; C4-c: 10 kg platform) ---
N_BUOY = 12
ARMS_TOTAL = 4.0 * cc.ARM_MASS_TOTAL  # 4 clusters x 12 kg = 48 kg
PLATFORM_MASS = 10.0  # C4-c assumption (48 + 10 < 60, S5)
M_TOTAL = N_BUOY * cc.M_BUOY + ARMS_TOTAL + PLATFORM_MASS  # 402.04 kg
M_PER_BUOY = M_TOTAL / N_BUOY  # 33.5033 kg

# Additional sink of every buoy below the isolated single-buoy waterline, to
# displace M_PER_BUOY at z=0. Re-derived on the mesh (derive_draft); cluster
# DZ2 = 0.17937 carried +4.0 kg/buoy, the platform carries +4.833 -> deeper.
PLATFORM_DZ = 0.21638


def buoy_centers() -> NDArray[np.float64]:
    """(12, 2) horizontal (x, y) centres: 4 clusters x 3 buoys, buoy k = 3c+b."""
    out = []
    for pc in np.deg2rad(CLUSTER_ANGLES_DEG):
        cx, cy = CLUSTER_ARM_RADIUS * np.cos(pc), CLUSTER_ARM_RADIUS * np.sin(pc)
        for tb in np.deg2rad(BUOY_ANGLES_DEG):
            out.append([cx + BUOY_RADIUS * np.cos(tb), cy + BUOY_RADIUS * np.sin(tb)])
    return np.asarray(out, dtype=np.float64)


def closest_cross_cluster_gap() -> float:
    """Minimum centre distance between buoys of DIFFERENT clusters (geom §3.6 = 0.620 m)."""
    c = buoy_centers()
    best = np.inf
    for i in range(N_BUOY):
        for j in range(i + 1, N_BUOY):
            if i // 3 != j // 3:
                best = min(best, float(np.hypot(*(c[i] - c[j]))))
    return best


def _single_displaced_mass(dz: float) -> float:
    """kg displaced (z<0) by ONE isolated hull sunk an extra ``dz``."""
    from floatsim.hydro.mesh_hygiene import load_gdf_panels

    p = load_gdf_panels(cc.SINGLE_EQDRAFT_MESH).panels.copy()
    p[..., 2] -= dz
    return cc.RHO * cc.displaced_volume_below_waterline(p)


def derive_draft(tol: float = 1.0e-4, max_iter: int = 8) -> float:
    """Newton (constant-waterplane) sink so one hull displaces ``M_PER_BUOY``.

    Identical 12 buoys at one draft, so the per-buoy balance sets the draft.
    Returns the additional sink; equals ``PLATFORM_DZ`` to the cached tol.
    """
    dz = cc.DZ2
    for _ in range(max_iter):
        err = _single_displaced_mass(dz) - M_PER_BUOY
        if abs(err / M_PER_BUOY) < tol:
            break
        dz -= err / (cc.RHO * cc.A_WP_SINGLE)
    return dz


# --- 17-body deck (12 buoys + 4 hubs + 1 platform, 16 yaw_locked joints) ------
# Body-frame reference z's. The buoy/hub values match the cluster convention the
# coupled hydro is referenced to (M10 PR1 / PR2); the platform sits above the
# hubs (all dry, S8). phi(rest)=0 depends only on the joint attach geometry, so
# these z's are a modelling choice, not a physical constraint.
Z_BUOY_REF = -1.1956674320202696
Z_HUB_REF = 0.4933695679797303
Z_PLATFORM_REF = 0.7


def build_platform_deck(shared_hydro_ref, hydrostatic_ref):  # type: ignore[no-untyped-def]
    """The 17-body / 16-joint platform Deck (M11b PR6).

    ``shared_hydro_ref`` / ``hydrostatic_ref`` are :class:`HydroDatabaseRef`
    placeholders on the deck (``build_system`` is called with the actual
    database objects). Bodies: 12 buoys (``hydro_body_label`` buoy1..12), 4
    cluster hubs + 1 platform cross (``structural``). Joints: 12 buoy->hub + 4
    hub->platform, all ``yaw_locked`` (S7) -> m = 64.
    """
    from floatsim.io.deck import (
        Body,
        Deck,
        Environment,
        Inertia,
        InitialConditions,
        Output,
        Simulation,
        YawLockedJoint,
    )
    from floatsim.io.deck import RegularWave as DeckWave

    bodies: list = []
    joints: list = []
    for c, pc in enumerate(np.deg2rad(CLUSTER_ANGLES_DEG)):
        cx, cy = CLUSTER_ARM_RADIUS * np.cos(pc), CLUSTER_ARM_RADIUS * np.sin(pc)
        for b, tb in enumerate(np.deg2rad(BUOY_ANGLES_DEG)):
            k = 3 * c + b
            bx, by = cx + BUOY_RADIUS * np.cos(tb), cy + BUOY_RADIUS * np.sin(tb)
            bodies.append(
                Body(
                    name=f"buoy{k + 1}",
                    reference_point=[bx, by, Z_BUOY_REF],
                    mass=cc.M_BUOY,
                    inertia=Inertia(Ixx=cc.I_XX_BUOY, Iyy=cc.I_YY_BUOY, Izz=cc.I_ZZ_BUOY),
                    hydro_body_label=f"buoy{k + 1}",
                    initial_conditions=InitialConditions(),
                )
            )
            joints.append(
                YawLockedJoint(
                    type="yaw_locked",
                    body_a=f"buoy{k + 1}",
                    body_b=f"hub{c + 1}",
                    attach_a_body=[0.0, 0.0, Z_HUB_REF - Z_BUOY_REF],
                    attach_b_body=[BUOY_RADIUS * np.cos(tb), BUOY_RADIUS * np.sin(tb), 0.0],
                    axis=[0.0, 0.0, 1.0],
                )
            )
        bodies.append(
            Body(
                name=f"hub{c + 1}",
                reference_point=[cx, cy, Z_HUB_REF],
                mass=cc.ARM_MASS_TOTAL,
                inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0),
                structural=True,
            )
        )
        joints.append(
            YawLockedJoint(
                type="yaw_locked",
                body_a=f"hub{c + 1}",
                body_b="platform",
                attach_a_body=[0.0, 0.0, 0.0],
                attach_b_body=[cx, cy, Z_HUB_REF - Z_PLATFORM_REF],
                axis=[0.0, 0.0, 1.0],
            )
        )
    bodies.append(
        Body(
            name="platform",
            reference_point=[0.0, 0.0, Z_PLATFORM_REF],
            mass=PLATFORM_MASS,
            inertia=Inertia(Ixx=10.0, Iyy=10.0, Izz=20.0),
            structural=True,
        )
    )
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=cc.RHO, gravity=cc.G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=bodies,
        shared_hydro_database=shared_hydro_ref,
        hydrostatic_database=hydrostatic_ref,
        joints=joints,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )
