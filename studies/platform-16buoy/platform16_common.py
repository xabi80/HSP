"""Shared constants + helpers for the 16-buoy platform (4 clusters x 4 buoys).

Variant of the M11b 12-buoy platform (``studies/platform-12buoy/platform_common``)
with **4 buoys per cluster** instead of 3:

  * 4 clusters (C4-a: 0/90/180/270 deg, 1.0 m arm radius) -- UNCHANGED from 12-buoy;
  * 4 buoys per cluster at **0/90/180/270 deg** on the same 0.5 m intra-cluster
    circle (a square, vs the 12-buoy's 120-deg triangle) -> 16 spar-fin hulls;
  * articulated ``yaw_locked`` buoy->hub and hub->platform, as the 12-buoy.

Spacing (checked): the min buoy-centre gap is 0.707 m (intra AND cross-cluster),
LARGER than the 12-buoy's 0.620 m -- fin-edge clearance +0.277 m at r_fin=0.215,
so no mesh overlap. Footprint radius 1.5 m (same as 12-buoy).

Mass (same rules as 12-buoy geometry §3.3, C4-c): every buoy carries the whole
floating mass (arms + platform dry). Arms scale per-arm (``cc.ARM_MASS_EACH`` =
4 kg): 4 arms/cluster x 4 clusters = 64 kg (vs 48 kg for the 12-buoy). Platform
cross kept at 10 kg (C4-c assumption; the 1.5 m footprint is unchanged). Per-buoy
support M_TOTAL/16 = 33.30 kg (12-buoy: 33.50) -> draft ~mm shallower, re-derived
on the mesh by :func:`derive_draft` (NOT copied).

Frame z's (``Z_*_REF``) are carried verbatim from the 12-buoy: they are a joint
modelling choice (``phi(rest)=0`` depends only on the attach geometry
Z_HUB-Z_BUOY, identical here), and the buoy hull is the same spar-fin, so the
per-buoy drag geometry is unchanged. Only ``PLATFORM_DZ`` (the BEM submergence)
is re-derived.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "cluster-3buoy-rigid"))

import cluster_common as cc  # noqa: E402

# --- Topology -----------------------------------------------------------------
N_CLUSTER = 4
N_PER_CLUSTER = 4
N_BUOY = N_CLUSTER * N_PER_CLUSTER  # 16
N_BODY = N_BUOY + N_CLUSTER + 1  # 16 buoys + 4 hubs + 1 platform = 21
N_DOF = 6 * N_BODY  # 126

# --- Geometry (radii UNCHANGED from 12-buoy; 4 buoys at 90 deg) ---------------
CLUSTER_ARM_RADIUS = 1.0  # m
CLUSTER_ANGLES_DEG = np.array([0.0, 90.0, 180.0, 270.0])
BUOY_ANGLES_DEG = np.array([0.0, 90.0, 180.0, 270.0])  # square (12-buoy was 0/120/240)
BUOY_RADIUS = cc.CLUSTER_RADIUS  # 0.5 m intra-cluster

# --- Mass balance (arms scale per-arm; platform cross fixed) ------------------
ARM_MASS_PER_CLUSTER = N_PER_CLUSTER * cc.ARM_MASS_EACH  # 4 x 4 = 16 kg
ARMS_TOTAL = N_CLUSTER * ARM_MASS_PER_CLUSTER  # 64 kg
PLATFORM_MASS = 10.0  # C4-c assumption (dry structural cross), same as 12-buoy
M_TOTAL = N_BUOY * cc.M_BUOY + ARMS_TOTAL + PLATFORM_MASS
M_PER_BUOY = M_TOTAL / N_BUOY

# Additional sink of every buoy below the isolated single-buoy waterline, to
# displace M_PER_BUOY at z=0. Re-derived on the mesh (derive_draft); cached.
PLATFORM_DZ = 0.20714  # derived on the mesh (derive_draft); 12-buoy was 0.21638 (deeper)

# --- Frame reference z's (carried from 12-buoy; joint modelling choice) -------
Z_BUOY_REF = -1.1956674320202696
Z_HUB_REF = 0.4933695679797303
Z_PLATFORM_REF = 0.7


def buoy_centers() -> NDArray[np.float64]:
    """(16, 2) horizontal (x, y) centres: 4 clusters x 4 buoys, buoy k = 4c+b."""
    out = []
    for pc in np.deg2rad(CLUSTER_ANGLES_DEG):
        cx, cy = CLUSTER_ARM_RADIUS * np.cos(pc), CLUSTER_ARM_RADIUS * np.sin(pc)
        for tb in np.deg2rad(BUOY_ANGLES_DEG):
            out.append([cx + BUOY_RADIUS * np.cos(tb), cy + BUOY_RADIUS * np.sin(tb)])
    return np.asarray(out, dtype=np.float64)


def buoy_body_index(buoy_k0: int) -> int:
    """Global deck-body index of buoy k (0-based). Deck order per cluster is
    [N_PER_CLUSTER buoys, 1 hub], so cluster c occupies N_PER_CLUSTER+1 slots."""
    c, b = divmod(buoy_k0, N_PER_CLUSTER)
    return (N_PER_CLUSTER + 1) * c + b


def platform_body_index() -> int:
    """Platform is the last deck body: N_CLUSTER*(N_PER_CLUSTER+1) = 20 -> index 20."""
    return N_CLUSTER * (N_PER_CLUSTER + 1)


def min_center_gap() -> float:
    """Minimum buoy-centre distance (any pair)."""
    c = buoy_centers()
    return min(
        float(np.hypot(*(c[i] - c[j]))) for i in range(N_BUOY) for j in range(i + 1, N_BUOY)
    )


def _single_displaced_mass(dz: float) -> float:
    """kg displaced (z<0) by ONE isolated hull sunk an extra ``dz``."""
    from floatsim.hydro.mesh_hygiene import load_gdf_panels

    p = load_gdf_panels(cc.SINGLE_EQDRAFT_MESH).panels.copy()
    p[..., 2] -= dz
    return cc.RHO * cc.displaced_volume_below_waterline(p)


def derive_draft(tol: float = 1.0e-4, max_iter: int = 8) -> float:
    """Newton (constant-waterplane) sink so one hull displaces ``M_PER_BUOY``.
    16 identical buoys at one draft, so the per-buoy balance sets the draft."""
    dz = cc.DZ2
    for _ in range(max_iter):
        err = _single_displaced_mass(dz) - M_PER_BUOY
        if abs(err / M_PER_BUOY) < tol:
            break
        dz -= err / (cc.RHO * cc.A_WP_SINGLE)
    return dz


def build_platform16_deck(shared_hydro_ref, hydrostatic_ref):  # type: ignore[no-untyped-def]
    """The 21-body / 20-joint 16-buoy platform Deck: 16 buoys
    (``hydro_body_label`` buoy1..16), 4 cluster hubs + 1 platform (``structural``);
    16 buoy->hub + 4 hub->platform ``yaw_locked`` joints."""
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
            k = N_PER_CLUSTER * c + b
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
                mass=ARM_MASS_PER_CLUSTER,
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


def main() -> None:
    """Verify: draft derivation, mass balance, spacing, deck shape."""
    print(f"16-buoy platform: {N_BUOY} buoys, {N_BODY} bodies, {N_DOF} DOF")
    print(f"  masses: {N_BUOY} x {cc.M_BUOY} buoy + {ARMS_TOTAL} arms + {PLATFORM_MASS} platform "
          f"= {M_TOTAL:.2f} kg;  per-buoy support {M_PER_BUOY:.4f} kg (12-buoy: 33.5033)")
    dz = derive_draft()
    print(f"  derived PLATFORM_DZ = {dz:.5f} m  (cached {PLATFORM_DZ}); "
          f"displaced/buoy = {_single_displaced_mass(dz):.4f} kg (target {M_PER_BUOY:.4f})")
    assert abs(dz - PLATFORM_DZ) < 1e-3, f"update PLATFORM_DZ to {dz:.5f}"
    print(f"  min buoy-centre gap = {min_center_gap():.4f} m (12-buoy: 0.620)")
    # placeholder refs so build succeeds without real DBs
    from floatsim.io.deck import HydroDatabaseRef

    ref = HydroDatabaseRef(format="capytaine", path="placeholder.nc")
    deck = build_platform16_deck(ref, ref)
    nb = sum(1 for b in deck.bodies if b.hydro_body_label is not None)
    print(f"  deck: {len(deck.bodies)} bodies ({nb} hydro buoys), {len(deck.joints)} joints; "
          f"platform index {platform_body_index()}")
    print(f"  buoy_body_index(0..3) = {[buoy_body_index(k) for k in range(4)]} "
          f"(cluster 0: buoys 0-3 -> slots 0-3, hub at 4)")


if __name__ == "__main__":
    main()
