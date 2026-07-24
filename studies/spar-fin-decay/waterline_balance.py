"""Waterline-balance diagnostic (M7.5 resumption, buoyancy interpretation).

The tier-2 check_hydrostatic_volume integrates the FULL closed mesh
(total enclosed volume = displacement-if-fully-submerged = genuine
reserve buoyancy ~40.9 kg). It does NOT give the displaced volume at
the design waterline. This script computes the actual displaced volume
below z=0 (the meshed waterline) and the equilibrium sink dz needed to
balance the 28.67 kg buoy.

(a) Manual clip of fullfix mesh at z=0: displaced volume of the part
    below the free surface, via the divergence theorem on the wetted
    surface (the z=0 waterplane lid contributes exactly zero to
    (1/6) sum v0.(v1 x v2) because r.n = 0 there).
(b) Capytaine immersed_part() displacement (independent check).
(c) Equilibrium sink dz = (M - m_disp) / (rho * A_wp).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from floatsim.hydro.mesh_hygiene import load_gdf_panels

_HERE = Path(__file__).resolve().parent
_FULLFIX = _HERE / "mesh" / "test2_spar_fin_fullfix.gdf"

_RHO = 1025.0
_G = 9.81
_M_BODY = 28.67
_R_SPAR = 0.0841
_A_WP = np.pi * _R_SPAR**2  # 0.022220 m^2


def _clip_triangle_below(tri: np.ndarray) -> list[np.ndarray]:
    """Clip a triangle against z=0, keeping the z<=0 part.

    Sutherland-Hodgman single-plane clip: walk the polygon edges in
    original winding order, emitting below-water vertices and edge
    crossings; then fan-triangulate the resulting polygon. Correct
    for all below-vertex configurations (winding preserved), so the
    signed-volume sum is exact for the wetted surface.
    """
    z = tri[:, 2]
    if np.all(z < 0.0):
        return [tri]
    if np.all(z >= 0.0):
        return []

    def intersect(p_in: np.ndarray, p_out: np.ndarray) -> np.ndarray:
        t = p_in[2] / (p_in[2] - p_out[2])  # linear interp to z=0
        return p_in + t * (p_out - p_in)

    poly: list[np.ndarray] = []
    n = 3
    for i in range(n):
        curr = tri[i]
        nxt = tri[(i + 1) % n]
        curr_in = curr[2] < 0.0
        nxt_in = nxt[2] < 0.0
        if curr_in:
            poly.append(curr)
            if not nxt_in:
                poly.append(intersect(curr, nxt))
        else:
            if nxt_in:
                poly.append(intersect(curr, nxt))
    if len(poly) < 3:
        return []
    # Fan-triangulate the below-water polygon (preserves winding).
    return [
        np.stack([poly[0], poly[k], poly[k + 1]])
        for k in range(1, len(poly) - 1)
    ]


def displaced_volume_below_waterline(panels: np.ndarray) -> float:
    """Displaced volume (z<0) of an outward-oriented quad mesh.

    V = (1/6) sum over clipped-below triangles of v0 . (v1 x v2).
    The z=0 lid contributes 0, so summing only the wetted surface
    yields the enclosed below-water volume.
    """
    total = 0.0
    for p in panels:
        for tri in ((p[0], p[1], p[2]), (p[0], p[2], p[3])):
            a, b, c = tri
            e1 = np.asarray(b) - np.asarray(a)
            e2 = np.asarray(c) - np.asarray(a)
            if np.linalg.norm(np.cross(e1, e2)) < 1.0e-15:
                continue
            for t in _clip_triangle_below(np.stack([a, b, c])):
                total += float(np.dot(t[0], np.cross(t[1], t[2])))
    return total / 6.0


def main() -> None:
    print("=" * 70)
    print("Waterline-balance diagnostic (fullfix mesh)")
    print("=" * 70)
    panels = load_gdf_panels(_FULLFIX).panels
    zmin = float(panels[..., 2].min())
    zmax = float(panels[..., 2].max())
    print(f"Mesh z-extent: [{zmin:.4f}, {zmax:.4f}] m  (waterline at z=0)")
    print(f"A_wp = pi*r_spar^2 = pi*{_R_SPAR}^2 = {_A_WP:.6f} m^2")
    print()

    # (a) manual clip
    v_disp = displaced_volume_below_waterline(panels)
    m_disp = _RHO * v_disp
    print("(a) MANUAL clip at z=0 (divergence theorem, wetted surface):")
    print(f"    displaced volume V_disp = {v_disp:.6e} m^3")
    print(f"    displaced mass  m_disp  = {m_disp:.4f} kg")
    print("    PREDICTION band: 2.35e-2..2.42e-2 m^3 (24.1..24.8 kg)")
    print()

    # (b) Capytaine immersed_part
    print("(b) Capytaine immersed_part() displacement (independent):")
    try:
        import capytaine as cpt

        mesh = cpt.load_mesh(str(_FULLFIX), file_format="gdf")
        body = cpt.FloatingBody(mesh=mesh, name="spar_fin_balance")
        immersed = body.immersed_part()
        # Capytaine mesh volume (immersed part is closed at the waterline).
        try:
            v_cpt = float(immersed.mesh.volume)
        except AttributeError:
            v_cpt = float(immersed.mesh.volumes.sum()) if hasattr(
                immersed.mesh, "volumes"
            ) else float("nan")
        m_cpt = _RHO * abs(v_cpt)
        print(f"    immersed volume = {abs(v_cpt):.6e} m^3")
        print(f"    displaced mass  = {m_cpt:.4f} kg")
    except Exception as exc:
        print(f"    Capytaine path failed ({type(exc).__name__}: {exc});")
        print("    relying on manual clip (a).")
        m_cpt = m_disp

    print()
    # (c) equilibrium sink
    dz = (_M_BODY - m_disp) / (_RHO * _A_WP)
    print("(c) Equilibrium sink dz = (M - m_disp) / (rho * A_wp):")
    print(f"    dz = ({_M_BODY} - {m_disp:.4f}) / ({_RHO} * {_A_WP:.6f})")
    print(f"    dz = {dz:.4f} m   (buoy floats this much DEEPER)")
    print("    PREDICTION band: 0.15..0.22 m")
    print()

    # Verdict on bands
    print("=" * 70)
    ok_a = 2.35e-2 <= v_disp <= 2.42e-2
    ok_c = 0.15 <= dz <= 0.22
    print(f"  (a) V_disp in [2.35e-2, 2.42e-2]:  {'PASS' if ok_a else 'OUT OF BAND'}")
    print(f"  (c) dz    in [0.15, 0.22]:         {'PASS' if ok_c else 'OUT OF BAND'}")
    print(f"  True unmoored draft = 1.09 + dz ~ {1.09 + dz:.4f} m "
          f"(mesh bottom at z={zmin:.3f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
