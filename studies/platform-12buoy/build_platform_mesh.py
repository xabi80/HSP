"""M11b PR6: build the 12-buoy platform mesh + mesh_hygiene screen.

12 copies of the spar-fin fullfix eqdraft hull, sunk to the platform draft
(``platform_common.derive_draft``), placed at the 12 buoy centres (4 clusters
x 3 buoys). Writes ``mesh/platform12_fullfix.gdf`` (17,856 = 12 x 1488 panels)
and validates 0 inward panels (copies of a validated hull cannot flip winding)
with a 1152-edge (12 x 96) open-boundary warning.

M11b Phase-1 STEP 2 established the numbers this script reproduces: draft
dz = 0.21638 m, closest cross-cluster pair 0.6197 m, 0 inward panels.
"""

from __future__ import annotations

import warnings

import numpy as np
import platform_common as pc

from floatsim.hydro.mesh_hygiene import (
    GdfMesh,
    load_gdf_panels,
    validate_panel_normals,
    write_gdf_panels,
)


def build(dz: float | None = None) -> GdfMesh:
    """Compose the 12-buoy platform mesh at additional sink ``dz`` (default
    ``PLATFORM_DZ``)."""
    if dz is None:
        dz = pc.PLATFORM_DZ
    single = load_gdf_panels(pc.cc.SINGLE_EQDRAFT_MESH)
    base = single.panels.copy()
    base[..., 2] -= dz
    parts = []
    for dx, dy in pc.buoy_centers():
        p = base.copy()
        p[..., 0] += dx
        p[..., 1] += dy
        parts.append(p)
    panels = np.concatenate(parts, axis=0)
    header = (
        "12-buoy platform (spar-fin fullfix x12, 4 clusters R=1m x 3 buoys R=0.5m)",
        single.header_lines[1],
        single.header_lines[2],
        f"  {panels.shape[0]}",
    )
    return GdfMesh(header_lines=header, panels=panels)


def main() -> None:
    print("=" * 72)
    print("M11b PR6: 12-buoy platform mesh")
    print("=" * 72)
    dz = pc.derive_draft()
    print(f"platform mass = {pc.M_TOTAL:.2f} kg, per-buoy {pc.M_PER_BUOY:.4f} kg")
    print(
        f"draft (additional sink) = {dz:.5f} m  (cached PLATFORM_DZ = {pc.PLATFORM_DZ:.5f}; "
        f"cluster DZ2 = {pc.cc.DZ2:.5f})"
    )
    print(f"closest cross-cluster pair = {pc.closest_cross_cluster_gap():.4f} m (expect ~0.620)")

    mesh = build(dz)
    print(f"\npanels = {mesh.n_panels}  (expect 17856 = 12 x 1488)")
    if mesh.n_panels != 17856:
        raise SystemExit(f"STOP: expected 17856 panels; got {mesh.n_panels}")
    z_bot = float(mesh.panels[..., 2].min())
    print(f"keel depth = {-z_bot:.4f} m below WL")

    print("\nmesh_hygiene ...")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        report = validate_panel_normals(mesh, return_report=True)
    ob = [w for w in rec if "open boundary" in str(w.message)]
    print(f"  inward panels        = {report.inward_indices.size}  (expect 0)")
    print(f"  indeterminate panels = {report.indeterminate_indices.size}")
    print(f"  n_open_edges         = {report.n_open_edges}  (expect 1152 = 12 x 96)")
    print(f"  open-boundary warning fired = {bool(ob)}")
    if report.inward_indices.size != 0:
        raise SystemExit(
            f"STOP: {report.inward_indices.size} inward panels -- copies of a "
            f"validated hull cannot flip winding; investigate the build."
        )

    pc.PLATFORM_MESH.parent.mkdir(parents=True, exist_ok=True)
    write_gdf_panels(mesh, pc.PLATFORM_MESH)
    print(f"\n  OK: 0 inward panels. Wrote {pc.PLATFORM_MESH}")


if __name__ == "__main__":
    main()
