"""Step 1: build the composite 3-buoy cluster mesh + mesh_hygiene check.

Places three copies of the spar-fin fullfix eqdraft hull on a 0.5 m
circle (one on +x, 120 deg apart), shifted DOWN by dz2 to the cluster
draft. Writes mesh/cluster3_fullfix.gdf (4464 panels) and validates:
0 inward panels expected (copies of a validated hull; translation and
duplication cannot flip winding) with a 288-edge open-boundary warning
(3 x 96).
"""

from __future__ import annotations

import warnings

import numpy as np

from floatsim.hydro.mesh_hygiene import (
    GdfMesh,
    load_gdf_panels,
    validate_panel_normals,
    write_gdf_panels,
)

import cluster_common as cc


def build(dz2: float = cc.DZ2) -> GdfMesh:
    """Compose the cluster mesh at additional sink dz2."""
    single = load_gdf_panels(cc.SINGLE_EQDRAFT_MESH)
    base = single.panels.copy()
    base[..., 2] -= dz2  # sink to cluster draft
    offsets = cc.buoy_offsets()
    parts = []
    for (dx, dy) in offsets:
        p = base.copy()
        p[..., 0] += dx
        p[..., 1] += dy
        parts.append(p)
    panels = np.concatenate(parts, axis=0)
    n = panels.shape[0]
    header = (
        "3-buoy rigid cluster (spar-fin fullfix x3, R=0.5 m)",
        single.header_lines[1],
        single.header_lines[2],
        f"  {n}",
    )
    return GdfMesh(header_lines=header, panels=panels)


def main() -> None:
    print("=" * 70)
    print("Step 1: composite 3-buoy cluster mesh")
    print("=" * 70)
    offsets = cc.buoy_offsets()
    print(f"Cluster radius: {cc.CLUSTER_RADIUS} m; buoy angles: "
          f"{cc.BUOY_ANGLES_DEG.tolist()} deg (one on +x, CCW)")
    for i, (dx, dy) in enumerate(offsets):
        print(f"  buoy {i}: centre = ({dx:+.4f}, {dy:+.4f}, 0)")
    print(f"Additional sink dz2 = {cc.DZ2} m (first pass; refined in Step 2)")

    mesh = build(cc.DZ2)
    print(f"\nComposite panels: {mesh.n_panels}  (expect 4464 = 3 x 1488)")
    if mesh.n_panels != 4464:
        raise SystemExit(f"STOP: expected 4464 panels; got {mesh.n_panels}")
    write_gdf_panels(mesh, cc.CLUSTER_MESH)
    print(f"Wrote {cc.CLUSTER_MESH}")

    print("\nmesh_hygiene on the composite ...")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        report = validate_panel_normals(mesh, return_report=True)
    ob = [str(w.message) for w in rec if "open boundary" in str(w.message)]
    print(f"  inward panels:        {report.inward_indices.size}  (expect 0)")
    print(f"  indeterminate panels: {report.indeterminate_indices.size}")
    print(f"  n_open_edges:         {report.n_open_edges}  (expect 288 = 3 x 96)")
    print(f"  open-boundary warning fired: {bool(ob)}")

    if report.inward_indices.size != 0:
        raise SystemExit(
            f"STOP: {report.inward_indices.size} inward panels in the "
            f"composite. Copies of a validated hull cannot flip winding; "
            f"investigate the build."
        )
    if report.n_open_edges != 288:
        print(f"  NOTE: n_open_edges {report.n_open_edges} != 288 "
              f"(hulls may share coincident edges at this spacing?).")
    print("\n  OK: 0 inward panels.")


if __name__ == "__main__":
    main()
