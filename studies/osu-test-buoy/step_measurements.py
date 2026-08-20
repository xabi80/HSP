"""Per-part geometry of the OSU Test Buoy STEP file, via gmsh's OpenCASCADE kernel.

Prints, per solid: name, bounding-box extents, volume, centroid — plus the total
envelope and solid volume. Used to measure the physical test buoy for the FloatSim
adaptation (see OSU-TEST-BUOY-GEOMETRY.md).

Requires: ``pip install gmsh`` (contributor tool; NOT a FloatSim runtime dependency).
Usage: ``python step_measurements.py ["path/to/OSU Test Buoy.stp"]``
"""
from __future__ import annotations

import sys

import gmsh

_DEFAULT = r"C:/Users/xlama/OneDrive/Documents/buoy/OSU Test Buoy.stp"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(step)
    gmsh.model.occ.synchronize()

    vols = gmsh.model.getEntities(3)
    bb = gmsh.model.getBoundingBox(-1, -1)  # mm
    print(f"{len(vols)} solids | envelope (mm): "
          f"{bb[3] - bb[0]:.0f} x {bb[4] - bb[1]:.0f} x {bb[5] - bb[2]:.0f} "
          f"(z {bb[2]:.0f}..{bb[5]:.0f})\n")
    print(f"{'tag':>4} {'name':30} {'vol(L)':>8} {'z_cog':>7} {'dz':>6} {'dx':>5} {'dy':>5}")
    rows = []
    for dim, tag in vols:
        b = gmsh.model.occ.getBoundingBox(dim, tag)
        vol = gmsh.model.occ.getMass(dim, tag)              # mm^3
        c = gmsh.model.occ.getCenterOfMass(dim, tag)
        rows.append((tag, gmsh.model.getEntityName(dim, tag), b, vol, c))
    for tag, name, b, vol, c in sorted(rows, key=lambda r: r[4][2]):
        print(f"{tag:>4} {name.split('/')[-1][:30]:30} {vol / 1e6:8.3f} {c[2]:7.0f} "
              f"{b[5] - b[2]:6.0f} {b[3] - b[0]:5.0f} {b[4] - b[1]:5.0f}")
    print(f"\nTotal solid (structural) volume: {sum(r[3] for r in rows) / 1e6:.3f} L")
    gmsh.finalize()


if __name__ == "__main__":
    main()
