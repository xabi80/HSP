"""Surface mesh of the OSU Test Buoy STEP via gmsh (OpenCASCADE), and a 3-view render.

Writes ``OSU_Test_Buoy.stl`` (regenerable, ~30 MB — not committed) and a 3-panel PNG.
The full-assembly mesh is for visualisation / a sanity check of the geometry; for a
Capytaine BEM, extract only the wetted hull (spar outer surface + heave-plate frame) and
clip at the waterline once the draft is known (see OSU-TEST-BUOY-GEOMETRY.md).

Requires: ``pip install gmsh``.
Usage: ``python mesh_from_step.py ["path/to/OSU Test Buoy.stp"] [out_dir]``
"""
from __future__ import annotations

import sys
from pathlib import Path

import gmsh
import numpy as np

_DEFAULT = r"C:/Users/xlama/OneDrive/Documents/buoy/OSU Test Buoy.stp"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(step).parent

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.occ.importShapes(step)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 22)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 4)
    gmsh.model.mesh.generate(2)
    gmsh.write(str(out / "OSU_Test_Buoy.stl"))

    nt, nc, _ = gmsh.model.mesh.getNodes()
    V = nc.reshape(-1, 3)
    lut = np.zeros(int(nt.max()) + 1, int)
    lut[nt.astype(int)] = np.arange(len(nt))
    et, _, en = gmsh.model.mesh.getElements(2)
    T = lut[np.vstack([e.reshape(-1, 3) for tp, e in zip(et, en) if tp == 2]).astype(int)]
    print(f"mesh: {len(V)} nodes, {len(T)} triangles")
    gmsh.finalize()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri = V[T]
    fig = plt.figure(figsize=(13, 8))
    for i, (az, el, ttl) in enumerate([(-60, 12, "3/4 view"), (0, 0, "front (x-z)"),
                                       (90, 0, "side (y-z)")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.add_collection3d(Poly3DCollection(tri, facecolor="#7fb4bb", edgecolor="#33525a",
                                             linewidths=0.05))
        ax.set_xlim(-170, 170); ax.set_ylim(-170, 170); ax.set_zlim(-450, 1850)
        ax.set_box_aspect((340, 340, 2300)); ax.view_init(elev=el, azim=az)
        ax.set_title(ttl, fontsize=10); ax.tick_params(labelsize=6)
    fig.suptitle("OSU Test Buoy — CAD surface mesh (gmsh from STEP)\n"
                 "rectangular webbed heave-plate/ballast at the base · 6\" pipe spar", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "OSU_Test_Buoy_mesh.png", dpi=140)
    print("wrote OSU_Test_Buoy.stl + OSU_Test_Buoy_mesh.png in", out)


if __name__ == "__main__":
    main()
