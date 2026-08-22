"""Render the Capytaine BEM mesh of the OSU Test Buoy.

Shows the exact geometry bem_database.py solves on: the full modelled body (6" spar
cylinder + a PLACEHOLDER solid equal-area disc for the perforated heave plate) and the
immersed (wetted) part that the boundary-element solver actually runs on (z < 0).

Writes OSU_buoy_mesh_capytaine.png next to this script. Requires capytaine.
See OSU-TEST-BUOY-GEOMETRY.md.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

warnings.simplefilter("ignore")
import capytaine as cpt  # noqa: E402

cpt.set_logging("ERROR")

HERE = Path(__file__).resolve().parent
RHO, G = 998.0, 9.806
M, ZG = 21.52, -0.907
R, Z_BOT, Z_TOP = 0.07965, -0.967, 0.717          # spar (6" pipe), immersed to WL at z=0
A_PLATE = float(np.sqrt(0.328 * 0.198 / np.pi))    # equal-area disc radius (placeholder)
Z_PLATE = -1.383
CY, PL, WV, INK = "#0c8b96", "#9aa7b0", "#2f74c0", "#20303a"


def _spar():
    return cpt.mesh_vertical_cylinder(length=Z_TOP - Z_BOT, radius=R,
                                      center=(0, 0, (Z_BOT + Z_TOP) / 2), resolution=(4, 48, 70))


def _plate():
    return cpt.mesh_vertical_cylinder(length=0.02, radius=A_PLATE, center=(0, 0, Z_PLATE),
                                      resolution=(12, 48, 2))


def _immersed(mesh):
    b = cpt.FloatingBody(mesh=mesh, mass=M, center_of_mass=(0, 0, ZG))
    b.rotation_center = np.array([0.0, 0.0, ZG])
    return b.immersed_part().mesh


def _add(ax, mesh, color, alpha=1.0):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    pc = Poly3DCollection([v[face] for face in f], facecolor=color,
                          edgecolor=INK, linewidths=0.12, alpha=alpha)
    ax.add_collection3d(pc)
    return v


def _frame(ax, zmin, zmax, title):
    rr = A_PLATE * 1.35
    ax.set_xlim(-rr, rr); ax.set_ylim(-rr, rr); ax.set_zlim(zmin, zmax)
    ax.set_box_aspect((2 * rr, 2 * rr, (zmax - zmin)))
    ax.view_init(elev=9, azim=-58)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_zlabel("z (m)", fontsize=9)
    ax.tick_params(axis="z", labelsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", color=CY, pad=2)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    spar, plate = _spar(), _plate()
    full = spar.join_meshes(plate)
    wetted = _immersed(full)
    n_full, n_wet = full.nb_faces, wetted.nb_faces
    print(f"full mesh: {n_full} panels   immersed (wetted): {n_wet} panels")

    fig = plt.figure(figsize=(11, 8.2))

    # (a) full modelled body + waterline
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    _add(ax, spar, CY)
    _add(ax, plate, PL)
    # waterline plane z=0
    g = A_PLATE * 1.3
    xx, yy = np.meshgrid([-g, g], [-g, g])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color=WV, alpha=0.18, zorder=0)
    ax.text(g * 0.2, g, 0.03, "still WL  z = 0", color=WV, fontsize=8)
    _frame(ax, Z_PLATE - 0.10, Z_TOP + 0.03, f"Full modelled body  ({n_full} panels)")
    ax.text(g * 1.05, 0, Z_TOP - 0.10, "freeboard 0.72 m", color=INK, fontsize=8)
    ax.text(0, 0, Z_PLATE - 0.13, "placeholder disc\n(heave plate)", color=PL, fontsize=8, ha="center")

    # (b) wetted mesh only (what the BEM solves)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    _add(ax, wetted, CY)
    _frame(ax, Z_PLATE - 0.06, 0.05, f"Wetted mesh solved by Capytaine  ({n_wet} panels)")
    ax.text(0, 0, 0.02, "draft 1.42 m ↓", color=INK, fontsize=8, ha="center")

    fig.suptitle("OSU Test Buoy — Capytaine BEM mesh  (6\" spar + placeholder equal-area disc)\n"
                 "the perforated/webbed heave plate is a solid stand-in; its real hydro comes from the tank",
                 fontsize=11.5, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / "OSU_buoy_mesh_capytaine.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
