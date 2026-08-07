"""Render the coupled 12-buoy platform BEM mesh (the panelisation actually fed to
Capytaine) as a static figure. Rebuilds the parametric spar+fin mesh via
``platform_fin_bem.make_combined`` -- no BEM solve, just the geometry -- and draws
three views:

  (a) full platform, isometric -- all 12 spar+fin buoys at their cluster positions;
  (b) plan (top-down) -- the 4-cluster / 12-buoy layout footprint;
  (c) a single spar+fin buoy, zoomed -- the annular heave-plate fin panelisation.

The mean waterline (z=0) is drawn as a translucent plane so the draft reads. Fin
radius is the 0.215 m baseline. Writes ``platform_bem_mesh.png``.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # platform-12buoy
sys.path.insert(0, str(_HERE.parent.parent / "cluster-3buoy-rigid"))

import platform_fin_bem as pfb  # noqa: E402  (adds its own sibling paths on import)
import cluster_fin_bem as cfb  # noqa: E402

_R_FIN = 0.215
_FACE = "#7ba7c9"
_EDGE = "#1f3a52"
_WATER = "#2f6f9f"


def _polys(mesh):  # type: ignore[no-untyped-def]
    """(n_faces, 4, 3) vertex coordinates for a Capytaine mesh or collection."""
    m = mesh.merged() if hasattr(mesh, "merged") else mesh
    return np.asarray(m.vertices)[np.asarray(m.faces)], m.nb_faces


def _equal_aspect(ax, pts):  # type: ignore[no-untyped-def]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ctr, span = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(ctr[0] - span, ctr[0] + span)
    ax.set_ylim(ctr[1] - span, ctr[1] + span)
    ax.set_zlim(ctr[2] - span, ctr[2] + span)
    ax.set_box_aspect((1, 1, 1))


def _waterplane(ax, pts, pad=0.15):  # type: ignore[no-untyped-def]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    xx, yy = np.meshgrid(
        np.linspace(lo[0] - pad, hi[0] + pad, 2), np.linspace(lo[1] - pad, hi[1] + pad, 2)
    )
    ax.plot_surface(xx, yy, np.zeros_like(xx), color=_WATER, alpha=0.12, zorder=0,
                    linewidth=0, shade=False)


def _draw(ax, polys, lw, alpha=0.9):  # type: ignore[no-untyped-def]
    pc = Poly3DCollection(polys, facecolor=_FACE, edgecolor=_EDGE, linewidths=lw, alpha=alpha)
    ax.add_collection3d(pc)


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        allb = pfb.make_combined(_R_FIN)
        cfb._NTHETA = pfb._NTHETA
        one = cfb.build_full_mesh(_R_FIN)

    full_polys, full_n = _polys(allb.mesh)
    one_polys, one_n = _polys(one)
    full_pts = full_polys.reshape(-1, 3)
    one_pts = one_polys.reshape(-1, 3)
    print(f"platform mesh: {full_n} panels ({allb.mesh.nb_faces}); single buoy: {one_n}", flush=True)

    fig = plt.figure(figsize=(17, 7.2))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    _draw(ax, full_polys, 0.12, alpha=0.82)
    _waterplane(ax, full_pts)
    _equal_aspect(ax, full_pts)
    ax.view_init(elev=22, azim=-52)
    ax.set_title(f"(a) 12-buoy platform BEM mesh\n{full_n} wetted-hull panels, "
                 f"fin R = {_R_FIN} m", fontsize=10, pad=0)
    for a in ("x", "y", "z"):
        getattr(ax, f"set_{a}label")(f"{a} (m)", fontsize=8, labelpad=1)
    ax.tick_params(labelsize=7)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    _draw(ax2, full_polys, 0.1, alpha=0.85)
    _equal_aspect(ax2, full_pts)
    ax2.view_init(elev=89, azim=-90)  # straight top-down
    ax2.set_title("(b) plan view -- 4 clusters x 3 buoys\n(wave heading 0 deg -> +x, right)",
                  fontsize=10, pad=0)
    ax2.set_xlabel("x (m)", fontsize=8, labelpad=1)
    ax2.set_ylabel("y (m)", fontsize=8, labelpad=1)
    ax2.set_zticks([])
    ax2.tick_params(labelsize=7)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    _draw(ax3, one_polys, 0.25, alpha=0.9)
    _waterplane(ax3, one_pts, pad=0.05)
    _equal_aspect(ax3, one_pts)
    ax3.view_init(elev=14, azim=-58)
    ax3.set_title(f"(c) one spar + annular fin\n{one_n} panels -- the R = {_R_FIN} m heave plate",
                  fontsize=10, pad=0)
    for a in ("x", "y", "z"):
        getattr(ax3, f"set_{a}label")(f"{a} (m)", fontsize=8, labelpad=1)
    ax3.tick_params(labelsize=7)

    fig.suptitle(
        "12-buoy articulated platform -- Capytaine coupled BEM mesh (parametric spar + heave-plate "
        f"fin, n_theta = {pfb._NTHETA})\ntranslucent plane = mean waterline (z = 0); "
        "the platform solves as a 72-DOF coupled radiation + diffraction problem",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = _HERE / "platform_bem_mesh.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}", flush=True)


if __name__ == "__main__":
    main()
