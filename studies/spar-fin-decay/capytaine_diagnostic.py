"""Diagnostic checks 1-3 for the Capytaine A_inf(heave) shortfall.

Per Xabier's 2026-06-28 instruction (updated study plan): the mesh
DOES contain the heave plate; the question is why Capytaine isn't
producing the expected ~27 kg heave added-mass contribution from
the plate.

  Check 1: Visualize the Capytaine-loaded mesh. Save to
           results/figures/capytaine_mesh_view.png.
  Check 2: Inspect panel normals on the plate's top (z ~ -0.955)
           and bottom (z ~ -0.959) faces.
  Check 3: Run a procedurally-built spar+disk reference geometry
           through the same script and compare A_inf.

STOP after Check 3 and report; Checks 4-5 gated on Xabier's review.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np

import capytaine as cpt

_HERE = Path(__file__).resolve().parent
_MESH_PATH = _HERE / "mesh" / "test2_spar_fin.gdf"
_FIGURES = _HERE / "results" / "figures"
_FIGURES.mkdir(parents=True, exist_ok=True)

_RHO = 1025.0
_G = 9.81
_COG = np.array([0.0, 0.0, -0.8317])

# Plate geometry from vertex inspection.
_PLATE_Z_TOP = -0.955  # approximate
_PLATE_Z_BOT = -0.959
_PLATE_R_INNER = 0.0841
_PLATE_R_OUTER = 0.215


def _load_real_body():
    mesh = cpt.load_mesh(str(_MESH_PATH), file_format="gdf")
    body = cpt.FloatingBody(mesh=mesh, center_of_mass=_COG, name="spar_fin_real")
    body.rotation_center = _COG
    body.add_all_rigid_body_dofs()
    return body


def check_1_visualize(body: cpt.FloatingBody) -> None:
    """Visualize the loaded mesh; emphasize the plate region."""
    print("\n" + "=" * 70)
    print("Check 1 -- visualize Capytaine-loaded mesh")
    print("=" * 70)
    mesh = body.mesh
    print(f"  faces: {mesh.nb_faces}")
    print(f"  vertices: {mesh.nb_vertices}")

    # Face centers + vertices for plotting.
    fc = mesh.faces_centers  # (n_faces, 3)
    fv = mesh.vertices  # (n_verts, 3)

    fig = plt.figure(figsize=(12, 5))

    # Panel 1: full mesh, X-Z projection.
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(fv[:, 0], fv[:, 2], s=0.3, c="steelblue", alpha=0.4, label="vertices")
    ax1.axhline(0, color="k", lw=0.7, ls="--", label="waterline")
    ax1.axhspan(_PLATE_Z_BOT, _PLATE_Z_TOP, color="orange", alpha=0.3, label="plate band")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("z (m)")
    ax1.set_aspect("equal")
    ax1.set_title("Full mesh, X-Z projection")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: plate-region close-up, X-Y projection at z ~ plate.
    ax2 = fig.add_subplot(1, 2, 2)
    plate_mask = (fv[:, 2] >= _PLATE_Z_BOT - 0.005) & (fv[:, 2] <= _PLATE_Z_TOP + 0.005)
    ax2.scatter(
        fv[plate_mask, 0],
        fv[plate_mask, 1],
        s=2,
        c="orange",
        label=f"vertices in plate band ({int(plate_mask.sum())})",
    )
    # Spar outline
    theta = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(_PLATE_R_INNER * np.cos(theta), _PLATE_R_INNER * np.sin(theta),
             "k--", lw=0.7, label=f"r = {_PLATE_R_INNER:.3f} (spar)")
    ax2.plot(_PLATE_R_OUTER * np.cos(theta), _PLATE_R_OUTER * np.sin(theta),
             "k-", lw=0.7, label=f"r = {_PLATE_R_OUTER:.3f} (plate)")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_aspect("equal")
    ax2.set_title(f"Plate band z in [{_PLATE_Z_BOT:.3f}, {_PLATE_Z_TOP:.3f}], X-Y view")
    ax2.legend(loc="upper right", fontsize=7)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = _FIGURES / "capytaine_mesh_view.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def check_2_normals(body: cpt.FloatingBody) -> None:
    """Inspect panel normals on the heave-plate top + bottom faces."""
    print("\n" + "=" * 70)
    print("Check 2 -- panel normals on heave-plate faces")
    print("=" * 70)

    mesh = body.mesh
    fc = mesh.faces_centers       # (n_faces, 3)
    fn = mesh.faces_normals       # (n_faces, 3) unit normals
    fa = mesh.faces_areas         # (n_faces,)

    # Identify plate-band panels. A face center within the plate band
    # AND with a horizontal radius > spar radius is "on the plate."
    fr_xy = np.hypot(fc[:, 0], fc[:, 1])
    on_plate = (
        (fc[:, 2] >= _PLATE_Z_BOT - 0.005)
        & (fc[:, 2] <= _PLATE_Z_TOP + 0.005)
        & (fr_xy > _PLATE_R_INNER + 0.001)
        & (fr_xy < _PLATE_R_OUTER + 0.001)
    )
    n_plate = int(on_plate.sum())
    print(f"  Plate panels detected: {n_plate}")
    if n_plate == 0:
        print("  WARNING: no plate panels detected; check filter ranges.")
        return

    # Split into "top face" (z ~ -0.955) and "bottom face" (z ~ -0.959).
    plate_z_mid = 0.5 * (_PLATE_Z_TOP + _PLATE_Z_BOT)  # ~ -0.957
    top_mask = on_plate & (fc[:, 2] > plate_z_mid)
    bot_mask = on_plate & (fc[:, 2] <= plate_z_mid)
    n_top = int(top_mask.sum())
    n_bot = int(bot_mask.sum())
    print(f"  Plate top (z > {plate_z_mid:.4f}): {n_top} panels")
    print(f"  Plate bot (z <= {plate_z_mid:.4f}): {n_bot} panels")

    if n_top > 0:
        nz_top = fn[top_mask, 2]
        print(f"  TOP face normals nz: mean = {nz_top.mean():+.4f}, "
              f"min = {nz_top.min():+.4f}, max = {nz_top.max():+.4f}")
        n_up = int((nz_top > 0.5).sum())
        n_dn = int((nz_top < -0.5).sum())
        print(f"    pointing UP (+z): {n_up} / {n_top}")
        print(f"    pointing DOWN (-z): {n_dn} / {n_top}")
    if n_bot > 0:
        nz_bot = fn[bot_mask, 2]
        print(f"  BOT face normals nz: mean = {nz_bot.mean():+.4f}, "
              f"min = {nz_bot.min():+.4f}, max = {nz_bot.max():+.4f}")
        n_up = int((nz_bot > 0.5).sum())
        n_dn = int((nz_bot < -0.5).sum())
        print(f"    pointing UP (+z): {n_up} / {n_bot}")
        print(f"    pointing DOWN (-z): {n_dn} / {n_bot}")

    # Expected (for a watertight closed body, normals point OUTWARD from
    # the fluid-side of the wetted surface):
    #   plate TOP face (z = -0.955): outward = +z (away from water above)
    #   plate BOT face (z = -0.959): outward = -z (away from water below)
    # Mismatch -> either reversed or both same -> plate is "transparent"
    # to BEM integration.
    print()
    print("  Expected (outward-from-body convention):")
    print(f"    TOP face nz ~ +1  (outward, away from spar interior)")
    print(f"    BOT face nz ~ -1  (outward, away from spar interior)")

    # Total plate area for a sanity-check vs analytical annulus area.
    A_top = float(fa[top_mask].sum()) if n_top > 0 else 0.0
    A_bot = float(fa[bot_mask].sum()) if n_bot > 0 else 0.0
    A_analytical = np.pi * (_PLATE_R_OUTER**2 - _PLATE_R_INNER**2)
    print(f"\n  Plate area (top, mesh sum): {A_top:.4f} m^2")
    print(f"  Plate area (bot, mesh sum): {A_bot:.4f} m^2")
    print(f"  Plate area (analytical annulus): {A_analytical:.4f} m^2")


def check_3_reference_geometry() -> None:
    """Run a procedural spar + thin disk through the same Capytaine pipeline.

    If the reference geometry produces the expected A_inf ~ 27 kg from
    the disk + ~1.6 kg from the spar, the GDF mesh is the issue.
    If the reference geometry ALSO produces a too-low A_inf, our
    Capytaine setup is at fault.
    """
    print("\n" + "=" * 70)
    print("Check 3 -- reference geometry: procedural spar + disk")
    print("=" * 70)

    # Build a slender spar from z = -0.95 to z = +0.75 with r = 0.0841.
    # resolution = (n_radial_disk, n_theta, n_axial)
    spar = cpt.mesh_vertical_cylinder(
        length=1.7,
        radius=0.0841,
        center=(0.0, 0.0, -0.10),  # z midpoint -> from -0.95 to +0.75
        resolution=(4, 40, 60),
        name="ref_spar",
    )
    # Build a thin horizontal heave plate (disk) at z = -0.957, R = 0.215.
    # Use mesh_disk for a flat horizontal disk -- this is the plate.
    plate_top = cpt.mesh_disk(
        radius=0.215,
        center=(0.0, 0.0, -0.955),
        normal=(0.0, 0.0, 1.0),
        resolution=(20, 40),
        name="plate_top",
    )
    plate_bot = cpt.mesh_disk(
        radius=0.215,
        center=(0.0, 0.0, -0.959),
        normal=(0.0, 0.0, -1.0),
        resolution=(20, 40),
        name="plate_bot",
    )
    # Combine all three into one mesh.
    combined_mesh = spar.join_meshes(plate_top, plate_bot)
    body = cpt.FloatingBody(
        mesh=combined_mesh, center_of_mass=_COG, name="ref_spar_disk"
    )
    body.rotation_center = _COG
    body.add_all_rigid_body_dofs()
    print(f"  Reference body faces: {body.mesh.nb_faces}")

    # Solve at a single high omega for A_inf.
    omegas = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    problems = [
        cpt.RadiationProblem(
            body=body, omega=float(w), radiating_dof="Heave",
            water_depth=float("inf"), rho=_RHO, g=_G,
        )
        for w in omegas
    ]
    solver = cpt.BEMSolver()
    print("  Solving reference geometry (a few seconds) ...")
    results = solver.solve_all(problems, n_jobs=1)
    ds = cpt.assemble_dataset(results)
    A_h = ds["added_mass"].sel(radiating_dof="Heave", influenced_dof="Heave")
    print(f"  Reference A(heave) at omega = {omegas[-1]:.1f} rad/s: "
          f"{float(A_h.isel(omega=-1)):.3f} kg")
    print(f"  Reference A(heave) sweep:")
    for i, w in enumerate(omegas):
        print(f"    omega = {w:5.2f}: A = {float(A_h.isel(omega=i)):7.3f} kg")
    print()
    print("  Analytical reference: slender spar + heave plate")
    R_plate = 0.215
    A_spar_end = (8.0 / 3.0) * _RHO * 0.0841**3
    A_plate_one_face = (8.0 / 3.0) * _RHO * R_plate**3
    # The two-face plate (top + bottom) acting in heave: each face
    # adds (8/3) * rho * R^3; total ~ 2 * 27 ~ 54 kg if BOTH faces
    # contribute. In practice, the disk acts as a single rigid plate
    # so the effective added mass is approx that of a single disk
    # of equivalent radius, ~27 kg. Reference both estimates.
    print(f"    Spar bottom (8/3 rho r^3, r = 0.0841): {A_spar_end:.3f} kg")
    print(f"    Disk one-face (8/3 rho R^3, R = 0.215): "
          f"{A_plate_one_face:.3f} kg")
    print(f"    Total expected (spar + plate one-face): "
          f"~ {A_spar_end + A_plate_one_face:.3f} kg")


def main() -> None:
    print("Capytaine diagnostic script for spar+fin study (post-Xabier-correction)")
    print(f"Capytaine version: {cpt.__version__}")

    # Real GDF body.
    body = _load_real_body()

    check_1_visualize(body)
    check_2_normals(body)
    check_3_reference_geometry()

    print("\n" + "=" * 70)
    print("Diagnostic checks 1-3 complete. Reporting per locked plan.")
    print("=" * 70)


if __name__ == "__main__":
    main()
