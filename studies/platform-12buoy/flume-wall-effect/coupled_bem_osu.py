"""Coupled 16-buoy OSU-plate BEM for the flume wall-effect articulated run.

OPEN or WALLED (method of images). 16 OSU-plate buoys (spar + equal-area heave-plate
disc) at the Phase-3 layout: 4 clusters x 4 buoys (square, 0/90/180/270 deg), buoy centres
on a 2.5 m circle. Per-buoy rigid DOFs (buoy{i}__Surge.. ) -> 96-DOF coupled radiation +
diffraction, hydrostatic C tiled from the single buoy. Saved in the FloatSim capytaine
schema (read_capytaine-compatible). Buoy count is parametric via BUOY_ANGLES_DEG (the
12-buoy triangle layout is [0,120,240]).

Usage: python coupled_bem_osu.py <open|walled> <test|full> [n_omega]
Walled: reflects each buoy across the flume walls y=+/-1.83 (2 image levels), with the
sign-correct mirrored rigid motion per DOF (surge/heave/pitch symmetric; sway/roll/yaw
antisymmetric), and reads the force on the REAL panels (two-DOF _all/_real trick).
"""
# ruff: noqa: E702  -- compact per-frequency BEM assembly/setup lines in a one-off generator.
from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.simplefilter("ignore")
import capytaine as cpt  # noqa: E402
from capytaine.bem.airy_waves import froude_krylov_force  # noqa: E402

cpt.set_logging("ERROR")
RHO, G = 998.0, 9.806
SPAR_R, Z_BOT, Z_TOP = 0.07965, -0.967, 0.717
PLATE_R, PLATE_Z, COG_Z = 0.1437, -1.383, -0.907
W = 3.66                                   # flume width (walls at y = +/- W/2)
# Depth for the COUPLED run: default deep. Finite depth (FLUME_DEPTH=2.7) is correct in principle
# but impractically slow in Capytaine at this panel count (>3 h/case), so the depth-sensitive
# wave-excitation wall effect is carried by the fast single-array freq-domain sweep
# (flume_wall_effect.py, native 2.7 m); the coupled run gives the multi-body/all-DOF free-decay
# + near-resonance response, where the wall effect is depth-robust. See README.
_FD = __import__("os").environ.get("FLUME_DEPTH", "inf")
DEPTH = np.inf if _FD == "inf" else float(_FD)
NT = 12                                    # spar/plate n_theta (coarse; wall ratio is mesh-robust)
CLUSTER_ANGLES_DEG = [0, 90, 180, 270]     # 4 clusters
BUOY_ANGLES_DEG = [0, 90, 180, 270]        # 4 buoys/cluster (square); 12-buoy used [0,120,240]
NB = len(CLUSTER_ANGLES_DEG) * len(BUOY_ANGLES_DEG)   # 16 buoys
NDOF = 6 * NB                              # 96 coupled rigid DOFs
DOF6 = ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"]
# rigid-DOF reflection across a y=const plane: translations flip y; rotations (pseudovec)
# flip x,z. So surge/heave/pitch keep sign under the image; sway/roll/yaw flip.
REFL_SIGN = {"Surge": +1, "Sway": -1, "Heave": +1, "Roll": -1, "Pitch": +1, "Yaw": -1}
HERE = Path(__file__).resolve().parent


def centers() -> list[tuple[float, float]]:
    s = 1.25 / 1.5
    out = []
    for pc in np.deg2rad(CLUSTER_ANGLES_DEG):
        cx, cy = s * np.cos(pc), s * np.sin(pc)
        for tb in np.deg2rad(BUOY_ANGLES_DEG):
            out.append((cx + 0.5 * s * np.cos(tb), cy + 0.5 * s * np.sin(tb)))
    return out


def buoy_mesh(cx: float, cy: float):  # type: ignore[no-untyped-def]
    spar = cpt.mesh_vertical_cylinder(length=Z_TOP - Z_BOT, radius=SPAR_R,
                                      center=(cx, cy, (Z_BOT + Z_TOP) / 2), resolution=(2, NT, 24))
    plate = cpt.mesh_vertical_cylinder(length=0.02, radius=PLATE_R, center=(cx, cy, PLATE_Z),
                                       resolution=(4, NT, 2))
    return cpt.FloatingBody(mesh=spar.join_meshes(plate)).immersed_part().mesh


def rigid_field(fc: np.ndarray, dof: str, c: tuple[float, float, float], sign: int) -> np.ndarray:
    """Unit rigid-DOF velocity field at panel centres fc about centre c, scaled by sign."""
    x, y, z = fc[:, 0] - c[0], fc[:, 1] - c[1], fc[:, 2] - c[2]
    zero = np.zeros(len(fc))
    fields = {
        "Surge": np.column_stack([np.ones(len(fc)), zero, zero]),
        "Sway": np.column_stack([zero, np.ones(len(fc)), zero]),
        "Heave": np.column_stack([zero, zero, np.ones(len(fc))]),
        "Roll": np.column_stack([zero, -z, y]),
        "Pitch": np.column_stack([z, zero, -x]),
        "Yaw": np.column_stack([-y, x, zero]),
    }
    return sign * fields[dof]


def single_buoy_c() -> np.ndarray:
    b = cpt.FloatingBody(mesh=buoy_mesh(0.0, 0.0), center_of_mass=(0, 0, COG_Z))
    b.rotation_center = np.array([0.0, 0.0, COG_Z])
    b.add_all_rigid_body_dofs()
    hs = b.compute_hydrostatics(rho=RHO, g=G)
    return np.array([[float(hs["hydrostatic_stiffness"].sel(radiating_dof=a, influenced_dof=b_))
                      for b_ in DOF6] for a in DOF6])


def build_open():  # type: ignore[no-untyped-def]
    bodies = []
    for i, (cx, cy) in enumerate(centers()):
        cog = (cx, cy, COG_Z)
        b = cpt.FloatingBody(mesh=buoy_mesh(cx, cy), center_of_mass=cog, name=f"buoy{i + 1}")
        b.rotation_center = np.asarray(cog)
        b.add_all_rigid_body_dofs()
        bodies.append(b)
    allb = bodies[0]
    for b in bodies[1:]:
        allb = allb + b
    return allb, list(allb.dofs)


def build_walled():  # type: ignore[no-untyped-def]
    """Combined mesh = NB real + Y-images; custom dofs {label}__{DOF}_all (real+mirrored
    image motion, enforces the wall BC) and _real (real panels only, for the force)."""
    cen = centers()
    real_meshes = [buoy_mesh(cx, cy) for cx, cy in cen]
    nreal = [m.nb_faces for m in real_meshes]
    # image centres per real buoy (2 nearest: reflect in +wall and -wall)
    img_cen = [[(cx, W - cy), (cx, -W - cy)] for cx, cy in cen]
    all_meshes = list(real_meshes)
    for row in img_cen:
        for (ix, iy) in row:
            all_meshes.append(buoy_mesh(ix, iy))
    mesh = all_meshes[0]
    for m in all_meshes[1:]:
        mesh = mesh + m
    body = cpt.FloatingBody(mesh=mesh)
    fc = body.mesh.faces_centers
    # panel index ranges: reals first (nreal), then images grouped per buoy
    starts = np.cumsum([0, *nreal]).tolist()
    real_slices = [slice(starts[i], starts[i + 1]) for i in range(NB)]
    npr = starts[NB]
    img_slices = [[None, None] for _ in range(NB)]
    off = npr
    per_img = nreal  # each image buoy has same face count as its real (same mesh)
    for i in range(NB):
        for j in range(2):
            img_slices[i][j] = slice(off, off + per_img[i]); off += per_img[i]
    dofs = {}
    for i, (cx, cy) in enumerate(cen):
        for d in DOF6:
            lab = f"buoy{i + 1}__{d}"
            f_all = np.zeros((len(fc), 3)); f_real = np.zeros((len(fc), 3))
            f_all[real_slices[i]] = rigid_field(fc[real_slices[i]], d, (cx, cy, COG_Z), +1)
            f_real[real_slices[i]] = f_all[real_slices[i]]
            for j, (ix, iy) in enumerate(img_cen[i]):
                f_all[img_slices[i][j]] = rigid_field(fc[img_slices[i][j]], d, (ix, iy, COG_Z),
                                                      REFL_SIGN[d])
            dofs[lab + "_all"] = f_all
            dofs[lab + "_real"] = f_real
    body.dofs = dofs
    real_labels = [f"buoy{i + 1}__{d}" for i in range(NB) for d in DOF6]
    return body, real_labels


def solve_freq(body, radiate_labels, read_labels, w, with_diff):  # type: ignore[no-untyped-def]
    solver = cpt.BEMSolver()
    probs = [cpt.RadiationProblem(body=body, omega=w, radiating_dof=r, water_depth=DEPTH,
                                  rho=RHO, g=G) for r in radiate_labels]
    if with_diff:
        probs.append(cpt.DiffractionProblem(body=body, omega=w, wave_direction=0.0,
                                            water_depth=DEPTH, rho=RHO, g=G))
    results = solver.solve_all(probs, progress_bar=False)
    del solver; gc.collect()
    return results


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    mode = sys.argv[1] if len(sys.argv) > 1 else "open"
    test = (len(sys.argv) > 2 and sys.argv[2] == "test")
    n_om = int(sys.argv[3]) if len(sys.argv) > 3 else 28
    om = np.array([1.5, 2.5]) if test else np.geomspace(0.1, 30.0, n_om)
    C6 = single_buoy_c()
    if mode == "walled":
        body, real_labels = build_walled()
        radiate = [f"{lab}_all" for lab in real_labels]
        read = [f"{lab}_real" for lab in real_labels]
    else:
        body, real_labels = build_open()
        radiate = real_labels; read = real_labels
    print(f"{mode}: {body.mesh.nb_faces} panels, {len(real_labels)} real DOF", flush=True)
    t0 = time.perf_counter()
    A = np.zeros((len(om) + 1, NDOF, NDOF)); B = np.zeros((len(om) + 1, NDOF, NDOF))
    Fexc = np.zeros((len(om) + 1, NDOF), complex)  # last (omega=inf) row stays 0 (reader strips it)
    for k, w in enumerate([*list(om), np.inf]):
        res = solve_freq(body, radiate, read, float(w), with_diff=np.isfinite(w))
        rad = res[:NDOF]
        for a, rr in enumerate(rad):
            A[k, a, :] = [float(np.real(rr.added_masses[rl])) for rl in read]
            if np.isfinite(w):
                B[k, a, :] = [float(np.real(rr.radiation_dampings[rl])) for rl in read]
        if np.isfinite(w):
            dif = res[NDOF]
            fk = froude_krylov_force(dif.problem)
            Fexc[k, :] = [complex(dif.forces[rl] + fk[rl]) for rl in read]
        if (k + 1) % 5 == 0 or test:
            print(f"  {k + 1}/{len(om) + 1} (w={w:.2f}) {(time.perf_counter() - t0) / 60:.1f} min",
                  flush=True)
    Ctiled = np.zeros((NDOF, NDOF))
    for i in range(NB):
        Ctiled[6 * i:6 * i + 6, 6 * i:6 * i + 6] = C6
    om_all = np.append(om, np.inf)
    ds = xr.Dataset(
        data_vars=dict(
            added_mass=(("omega", "radiating_dof", "influenced_dof"), A),
            radiation_damping=(("omega", "radiating_dof", "influenced_dof"), B),
            hydrostatic_stiffness=(("radiating_dof", "influenced_dof"), Ctiled),
            excitation_force=(("complex", "omega", "wave_direction", "influenced_dof"),
                              np.stack([Fexc.real, Fexc.imag], 0)[:, :, None, :]),
        ),
        coords=dict(omega=("omega", om_all), wave_direction=("wave_direction", [0.0]),
                    radiating_dof=("radiating_dof", real_labels),
                    influenced_dof=("influenced_dof", real_labels),
                    complex=("complex", ["re", "im"])),
        attrs=dict(rho=RHO, g=G, water_depth=DEPTH, body_name=f"osu{NB}_{mode}"),
    )
    out = HERE / (f"coupled_osu_{mode}{'_test' if test else ''}.nc")
    ds.to_netcdf(out)
    print(f"wrote {out}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    sys.path.insert(0, str(Path("C:/Users/xlama/OneDrive/Documents/buoy/HSP_code")))
    from floatsim.hydro.readers.capytaine import read_capytaine
    h = read_capytaine(out)
    print(f"read_capytaine OK: {np.asarray(h.omega).size} om, A {np.asarray(h.A).shape}, "
          f"A_inf[2,2]={float(np.asarray(h.A_inf)[2, 2]):.2f}, "
          f"C[2,2]={float(np.asarray(h.C)[2, 2]):.1f}")


if __name__ == "__main__":
    main()
