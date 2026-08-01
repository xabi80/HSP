"""M11b PR7: 12-buoy platform BEM at scale, with the two-detector conditioning
screening EMBEDDED in the solve path (OPTION 2).

12 separate FloatingBodies (72 DOF) on the regenerated platform mesh, at the
platform draft. Radiation + diffraction on the 13-omega reduced grid bracketing
the resonances. cond(K) is emitted per frequency by a custom linear_solver hook
(zgecon on the LU the solver already computes -- no re-factorize); the
symmetrized-B min-eig is computed from the assembled dataset. Every retained
slice must clear BOTH detectors (platform_screening.screen); contaminated
slices are EXCLUDED (grid selection, M8 PR3 pattern), never value-modified.

Modes:
  python platform_bem.py validate   # STEP D: re-validate both detectors on the
                                     # known cluster-draft cases (16.837, 4.934)
  python platform_bem.py run         # STEP 3: full 12-buoy solve + screen + save
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import capytaine as cpt
import numpy as np
import platform_common as pc
import platform_screening as ps
from capytaine.bem.engines import BasicMatrixEngine
from scipy import linalg as sl
from scipy.linalg.lapack import zgecon

_HERE = Path(__file__).resolve().parent
_OUT_NC = _HERE / "platform12_bem.nc"
_CLUSTER_MESH = _HERE.parent / "cluster-3buoy-rigid" / "mesh" / "cluster3_fullfix.gdf"

# 13-omega reduced grid bracketing the resonances (heave ~T=3.11s -> w=2.02,
# rotational ~T=3.26s -> w=1.93). Fine over 1.7-2.3 (5 pts), coarse to 30 for
# A_inf/tail. Deliberately avoids the known high-w irregular frequencies
# (16.8/20.9) and the 4.934 output-contamination slice -- they are not grid
# points -- so the detectors act as a SAFETY NET confirming no NEW 12-buoy
# contamination lands on a retained slice.
OMEGAS_13 = np.array([0.5, 1.0, 1.5, 1.75, 1.9, 2.0, 2.1, 2.25, 2.5, 3.0, 5.0, 12.0, 30.0])


class CondLUSolver:
    """capytaine linear_solver that emits cond(K)=1/rcond via zgecon on its own
    LU (linear_solvers.py:86 style) -- no second factorization."""

    def __init__(self) -> None:
        self.cached_matrix = None
        self.cached_decomp = None
        self.last_cond = float("nan")

    def __call__(self, A, b):  # type: ignore[no-untyped-def]
        if A is not self.cached_matrix:
            self.cached_matrix = A
            self.cached_decomp = sl.lu_factor(A)
            rc, _ = zgecon(self.cached_decomp[0], float(np.linalg.norm(A, 1)), norm="1")
            self.last_cond = (1.0 / rc) if rc > 0 else np.inf
        return sl.lu_solve(self.cached_decomp, b)


def cond_k_sweep(mesh_path, omegas, cog=(0.0, 0.0, 0.0)):  # type: ignore[no-untyped-def]
    """cond(K) per omega for a single-body mesh (heave DOF suffices to trigger
    the factorization). Used by STEP D validation and diagnostics."""
    mesh = cpt.load_mesh(str(mesh_path), file_format="gdf")
    hook = CondLUSolver()
    bem = cpt.BEMSolver(engine=BasicMatrixEngine(linear_solver=hook))
    cond = np.full(len(omegas), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, w in enumerate(omegas):
            body = cpt.FloatingBody(mesh=mesh, name="m")
            body.add_translation_dof(name="Heave")
            bem.solve(
                cpt.RadiationProblem(
                    body=body,
                    omega=float(w),
                    radiating_dof="Heave",
                    water_depth=float("inf"),
                    rho=pc.cc.RHO,
                    g=pc.cc.G,
                ),
                keep_details=False,
            )
            cond[i] = hook.last_cond
    return cond


def _build_bodies():  # type: ignore[no-untyped-def]
    base = cpt.load_mesh(str(pc.cc.SINGLE_EQDRAFT_MESH), file_format="gdf")
    base = base.translated([0.0, 0.0, -pc.PLATFORM_DZ])
    cogz = pc.cc.CoG_Z_SINGLE - pc.PLATFORM_DZ
    bodies = []
    for k, (dx, dy) in enumerate(pc.buoy_centers()):
        m = base.translated([float(dx), float(dy), 0.0])
        cog = [float(dx), float(dy), cogz]
        b = cpt.FloatingBody(mesh=m, center_of_mass=cog, name=f"buoy{k + 1}")
        b.rotation_center = np.asarray(cog)
        b.add_all_rigid_body_dofs()
        bodies.append(b)
    allb = bodies[0]
    for b in bodies[1:]:
        allb = allb + b
    return allb


def run() -> None:
    print("=" * 72)
    print("M11b PR7: 12-buoy platform BEM (13-omega) + embedded screening")
    print(f"grid: {OMEGAS_13.tolist()} rad/s (+inf)")
    allb = _build_bodies()
    dofs = list(allb.dofs)
    print(f"combined body: {len(dofs)} DOF, {allb.mesh.nb_faces} faces")

    hook = CondLUSolver()
    bem = cpt.BEMSolver(engine=BasicMatrixEngine(linear_solver=hook))
    omegas_inf = [*OMEGAS_13.tolist(), float("inf")]
    cond_per_omega: dict[float, float] = {}

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rad, dif = [], []
        for w in omegas_inf:
            for d in dofs:
                rad.append(
                    cpt.RadiationProblem(
                        body=allb,
                        omega=float(w),
                        radiating_dof=d,
                        water_depth=float("inf"),
                        rho=pc.cc.RHO,
                        g=pc.cc.G,
                    )
                )
        for w in OMEGAS_13:
            dif.append(
                cpt.DiffractionProblem(
                    body=allb,
                    omega=float(w),
                    wave_direction=0.0,
                    water_depth=float("inf"),
                    rho=pc.cc.RHO,
                    g=pc.cc.G,
                )
            )
        # Solve per-omega so cond(K) maps to omega (one factorization per omega).
        results = []
        for w in omegas_inf:
            probs = [p for p in rad if p.omega == float(w)] + [
                p for p in dif if p.omega == float(w)
            ]
            results += bem.solve_all(probs, n_jobs=1)
            cond_per_omega[float(w)] = hook.last_cond
    dt = time.perf_counter() - t0
    print(f"solved {len(results)} problems in {dt:.0f} s = {dt/60:.1f} min")

    ds = cpt.assemble_dataset(results)
    import xarray as xr  # noqa: F401

    for c in ("radiating_dof", "influenced_dof"):
        if c in ds.coords:
            ds = ds.assign_coords({c: ds[c].astype(str)})
    omega_finite = OMEGAS_13
    B = np.stack([ds["radiation_damping"].sel(omega=w).values for w in omega_finite], axis=-1)
    cond_k = np.array([cond_per_omega[float(w)] for w in omega_finite])

    verdicts = ps.screen(omega_finite, cond_k, B)
    print("\nSCREENING TABLE (STEP 4):")
    hdr = ("omega", "cond(K)", "cond_z", "B_mineig", "bmineig_z", "sig", "verdict", "excl")
    print(
        f"  {hdr[0]:>7} {hdr[1]:>10} {hdr[2]:>7} {hdr[3]:>11} {hdr[4]:>9} {hdr[5]:>4} {hdr[6]:>16}"
    )
    for v in verdicts:
        print(
            f"  {v.omega:7.3f} {v.cond_k:10.3e} {v.cond_z:7.2f} {v.bmineig:+11.3e} "
            f"{v.bmineig_z:9.2f} {v.psd_fires!s:>4} {v.verdict:>16} {v.exclude}"
        )
    excl = [v.omega for v in verdicts if v.exclude]
    both = [v.omega for v in verdicts if v.verdict == "both"]
    print(f"\nexcluded: {excl if excl else 'NONE (all retained)'}")
    if both:
        print(f"*** BOTH-FIRE (new class) at {both} -- REPORT, distinct observation ***")
    kpk = int(np.nanargmax(cond_k))
    print(f"peak cond(K) = {np.nanmax(cond_k):.3e} at omega={omega_finite[kpk]:.3f}")

    # Buoyancy-only hydrostatic C (72x72); gravity added downstream (M5 lesson).
    immersed = allb.immersed_part()
    ds["hydrostatic_stiffness"] = immersed.compute_hydrostatic_stiffness(rho=pc.cc.RHO, g=pc.cc.G)

    from capytaine.io.xarray import separate_complex_values

    separate_complex_values(ds).to_netcdf(str(_OUT_NC))
    print(f"wrote {_OUT_NC.name}")


def add_hydrostatic() -> None:
    """Post-process: add hydrostatic_stiffness to an existing platform12_bem.nc
    (the run() that produced it predated the code above), avoiding a re-solve."""
    import xarray as xr
    from capytaine.io.xarray import separate_complex_values

    allb = _build_bodies()
    dofs = list(allb.dofs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        K = allb.immersed_part().compute_hydrostatic_stiffness(rho=pc.cc.RHO, g=pc.cc.G)
    ds = xr.open_dataset(str(_OUT_NC))
    ds.load()
    ds.close()
    ds["hydrostatic_stiffness"] = xr.DataArray(
        np.asarray(K),
        dims=("radiating_dof", "influenced_dof"),
        coords={"radiating_dof": [str(d) for d in dofs], "influenced_dof": [str(d) for d in dofs]},
    )
    separate_complex_values(ds).to_netcdf(str(_OUT_NC))
    c33 = float(
        np.asarray(K)[np.ix_([6 * k + 2 for k in range(12)], [6 * k + 2 for k in range(12)])].sum()
    )
    print(f"added hydrostatic_stiffness; C33 composite = {c33:.1f} N/m (expect ~12x221=2652)")


def validate() -> None:
    """STEP D: both detectors on the known cluster-draft cases."""
    grid = np.geomspace(0.1, 30.0, 80)
    k4 = int(np.argmin(np.abs(grid - 4.934)))
    k16 = int(np.argmin(np.abs(grid - 16.837)))
    band = sorted({*range(k4 - 3, k4 + 4), *range(k16 - 3, k16 + 4)})
    ws = grid[band]
    cond = cond_k_sweep(_CLUSTER_MESH, ws)
    cz = ps.neighbour_trend_z(np.log10(cond))
    i4 = list(band).index(k4)
    i16 = list(band).index(k16)
    print(
        f"cond(K) STEP D: 16.837 z={cz[i16]:.2f} (must fire, thr {ps.COND_Z_THRESHOLD}); "
        f"4.934 z={cz[i4]:.2f} (must stay flat)"
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    {"run": run, "hydrostatic": add_hydrostatic, "validate": validate}[mode]()
