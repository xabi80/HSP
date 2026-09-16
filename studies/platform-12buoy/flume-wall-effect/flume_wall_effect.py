"""Sidewall (blockage) + finite-depth effect on the Phase-3 12-buoy platform in the
OSU Hinsdale Large Wave Flume (LWF) — the analysis behind REBUTTAL-sidewall.md.

Context. A TEAMER reviewer flagged: "Phase 3 platform (2.50 m) inside HWRL's 3.67 m
flume leaves 0.6 m side clearance, inducing severe sidewall reflections that will
corrupt dynamic data." Phase 3 = the platform (Phase 1 = single buoy, Phase 2 =
cluster), tested by free-decay and by regular-wave sweeps. This module quantifies the
sidewall effect with potential-flow BEM (Capytaine) and shows it is small.

Framing. The reviewer's claim is about the SIDE WALLS, so every comparison here is
walls-in vs walls-out **at the same 2.7 m operating depth** — this isolates the wall
effect (the separate, larger finite-depth effect on long waves is common to both and
is not what the comment is about).

Geometry. 12 buoys (4 clusters x 3), buoy centres on a 2.5 m circle (the Phase-3
"2.5 m diameter to centres"; the sim layout in ``platform_common.py`` has centre radius
1.5 m, so scaled x 0.833). Each buoy uses the Phase-1 decay-correlated hull
(``osu-test-buoy/osu_buoy_common.py``: 0.159 m spar, equal-area heave plate r = 0.1437 m)
— the geometry that reproduced the Phase-1 free-decay (T ~ 2.5 s, zeta ~ 13 %).

Method. Finite depth is native in Capytaine (``water_depth``). Side walls (W = 3.66 m)
are imposed by the METHOD OF IMAGES: mirror the buoy centres across the walls and drive
the images with the same DOF field (a wall parallel to the motion has an in-phase image),
then read the hydrodynamic force on the REAL panels only via a two-DOF trick
(radiate/excite ``*_all`` over real+image panels, integrate over ``*_real``). The two-wall
image series alternates and converges over ~3 reflection levels.

Run ``python flume_wall_effect.py`` (BEM sweep, ~10-15 min) to regenerate the numbers and
figures. Requires capytaine (contributor tool, not a FloatSim runtime dep).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

warnings.simplefilter("ignore")
HERE = Path(__file__).resolve().parent

# --- flume (OSU Hinsdale Large Wave Flume) ---
FLUME_W = 3.66          # m, width (walls at y = +/- W/2)
FLUME_H = 2.7           # m, operating still-water depth (max ~2.74 m)

# --- single buoy (Phase-1 decay-correlated hull; osu_buoy_common.py) ---
SPAR_R = 0.1593 / 2     # m, 6" spar radius
SPAR_BOT = -0.967       # m, spar bottom (waterline frame)
PLATE_R = 0.1437        # m, equal-area heave-plate disc
PLATE_Z = -1.383        # m, plate depth
C33_BUOY = 194.5        # N/m, single-buoy heave waterplane stiffness
T_HEAVE = 2.52          # s, Phase-1 correlated heave period
ZETA = 0.13             # -, Phase-1 correlated heave damping (viscous)

# --- platform layout (12-buoy, 2.5 m to centres) ---
N_BUOY = 12
_SCALE = 1.25 / 1.5     # centre radius 1.5 m (sim) -> 1.25 m (2.5 m diameter to centres)
ARM, INTRA = 1.0 * _SCALE, 0.5 * _SCALE
RHO, G = 998.0, 9.806


def buoy_centers() -> NDArray[np.float64]:
    """(12, 2) buoy centres: 4 clusters (0/90/180/270 deg) x 3 buoys (0/120/240 deg)."""
    out = []
    for pc in np.deg2rad([0.0, 90.0, 180.0, 270.0]):
        cx, cy = ARM * np.cos(pc), ARM * np.sin(pc)
        for tb in np.deg2rad([0.0, 120.0, 240.0]):
            out.append([cx + INTRA * np.cos(tb), cy + INTRA * np.sin(tb)])
    return np.asarray(out, dtype=np.float64)


def clearance() -> float:
    """Side clearance (m) from the outermost fin edge to the wall (as-built ~0.44 m)."""
    c = buoy_centers()
    return FLUME_W / 2 - (np.abs(c[:, 0]).max() + PLATE_R)


def transverse_cutoff_periods(n_max: int = 3, h: float = FLUME_H) -> list[float]:
    """Flume transverse (cross-width) sloshing cut-on periods T_n (s), k_yn = n*pi/W.

    For T > T_1 the transverse modes are evanescent -> the wall effect is small and
    smooth; near T_n the scattered field resonates across the width. Nearly h-independent.
    """
    out = []
    for n in range(1, n_max + 1):
        k = n * np.pi / FLUME_W
        out.append(float(2 * np.pi / np.sqrt(G * k * np.tanh(k * h))))
    return out


# ---- BEM (import capytaine lazily so the geometry helpers work without it) ----
def buoy_mesh(cx: float, cy: float):  # type: ignore[no-untyped-def]
    """One immersed buoy mesh (surface-piercing spar + thin closed heave-plate disc)."""
    import capytaine as cpt

    spar = cpt.mesh_vertical_cylinder(
        length=abs(SPAR_BOT) + 0.1, radius=SPAR_R,
        center=(cx, cy, (SPAR_BOT + 0.1) / 2), resolution=(1, 10, 6))
    plate = cpt.mesh_vertical_cylinder(
        length=0.01, radius=PLATE_R, center=(cx, cy, PLATE_Z), resolution=(2, 14, 1))
    return cpt.FloatingBody(mesh=spar + plate).immersed_part().mesh


def array_mesh(centres: NDArray[np.float64]):  # type: ignore[no-untyped-def]
    m = buoy_mesh(*centres[0])
    for c in centres[1:]:
        m = m + buoy_mesh(*c)
    return m


def wall_images(centres: NDArray[np.float64], level: int, axis: int) -> list[tuple[float, float]]:
    """Image buoy centres for walls at +/- W/2 on ``axis`` (0=x for decay, 1=y for waves),
    up to ``level`` reflection levels. The two-wall image set is two interleaved families."""
    w = FLUME_W
    im: list[tuple[float, float]] = []

    def add(coord_fn) -> None:  # type: ignore[no-untyped-def]
        for cx, cy in centres:
            v = coord_fn(cy if axis else cx)
            im.append((cx, v) if axis else (v, cy))

    add(lambda c: w - c)             # reflect in +W/2 wall
    add(lambda c: -w - c)            # reflect in -W/2 wall
    if level >= 2:
        add(lambda c: c + 2 * w)
        add(lambda c: c - 2 * w)
    if level >= 3:
        add(lambda c: 3 * w - c)
        add(lambda c: -3 * w - c)
    return im


def _dofs(body, n_real: int) -> None:  # type: ignore[no-untyped-def]
    """Attach heave/surge/pitch DOFs in ``*_all`` (real+image, enforces the wall BC) and
    ``*_real`` (real panels only, for the force integral) forms."""
    fc = body.mesh.faces_centers
    nf = len(fc)
    d = {
        "He_all": np.tile([0.0, 0.0, 1.0], (nf, 1)),
        "Su_all": np.tile([1.0, 0.0, 0.0], (nf, 1)),
        "Pi_all": np.column_stack([fc[:, 2], np.zeros(nf), -fc[:, 0]]),
    }
    for k in list(d):
        real = d[k].copy()
        real[n_real:] = 0.0
        d[k[:2] + "_real"] = real
    body.dofs = d


def coefficients(centres, n_real, omega, depth, solver):  # type: ignore[no-untyped-def]
    """Diagonal added mass, radiation damping and |excitation| for heave/surge/pitch."""
    import capytaine as cpt

    body = cpt.FloatingBody(mesh=array_mesh(centres))
    _dofs(body, n_real)
    out: dict[tuple[str, str], float] = {}
    for dof in ("He", "Su", "Pi"):
        r = solver.solve(cpt.RadiationProblem(
            body=body, radiating_dof=dof + "_all", omega=omega, water_depth=depth, rho=RHO, g=G))
        out[("A", dof)] = float(np.real(r.added_masses[dof + "_real"]))
        out[("B", dof)] = float(np.real(r.radiation_dampings[dof + "_real"]))
    dif = solver.solve(cpt.DiffractionProblem(
        body=body, wave_direction=0.0, omega=omega, water_depth=depth, rho=RHO, g=G))
    for dof in ("He", "Su", "Pi"):
        out[("F", dof)] = abs(dif.forces[dof + "_real"])
    return out


def run_sweep(periods: NDArray[np.float64], wall_level: int = 2):  # type: ignore[no-untyped-def]
    """Walls-out (unbounded, h) vs walls-in (h + side walls) at every period; cached to .npz."""
    import capytaine as cpt

    solver = cpt.BEMSolver()
    cen = buoy_centers()
    n_real = _n_real(cen)
    img = np.vstack([cen, *wall_images(cen, wall_level, axis=1)])  # waves along +x -> y-walls
    rows = []
    for t in periods:
        w = 2 * np.pi / t
        o = coefficients(cen, n_real, w, np.inf, solver)
        wl = coefficients(img, n_real, w, FLUME_H, solver)
        rows.append((float(t), o, wl))
        print(f"T={t:.2f}s  heave dA={_pct(o, wl, 'A', 'He'):+.1f}% "
              f"dFexc={_pct(o, wl, 'F', 'He'):+.1f}%  "
              f"pitch dFexc={_pct(o, wl, 'F', 'Pi'):+.1f}%  "
              f"surge dFexc={_pct(o, wl, 'F', 'Su'):+.1f}%")
    np.save(HERE / "sweep_results.npy", np.array(rows, dtype=object), allow_pickle=True)
    return rows


def _n_real(centres) -> int:  # type: ignore[no-untyped-def]
    return len(array_mesh(centres).faces_centers)


def _pct(o, wl, kind, dof) -> float:  # type: ignore[no-untyped-def]
    return 100.0 * (wl[(kind, dof)] / o[(kind, dof)] - 1.0) if abs(o[(kind, dof)]) > 1e-9 else 0.0


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    outer = 2 * (np.abs(buoy_centers()[:, 0]).max() + PLATE_R)
    cutoffs = [round(t, 2) for t in transverse_cutoff_periods()]
    print(f"12-buoy platform (2.5 m to centres) in the OSU LWF ({FLUME_W} m x {FLUME_H} m)")
    print(f"  outer fin extent {outer:.2f} m -> side clearance {clearance() * 100:.0f} cm/side "
          f"({100 * outer / FLUME_W:.0f}% of width)")
    print(f"  transverse cut-on periods T_n = {cutoffs} s")
    periods = np.array([1.40, 1.53, 1.80, 2.19, 2.52, 3.00, 3.50, 4.00])
    run_sweep(periods, wall_level=2)


if __name__ == "__main__":
    main()
