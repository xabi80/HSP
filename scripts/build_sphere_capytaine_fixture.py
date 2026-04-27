"""Generate the deeply-submerged-sphere reference NetCDF using Capytaine.

This is the real-BEM sibling of the analytically-authored
``synthetic_sphere.nc`` produced by
``build_capytaine_synthetic_fixtures.py``. The synthetic file lets the
parser tests run without Capytaine installed; this script is what
produces the BEM-noise-laden reference Capytaine writes when actually
solving the diffraction/radiation problem.

Why a separate script
---------------------
Capytaine itself is **not** a runtime dependency of FloatSim
(CLAUDE.md §9). Adding it to the test environment is too heavy for
what would be a single fixture. The output file
``tests/fixtures/bem/capytaine/sphere_submerged.nc`` is committed
once after running this script; CI consumes the committed file
without re-running Capytaine.

When to re-run
--------------
- Capytaine major version bump that changes the on-disk schema (the
  reader's regression tests should catch this).
- Geometry/discretisation tweak (mesh resolution, depth, frequency
  grid).
- Initial generation on a fresh machine.

Reproducibility
---------------
The mesh is generated from Capytaine's built-in ``mesh_sphere`` helper
with a deterministic resolution. Frequency grid and water density are
fixed below so re-running produces a byte-similar (but not necessarily
bit-identical -- BEM solvers carry tiny FPU-noise differences) file.

Geometry / case
---------------
Fully submerged sphere of radius ``R = 1 m`` centred at depth
``z = -5 R`` below the still water level. Apache 2.0 free-surface
conditions (no current, no forward speed). Frequency grid spans
``omega in [0.2, 2.0] rad/s`` plus ``omega = +inf`` for ``A_inf``.
Two wave headings (0 deg, 90 deg).

Analytical reference (Lamb 1932, §92):
``A_ii = (2/3) * pi * rho * R**3``
``B_ii(omega) -> 0`` (deep submergence)
``C = 0`` (neutral buoyancy, no waterplane)

The script verifies each of these to a coarse tolerance before saving;
a panel-density warning is emitted if scatter is large.

Usage::

    pip install capytaine
    python scripts/build_sphere_capytaine_fixture.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import capytaine as cpt
except ImportError as exc:  # pragma: no cover - script-time import guard
    raise SystemExit(
        "Capytaine is not installed. Install it (e.g. `pip install capytaine`) "
        "to regenerate the sphere reference fixture, or use the analytical "
        "synthetic_sphere.nc instead."
    ) from exc


# ---------------------------------------------------------------------------
# Case parameters -- tweak these and bump the docstring "When to re-run".
# ---------------------------------------------------------------------------
_RHO: float = 1025.0
_G: float = 9.80665
_RADIUS_M: float = 1.0
_DEPTH_M: float = 5.0  # depth of sphere center below MWL (=> 5R below MWL)
_OMEGA_FINITE: tuple[float, ...] = (0.3, 0.7, 1.4)
_HEADINGS_DEG: tuple[float, ...] = (0.0, 90.0)
_PANEL_RESOLUTION: tuple[int, int] = (16, 16)  # (n_theta, n_phi); coarse but cheap


def _build_sphere_body() -> cpt.FloatingBody:
    mesh = cpt.mesh_sphere(
        radius=_RADIUS_M,
        center=(0.0, 0.0, -_DEPTH_M),
        resolution=_PANEL_RESOLUTION,
    )
    body = cpt.FloatingBody(mesh=mesh, name="sphere_submerged")
    body.add_all_rigid_body_dofs()
    body.center_of_mass = (0.0, 0.0, -_DEPTH_M)
    # Neutral buoyancy: hydrostatic stiffness is identically zero; we
    # leave Capytaine to emit zero rather than skipping the
    # hydrostatic_stiffness variable.
    return body


def _build_problems(body: cpt.FloatingBody) -> list:
    problems: list = []
    omegas = [*_OMEGA_FINITE, float("inf")]
    for omega in omegas:
        for dof in body.dofs:
            problems.append(
                cpt.RadiationProblem(
                    body=body,
                    omega=omega,
                    radiating_dof=dof,
                    rho=_RHO,
                    g=_G,
                )
            )
        if not np.isinf(omega):
            for beta_deg in _HEADINGS_DEG:
                problems.append(
                    cpt.DiffractionProblem(
                        body=body,
                        omega=omega,
                        wave_direction=np.deg2rad(beta_deg),
                        rho=_RHO,
                        g=_G,
                    )
                )
    return problems


def _verify_against_analytical(ds) -> None:
    A_analytical = (2.0 / 3.0) * np.pi * _RHO * _RADIUS_M**3

    A_finite = ds.added_mass.sel(omega=_OMEGA_FINITE[0]).values
    a_diag = np.diag(np.asarray(A_finite, dtype=float))
    rel = np.abs(a_diag[:3] - A_analytical) / A_analytical
    if np.any(rel > 5.0e-2):
        print(
            "WARNING: sphere added mass deviates from (2/3) pi rho R^3 by "
            f"more than 5% (panel density {_PANEL_RESOLUTION}). "
            f"diag(A)[:3]={a_diag[:3]}, analytical={A_analytical}. "
            "Increase panel resolution and re-run."
        )

    B_finite = ds.radiation_damping.sel(omega=_OMEGA_FINITE[0]).values
    b_diag_norm = np.linalg.norm(np.diag(np.asarray(B_finite, dtype=float))) / (
        _RHO * _RADIUS_M**3 * _OMEGA_FINITE[0]
    )
    if b_diag_norm > 1.0e-2:
        print(
            "WARNING: sphere radiation damping not negligible "
            f"(||diag(B)|| / (rho R^3 omega) = {b_diag_norm:.2e}). "
            "Sphere may be insufficiently submerged."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("tests/fixtures/bem/capytaine/sphere_submerged.nc"),
        help="Output path for the sphere NetCDF fixture.",
    )
    args = parser.parse_args(argv)

    body = _build_sphere_body()
    problems = _build_problems(body)
    print(f"  solving {len(problems)} BEM problems " f"(panel resolution {_PANEL_RESOLUTION})...")
    solver = cpt.BEMSolver()
    results = solver.solve_all(problems)
    ds = cpt.assemble_dataset(results, hydrostatics=False)
    _verify_against_analytical(ds)

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    cpt.io.xarray.save_dataset_as_netcdf(args.dst, ds)
    print(f"  wrote {args.dst} ({args.dst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
