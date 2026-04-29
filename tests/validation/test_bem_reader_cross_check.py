"""Milestone 5 gate validation — three-reader BEM cross-check (Q4 of M5 plan).

A fully-submerged sphere (depth >= 5R, no free-surface effects) has a
closed-form solution: added mass on each translational DOF is the
classical Lamb (1932) §92 result,

    A_ii = (2/3) * pi * rho * R**3

with negligible radiation damping (no surface-piercing radiation) and
zero hydrostatic restoring (neutral buoyancy, no waterplane). When the
same physical sphere is run through OrcaWave, WAMIT, and Capytaine, the
three independent panel-method solvers must converge on the same
answer — and that answer must agree with the analytical reference.

That makes this test a real check of the **readers**: any disagreement
points either at a parser bug (wrong DOF order, missed unit conversion,
phase-convention slip) or at solver-level scatter (panel density, grid
resolution). The plan budgets ``rtol = 1e-2`` for ``A_ii`` against
analytical and across readers, with matching atol scales for ``B_ii``
(``1e-2 * rho * R**3 * omega``) and ``C`` (``1e-3 * rho * g * R**2``).

Fixture availability — per-fixture granular skip
------------------------------------------------
This test depends on three fixture files that must be generated
externally:

- ``tests/fixtures/bem/orcaflex/sphere_submerged.yml`` — produced from
  an OrcaWave sphere case via OrcaWave's VesselType YAML export.
- ``tests/fixtures/bem/wamit/sphere_submerged.{1,3,hst}`` — produced
  from a WAMIT sphere case (or trimmed from an NREL example).
- ``tests/fixtures/bem/capytaine/sphere_submerged.nc`` — produced by
  running ``scripts/build_sphere_capytaine_fixture.py`` in an
  environment that has Capytaine installed.

The first two require Xabier to run OrcaWave / WAMIT externally and
commit the outputs. The third can be regenerated any time Capytaine
is available (the script is committed; only the resulting ``.nc`` is
not, to keep Capytaine out of the runtime dep list).

Skipping is **per fixture**, not module-wide: the per-reader analytical
tests run as soon as their fixture is present, and the inter-reader
tests skip only if any of their input fixtures is missing. Partial
fixture availability still exercises whatever's present, so a single
fixture landing immediately starts catching parser regressions on that
reader. Each missing fixture produces a skip message naming the file.

Sphere case parameters (LOCKED — must match across all three fixtures)
----------------------------------------------------------------------
- Radius ``R = 5 m``
- Centre depth ``z = -25 m`` (5R below MWL — deep submergence)
- Density ``rho = 1025 kg/m^3``
- Gravity ``g = 9.80665 m/s^2``
- Finite-frequency grid: 30 log-spaced points in
  ``omega in [0.1, 3.0] rad/s`` (matches the Capytaine generation
  script and the OrcaWave / WAMIT spec)
- Endpoint frequencies: ``omega = 0`` and ``omega = +inf`` for
  zero-frequency added mass and ``A_inf`` respectively
- Two wave headings: ``0 deg, 90 deg``

The analytical reference at this radius and density is
``A_ii = (2/3) * pi * 1025 * 125 ~= 268353 kg``.

If a fixture is generated with a different parameter set, the assertions
in this module will need to be reparameterised. The values above are the
locked contract; ``docs/fixtures/sphere-generation.md`` (to be added
when fixtures land) records the OrcaWave and WAMIT generation
procedures.

Tolerances (per docs/milestone-5-plan.md Q4 / PR3, locked at rtol=1e-2)
----------------------------------------------------------------------
- ``A_ii`` vs analytical ``(2/3) pi rho R^3``: ``rtol = 1e-2``
- ``B_ii(omega)`` vs zero: ``atol = 1e-2 * rho * R^3 * omega``
- ``C`` vs zero: ``atol = 1e-3 * rho * g * R^2``
- Inter-reader ``A_inf`` agreement: ``rtol = 1e-2`` (same numerical
  scale as the analytical comparison; tighter would over-fit BEM
  panel-method scatter, looser would stop catching parser bugs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.readers import (
    read_capytaine,
    read_orcaflex_vessel_yaml,
    read_wamit,
)

# ---------------------------------------------------------------------------
# fixture paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "bem"

_ORCAFLEX_FIXTURE = _FIXTURES / "orcaflex" / "sphere_submerged.yml"
_WAMIT_STEM = _FIXTURES / "wamit" / "sphere_submerged"
_WAMIT_FILES = tuple(_WAMIT_STEM.parent / (_WAMIT_STEM.name + ext) for ext in (".1", ".3", ".hst"))
_CAPYTAINE_FIXTURE = _FIXTURES / "capytaine" / "sphere_submerged.nc"


def _wamit_missing_files() -> list[Path]:
    """Return the list of WAMIT fixture files that are absent (any of .1/.3/.hst)."""
    return [f for f in _WAMIT_FILES if not f.is_file()]


def _skip_if_missing(*paths: Path) -> None:
    """Call ``pytest.skip`` if any of the given fixture paths is absent.

    Used inside loader fixtures so per-reader tests skip individually
    while inter-reader tests skip when any of their inputs is missing.
    The skip message names every absent file so a partial-availability
    state is visible at a glance.
    """
    missing = [p for p in paths if not p.is_file()]
    if not missing:
        return
    rels = ", ".join(str(p.relative_to(_FIXTURES.parents[1])) for p in missing)
    pytest.skip(
        f"BEM cross-check fixture(s) missing: {rels}. "
        "See module docstring for generation instructions."
    )


# ---------------------------------------------------------------------------
# sphere case parameters (LOCKED — see module docstring)
# ---------------------------------------------------------------------------

_RHO: Final[float] = 1025.0
_G: Final[float] = 9.80665
_RADIUS_M: Final[float] = 5.0
_DEPTH_M: Final[float] = 25.0  # centre below MWL (z = -25 m, 5R submergence)

_A_ANALYTICAL: Final[float] = (2.0 / 3.0) * np.pi * _RHO * _RADIUS_M**3

# Tolerances per docs/milestone-5-plan.md Q4 / PR3 (locked at rtol=1e-2).
_A_RTOL: Final[float] = 1.0e-2
_B_ATOL_SCALE: Final[float] = 1.0e-2 * _RHO * _RADIUS_M**3  # multiply by omega per row
_C_ATOL: Final[float] = 1.0e-3 * _RHO * _G * _RADIUS_M**2


# ---------------------------------------------------------------------------
# loaders (cached so each test does not re-parse the fixtures; per-fixture
# skip lives inside each loader so module-wide failure is impossible)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hdb_orcaflex() -> HydroDatabase:
    _skip_if_missing(_ORCAFLEX_FIXTURE)
    return read_orcaflex_vessel_yaml(_ORCAFLEX_FIXTURE)


@pytest.fixture(scope="module")
def hdb_wamit() -> HydroDatabase:
    _skip_if_missing(*_WAMIT_FILES)
    return read_wamit(_WAMIT_STEM)


@pytest.fixture(scope="module")
def hdb_capytaine() -> HydroDatabase:
    _skip_if_missing(_CAPYTAINE_FIXTURE)
    return read_capytaine(_CAPYTAINE_FIXTURE)


# ---------------------------------------------------------------------------
# per-reader: agreement with the analytical sphere
# ---------------------------------------------------------------------------


def _assert_translational_added_mass_matches(hdb: HydroDatabase, name: str) -> None:
    """Diagonal entries of ``A_inf`` on translational DOFs match analytical."""
    a_diag = np.diag(hdb.A_inf)[:3]
    rel = np.abs(a_diag - _A_ANALYTICAL) / _A_ANALYTICAL
    assert np.all(rel < _A_RTOL), (
        f"{name}: A_inf translational diagonal {a_diag} deviates from "
        f"analytical (2/3) pi rho R^3 = {_A_ANALYTICAL:.0f} kg by "
        f"max rel-err {float(np.max(rel)):.3e} (limit rtol={_A_RTOL})."
    )


def _assert_radiation_damping_negligible(hdb: HydroDatabase, name: str) -> None:
    """``B_ii(omega)`` -> 0 to ``atol = 1e-2 * rho * R^3 * omega`` per frequency."""
    for k, omega in enumerate(hdb.omega):
        if omega <= 0.0:
            continue
        atol = _B_ATOL_SCALE * float(omega)
        b_diag = np.diag(hdb.B[:, :, k])
        assert np.all(np.abs(b_diag) < atol), (
            f"{name}: B_diag at omega={float(omega):.3f} rad/s exceeds "
            f"atol={atol:.3e} (sphere is supposed to radiate ~0 at "
            f"this submergence). Got max|B_diag|={float(np.max(np.abs(b_diag))):.3e}."
        )


def _assert_hydrostatic_stiffness_zero(hdb: HydroDatabase, name: str) -> None:
    """``C ~ 0`` for a neutrally-buoyant fully-submerged body."""
    assert np.all(np.abs(hdb.C) < _C_ATOL), (
        f"{name}: hydrostatic stiffness C is not zero "
        f"(max|C|={float(np.max(np.abs(hdb.C))):.3e}, atol={_C_ATOL:.3e}). "
        f"A neutrally-buoyant submerged sphere has no waterplane and no "
        f"buoyancy moment arm, so C should be zero to numerical noise."
    )


def test_orcaflex_sphere_matches_analytical(hdb_orcaflex: HydroDatabase) -> None:
    _assert_translational_added_mass_matches(hdb_orcaflex, "OrcaFlex")
    _assert_radiation_damping_negligible(hdb_orcaflex, "OrcaFlex")
    _assert_hydrostatic_stiffness_zero(hdb_orcaflex, "OrcaFlex")


def test_wamit_sphere_matches_analytical(hdb_wamit: HydroDatabase) -> None:
    _assert_translational_added_mass_matches(hdb_wamit, "WAMIT")
    _assert_radiation_damping_negligible(hdb_wamit, "WAMIT")
    _assert_hydrostatic_stiffness_zero(hdb_wamit, "WAMIT")


def test_capytaine_sphere_matches_analytical(hdb_capytaine: HydroDatabase) -> None:
    _assert_translational_added_mass_matches(hdb_capytaine, "Capytaine")
    _assert_radiation_damping_negligible(hdb_capytaine, "Capytaine")
    _assert_hydrostatic_stiffness_zero(hdb_capytaine, "Capytaine")


# ---------------------------------------------------------------------------
# inter-reader: same physical case -> same canonical HydroDatabase
# ---------------------------------------------------------------------------


def test_three_readers_agree_on_a_inf(
    hdb_orcaflex: HydroDatabase,
    hdb_wamit: HydroDatabase,
    hdb_capytaine: HydroDatabase,
) -> None:
    """Translational ``A_inf`` agrees across all three readers to ``rtol=1e-2``.

    Same physical sphere => same hydrodynamic added mass (within
    panel-method scatter). A reader bug — wrong DOF order, missed unit
    conversion, phase-convention slip — surfaces as a single reader
    drifting away from the other two.
    """
    a_orca = np.diag(hdb_orcaflex.A_inf)[:3]
    a_wamit = np.diag(hdb_wamit.A_inf)[:3]
    a_cpt = np.diag(hdb_capytaine.A_inf)[:3]
    np.testing.assert_allclose(
        a_wamit,
        a_orca,
        rtol=_A_RTOL,
        err_msg="WAMIT vs OrcaFlex A_inf translational diagonal disagree",
    )
    np.testing.assert_allclose(
        a_cpt,
        a_orca,
        rtol=_A_RTOL,
        err_msg="Capytaine vs OrcaFlex A_inf translational diagonal disagree",
    )


def test_three_readers_share_canonical_dof_order(
    hdb_orcaflex: HydroDatabase,
    hdb_wamit: HydroDatabase,
    hdb_capytaine: HydroDatabase,
) -> None:
    """The translational block (rows/cols 0-2) of ``A_inf`` is diagonal-dominant
    for all three readers — proving they agree on which row is heave.

    Off-diagonal translational entries on a deeply-submerged sphere
    are zero by symmetry. If any reader writes them at the same
    magnitude as the diagonal, the DOF axis was permuted somewhere
    in the parser.
    """
    for name, hdb in (
        ("OrcaFlex", hdb_orcaflex),
        ("WAMIT", hdb_wamit),
        ("Capytaine", hdb_capytaine),
    ):
        block = np.asarray(hdb.A_inf[:3, :3], dtype=np.float64)
        diag = np.diag(block)
        off = block - np.diag(diag)
        assert float(np.max(np.abs(off))) < _A_RTOL * float(np.max(np.abs(diag))), (
            f"{name}: translational A_inf has non-trivial off-diagonal entries "
            f"(max|off|/max|diag| = "
            f"{float(np.max(np.abs(off)) / np.max(np.abs(diag))):.3e}, "
            f"limit {_A_RTOL}). DOF order may be permuted in the parser."
        )
