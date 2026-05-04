"""M6 PR2 -- S1 static-equilibrium cross-check (OpenFAST vs FloatSim).

The first OpenFAST cross-check assertion fires here. Following Path
IV.b of Xabier's M6 PR2 review (locked 2026-05-04), this test
**reframes** S1 from "validate equilibrium agreement" to "validate
linear response to deck residual":

- FloatSim's :func:`floatsim.solver.equilibrium.solve_static_equilibrium`
  returns ``xi=0`` for any deck because equilibrium IS the
  linearisation point of the Cummins formulation
  (``docs/openfast-cross-check-conventions.md`` Item 15).
- OpenFAST's nonlinear solver settles into a ~0.488 m heave offset
  whenever the deck's total mass and displaced volume don't exactly
  balance at the BEM reference; that offset is a deck-bookkeeping
  artifact, not physics disagreement.
- The cross-check that *is* physics-meaningful: compute the
  net residual force from OpenFAST's input files, apply it to
  FloatSim's solver as ``F_external``, and assert the resulting
  displacement matches OpenFAST's last-30-s mean.

That last assertion validates, in one shot:
- Parsing of OpenFAST input files (mass tables, distributed
  station tables, named scalars).
- The deck-mass aggregation chain (platform + ballast + tower +
  RNA + blades, with the right CoG bookkeeping).
- The buoyancy calculation (rho * V0 * g at the BEM reference).
- The Cummins linearisation (equilibrium IS the linearisation
  point; non-zero ``F_external`` produces displacement
  proportional to ``C^-1 F_external``).
- The hydrostatic-gravity decomposition surfaced by the
  PR1 audit (Items 5 + 14).

DOFs asserted: HEAVE, ROLL, PITCH only (Item 14)
------------------------------------------------
OC4 unmoored has zero hydrostatic stiffness on surge / sway / yaw.
Asserting equilibrium agreement on those would compare FloatSim's
``xi_eq=0`` (regularised) against OpenFAST's slow numerical drift
(~ -0.6 mm/s on surge over 600 s, accumulating to -0.34 m). The
assertions in this module iterate only over ``[heave, roll, pitch]``;
surge / sway / yaw are skipped per Item 14. They will be cross-checked
in S3 (wave-excited) and S4 (mooring-restored).

PITCH borderline note
---------------------
For OC4, pitch is borderline because the M2 fixture's
``platform_small.yml`` was authored against the Robertson 2014
"platform-only" mass (1.347e7 kg, z_G=-13.46 m), whereas OpenFAST's
S1 deck combines platform + tower + RNA at a much shallower combined
CoG (z_G≈-5.7 m). The pitch restoring stiffness is dominated by
``-m_total * g * z_G_combined``, which differs between the two
deck-mass conventions. Pitch passes the ±0.5° tolerance only because
both predictions and observed are small (< 0.5°) — the assertion is
on AGREEMENT, not on the precise value. A future deck-identity
refinement (re-construct C_55 for the OpenFAST mass distribution)
would tighten this; for PR2 we accept the boundary.

Reference values from the committed S1 CSV (TMax=600, last-30-s mean):
- heave = 0.4882 m (std 0.0552 m)
- roll  = -0.0000° (std 0.0005°)
- pitch = -0.0814° (std 0.0094°)

Tolerances per Item 13:
- heave: ``atol = 0.15 m``
- roll:  ``atol = 0.5°``
- pitch: ``atol = 0.5°``
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.readers.orcaflex_vessel_yaml import read_orcaflex_vessel_yaml
from floatsim.solver.equilibrium import solve_static_equilibrium
from tests.support.openfast_csv import load_openfast_history
from tests.support.openfast_deck import compute_openfast_deck_residual
from tests.validation.test_oc4_natural_periods import _oc4_rigid_body_mass_matrix

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURE_ROOT: Final[Path] = Path(__file__).resolve().parents[1] / "fixtures"
_OPENFAST_S1: Final[Path] = _FIXTURE_ROOT / "openfast" / "oc4_deepcwind" / "inputs" / "s1_static_eq"
_PLATFORM_SMALL: Final[Path] = _FIXTURE_ROOT / "bem" / "orcaflex" / "platform_small.yml"

# ---------------------------------------------------------------------------
# Per-DOF tolerances locked by docs/openfast-cross-check-conventions.md Item 13
# ---------------------------------------------------------------------------

_HEAVE_ATOL_M: Final[float] = 0.15
_ROLL_ATOL_RAD: Final[float] = np.deg2rad(0.5)
_PITCH_ATOL_RAD: Final[float] = np.deg2rad(0.5)

# Index map for ξ = (surge, sway, heave, roll, pitch, yaw)
_HEAVE: Final[int] = 2
_ROLL: Final[int] = 3
_PITCH: Final[int] = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_30s_mean(t: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    """Return the last-30-s mean of ``x`` per Item 12."""
    mask = t >= t[-1] - 30.0
    return float(np.mean(x[mask]))


def _make_state_force(
    F_external: NDArray[np.float64],
):
    """Wrap a constant 6-vector as the ``state_force`` callable signature."""

    def _f(
        _t: float, _xi: NDArray[np.float64], _xi_dot: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return F_external

    return _f


# ---------------------------------------------------------------------------
# Cached setup (one solver run per pytest session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def s1_reference() -> dict[str, float]:
    """OpenFAST S1 reference values: last-30-s mean of each DOF."""
    history = load_openfast_history(_OPENFAST_S1 / "s1_static_eq.csv")
    return {
        "heave_m": _last_30s_mean(history.t, history.xi[:, _HEAVE]),
        "roll_rad": _last_30s_mean(history.t, history.xi[:, _ROLL]),
        "pitch_rad": _last_30s_mean(history.t, history.xi[:, _PITCH]),
        "duration_s": float(history.duration_s),
        "n_samples": int(history.n_samples),
    }


@pytest.fixture(scope="module")
def floatsim_static_eq() -> dict[str, float]:
    """Run FloatSim's static-equilibrium solver with the OpenFAST deck residual."""
    # Load the OC4-shaped Cummins LHS via the M2 fixture (full-restoring
    # OrcaFlex VesselType). This carries the BEM-derived A_inf and C
    # for the OC4 platform.
    hdb = read_orcaflex_vessel_yaml(_PLATFORM_SMALL)
    M_rigid = _oc4_rigid_body_mass_matrix()
    lhs = assemble_cummins_lhs(rigid_body_mass=M_rigid, hdb=hdb)

    # Compute OpenFAST's deck residual from the S1 input files.
    residual = compute_openfast_deck_residual(_OPENFAST_S1)

    # Solve C xi = F_residual.
    result = solve_static_equilibrium(
        lhs=lhs,
        state_force=_make_state_force(residual.F_residual),
        xi0=np.zeros(6),
    )
    return {
        "xi_eq": result.xi_eq,
        "converged": result.converged,
        "residual_norm": result.residual_norm,
        "iterations": result.iterations,
        "F_residual": residual.F_residual,
        "m_total_kg": residual.m_total_kg,
        "buoyancy_n": residual.buoyancy_n,
        "weight_n": residual.weight_n,
    }


# ---------------------------------------------------------------------------
# The cross-check assertions
# ---------------------------------------------------------------------------


def test_solver_converges(floatsim_static_eq: dict[str, float]) -> None:
    """FloatSim's solver must converge with the deck residual applied.

    This is a precondition: failure here means the test below cannot
    distinguish a physics mismatch from a numerical-solver issue.
    """
    assert floatsim_static_eq["converged"], (
        f"static_equilibrium_solver did not converge "
        f"(iters={floatsim_static_eq['iterations']}, "
        f"|r|_inf={floatsim_static_eq['residual_norm']:.3e})"
    )


def test_zero_external_force_returns_zero_xi() -> None:
    """Sanity (Item 15): with no F_external, xi_eq = 0 by construction.

    Confirms FloatSim's static equilibrium is the linearisation point
    of the Cummins formulation. This is the framing premise of Path
    IV.b -- without it, the residual-as-external-force approach makes
    no sense.
    """
    hdb = read_orcaflex_vessel_yaml(_PLATFORM_SMALL)
    M_rigid = _oc4_rigid_body_mass_matrix()
    lhs = assemble_cummins_lhs(rigid_body_mass=M_rigid, hdb=hdb)
    result = solve_static_equilibrium(lhs=lhs, xi0=np.zeros(6))
    assert result.converged
    assert np.allclose(result.xi_eq, np.zeros(6), atol=1e-9), (
        f"xi_eq with zero F_external = {result.xi_eq}; expected 0 "
        "(equilibrium IS the linearisation point of the Cummins formulation; "
        "see conventions doc Item 15)."
    )


def test_heave_matches_openfast_with_deck_residual(
    floatsim_static_eq: dict[str, float],
    s1_reference: dict[str, float],
) -> None:
    """Heave equilibrium with OpenFAST deck residual matches OpenFAST's
    last-30-s mean within ±0.15 m (Item 13)."""
    xi_eq = floatsim_static_eq["xi_eq"]
    heave_floatsim = float(xi_eq[_HEAVE])
    heave_openfast = s1_reference["heave_m"]
    delta = abs(heave_floatsim - heave_openfast)
    assert delta < _HEAVE_ATOL_M, (
        f"Heave equilibrium disagrees with OpenFAST: "
        f"FloatSim={heave_floatsim:.4f} m, OpenFAST={heave_openfast:.4f} m, "
        f"|delta|={delta:.4f} m, tolerance={_HEAVE_ATOL_M:.3f} m. "
        f"F_residual[heave]={floatsim_static_eq['F_residual'][_HEAVE]:.3e} N "
        f"(buoyancy={floatsim_static_eq['buoyancy_n']:.3e} N - "
        f"weight={floatsim_static_eq['weight_n']:.3e} N = "
        f"{floatsim_static_eq['buoyancy_n'] - floatsim_static_eq['weight_n']:.3e} N). "
        "Likely failure modes: deck-mass parsing in compute_openfast_deck_residual, "
        "C_33 mismatch between platform_small.yml and OpenFAST's WAMIT, or a "
        "regression in the gravity-decomposition path (Item 5)."
    )


def test_roll_matches_openfast_with_deck_residual(
    floatsim_static_eq: dict[str, float],
    s1_reference: dict[str, float],
) -> None:
    """Roll equilibrium agreement within ±0.5° (Item 13).

    OC4 is axisymmetric so OpenFAST's roll is essentially zero
    (~1e-5 rad). FloatSim's prediction is also exactly zero by
    symmetry of the deck-residual computation (no y-axis CoG
    asymmetry). The test asserts both stay within the band of zero.
    """
    roll_floatsim = float(floatsim_static_eq["xi_eq"][_ROLL])
    roll_openfast = s1_reference["roll_rad"]
    delta = abs(roll_floatsim - roll_openfast)
    assert delta < _ROLL_ATOL_RAD, (
        f"Roll disagrees with OpenFAST: "
        f"FloatSim={np.rad2deg(roll_floatsim):.4f}°, "
        f"OpenFAST={np.rad2deg(roll_openfast):.4f}°, "
        f"|delta|={np.rad2deg(delta):.4f}°, "
        f"tolerance={np.rad2deg(_ROLL_ATOL_RAD):.3f}°"
    )


def test_pitch_matches_openfast_with_deck_residual(
    floatsim_static_eq: dict[str, float],
    s1_reference: dict[str, float],
) -> None:
    """Pitch equilibrium agreement within ±0.5° (Item 13).

    Borderline case: OpenFAST settles at -0.08° while FloatSim's
    deck-residual model predicts ~+0.37° (driven by the off-axis
    NacCMxn weight at +1.9 m downwind). The 0.45° magnitude
    difference passes ±0.5° tolerance but flags a deck-identity
    limitation worth noting in the M6 cross-check report:

    - The M2 ``platform_small.yml`` C_55 was authored against the
      Robertson 2014 platform-only mass (z_G=-13.46 m); the gravity
      contribution to pitch stiffness ``-m·g·z_G`` differs from
      OpenFAST's combined-CoG configuration (z_G≈-5.7 m at
      platform+tower+RNA combined CoG).
    - A deck-identity refinement (rebuild C_55 with OpenFAST's
      m_total at the combined CoG) would tighten this. Out of scope
      for PR2.
    """
    pitch_floatsim = float(floatsim_static_eq["xi_eq"][_PITCH])
    pitch_openfast = s1_reference["pitch_rad"]
    delta = abs(pitch_floatsim - pitch_openfast)
    assert delta < _PITCH_ATOL_RAD, (
        f"Pitch disagrees with OpenFAST beyond ±0.5°: "
        f"FloatSim={np.rad2deg(pitch_floatsim):.4f}°, "
        f"OpenFAST={np.rad2deg(pitch_openfast):.4f}°, "
        f"|delta|={np.rad2deg(delta):.4f}°"
    )


# ---------------------------------------------------------------------------
# Diagnostic-only: log surge/sway/yaw values for the cross-check report,
# but do NOT assert on them per Item 14.
# ---------------------------------------------------------------------------


def test_surge_sway_yaw_are_skipped_per_item_14(
    floatsim_static_eq: dict[str, float],
) -> None:
    """Item 14: zero-stiffness DOFs have no defined equilibrium and are
    skipped in static-equilibrium cross-checks. This test confirms
    FloatSim's solver returns near-zero on those DOFs (the regularised
    solution to a rank-deficient system) -- not asserted against
    OpenFAST's drift.
    """
    xi_eq = floatsim_static_eq["xi_eq"]
    # Regularisation pulls unrestored DOFs toward zero. The exact
    # value depends on lambda_reg (default ~1e-8 of max C_diag); we
    # only assert order-of-magnitude smallness.
    for i, name in [(0, "surge"), (1, "sway"), (5, "yaw")]:
        assert abs(float(xi_eq[i])) < 1.0, (
            f"FloatSim's solver returned {name}={float(xi_eq[i]):.3e} "
            "on a zero-stiffness DOF; the regularised solution should "
            "be near zero (see Item 14 + equilibrium.py docstring)."
        )
