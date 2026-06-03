"""M6 PR5 -- S4 moored static equilibrium cross-check (FloatSim vs OpenFAST + MoorDyn).

Validates FloatSim's analytic catenary mooring against OpenFAST's
MoorDyn-converged steady state. Quasi-static comparison only --
F-DAMP-MATCH (Item 26) means time-domain coupled comparison is
ill-defined here, but the converged steady state IS well-defined
and is what PR5 cross-checks.

Scope at PR5 (P-equivalent, locked plan)
----------------------------------------
Cross-check FloatSim's moored static equilibrium (heave, surge,
fairlead tensions) against OpenFAST's MoorDyn-converged steady
state. Pre-flight diagnostics validated the prediction at
sub-0.15 % rel-err on tensions and sub-cm on heave -- the
strongest M6 PR pre-flight result. Step D consumes the
pre-flight calibration and asserts at planned (Q4) tolerances.

Pre-flight findings (documented in scripts/m6_pr5_mooring_prediction.py)
------------------------------------------------------------------------
1. **Submerged line weight (Item 32)**: MoorDyn's MassDen is the
   AIR mass per unit length, not the submerged weight. Submerged
   w = (m_air - rho_water * A_cross) * g; for OC4's 76.6mm chain,
   this is 4.2 % less than the naive m_air * g.
2. **Surge averaging window (Item 33)**: OC4 moored surge has a
   ~ 100 s natural period with very slow damping (radiation +
   MoorDyn line drag only); a 30-s OF averaging window samples
   one half-cycle and is biased by oscillation phase. Use the
   last 200 s for surge (covers 2 full periods).
3. **S4 fixture TMax bumped from 200 s to 1200 s** (R1b) so the
   slow surge mode has time to settle to << 1m amplitude.

Setup
-----
- OC4 platform: M_TOTAL = 1.4074e7 kg (Setup B combined-deck mass,
  matching PR3/PR4/PR6), C_33 = 3.836e6 N/m (Robertson),
  PtfmVol0 = 13917 m³ (HydroDyn deck)
- 3 catenary mooring lines (D=76.6 mm chain, L=835.35 m,
  m_air=113.35 kg/m, EA=7.536e8 N), anchors at 837 m radius at
  120° spacing, fairleads at 40.8 m radius (body-fixed)
- Seabed at z=-200 m (anchor depth)
- Still water (WaveMod=0), all DOFs free in OF (PR5 method)
- FloatSim: analytic catenary solve per
  ``floatsim.mooring.catenary_analytic.solve_catenary`` +
  iterative heave-equilibrium close

Assertion structure (6 assertions)
-----------------------------------
1. solver converges on moored equilibrium (sanity)
2. heave_eq within atol=5cm of OF last-30s mean
3. surge_eq within atol=10cm of OF last-200s mean
4-6. fairlead tensions line 1/2/3 within rtol=5e-2 of OF
     last-30s means

Anchor tensions are computed and logged in the diagnostic output
but NOT asserted (over-constraint; coupled to fair tensions via
catenary mechanics).

Per Decision B discipline: any failure pauses for diagnosis, not
silent xfail.

Tolerance discipline (Item 16 + Q4)
------------------------------------
- Tolerances at planned values (5cm / 10cm / 5e-2), NOT tightened
  to the measured 0.15 % on tensions. Tight-headroom signal is
  preserved as test diagnostic output (visible in -v mode).

Inherits
--------
- Item 13: tolerances accommodate residual oscillation (heave
  clean at 30s; surge needs 200s window per Item 33)
- Item 16: damping tolerance depends on dissipation regime --
  here in the quasi-static regime, dissipation is irrelevant
- Item 26: MoorDyn vs analytic catenary -- F-DAMP-MATCH; time-
  domain comparison NOT done; only the steady state
- Item 31: MoorDyn FairTen/AnchTen conventions
- Item 32: submerged line weight correction (PR5 finding)
- Item 33: surge averaging window discipline (PR5 finding)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from floatsim.mooring.catenary_analytic import (
    CatenaryAttachment,
    CatenaryLine,
    CatenarySolution,
    make_catenary_state_force,
    solve_catenary,
)
from tests.support.openfast_csv import load_openfast_history

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_S4_DECK_DIR: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "openfast"
    / "oc4_deepcwind"
    / "inputs"
    / "s4_moored_eq"
)

# ---------------------------------------------------------------------------
# OC4 mooring geometry (from s4_moored_eq_MoorDyn.dat)
# ---------------------------------------------------------------------------

_RHO_KG_M3: Final[float] = 1025.0
_G_M_S2: Final[float] = 9.80665

# Line properties: 76.6mm chain, 835.35m unstretched, EA = 7.536e8 N.
# Submerged weight per Item 32: w_sub = (m_air - rho * A_cross) * g.
_LINE_DIAM_M: Final[float] = 0.0766
_LINE_MASS_AIR_KG_PER_M: Final[float] = 113.35
_LINE_A_CROSS_M2: Final[float] = float(np.pi * _LINE_DIAM_M**2 / 4.0)
_LINE_W_SUB_N_PER_M: Final[float] = (
    _LINE_MASS_AIR_KG_PER_M - _RHO_KG_M3 * _LINE_A_CROSS_M2
) * _G_M_S2

_LINE_PROPS: Final[CatenaryLine] = CatenaryLine(
    length=835.35,
    weight_per_length=_LINE_W_SUB_N_PER_M,
    EA=7.536e8,
)
_SEABED_DEPTH_M: Final[float] = 200.0

# Anchor positions (Fixed, MoorDyn POINTS section IDs 1-3).
_ANCHORS_3D: Final[NDArray[np.float64]] = np.array(
    [
        [+418.80, +725.38, -200.0],
        [-837.60, 0.00, -200.0],
        [+418.80, -725.38, -200.0],
    ],
    dtype=np.float64,
)

# Fairlead positions on vessel body (body-frame offsets, IDs 4-6).
_FAIRLEADS_BODY: Final[NDArray[np.float64]] = np.array(
    [
        [+20.43, +35.39, -14.0],
        [-40.87, 0.00, -14.0],
        [+20.43, -35.39, -14.0],
    ],
    dtype=np.float64,
)

# Platform properties.
_M_TOTAL_KG: Final[float] = 1.4074e7  # Setup B combined-deck mass
_C33_HEAVE_N_PER_M: Final[float] = 3.836e6
_PTFM_VOL0_M3: Final[float] = 13917.0

# Tolerances per Q4 + Item 13.
_HEAVE_ATOL_M: Final[float] = 0.05  # 5 cm
_SURGE_ATOL_M: Final[float] = 0.10  # 10 cm (looser per residual surge oscillation)
_TENSION_RTOL: Final[float] = 5.0e-2

# OF averaging windows.
_HEAVE_AVG_WINDOW_S: Final[float] = 30.0  # heave settles cleanly
_SURGE_AVG_WINDOW_S: Final[float] = 200.0  # Item 33
_TENSION_AVG_WINDOW_S: Final[float] = 30.0


# ---------------------------------------------------------------------------
# Per-line catenary solver (3D → 2D rotation, then back)
# ---------------------------------------------------------------------------


def _solve_line_at_body_offset(
    line_idx: int,
    surge_m: float,
    heave_m: float,
) -> tuple[CatenarySolution, float]:
    """Solve catenary for line ``line_idx`` (0/1/2) with body offset.

    Returns (CatenarySolution, azimuth_deg).
    """
    anchor = _ANCHORS_3D[line_idx]
    fairlead = _FAIRLEADS_BODY[line_idx] + np.array([surge_m, 0.0, heave_m])
    dxy = anchor[:2] - fairlead[:2]
    horizontal_span = float(np.hypot(dxy[0], dxy[1]))
    azimuth_rad = float(np.arctan2(dxy[1], dxy[0]))
    anchor_2d = np.array([0.0, float(anchor[2])])
    fairlead_2d = np.array([horizontal_span, float(fairlead[2])])
    sol = solve_catenary(
        line=_LINE_PROPS,
        anchor_pos=anchor_2d,
        fairlead_pos=fairlead_2d,
        seabed_depth=_SEABED_DEPTH_M,
    )
    return sol, float(np.degrees(azimuth_rad))


# ---------------------------------------------------------------------------
# F3 composer wiring (M7-Foundation PR3 refactor)
# ---------------------------------------------------------------------------
#
# Post-M7-Foundation PR3 (commit TBD), the catenary 6-vector force on
# body 0 is computed by floatsim.mooring.catenary_analytic.make_catenary_state_force
# rather than summed by hand from per-line CatenarySolution objects. The
# composer's force agrees with the prior hand-wired path at rtol = 1e-12
# (pinned by tests/unit/test_catenary_state_force.py at the M6 PR5
# geometry; see also scripts/m7_pr3_catenary_prediction.py for the
# Step A hand-derived 6-vector targets).
#
# Per-line CatenarySolution objects are still needed for the tension
# assertions (T_fairlead per line). The composer returns the resultant
# generalised force on the body; it does NOT decompose into per-line
# tensions. _solve_line_at_body_offset is preserved for that purpose.


def _build_oc4_attachments() -> list[CatenaryAttachment]:
    """Build the 3 OC4 CatenaryAttachment instances from the locked geometry."""
    return [
        CatenaryAttachment(
            body_index=0,
            fairlead_body=_FAIRLEADS_BODY[i].copy(),
            anchor_global=_ANCHORS_3D[i].copy(),
            line=_LINE_PROPS,
            seabed_depth=_SEABED_DEPTH_M,
        )
        for i in range(3)
    ]


# Module-level cached composer (single body, n_dof = 6).
_CATENARY_STATE_FORCE: Final = make_catenary_state_force(_build_oc4_attachments(), n_dof=6)


def _net_z_force_on_body(
    heave_m: float, surge_m: float = 0.0
) -> tuple[float, list[CatenarySolution], list[float]]:
    """Net vertical force on body at trial (surge, heave).

    F_z = rho * V_0 * g - C_33 z - m g + F_mooring_z

    F_mooring_z is read from the composer's 6-vector (negative because
    the lines pull body DOWN at the fairleads). The per-line
    CatenarySolution list is still produced for downstream tension
    assertions; it is NOT used in the Newton residual itself, which
    flows through the composer to exercise the F3 code path the
    integrator will use in dynamic runs.
    """
    xi = np.array([surge_m, 0.0, heave_m, 0.0, 0.0, 0.0], dtype=np.float64)
    F_6 = _CATENARY_STATE_FORCE(0.0, xi, np.zeros(6))
    F_mooring_z = float(F_6[2])

    # Per-line solutions for the eventual tension assertions.
    sols: list[CatenarySolution] = []
    azimuths: list[float] = []
    for i in range(3):
        sol, az = _solve_line_at_body_offset(i, surge_m, heave_m)
        sols.append(sol)
        azimuths.append(az)

    F_buoyancy = _RHO_KG_M3 * _PTFM_VOL0_M3 * _G_M_S2 - _C33_HEAVE_N_PER_M * heave_m
    F_weight = -_M_TOTAL_KG * _G_M_S2
    F_net = F_buoyancy + F_weight + F_mooring_z
    return F_net, sols, azimuths


def _solve_heave_equilibrium(
    tol_n: float = 1.0e2, max_iter: int = 50
) -> tuple[float, list[CatenarySolution], list[float], bool]:
    """Newton iterate on heave until |F_z_net| < tol_n.

    Returns (heave_eq, per_line_solutions, per_line_azimuths_deg,
    converged).
    """
    heave = 0.0
    for _ in range(max_iter):
        F_net, sols, azimuths = _net_z_force_on_body(heave)
        if abs(F_net) < tol_n:
            return heave, sols, azimuths, True
        dz = F_net / _C33_HEAVE_N_PER_M
        heave += dz
    return heave, sols, azimuths, False


# ---------------------------------------------------------------------------
# OpenFAST reference (lazy-loaded)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openfast_s4_eq() -> dict[str, float]:
    """Load OF S4 CSV and compute the cross-check reference means."""
    csv = next(_S4_DECK_DIR.glob("*.csv"))
    h = load_openfast_history(csv)
    t = h.t
    heave_mask = t >= t[-1] - _HEAVE_AVG_WINDOW_S
    surge_mask = t >= t[-1] - _SURGE_AVG_WINDOW_S
    ten_mask = t >= t[-1] - _TENSION_AVG_WINDOW_S

    refs: dict[str, float] = {
        "heave_mean": float(np.mean(h.xi[heave_mask, 2])),
        "heave_std": float(np.std(h.xi[heave_mask, 2])),
        "surge_mean": float(np.mean(h.xi[surge_mask, 0])),
        "surge_std": float(np.std(h.xi[surge_mask, 0])),
        "t_max": float(t[-1]),
    }
    for i in (1, 2, 3):
        f_ch = f"fair_ten_line{i}_n"
        a_ch = f"anch_ten_line{i}_n"
        if f_ch in h.extra_columns:
            x = h.extra_columns[f_ch]
            refs[f"fair_ten_line{i}_mean"] = float(np.mean(x[ten_mask]))
            refs[f"fair_ten_line{i}_std"] = float(np.std(x[ten_mask]))
        if a_ch in h.extra_columns:
            x = h.extra_columns[a_ch]
            refs[f"anch_ten_line{i}_mean"] = float(np.mean(x[ten_mask]))
    return refs


# ---------------------------------------------------------------------------
# FloatSim prediction (lazy-computed)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def floatsim_s4_prediction() -> dict[str, object]:
    """Solve FloatSim moored equilibrium + per-line tensions."""
    heave_eq, sols, azimuths, converged = _solve_heave_equilibrium()
    out: dict[str, object] = {
        "heave_eq": heave_eq,
        "surge_eq": 0.0,  # by 3-fold symmetry; PR5 doesn't surge-iterate
        "converged": converged,
        "solutions": sols,
        "azimuths_deg": azimuths,
    }
    for i, sol in enumerate(sols, start=1):
        T_F = sol.T_fairlead
        T_A = sol.H if sol.regime == "touchdown" else float(np.hypot(sol.H, sol.V_anchor))
        out[f"fair_ten_line{i}_pred"] = T_F
        out[f"anch_ten_line{i}_pred"] = T_A
        out[f"line{i}_regime"] = sol.regime
    return out


# ---------------------------------------------------------------------------
# Assertions (6 total per locked plan)
# ---------------------------------------------------------------------------


def test_floatsim_solver_converges_on_moored_equilibrium(
    floatsim_s4_prediction: dict[str, object],
) -> None:
    """FloatSim's iterative heave-equilibrium close converges (Newton on
    F_net = rho * V_0 * g - C_33 z - m g - Σ V_F).
    """
    assert floatsim_s4_prediction["converged"] is True, (
        "FloatSim moored-equilibrium solver failed to converge. "
        f"Last heave = {floatsim_s4_prediction['heave_eq']} m. "
        "Check catenary line properties + platform mass + buoyancy "
        "reference (PtfmVol0)."
    )
    # All 3 lines should be in touchdown regime for OC4's 17m-slack
    # configuration. A "suspended" result would indicate an arithmetic
    # error in line geometry.
    for i in (1, 2, 3):
        assert floatsim_s4_prediction[f"line{i}_regime"] == "touchdown", (
            f"line {i} regime = {floatsim_s4_prediction[f'line{i}_regime']!r}; "
            "expected 'touchdown'. Verify line length vs anchor-fairlead "
            "geometry."
        )


def test_heave_equilibrium_matches_openfast(
    openfast_s4_eq: dict[str, float],
    floatsim_s4_prediction: dict[str, object],
) -> None:
    """Heave equilibrium agrees with OF last-30s mean within atol=5cm."""
    fs = float(floatsim_s4_prediction["heave_eq"])
    of = openfast_s4_eq["heave_mean"]
    delta = abs(fs - of)
    print(
        f"\nDIAGNOSTIC heave: FS = {fs:+.5f} m, OF = {of:+.5f} m, "
        f"|delta| = {delta*100:.3f} cm (gate = {_HEAVE_ATOL_M*100:.0f} cm)"
    )
    assert delta < _HEAVE_ATOL_M, (
        f"heave equilibrium: FS = {fs:+.5f} m vs OF = {of:+.5f} m; "
        f"|delta| = {delta*100:.3f} cm exceeds atol = {_HEAVE_ATOL_M*100:.0f} cm. "
        "Per Decision B: pause for diagnosis. Pre-flight measured "
        "|delta| = 0.27 cm; large deviation suggests a regression in "
        "the catenary solver, line properties, or platform buoyancy."
    )


def test_surge_equilibrium_matches_openfast(
    openfast_s4_eq: dict[str, float],
    floatsim_s4_prediction: dict[str, object],
) -> None:
    """Surge equilibrium agrees with OF last-200s mean within atol=10cm.

    OF averaging window per Item 33: surge has ~ 100s natural period
    with slow damping; need >= 2 periods to wash out oscillation phase.
    """
    fs = float(floatsim_s4_prediction["surge_eq"])
    of = openfast_s4_eq["surge_mean"]
    delta = abs(fs - of)
    print(
        f"\nDIAGNOSTIC surge: FS = {fs:+.5f} m, OF = {of:+.5f} m, "
        f"|delta| = {delta*100:.3f} cm (gate = {_SURGE_ATOL_M*100:.0f} cm; "
        f"OF window = last {_SURGE_AVG_WINDOW_S:.0f}s, std = "
        f"{openfast_s4_eq['surge_std']*100:.2f} cm)"
    )
    assert delta < _SURGE_ATOL_M, (
        f"surge equilibrium: FS = {fs:+.5f} m vs OF = {of:+.5f} m "
        f"(last {_SURGE_AVG_WINDOW_S:.0f}s window); |delta| = "
        f"{delta*100:.3f} cm exceeds atol = {_SURGE_ATOL_M*100:.0f} cm."
    )


@pytest.mark.parametrize("line", [1, 2, 3])
def test_fair_tension_matches_openfast(
    line: int,
    openfast_s4_eq: dict[str, float],
    floatsim_s4_prediction: dict[str, object],
) -> None:
    """Fairlead tension per line agrees with OF last-30s mean within
    rtol = 5e-2 (Q4). Anchor tensions logged but not asserted (coupled
    to fair tension via catenary mechanics).
    """
    fs = float(floatsim_s4_prediction[f"fair_ten_line{line}_pred"])
    of = openfast_s4_eq[f"fair_ten_line{line}_mean"]
    of_std = openfast_s4_eq[f"fair_ten_line{line}_std"]
    rel = abs(fs - of) / of
    # Diagnostic: include anchor tension comparison too.
    fs_anch = float(floatsim_s4_prediction[f"anch_ten_line{line}_pred"])
    of_anch = openfast_s4_eq[f"anch_ten_line{line}_mean"]
    rel_anch = abs(fs_anch - of_anch) / of_anch
    print(
        f"\nDIAGNOSTIC line {line} FairTen: FS = {fs:.4e}, OF = {of:.4e} "
        f"(std/mean = {of_std/of*100:.2f}%), rel-err = {rel*100:+.3f}% "
        f"(gate = {_TENSION_RTOL*100:.0f}%)"
        f"\nDIAGNOSTIC line {line} AnchTen: FS = {fs_anch:.4e}, "
        f"OF = {of_anch:.4e}, rel-err = {rel_anch*100:+.3f}% (informational)"
    )
    assert rel < _TENSION_RTOL, (
        f"line {line} FairTen: FS = {fs:.4e} N vs OF = {of:.4e} N; "
        f"rel-err = {rel*100:+.3f}% exceeds rtol = {_TENSION_RTOL*100:.0f}%. "
        "Per Decision B: pause for diagnosis. Pre-flight measured "
        "rel-err ~ 0.1%; large deviation suggests a regression in the "
        "catenary solver or line submerged-weight bookkeeping (Item 32)."
    )
