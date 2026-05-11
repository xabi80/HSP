"""M6 PR6 -- S5 heave drag decay (hyperbolic envelope cross-check vs OpenFAST).

Validates FloatSim's Morison-drag time-domain pipeline against OpenFAST's
S5 reference using the **hyperbolic-envelope** signature characteristic of
drag-dominated free decay (Faltinsen 1990, Ch. 4). Per conventions doc
Item 16's regime classification, this scenario is the canonical Phase 1
forced-vs-radiation-damping discriminator: heave drag from the OC4
heave plates dominates the radiation damping (which is ζ ~ 0.057 %
radiation-only -- see Item 26 + 27 + the M6 PR4 narrative).

Scope at PR6 (P2 narrowing per the locked plan)
------------------------------------------------
Rather than wiring the full 25-Morison-element + 3-axial-drag-joint
deck identity into FloatSim (which would require parsing the entire
HydroDyn deck plus 3 MoorDyn catenary lines), this PR uses a
**calibrated equivalent Morison element** whose lumped ``Cd · D · L``
matches the aggregated heave drag from the full OC4 deck. Pre-step
diagnostics (``scripts/m6_pr6_drag_aggregation.py``) derived the
aggregate from first principles (using HydroDyn's actual axial-drag
formula per Item 30) and validated against OpenFAST's measured
δ = 0.309 1/m at 1.28 % rel-err. PR6 then uses that aggregate.

The full deck-identity test (with all 25 Morison elements + 3 joints
+ MoorDyn-equivalent mooring) is a future PR; PR6 establishes the
regime classification and the drag mechanism's quantitative agreement.

Aggregate equivalent
--------------------
From ``scripts/m6_pr6_drag_aggregation.py``:

  R_total = 3.4073e+06 kg/m
    cylindrical contribution: 6.87e+04 (2 %, from cross-braces 11-22)
    axial contribution:       3.34e+06 (98 %, from 3 heave plates at z=-20)

  To reproduce in FloatSim's standard Morison module (which applies
  ``F = 0.5 rho Cd D L |v_n| v_n``):
      Cd · D · L = 2 · R_total / rho = 6648 m²

Equivalent element (horizontal cylinder at body reference, body heave
motion perpendicular to axis):
  D = 24 m            (heave plate diameter; physical anchor)
  L = 24 m            (square element)
  Cd = 6648 / (D·L) = 11.54

The Cd > 1 reflects that the equivalent element represents three
heave plates with HydroDyn AxCd = 9.6 (per-face Morison-equivalent
Cd = 4.8 each), aggregated by axial-drag formula factor 1/4 vs the
standard Morison 1/2.

Setup
-----
- Combined-deck rigid mass + Robertson C_33 (Setup B; same as PR3/PR4)
- marin_semi BEM (A_inf_33 = 1.496e7 kg, radiation damping near-zero
  at ω_n_heave = 0.057 % ζ)
- One equivalent Morison element (per above)
- **No mooring** (test isolates the drag mechanism; equilibrium offset
  from OpenFAST's MoorDyn pre-tension is removed by initialising at
  IC = 0.519 m = OF's amplitude-from-equilibrium)
- Calm sea (``WaveMod=0`` in OF deck → u_fluid = 0)
- Heave-only DOF (other DOFs locked via mass-matrix structure;
  initial conditions zero on other DOFs)

Numerical setup
---------------
- ``dt = 0.05 s`` to match OpenFAST sample rate.
- ``rho_inf = 1.0`` (trapezoidal limit; isolates physical hyperbolic
  agreement from numerical damping).
- ``t_max_kernel = 200 s``: clears the M6 PR3 three-check gate (Item 25)
  for marin_semi.
- Heave free-decay e-folding time is ~ 81 min (zeta = 0.057 %); over
  600 s we get ~ 35 peaks within the drag-dominated regime (where
  drag ~v² exceeds radiation ~v).

Fit window
----------
The drag-dominated regime applies where ``Cd·D·L·v² > B·v``, i.e.
where the body's velocity is high enough that quadratic drag
exceeds linear radiation damping. For OC4 S5 this regime spans
peaks 0-15 (amplitudes from 0.44 m down to 0.14 m). Below 0.14 m
the envelope flattens as radiation + mooring (in OF) take over.
PR6 fits peaks 0-15 only -- this is the canonical regime-classification
test window.

Tolerances per Q4
-----------------
- δ agreement: ``rtol = 5e-2`` (5 % per Item 16 regime classification)
- Envelope shape: hyperbolic prediction beats exponential by ≥ 5x
  RMS over peaks 0-15

Per Decision B discipline: any failure pauses for diagnosis, not
silent xfail. The pre-step aggregation already passed at 1.28 %
rel-err; any disagreement in Step D would indicate a FloatSim
pipeline bug (kernel, integrator, Morison force assembly) rather
than the drag aggregation.

Inherits
--------
- Item 16: damping tolerance depends on dissipation regime
- Item 25: three-check kernel gate (t_max=200 s)
- Item 26: MoorDyn vs analytic catenary (here: avoided by no-mooring
  setup, isolating the drag mechanism)
- Item 27: free-decay vs forced-response damping tolerance
- Item 30: HydroDyn JAxCd uses 1/4 factor (calibrated into the
  aggregate via scripts/m6_pr6_drag_aggregation.py)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.morison import MorisonElement, make_morison_state_force
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.readers.wamit import read_added_mass_and_damping
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import integrate_cummins
from tests.support.openfast_csv import load_openfast_history
from tests.validation.test_m6_openfast_free_decay import (
    _MARIN_SEMI as _MARIN_SEMI_PATH,
)
from tests.validation.test_m6_openfast_free_decay import (
    _build_setup_b,
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_S5_DECK_DIR: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "openfast"
    / "oc4_deepcwind"
    / "inputs"
    / "s5_drag_decay"
)

# ---------------------------------------------------------------------------
# Aggregate equivalent Morison element (calibrated in
# scripts/m6_pr6_drag_aggregation.py)
# ---------------------------------------------------------------------------

# R_total = 3.4073e+06 kg/m from the aggregation script (HydroDyn 1/4 factor +
# cylindrical Morison contributions from cross-braces 11-22, summed).
_R_AGGREGATE_KG_PER_M: Final[float] = 3.4073e6

# Equivalent horizontal cylinder: F = 0.5 rho Cd D L v|v|, so Cd·D·L = 2R/rho.
_RHO_KG_PER_M3: Final[float] = 1025.0
_EQUIV_D_M: Final[float] = 24.0  # heave plate diameter (physical anchor)
_EQUIV_L_M: Final[float] = 24.0
_EQUIV_CD: Final[float] = 2.0 * _R_AGGREGATE_KG_PER_M / (_RHO_KG_PER_M3 * _EQUIV_D_M * _EQUIV_L_M)
# ≈ 11.54 (>> 1 -- reflects 3 heave plates with AxCd=9.6 aggregated)

# Integration parameters.
_DT_S: Final[float] = 0.05
_DURATION_S: Final[float] = 600.0
_KERNEL_T_MAX_S: Final[float] = 200.0

# OpenFAST reference (S5 measured per pre-flight diagnostic at
# docs/diagnostics/m6-pr6-drag-aggregation.md).
_OF_DELTA_REFERENCE: Final[float] = 0.3090
_AMP_FROM_EQ_M: Final[float] = 0.519  # OF amplitude-from-equilibrium

# Tolerances per Q4 / Item 16.
_DELTA_RTOL: Final[float] = 5.0e-2
_ENVELOPE_DISCRIM_RATIO: Final[float] = 5.0

# Fit window: peaks 0-15 (drag-dominated; envelope transitions to
# radiation-dominated below ~0.14 m).
_FIT_PEAK_INDEX_MAX: Final[int] = 15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_marin_semi_hdb(C: NDArray[np.float64]) -> HydroDatabase:
    """marin_semi.{1} BEM combined with the caller-supplied hydrostatic C.

    The S5 cross-check does not consume F_exc (calm sea), but the kernel
    construction needs A(omega), B(omega), A_inf from marin_semi.1.
    """
    omega, A, B, A_inf = read_added_mass_and_damping(_MARIN_SEMI_PATH)
    n_w = omega.size
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0]),
        A=A,
        B=B,
        A_inf=A_inf,
        C=C,
        RAO=np.zeros((6, n_w, 1), dtype=np.complex128),
        reference_point=np.array([0.0, 0.0, 0.0]),
        C_source="full",
    )


def _positive_peaks(
    t: NDArray[np.float64], x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find positive local maxima (peaks above zero)."""
    is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0)
    idx = np.where(is_peak)[0] + 1
    return t[idx], x[idx]


def _calibrate_delta_hyperbolic(peaks: NDArray[np.float64]) -> float:
    """Calibrate δ from xi_0 -> xi_1 such that
    xi_1 = xi_0 / (1 + 1·xi_0·δ).

    Equivalent: δ = (1/xi_1 - 1/xi_0) (per-cycle inverse-amplitude jump).
    """
    if peaks.size < 2:
        raise ValueError(f"need >= 2 peaks; got {peaks.size}")
    xi0 = float(peaks[0])
    xi1 = float(peaks[1])
    if xi0 <= 0.0 or xi1 <= 0.0 or xi1 >= xi0:
        raise ValueError(f"peaks must be positive and decreasing (xi0={xi0}, xi1={xi1})")
    return 1.0 / xi1 - 1.0 / xi0


def _hyperbolic_predict(xi0: float, delta: float, n_max: int) -> NDArray[np.float64]:
    """xi_n = xi_0 / (1 + n · xi_0 · δ) for n = 0..n_max."""
    ns = np.arange(n_max + 1, dtype=np.float64)
    return xi0 / (1.0 + ns * xi0 * delta)


def _exponential_predict(xi0: float, eta: float, n_max: int) -> NDArray[np.float64]:
    """xi_n = xi_0 · exp(-n · η) for n = 0..n_max."""
    ns = np.arange(n_max + 1, dtype=np.float64)
    return xi0 * np.exp(-ns * eta)


# ---------------------------------------------------------------------------
# OpenFAST reference fixture (regime classification + δ measurement)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openfast_s5_decay() -> dict[str, object]:
    """Load the OpenFAST S5 reference and extract centred peaks 0-15."""
    csv = next(_S5_DECK_DIR.glob("*.csv"))
    history = load_openfast_history(csv)
    heave = history.xi[:, 2]
    t = history.t

    # OF heave has a non-zero equilibrium due to MoorDyn pretension.
    # Subtract late-time mean (last 60 s) to centre the oscillation.
    eq = float(np.mean(heave[t >= t[-1] - 60.0]))
    peak_t, peaks_raw = _positive_peaks(t, heave - eq)
    if peaks_raw.size < _FIT_PEAK_INDEX_MAX + 1:
        raise ValueError(f"need >= {_FIT_PEAK_INDEX_MAX + 1} peaks; got {peaks_raw.size}")
    peaks_fit = peaks_raw[: _FIT_PEAK_INDEX_MAX + 1]
    delta = _calibrate_delta_hyperbolic(peaks_fit)
    return {
        "t": t,
        "heave_centred": heave - eq,
        "equilibrium_m": eq,
        "peak_t": peak_t,
        "peaks": peaks_raw,
        "peaks_fit": peaks_fit,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# FloatSim free-decay run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def floatsim_s5_decay() -> dict[str, object]:
    """Run FloatSim's heave free-decay with the calibrated equivalent
    Morison element.
    """
    setup = _build_setup_b(_S5_DECK_DIR)
    hdb = _build_marin_semi_hdb(setup.C)
    lhs = assemble_cummins_lhs(rigid_body_mass=setup.M, hdb=hdb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        kernel = compute_retardation_kernel(hdb, t_max=_KERNEL_T_MAX_S, dt=_DT_S)

    # Single equivalent horizontal Morison element at body reference.
    # Axis along x (horizontal); body heave motion perpendicular -> full drag.
    element = MorisonElement(
        body_index=0,
        node_a_body=np.array([-_EQUIV_L_M / 2.0, 0.0, 0.0]),
        node_b_body=np.array([+_EQUIV_L_M / 2.0, 0.0, 0.0]),
        diameter=_EQUIV_D_M,
        Cd=_EQUIV_CD,
        include_inertia=False,
    )

    # Calm sea: u_fluid = 0 everywhere. Note signature is (point, t) per
    # floatsim/hydro/morison.py FluidFieldFn alias.
    def _zero_fluid_velocity(point: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        del point, t
        return np.zeros(3, dtype=np.float64)

    morison_force = make_morison_state_force(
        elements=[element],
        n_dof=6,
        fluid_velocity_fn=_zero_fluid_velocity,
        rho=_RHO_KG_PER_M3,
    )

    xi0 = np.zeros(6, dtype=np.float64)
    xi0[2] = _AMP_FROM_EQ_M  # release from OF's amplitude-from-equilibrium

    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=_DURATION_S,
        state_force=morison_force,
        rho_inf=1.0,
    )

    heave = res.xi[:, 2]
    peak_t, peaks_raw = _positive_peaks(res.t, heave)
    if peaks_raw.size < _FIT_PEAK_INDEX_MAX + 1:
        raise ValueError(f"need >= {_FIT_PEAK_INDEX_MAX + 1} peaks; got {peaks_raw.size}")
    peaks_fit = peaks_raw[: _FIT_PEAK_INDEX_MAX + 1]
    delta = _calibrate_delta_hyperbolic(peaks_fit)

    return {
        "result": res,
        "peak_t": peak_t,
        "peaks": peaks_raw,
        "peaks_fit": peaks_fit,
        "delta": delta,
    }


# ---------------------------------------------------------------------------
# Test 1 -- OpenFAST reference passes regime classification
# ---------------------------------------------------------------------------


def test_openfast_envelope_is_hyperbolic_drag_dominated(
    openfast_s5_decay: dict[str, object],
) -> None:
    """OpenFAST S5 envelope must match hyperbolic over peaks 0-15 AND
    diverge from exponential (drag-dominated regime classification per
    Item 16). Pins the cross-check reference's regime.
    """
    peaks = openfast_s5_decay["peaks_fit"]
    delta = openfast_s5_decay["delta"]
    xi0 = float(peaks[0])

    hyp_pred = _hyperbolic_predict(xi0, delta, _FIT_PEAK_INDEX_MAX)
    # Calibrate exponential from xi_0 -> xi_1.
    eta = float(np.log(xi0 / peaks[1]))
    exp_pred = _exponential_predict(xi0, eta, _FIT_PEAK_INDEX_MAX)

    hyp_rms = float(np.sqrt(np.mean((peaks - hyp_pred) ** 2)) / xi0)
    exp_rms = float(np.sqrt(np.mean((peaks - exp_pred) ** 2)) / xi0)

    # Hyperbolic must fit within 5 % rel-RMS (well below the per-peak
    # measured 1 % in the pre-flight diagnostic).
    assert hyp_rms < 0.05, (
        f"OpenFAST hyperbolic-envelope RMS residual {hyp_rms:.4%} exceeds 5%. "
        f"Pre-flight measured < 1% per peak over 0-15; sweep may need "
        "re-running. See docs/diagnostics/m6-pr6-drag-aggregation.md."
    )
    # Exponential must be substantially worse -- > 5x the hyperbolic RMS.
    assert exp_rms > _ENVELOPE_DISCRIM_RATIO * hyp_rms, (
        f"OpenFAST envelope exponential RMS {exp_rms:.4%} not sufficiently "
        f"worse than hyperbolic RMS {hyp_rms:.4%}; ratio "
        f"{exp_rms / max(hyp_rms, 1e-30):.2f}x must exceed "
        f"{_ENVELOPE_DISCRIM_RATIO:.0f}x for clean regime classification."
    )


# ---------------------------------------------------------------------------
# Test 2 -- FloatSim's hyperbolic δ agrees with OpenFAST's
# ---------------------------------------------------------------------------


def test_floatsim_delta_matches_openfast(
    openfast_s5_decay: dict[str, object],
    floatsim_s5_decay: dict[str, object],
) -> None:
    """FloatSim's hyperbolic δ agrees with OpenFAST's within rtol = 5e-2.

    Primary cross-check assertion. Pre-step diagnostic
    (scripts/m6_pr6_drag_aggregation.py) predicted δ = 0.313 from the
    aggregate; the time-domain Cummins + Morison run should reproduce
    this and match OF's measured 0.309 at the test gate.
    """
    delta_fs = float(floatsim_s5_decay["delta"])
    delta_of = float(openfast_s5_decay["delta"])
    rel_err = abs(delta_fs - delta_of) / delta_of
    assert rel_err < _DELTA_RTOL, (
        f"FloatSim δ = {delta_fs:.4f} 1/m vs OpenFAST δ = {delta_of:.4f} 1/m; "
        f"rel-err = {rel_err:.4%} exceeds rtol = {_DELTA_RTOL:.0%}. "
        f"Per Decision B discipline this is a diagnosis stop, not an xfail. "
        "Pre-step aggregation matched at 1.28 % rel-err; the disagreement "
        "is in the FloatSim time-domain pipeline (kernel, integrator, "
        "Morison force assembly)."
    )


# ---------------------------------------------------------------------------
# Test 3 -- FloatSim envelope is hyperbolic, not exponential (M5-style
# discriminator on real-deck-equivalent setup)
# ---------------------------------------------------------------------------


def test_floatsim_envelope_is_hyperbolic_not_exponential(
    floatsim_s5_decay: dict[str, object],
) -> None:
    """FloatSim's heave free-decay envelope must be hyperbolic (drag-
    dominated regime), not exponential (linear-damping regime).

    Item 16 regime classification on FloatSim itself. The M5 PR5
    synthetic test pinned this on a clean setup; PR6 validates it on
    the OC4-deck-equivalent aggregate.
    """
    peaks = floatsim_s5_decay["peaks_fit"]
    delta = float(floatsim_s5_decay["delta"])
    xi0 = float(peaks[0])

    hyp_pred = _hyperbolic_predict(xi0, delta, _FIT_PEAK_INDEX_MAX)
    eta = float(np.log(xi0 / peaks[1]))
    exp_pred = _exponential_predict(xi0, eta, _FIT_PEAK_INDEX_MAX)

    hyp_rms = float(np.sqrt(np.mean((peaks - hyp_pred) ** 2)) / xi0)
    exp_rms = float(np.sqrt(np.mean((peaks - exp_pred) ** 2)) / xi0)

    assert hyp_rms < 0.05, (
        f"FloatSim hyperbolic-envelope RMS residual {hyp_rms:.4%} exceeds 5%. "
        "Either the FloatSim Morison force has a bug or the calibrated "
        "aggregate doesn't capture the drag dynamics correctly."
    )
    assert exp_rms > _ENVELOPE_DISCRIM_RATIO * hyp_rms, (
        f"FloatSim envelope exponential RMS {exp_rms:.4%} not sufficiently "
        f"worse than hyperbolic RMS {hyp_rms:.4%}; ratio "
        f"{exp_rms / max(hyp_rms, 1e-30):.2f}x must exceed "
        f"{_ENVELOPE_DISCRIM_RATIO:.0f}x for clean regime classification. "
        "The decay may be linear-damping-dominated rather than drag, "
        "indicating the equivalent Morison element under-applies the drag."
    )
