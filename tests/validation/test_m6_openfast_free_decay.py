"""M6 PR3 -- S2 pitch free-decay cross-check (FloatSim vs OpenFAST).

Free-decay cross-check on the regenerated S2 scenario (Morison drag
disabled, ``PtfmSurge=0`` IC, ``PtfmPitch=5°`` IC). Validates the
post-fix retardation kernel (`docs/post-mortems/m6-pr3-radiation-kernel-bug.md`)
against the OpenFAST reference at the cross-check level on real
BEM data.

Re-scope from the original plan
-------------------------------
The locked workflow paused PR3 after Mod 2 (damping stability)
discovered S2's OpenFAST reference had Morison drag active. With
drag, the pitch envelope is hyperbolic (amplitude-dependent ζ),
not exponential — making a tight ζ cross-check apples-to-oranges
against FloatSim's radiation-only setup.

Per Option A of the locked re-scope (`docs/diagnostics/m6-pr3-damping-stability.md`):

- The S2 deck was regenerated with Cd=0 on all 25 Morison members,
  isolating the comparison to **radiation-only** physics on both
  sides.
- With drag off, OC4 pitch radiation damping is essentially zero
  (ζ ~ 1e-9 in FloatSim, ζ at numerical noise floor in OpenFAST).
  A tight ζ assertion is no longer meaningful — replaced here by
  a non-negativity check (radiation must dissipate, not inject
  energy; this is the kernel-fix validation at the cross-check
  level).
- The Pre-step period gap diagnostic
  (`docs/diagnostics/m6-pr3-period-gap-diagnostic.md`) classified
  the period mismatch as F1-mostly-explains: combined-deck
  (platform + tower + RNA) FloatSim setup gives 25.67 s vs
  OpenFAST's 26.83 s, rel-err 4.29 % (within 5e-2, beyond 2e-2).
  Period assertion fires xfail-strict under "F1-residual".

What this PR does and doesn't validate
--------------------------------------
**Validates**:
- Post-fix retardation kernel produces a stable, non-negative
  damping response on real BEM data (marin_semi.1) at the cross-
  check level. This is the M6-level acceptance test for the
  M6 PR3 radiation-kernel fix.
- Combined-deck mass aggregation + Cummins linearisation produce
  an OC4 pitch period within 5 % of OpenFAST.

**Does not validate**:
- Quantitative damping match. OC4 pitch radiation damping is
  ~1e-9 at the natural frequency; both tools are at numerical
  noise. The damping cross-check that *does* discriminate physics
  belongs to S5 (drag-on heave decay; M6 PR6) where the dominant
  dissipation mechanism is matched in both tools.
- Tight period match (the 4.29 % residual is documented
  follow-up F1-residual in the cross-check report).

Tolerances per Q4
-----------------
- Period: ``rtol = 2e-2`` per the M6 plan v2 Q4. Asserted with
  ``@pytest.mark.xfail(strict=True)`` because the F1-residual
  pre-step result (4.29 %) sits outside this gate. Strict-xfail
  catches the day F1-residual is closed and the test starts
  passing — the marker should come off then.
- Damping non-negativity: ``ζ ≥ -1e-6`` (allow numerical noise
  but no systematic energy injection).

Mods applied (per Xabier's PR3 plan)
------------------------------------
- Mod 3 envelope check: trend-based (geometric-mean comparison
  across the run), not strict monotone. Catches "kernel pumps
  energy" without false-failing on essentially-undamped peaks.
- Mod 5 IC checks: pitch[0] = 5° exact, pitch[1] essentially
  unchanged from t=0 (zero initial velocity). Cheap setup-bug
  catchers.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.readers.wamit import read_added_mass_and_damping
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import IntegrationResult, integrate_cummins
from tests.support.openfast_csv import load_openfast_history
from tests.support.openfast_deck import (
    _integrate_distributed_mass,
    _scan_named_float,
    _scan_named_path,
)
from tests.validation.test_oc4_natural_periods import (
    OC4_C33_HEAVE_N_PER_M,
    OC4_C44_ROLL_NM_PER_RAD,
    OC4_C55_PITCH_NM_PER_RAD,
    OC4_COG_BELOW_SWL_M,
    OC4_IXX_COG,
    OC4_IYY_COG,
    OC4_IZZ_COG,
    OC4_PLATFORM_MASS_KG,
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURE_ROOT: Final[Path] = Path(__file__).resolve().parents[1] / "fixtures"
_S2_DECK_DIR: Final[Path] = (
    _FIXTURE_ROOT / "openfast" / "oc4_deepcwind" / "inputs" / "s2_pitch_decay"
)
_MARIN_SEMI: Final[Path] = (
    _FIXTURE_ROOT
    / "openfast"
    / "oc4_deepcwind"
    / "baseline"
    / "5MW_Baseline"
    / "HydroData"
    / "marin_semi.1"
)

# ---------------------------------------------------------------------------
# Setup-B physical constants (combined-deck convention; see Pre-step doc)
# ---------------------------------------------------------------------------

# Robertson Table 3-1 platform-with-ballast CoG depth (NOT OpenFAST's
# PtfmCMzt = -8.66 m which is steel-only). Required to be consistent
# with OC4_PLATFORM_MASS_KG = 1.3473e7 kg which includes ballast.
# See docs/diagnostics/m6-pr3-period-gap-diagnostic.md "Convention note".
_PLATFORM_WITH_BALLAST_Z_G_M: Final[float] = -OC4_COG_BELOW_SWL_M

# Standard gravity (matches OpenFAST default).
_G: Final[float] = 9.80665

# Buoyancy-only pitch hydrostatic stiffness for OC4 marin_semi (.hst-style):
#   C_55_buoy = OC4_C55_full - m_platform * g * |z_G_platform|
# This is the BEM-derived buoyancy contribution alone; the gravity term
# -m * g * z_G is added per-deck depending on the mass distribution.
_C55_BUOYANCY_ONLY: Final[float] = OC4_C55_PITCH_NM_PER_RAD - OC4_PLATFORM_MASS_KG * _G * (
    -_PLATFORM_WITH_BALLAST_Z_G_M
)
# similarly for roll (axisymmetric):
_C44_BUOYANCY_ONLY: Final[float] = OC4_C44_ROLL_NM_PER_RAD - OC4_PLATFORM_MASS_KG * _G * (
    -_PLATFORM_WITH_BALLAST_Z_G_M
)

# Tolerances per Q4 of docs/milestone-6-plan.md.
_PERIOD_RTOL: Final[float] = 2.0e-2
_DAMPING_NONNEG_TOL: Final[float] = -1.0e-6

# Pitch-decay analytical references from the regenerated S2 reference,
# extracted in the Pre-step diagnostic. Used only by the diagnostic-log
# test for traceability; assertions read fresh from the CSV.
_OPENFAST_PITCH_PERIOD_S: Final[float] = 26.8257
_OPENFAST_PITCH_PEAK_DEG: Final[float] = 5.0  # peaks barely decay (radiation-only OC4 pitch)


# ---------------------------------------------------------------------------
# Setup B — combined-deck rigid mass + recomputed C_55 (locked Pre-step setup)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombinedDeckSetup:
    """Combined-deck rigid-body mass and pitch hydrostatic stiffness."""

    M: NDArray[np.float64]
    C: NDArray[np.float64]
    m_total_kg: float
    z_G_combined_m: float
    x_G_combined_m: float
    I_55_at_SWL: float
    components: dict[str, dict[str, float]]


def _build_setup_b(deck_dir: Path) -> CombinedDeckSetup:
    """Parse the OpenFAST S2 deck and build Setup B (combined deck)."""
    fst = next(deck_dir.glob("*.fst"))
    elastodyn = _scan_named_path(fst, "EDFile")
    ed_hub_mass = _scan_named_float(elastodyn, "HubMass")
    ed_nac_mass = _scan_named_float(elastodyn, "NacMass")
    ed_yawbr_mass = _scan_named_float(elastodyn, "YawBrMass")
    ed_nac_cmxn = _scan_named_float(elastodyn, "NacCMxn")
    ed_nac_cmzn = _scan_named_float(elastodyn, "NacCMzn")
    ed_tower_ht = _scan_named_float(elastodyn, "TowerHt")
    ed_tower_bs_ht = _scan_named_float(elastodyn, "TowerBsHt")

    # Tower distributed mass via TwrFile.
    twr_span = ed_tower_ht - ed_tower_bs_ht
    twr_file = _scan_named_path(elastodyn, "TwrFile")
    tower_mass, tower_centroid_frac = _integrate_distributed_mass(twr_file, span_m=twr_span)
    tower_cog_z = ed_tower_bs_ht + tower_centroid_frac * twr_span

    # Blade mass: 3 identical blades via BldFile(1).
    bld_file = _scan_named_path(elastodyn, "BldFile(1)")
    try:
        adj_bl_ms = _scan_named_float(bld_file, "AdjBlMs")
    except ValueError:
        adj_bl_ms = 1.0
    blade_mass_each_raw, _ = _integrate_distributed_mass(
        bld_file, span_m=63.0, htfract_col=0, tmassden_col=3
    )
    blade_mass_each = adj_bl_ms * blade_mass_each_raw
    blade_total = 3.0 * blade_mass_each

    tower_top_z = ed_tower_ht
    nac_cog_x = ed_nac_cmxn
    nac_cog_z = tower_top_z + ed_nac_cmzn
    hub_cog_x = nac_cog_x
    hub_cog_z = tower_top_z
    blade_cog_x = nac_cog_x
    blade_cog_z = tower_top_z

    masses: dict[str, float] = {
        "platform_with_ballast": OC4_PLATFORM_MASS_KG,
        "tower": tower_mass,
        "hub": ed_hub_mass,
        "yaw_bearing": ed_yawbr_mass,
        "nacelle": ed_nac_mass,
        "blades_total": blade_total,
    }
    cogs_z = {
        "platform_with_ballast": _PLATFORM_WITH_BALLAST_Z_G_M,
        "tower": tower_cog_z,
        "hub": hub_cog_z,
        "yaw_bearing": tower_top_z,
        "nacelle": nac_cog_z,
        "blades_total": blade_cog_z,
    }
    cogs_x = {
        "platform_with_ballast": 0.0,
        "tower": 0.0,
        "hub": hub_cog_x,
        "yaw_bearing": 0.0,
        "nacelle": nac_cog_x,
        "blades_total": blade_cog_x,
    }

    m_total = float(sum(masses.values()))
    z_G = float(sum(masses[k] * cogs_z[k] for k in masses) / m_total)
    x_G = float(sum(masses[k] * cogs_x[k] for k in masses) / m_total)

    # Inertia at SWL via point-mass parallel-axis from each component's
    # CoG, plus the platform's intrinsic Robertson I_yy_cog.
    I_55 = OC4_IYY_COG + masses["platform_with_ballast"] * (cogs_z["platform_with_ballast"] ** 2)
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_55 += masses[k] * (cogs_x[k] ** 2 + cogs_z[k] ** 2)
    I_44 = OC4_IXX_COG + masses["platform_with_ballast"] * (cogs_z["platform_with_ballast"] ** 2)
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_44 += masses[k] * cogs_z[k] ** 2  # roll about x: x-coord doesn't matter
    I_66 = OC4_IZZ_COG
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_66 += masses[k] * cogs_x[k] ** 2

    inertia_at_ref = np.diag([I_44, I_55, I_66]).astype(np.float64)
    cog_offset = np.array([x_G, 0.0, z_G], dtype=np.float64)
    M = rigid_body_mass_matrix(
        mass=m_total,
        inertia_at_reference=inertia_at_ref,
        cog_offset_body=cog_offset,
    )

    # C with combined CoG: gravity term -m * g * z_G replaces the
    # platform-only term, on top of the buoyancy-only contribution.
    C = np.zeros((6, 6), dtype=np.float64)
    C[2, 2] = OC4_C33_HEAVE_N_PER_M
    C[3, 3] = _C44_BUOYANCY_ONLY + (-m_total * _G * z_G)
    C[4, 4] = _C55_BUOYANCY_ONLY + (-m_total * _G * z_G)

    components = {k: {"mass_kg": masses[k], "z_G_m": cogs_z[k], "x_G_m": cogs_x[k]} for k in masses}
    return CombinedDeckSetup(
        M=M,
        C=C,
        m_total_kg=m_total,
        z_G_combined_m=z_G,
        x_G_combined_m=x_G,
        I_55_at_SWL=float(I_55),
        components=components,
    )


def _build_marin_semi_hdb(C: NDArray[np.float64]) -> HydroDatabase:
    """marin_semi.1 BEM combined with the caller-supplied hydrostatic C."""
    omega, A, B, A_inf = read_added_mass_and_damping(_MARIN_SEMI)
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0, 90.0]),
        A=A,
        B=B,
        A_inf=A_inf,
        C=C,
        RAO=np.zeros((6, omega.size, 2), dtype=np.complex128),
        reference_point=np.array([0.0, 0.0, 0.0]),
        C_source="full",
    )


# ---------------------------------------------------------------------------
# Trajectory analysis helpers
# ---------------------------------------------------------------------------


def _last_60s_mean(t: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    return float(np.mean(x[t >= t[-1] - 60.0]))


def _upward_zero_crossings(t: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
    signs = np.sign(x)
    signs[signs == 0] = 1.0
    transitions = np.where(np.diff(signs) > 0)[0]
    out = []
    for i in transitions:
        denom = x[i + 1] - x[i]
        frac = -x[i] / denom if denom != 0 else 0.0
        out.append(t[i] + frac * (t[i + 1] - t[i]))
    return np.asarray(out, dtype=np.float64)


def _fit_period(t: NDArray[np.float64], x_zm: NDArray[np.float64], n_cycles: int = 10) -> float:
    zeros = _upward_zero_crossings(t, x_zm)
    if zeros.size < n_cycles + 1:
        raise AssertionError(
            f"need >= {n_cycles + 1} upward zero crossings for period fit; got {zeros.size}"
        )
    return float(np.mean(np.diff(zeros[: n_cycles + 1])))


def _positive_peaks(
    t: NDArray[np.float64], x_zm: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    is_peak = (x_zm[1:-1] > x_zm[:-2]) & (x_zm[1:-1] > x_zm[2:]) & (x_zm[1:-1] > 0)
    idx = np.where(is_peak)[0] + 1
    return t[idx], x_zm[idx]


def _zeta_log_decrement_first_n(peaks: NDArray[np.float64], n: int = 5) -> float:
    if peaks.size < n + 1:
        raise AssertionError(f"need >= {n + 1} peaks for log-decrement; got {peaks.size}")
    delta = float(np.log(peaks[0] / peaks[n]) / n)
    return float(delta / np.sqrt(delta * delta + 4.0 * np.pi * np.pi))


# ---------------------------------------------------------------------------
# Module-scoped fixtures (heavy work, run once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def openfast_reference() -> dict[str, float | NDArray[np.float64]]:
    """OpenFAST S2 (drag-off) reference quantities."""
    history = load_openfast_history(_S2_DECK_DIR / "s2_pitch_decay.csv")
    pitch = history.xi[:, 4]
    pitch_eq = _last_60s_mean(history.t, pitch)
    pitch_zm = pitch - pitch_eq
    period = _fit_period(history.t, pitch_zm, n_cycles=10)
    return {
        "t": history.t,
        "pitch_rad": pitch,
        "pitch_zm_rad": pitch_zm,
        "pitch_eq_rad": pitch_eq,
        "period_s": period,
        "pitch_ic_rad": float(pitch[0]),
    }


@pytest.fixture(scope="module")
def floatsim_run() -> dict[str, object]:
    """Run FloatSim's combined-deck (Setup B) S2 free decay."""
    setup = _build_setup_b(_S2_DECK_DIR)
    hdb = _build_marin_semi_hdb(setup.C)
    lhs = assemble_cummins_lhs(rigid_body_mass=setup.M, hdb=hdb)
    dt = 0.05  # match OpenFAST sample rate
    with warnings.catch_warnings():
        # Radiation-only OC4 pitch decays slowly enough that the §9.1
        # diagnostic fires on long-duration runs; informational here.
        warnings.simplefilter("ignore", UserWarning)
        kernel = compute_retardation_kernel(hdb, t_max=200.0, dt=dt)
    xi0 = np.zeros(6)
    xi0[4] = np.deg2rad(5.0)  # PtfmPitch IC = 5° (matches OpenFAST S2)
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=600.0,  # match OpenFAST reference duration
        rho_inf=1.0,  # trapezoidal limit; isolates radiation damping
    )
    return {
        "result": res,
        "setup": setup,
        "dt_s": dt,
        "t_max_kernel_s": 200.0,
    }


# ---------------------------------------------------------------------------
# Sanity / IC tests
# ---------------------------------------------------------------------------


def test_floatsim_run_is_finite(floatsim_run: dict[str, object]) -> None:
    """Sanity: the integration produces finite values (no NaN / inf)."""
    res: IntegrationResult = floatsim_run["result"]  # type: ignore[assignment]
    assert np.all(np.isfinite(res.xi)), "FloatSim xi has non-finite values"
    assert np.all(np.isfinite(res.xi_dot)), "FloatSim xi_dot has non-finite values"
    assert np.all(np.isfinite(res.xi_ddot)), "FloatSim xi_ddot has non-finite values"


def test_initial_pitch_ic_is_5_deg(floatsim_run: dict[str, object]) -> None:
    """Mod 5: PtfmPitch IC = 5° applied exactly at t=0."""
    res: IntegrationResult = floatsim_run["result"]  # type: ignore[assignment]
    pitch_t0_rad = float(res.xi[0, 4])
    pitch_t0_deg = np.rad2deg(pitch_t0_rad)
    assert pitch_t0_deg == pytest.approx(5.0, abs=1.0e-6), (
        f"FloatSim pitch[t=0] = {pitch_t0_deg:.6f}° (expected 5°). "
        "Test harness IC application is broken."
    )


def test_initial_pitch_velocity_is_zero(floatsim_run: dict[str, object]) -> None:
    """Mod 5: zero initial velocity -> pitch[1] essentially unchanged from pitch[0].

    With ξ_dot(0)=0, the first integrator step's pitch change is
    O(dt² · ξ_ddot[0]). For dt=0.05 s and the computed ξ_ddot[0]
    (~ -ω_n² · pitch[0]), the change is O(1e-3 °) -- much smaller
    than 1 % of the IC.
    """
    res: IntegrationResult = floatsim_run["result"]  # type: ignore[assignment]
    delta_deg = abs(np.rad2deg(res.xi[1, 4] - res.xi[0, 4]))
    assert delta_deg < 0.05, (
        f"|pitch[1] - pitch[0]| = {delta_deg:.5f}° exceeds 0.05° gate; "
        "either the integrator is broken or ξ_dot(0) is not zero."
    )


# ---------------------------------------------------------------------------
# Period assertion (xfail-strict under F1-residual)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "F1-residual: combined-deck FloatSim period 25.67 s vs OpenFAST "
        "26.83 s = -4.29% rel-err. Beyond rtol=2e-2; within rtol=5e-2. "
        "Pre-step diagnostic classified as F1-mostly-explains "
        "(docs/diagnostics/m6-pr3-period-gap-diagnostic.md). Closing "
        "F1-residual requires distributed-inertia integration of platform "
        "ballast and tower/RNA components; out of PR3 scope."
    ),
)
def test_pitch_period_matches_openfast(
    floatsim_run: dict[str, object],
    openfast_reference: dict[str, float | NDArray[np.float64]],
) -> None:
    """Pitch period agreement at rtol=2e-2 per Q4 of milestone-6-plan.md.

    Expected to fail under F1-residual (4.29 % rel-err). xfail-strict
    catches the day F1-residual is closed and the test starts passing
    -- the marker should come off then.
    """
    res: IntegrationResult = floatsim_run["result"]  # type: ignore[assignment]
    pitch = res.xi[:, 4]
    pitch_eq = _last_60s_mean(res.t, pitch)
    pitch_zm = pitch - pitch_eq
    fs_period = _fit_period(res.t, pitch_zm, n_cycles=10)

    of_period = float(openfast_reference["period_s"])  # type: ignore[arg-type]
    rel_err = abs(fs_period - of_period) / of_period

    assert rel_err < _PERIOD_RTOL, (
        f"FloatSim period {fs_period:.4f} s vs OpenFAST {of_period:.4f} s; "
        f"rel-err {rel_err:.4f} exceeds rtol {_PERIOD_RTOL}. "
        "F1-residual classification: distributed-inertia bookkeeping of "
        "platform ballast + tower/RNA components."
    )


# ---------------------------------------------------------------------------
# Damping non-negativity (replaces tight ζ assertion; kernel-fix validation)
# ---------------------------------------------------------------------------


def test_pitch_damping_is_non_negative(floatsim_run: dict[str, object]) -> None:
    """Radiation must dissipate, not inject energy. ζ ≥ -1e-6.

    On OC4 pitch the radiation damping at the natural frequency is
    essentially zero (ζ ~ 1e-9 in both FloatSim and OpenFAST), so a
    tight match assertion has no signal. Non-negativity catches the
    pre-fix kernel pathology (sustained oscillation / energy injection)
    on real BEM data without requiring a tight match that the physics
    setup doesn't support.

    Tight damping cross-checks belong in scenarios where the dominant
    dissipation mechanism is matched in both tools (S5: drag-on heave
    decay, M6 PR6).
    """
    res: IntegrationResult = floatsim_run["result"]  # type: ignore[assignment]
    pitch = res.xi[:, 4]
    pitch_eq = _last_60s_mean(res.t, pitch)
    pitch_zm = pitch - pitch_eq
    _, peaks = _positive_peaks(res.t, pitch_zm)
    if peaks.size < 6:
        pytest.skip(f"insufficient peaks for log-decrement: {peaks.size} (need 6)")
    zeta = _zeta_log_decrement_first_n(peaks, n=5)
    assert zeta >= _DAMPING_NONNEG_TOL, (
        f"FloatSim pitch damping ζ = {zeta:.3e} is below {_DAMPING_NONNEG_TOL}. "
        "Radiation must dissipate energy; a negative ζ means the kernel is "
        "pumping energy into the system (cf. the M6 PR3 pre-fix bug)."
    )


# ---------------------------------------------------------------------------
# Mod 3 envelope-trend check
# ---------------------------------------------------------------------------


def test_pitch_envelope_trend_does_not_grow(floatsim_run: dict[str, object]) -> None:
    """Mod 3: amplitude envelope must not grow over the run.

    Strict monotonicity false-fails on numerical noise on
    essentially-undamped peaks. The trend check still catches
    'kernel injects energy' (sustained amplitude growth) which is
    the pathology we actually want to validate.

    Geometric mean of peaks N+1..N+3 must be < geometric mean of
    peaks N..N+2 plus a small noise tolerance, for all N where
    N+3 exists. With OC4-pitch radiation-only (ζ ~ 1e-9), peaks
    are essentially constant; the check reduces to "no triple's
    geometric mean grows by more than ε relative to the previous
    triple".
    """
    res: IntegrationResult = floatsim_run["result"]  # type: ignore[assignment]
    pitch = res.xi[:, 4]
    pitch_eq = _last_60s_mean(res.t, pitch)
    pitch_zm = pitch - pitch_eq
    _, peaks = _positive_peaks(res.t, pitch_zm)
    if peaks.size < 6:
        pytest.skip(f"insufficient peaks for trend check: {peaks.size} (need 6)")

    # Tolerance: 1% growth between adjacent geometric-mean triples.
    # Calibrated to allow numerical noise on a near-zero-damping run
    # while still catching a 1-2% per-cycle amplitude growth (the
    # signature of the pre-fix kernel pumping energy at the resonance).
    eps = 1.0e-2
    n_violations = 0
    worst_growth = 0.0
    for i in range(peaks.size - 3):
        gm_now = float(np.exp(np.mean(np.log(peaks[i : i + 3]))))
        gm_next = float(np.exp(np.mean(np.log(peaks[i + 1 : i + 4]))))
        growth = (gm_next - gm_now) / gm_now
        worst_growth = max(worst_growth, growth)
        if growth > eps:
            n_violations += 1
    assert n_violations == 0, (
        f"Pitch envelope grows by more than {eps:.3%} on {n_violations} "
        f"adjacent triples (worst: {worst_growth:.4%}). The kernel is "
        "injecting energy or there is a numerical instability."
    )


# ---------------------------------------------------------------------------
# Diagnostic logging (always passes; emits per-window ζ + first-N peaks)
# ---------------------------------------------------------------------------


def test_diagnostic_log_pitch_envelope(
    floatsim_run: dict[str, object],
    openfast_reference: dict[str, float | NDArray[np.float64]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Always-passing log of pitch peaks + window ζ values for the report.

    Outputs go to pytest's captured stdout; visible with `pytest -s` or
    in CI logs. Also accessible to the cross-check report generator.
    """
    res: IntegrationResult = floatsim_run["result"]  # type: ignore[assignment]
    setup: CombinedDeckSetup = floatsim_run["setup"]  # type: ignore[assignment]
    pitch = res.xi[:, 4]
    pitch_eq = _last_60s_mean(res.t, pitch)
    pitch_zm = pitch - pitch_eq
    fs_period = _fit_period(res.t, pitch_zm, n_cycles=10)
    peak_t, peaks = _positive_peaks(res.t, pitch_zm)

    print()  # newline so pytest -s output is readable
    print("=" * 64)
    print("M6 PR3 -- diagnostic log: FloatSim pitch decay vs OpenFAST")
    print("=" * 64)
    print(
        f"FloatSim setup: combined deck (m_total={setup.m_total_kg:.4e} kg, "
        f"z_G={setup.z_G_combined_m:+.3f} m, I_55={setup.I_55_at_SWL:.4e} kg*m^2)"
    )
    print(
        f"  C_55 = {setup.C[4, 4]:.4e} N*m/rad " f"(buoy {_C55_BUOYANCY_ONLY:.3e} + gravity term)"
    )
    print()
    print(f"FloatSim pitch period (10 cycles): {fs_period:.4f} s")
    print(f"OpenFAST pitch period (10 cycles): " f"{float(openfast_reference['period_s']):.4f} s")
    rel_err = abs(fs_period - float(openfast_reference["period_s"])) / float(
        openfast_reference["period_s"]
    )
    print(f"  rel-err: {rel_err:.4f} (rtol target {_PERIOD_RTOL})")
    print()
    print(f"FloatSim positive-peak count: {peaks.size}")
    if peaks.size >= 6:
        print(f"  peak[0] = {np.rad2deg(peaks[0]):.5f}° at t = {peak_t[0]:.2f} s")
        print(f"  peak[5] = {np.rad2deg(peaks[5]):.5f}° at t = {peak_t[5]:.2f} s")
        if peaks.size >= 21:
            print(f"  peak[20] = {np.rad2deg(peaks[20]):.5f}° " f"at t = {peak_t[20]:.2f} s")
        for window_label, (start, end) in (
            ("peaks 1-5", (0, 5)),
            ("peaks 5-10", (5, 10)),
            ("peaks 10-20", (10, 20)),
        ):
            try:
                z = _zeta_log_decrement_first_n(peaks[start:], n=end - start)
                print(f"  zeta over {window_label}: {z:.4e}")
            except AssertionError as e:
                print(f"  zeta over {window_label}: --  ({e})")
    print("=" * 64)
    captured = capsys.readouterr().out
    # Re-emit so the captured output also reaches the live log.
    print(captured)
