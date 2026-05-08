"""M6 PR4 Pre-3 -- RAO definition lock-down + dual-path verification.

What this script verifies
-------------------------

For TWO wave frequencies (per the locked Pre-3 plan) -- ``WaveTp = 10 s``
(super-resonant for OC4 heave; mass-controlled regime) and
``WaveTp = 25 s`` (near OC4 pitch natural; stiffness-controlled-with-
resonance regime) -- compute the heave RAO amplitude and phase TWO
ways and confirm they agree:

  Path A -- WAMIT-impedance path
    Read ``marin_semi.{1,3,hst}`` BEM coefficients.
    Combine with the Pre-step Setup B mass matrix and Robertson C
    (with-ballast platform mass at ``z_G = -13.46 m``, plus
    tower/RNA/blades aggregated from the OpenFAST deck per
    ``compute_openfast_deck_residual``).
    Build the steady-state impedance ``Z(omega) = -omega^2 (M + A_full(omega))
    + i omega B(omega) + C`` and solve ``xi_hat = Z^{-1} F_exc``.
    Heave RAO = ``|xi_hat[2]|`` and ``arg(xi_hat[2])`` (per unit wave
    amplitude, with phase reference at the BEM origin = SWL).

  Path B -- OpenFAST .outb time-series path
    Load the OpenFAST CSV from the regenerated S3 fixture (post-
    fix-s3-wavemod). Sinusoidal lstsq fit on heave and wave_elev at
    the OpenFAST IFFT-quantised wave frequency (Item 21). RAO
    amplitude = amp(heave) / amp(wave_elev); RAO phase =
    circular-subtraction of phases.

What this verifies / does NOT verify
------------------------------------

This is a CONSISTENCY check between FloatSim's WAMIT-.3 path and
OpenFAST's HydroDyn time-domain path. It establishes that
**FloatSim's RAO extraction uses the same convention as OpenFAST**
(BEM reference at SWL, Cummins linearisation, F_exc per unit wave
amplitude, phase relative to wave elevation at the body origin,
quantised-omega lstsq fit).

It does NOT verify ABSOLUTE correctness. Both tools could share a
common sign error, factor of 2, or reference-point convention and
still pass this check. Absolute correctness against an analytical
reference (Froude-Krylov on a submerged body, for instance) is out
of Pre-3 scope.

The test passes if amplitude agreement is within rtol = 1e-2 AND
phase agreement is within atol = 1 deg, at BOTH frequencies. If
only one passes, the convention is regime-dependent and worth
diagnosing before PR4 starts.

Run from repo root:
    python scripts/m6_pr4_pre3_rao_verification.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from floatsim.bodies.mass_properties import rigid_body_mass_matrix  # noqa: E402
from floatsim.hydro.readers.wamit import (  # noqa: E402
    read_added_mass_and_damping,
)
from tests.support.openfast_csv import load_openfast_history  # noqa: E402
from tests.support.openfast_deck import (  # noqa: E402
    OC4_PLATFORM_COG_Z_M,
    OC4_PLATFORM_TOTAL_MASS_KG,
    _integrate_distributed_mass,
    _scan_named_float,
    _scan_named_path,
)
from tests.support.rao_extraction import (  # noqa: E402
    extract_rao_from_history,
    quantised_wave_period_s,
    read_wave_tmax_from_seastate,
)
from tests.validation.test_oc4_natural_periods import (  # noqa: E402
    OC4_C33_HEAVE_N_PER_M,
    OC4_C44_ROLL_NM_PER_RAD,
    OC4_C55_PITCH_NM_PER_RAD,
    OC4_IXX_COG,
    OC4_IYY_COG,
    OC4_IZZ_COG,
)

S3_INPUTS = REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/inputs/s3_rao_sweep"
MARIN_DIR = REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/baseline/5MW_Baseline/HydroData"
MARIN_STEM = MARIN_DIR / "marin_semi"

# Verification frequencies per the locked plan.
VERIFICATION_PERIODS_S = (10.0, 25.0)

# Acceptance gates.
AMP_RTOL = 1.0e-2
PHASE_ATOL_DEG = 1.0

# Configuration consistent with M6 PR3 / PR4 (combined-deck Setup B).
G_M_S2 = 9.80665
HEAVE_DOF = 2
ROLL_DOF = 3
PITCH_DOF = 4

# Buoyancy-only C from Robertson decomposition (M6 PR3 Pre-step doc).
C55_BUOYANCY_ONLY = OC4_C55_PITCH_NM_PER_RAD - OC4_PLATFORM_TOTAL_MASS_KG * G_M_S2 * (
    -OC4_PLATFORM_COG_Z_M
)
C44_BUOYANCY_ONLY = OC4_C44_ROLL_NM_PER_RAD - OC4_PLATFORM_TOTAL_MASS_KG * G_M_S2 * (
    -OC4_PLATFORM_COG_Z_M
)


@dataclass(frozen=True)
class CombinedDeckSetup:
    """Setup B: combined-deck rigid-body M plus combined-CoG C (Robertson convention)."""

    M: np.ndarray
    C: np.ndarray
    m_total_kg: float
    z_G_combined_m: float


def _build_combined_setup(deck_dir: Path) -> CombinedDeckSetup:
    """Aggregate platform + tower + RNA from an OpenFAST S3 deck.

    Mirrors ``tests/validation/test_m6_openfast_free_decay._build_setup_b``
    but runs against any S3 sweep variant (the mass aggregation is
    independent of WaveTp).
    """
    fst = next(deck_dir.glob("*.fst"))
    elastodyn = _scan_named_path(fst, "EDFile")
    ed_hub_mass = _scan_named_float(elastodyn, "HubMass")
    ed_nac_mass = _scan_named_float(elastodyn, "NacMass")
    ed_yawbr_mass = _scan_named_float(elastodyn, "YawBrMass")
    ed_nac_cmxn = _scan_named_float(elastodyn, "NacCMxn")
    ed_nac_cmzn = _scan_named_float(elastodyn, "NacCMzn")
    ed_tower_ht = _scan_named_float(elastodyn, "TowerHt")
    ed_tower_bs_ht = _scan_named_float(elastodyn, "TowerBsHt")

    twr_span = ed_tower_ht - ed_tower_bs_ht
    twr_file = _scan_named_path(elastodyn, "TwrFile")
    tower_mass, tower_centroid_frac = _integrate_distributed_mass(twr_file, span_m=twr_span)
    tower_cog_z = ed_tower_bs_ht + tower_centroid_frac * twr_span

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

    masses = {
        "platform_with_ballast": OC4_PLATFORM_TOTAL_MASS_KG,
        "tower": tower_mass,
        "hub": ed_hub_mass,
        "yaw_bearing": ed_yawbr_mass,
        "nacelle": ed_nac_mass,
        "blades_total": blade_total,
    }
    cogs_z = {
        "platform_with_ballast": OC4_PLATFORM_COG_Z_M,
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

    I_55 = OC4_IYY_COG + masses["platform_with_ballast"] * (cogs_z["platform_with_ballast"] ** 2)
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_55 += masses[k] * (cogs_x[k] ** 2 + cogs_z[k] ** 2)
    I_44 = OC4_IXX_COG + masses["platform_with_ballast"] * (cogs_z["platform_with_ballast"] ** 2)
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_44 += masses[k] * cogs_z[k] ** 2
    I_66 = OC4_IZZ_COG
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_66 += masses[k] * cogs_x[k] ** 2

    M = rigid_body_mass_matrix(
        mass=m_total,
        inertia_at_reference=np.diag([I_44, I_55, I_66]).astype(np.float64),
        cog_offset_body=np.array([x_G, 0.0, z_G], dtype=np.float64),
    )

    C = np.zeros((6, 6), dtype=np.float64)
    C[2, 2] = OC4_C33_HEAVE_N_PER_M
    C[3, 3] = C44_BUOYANCY_ONLY + (-m_total * G_M_S2 * z_G)
    C[4, 4] = C55_BUOYANCY_ONLY + (-m_total * G_M_S2 * z_G)
    return CombinedDeckSetup(M=M, C=C, m_total_kg=m_total, z_G_combined_m=z_G)


def _interp_complex_at_omega(omega_grid: np.ndarray, F: np.ndarray, omega_target: float) -> complex:
    """Linear interpolation of a complex-valued spectrum at a target frequency."""
    re = float(np.interp(omega_target, omega_grid, F.real))
    im = float(np.interp(omega_target, omega_grid, F.imag))
    return complex(re, im)


def _interp_real_at_omega(omega_grid: np.ndarray, X: np.ndarray, omega_target: float) -> float:
    return float(np.interp(omega_target, omega_grid, X))


def _read_excitation_direct(path: Path, omega_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direct WAMIT .3 reader -- bypasses FloatSim's strict Re/Im vs
    Mod*exp(i*Pha) cross-check (too tight for marin_semi.3's printed
    precision) but applies the SAME dimensionalisation factor that the
    public reader does (post-fix-wamit-dimensionalisation, conventions
    doc Item 22).

    Per WAMIT v7 manual §4.2: F_exc_i_dim = rho * g * ULEN^k_i * F_exc_i_nondim,
    where k_i = 2 for translational modes (i=1..3) and k_i = 3 for
    rotational (i=4..6). For OC4 marin_semi: rho=1025, g=9.80665, ULEN=1.

    Returns (heading_deg, F_exc) where F_exc has shape (6, n_omega, n_h),
    in SI units (N/m wave amplitude for translational; N*m/m for
    rotational).
    """
    rho = 1025.0
    g = 9.80665
    ulen = 1.0
    rows: list[tuple[float, float, int, float, float]] = []
    headings: set[float] = set()
    with path.open() as fh:
        for line in fh:
            tokens = line.split()
            if not tokens or len(tokens) != 7:
                continue
            try:
                per = float(tokens[0])
                beta = float(tokens[1])
                i = int(tokens[2])
                re_x = float(tokens[5])
                im_x = float(tokens[6])
            except ValueError:
                continue
            rows.append((per, beta, i, re_x, im_x))
            headings.add(beta)
    heading_deg = np.asarray(sorted(headings), dtype=np.float64)
    F = np.zeros((6, omega_grid.size, heading_deg.size), dtype=np.complex128)
    for per, beta, i, re_x, im_x in rows:
        omega = 2.0 * np.pi / per
        w_idx = int(np.argmin(np.abs(omega_grid - omega)))
        if abs(omega_grid[w_idx] - omega) > 1.0e-6:
            continue  # row not on the grid -- skip
        h_idx = int(np.argmin(np.abs(heading_deg - beta)))
        # Apply dimensionalisation: rho * g * ULEN^k where k = 2 for
        # translational (i in {1,2,3}), 3 for rotational (i in {4,5,6}).
        k_power = 2 if i <= 3 else 3
        factor = rho * g * (ulen**k_power)
        F[i - 1, w_idx, h_idx] = complex(re_x * factor, im_x * factor)
    return heading_deg, F


def _path_a_impedance_rao(
    setup: CombinedDeckSetup,
    omega_quantised: float,
    heading_deg: float = 0.0,
) -> tuple[float, float]:
    """Heave RAO via Z xi = F_exc at the quantised frequency.

    Returns (amplitude_m_per_m, phase_rad).

    Phase convention: returns the **lag** of response behind wave
    elevation (positive = response peaks AFTER wave peak), matching
    Path B's ``atan2(B, A)`` lstsq convention. Under the +i convention
    that FloatSim's WAMIT reader uses, the impedance solution has
    ``arg(xi_hat)`` representing the LEAD; the lag is ``-arg(xi_hat)``.
    See conventions doc Item 24 and
    ``docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.md``.
    """
    # BEM data on its native grid.
    omega_bem, A_full, B_full, _A_inf = read_added_mass_and_damping(MARIN_STEM.with_suffix(".1"))
    headings, F_exc = _read_excitation_direct(MARIN_STEM.with_suffix(".3"), omega_bem)

    # Closest heading column.
    h_idx = int(np.argmin(np.abs(headings - heading_deg)))
    if abs(headings[h_idx] - heading_deg) > 1.0e-6:
        raise ValueError(f"heading {heading_deg} deg not in marin_semi.3 headings {headings}")

    # Interpolate per-DOF A, B, F_exc at the quantised frequency.
    A_q = np.zeros((6, 6), dtype=np.float64)
    B_q = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            A_q[i, j] = _interp_real_at_omega(omega_bem, A_full[i, j, :], omega_quantised)
            B_q[i, j] = _interp_real_at_omega(omega_bem, B_full[i, j, :], omega_quantised)
    F_q = np.zeros(6, dtype=np.complex128)
    for i in range(6):
        F_q[i] = _interp_complex_at_omega(omega_bem, F_exc[i, :, h_idx], omega_quantised)

    Z = -(omega_quantised**2) * (setup.M + A_q) + 1j * omega_quantised * B_q + setup.C
    xi_hat = np.linalg.solve(Z, F_q)
    # Negate arg(xi_hat) to convert from LEAD (+i convention's arg) to LAG
    # (cos+sin lstsq's atan2(B, A)). See module docstring + Item 24.
    return float(np.abs(xi_hat[HEAVE_DOF])), -float(np.angle(xi_hat[HEAVE_DOF]))


def _path_b_lstsq_rao(
    deck_dir: Path, omega_quantised: float, dof_idx: int
) -> tuple[float, float, float, float]:
    """Heave / roll / pitch RAO via lstsq fit on the OpenFAST CSV.

    ``dof_idx`` is the 6-DOF index into ``history.xi`` (0=surge, ..., 5=yaw),
    matching the FloatSim convention. The CSV's column layout
    [time, surge, sway, heave, roll, pitch, yaw, wave_elev] is handled by
    ``load_openfast_history`` which packs (surge, sway, heave, roll,
    pitch, yaw) into ``xi``.

    Returns (amplitude_per_m_wave, phase_rad, response_residual, wave_residual).
    """
    csv = next(deck_dir.glob("*.csv"))
    history = load_openfast_history(csv)
    response = history.xi[:, dof_idx]
    if "wave_elev_m" not in history.extra_columns:
        raise ValueError(f"{csv.name} has no wave_elev_m channel")
    wave_elev = history.extra_columns["wave_elev_m"]
    rao_amp, rao_phase, resp_resid, wave_resid = extract_rao_from_history(
        history.t, response, wave_elev, omega_quantised
    )
    return rao_amp, rao_phase, resp_resid, wave_resid


def _wrap_diff_deg(a_rad: float, b_rad: float) -> float:
    """Circular phase difference, in degrees."""
    diff = a_rad - b_rad
    diff_wrapped = ((diff + np.pi) % (2.0 * np.pi)) - np.pi
    return np.rad2deg(diff_wrapped)


def main() -> None:
    print("M6 PR4 Pre-3 -- RAO definition lock-down + dual-path verification")
    print("=" * 78)
    print()
    print("Verifying CONSISTENCY (not absolute correctness) between:")
    print("  Path A: WAMIT marin_semi.{1,3,hst} -> impedance Z(omega) xi = F_exc")
    print("  Path B: OpenFAST .outb -> sinusoidal lstsq fit on heave / wave_elev")
    print()
    print(f"Acceptance: amp rtol < {AMP_RTOL:.0e}, phase atol < {PHASE_ATOL_DEG:.1f} deg")
    print(f"Frequencies: {[f'{t:.0f}s' for t in VERIFICATION_PERIODS_S]}")
    print()

    print(
        f"{'WaveTp_label':>12}  {'WaveTp_actual':>14}  "
        f"{'Path A amp':>12}  {'Path B amp':>12}  {'amp rel-err':>12}  "
        f"{'Path A phase':>14}  {'Path B phase':>14}  {'phase err':>10}  {'verdict':>8}"
    )
    print("-" * 122)

    all_pass = True
    for wave_tp in VERIFICATION_PERIODS_S:
        wt_str = f"{wave_tp:05.1f}".replace(".", "p")
        deck_dir = S3_INPUTS / f"WaveTp_{wt_str}"
        seastate = next(deck_dir.glob("*_SeaState.dat"))
        wave_tmax = read_wave_tmax_from_seastate(seastate)
        t_actual = quantised_wave_period_s(wave_tp, wave_tmax)
        omega_q = 2.0 * np.pi / t_actual

        setup = _build_combined_setup(deck_dir)
        amp_a, phase_a = _path_a_impedance_rao(setup, omega_q, heading_deg=0.0)
        amp_b, phase_b, resp_resid, wave_resid = _path_b_lstsq_rao(deck_dir, omega_q, HEAVE_DOF)

        amp_rel_err = (amp_a - amp_b) / amp_b
        phase_err_deg = _wrap_diff_deg(phase_a, phase_b)
        verdict_amp = abs(amp_rel_err) < AMP_RTOL
        verdict_phase = abs(phase_err_deg) < PHASE_ATOL_DEG
        verdict = "PASS" if (verdict_amp and verdict_phase) else "FAIL"
        all_pass = all_pass and (verdict_amp and verdict_phase)

        # Highlight the IFFT-quantisation if the configured/actual differ > 0.5%.
        flag = " *" if abs((t_actual - wave_tp) / wave_tp) > 5.0e-3 else "  "
        print(
            f"  {wave_tp:>9.1f}{flag}  {t_actual:>14.5f}  "
            f"{amp_a:>12.6f}  {amp_b:>12.6f}  {amp_rel_err:>+12.5f}  "
            f"{np.rad2deg(phase_a):>+14.4f}  {np.rad2deg(phase_b):>+14.4f}  "
            f"{phase_err_deg:>+10.4f}  {verdict:>8}"
        )
        print(
            f"     {'':>9}    {'':>14}  "
            f"{'':>12}  {'':>12}  {'':>12}  "
            f"{'':>14}  {'':>14}  "
            f"resp_resid={resp_resid:.4f}, wave_resid={wave_resid:.4f}"
        )

    print()
    print("Legend:  * configured vs actual WaveTp differ by >0.5% (Item 21 visible)")
    print()
    if all_pass:
        print(
            "All Pre-3 checks PASSED. RAO convention consistent with OpenFAST "
            "across regimes; PR4 may proceed."
        )
    else:
        print("Pre-3 checks FAILED. Diagnose convention before PR4 starts.")


if __name__ == "__main__":
    main()
