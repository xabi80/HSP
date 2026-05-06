"""M6 PR3 Pre-step -- pitch period gap diagnostic.

Locked workflow (Option A): before any test code lands, classify the
period gap between FloatSim and OpenFAST. Two FloatSim setups are run
against the regenerated drag-off S2 reference:

  Setup A: platform-only mass (Robertson 2014) + Robertson C_55.
           This is what the original PR3 plan used. Block-diagonal
           rigid-body mass at SWL via _oc4_rigid_body_mass_matrix().

  Setup B: combined mass (platform + tower + RNA from OpenFAST
           ElastoDyn / TwrFile / BldFile parsing) + C_55 recomputed
           with the combined CoG. Mass matrix uses cog_offset_body
           = (x_G_combined, 0, z_G_combined) so the surge-pitch
           coupling (M[0,4] = -m * z_G) is captured.

For each setup: run a 600-s Cummins free-decay from a 5-deg pitch IC
on the post-fix kernel (marin_semi.1 BEM). Fit pitch period from
upward zero crossings (mean-subtracted) over the first 10 cycles.

Compare to OpenFAST's measured period from the drag-off S2 CSV.

Decision tree (per Xabier's locked plan):
  - Setup B period within rtol=2e-2 of OpenFAST -> F1 fully explains
    the gap. Use Setup B for PR3; period assertion fires GREEN.
  - Setup B period within rtol=5e-2 -> F1 mostly explains. Period
    xfail-strict under "F1-residual"; PR3 proceeds.
  - Setup B period beyond rtol=5e-2 -> second deck-identity effect
    exists; pause and name it F2.

Run from the repo root:
    python scripts/m6_pr3_prestep_period_gap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from floatsim.bodies.mass_properties import rigid_body_mass_matrix  # noqa: E402
from floatsim.hydro.database import HydroDatabase  # noqa: E402
from floatsim.hydro.radiation import assemble_cummins_lhs  # noqa: E402
from floatsim.hydro.readers.wamit import read_added_mass_and_damping  # noqa: E402
from floatsim.hydro.retardation import compute_retardation_kernel  # noqa: E402
from floatsim.solver.newmark import integrate_cummins  # noqa: E402
from tests.support.openfast_deck import (  # noqa: E402
    _integrate_distributed_mass,
    _scan_named_float,
    _scan_named_path,
)
from tests.validation.test_oc4_natural_periods import (  # noqa: E402
    OC4_C33_HEAVE_N_PER_M,
    OC4_C44_ROLL_NM_PER_RAD,
    OC4_C55_PITCH_NM_PER_RAD,
    OC4_COG_BELOW_SWL_M,
    OC4_IXX_COG,
    OC4_IYY_COG,
    OC4_IZZ_COG,
    OC4_PLATFORM_MASS_KG,
    _oc4_rigid_body_mass_matrix,
)

OC4_PLATFORM_TOTAL_MASS_KG = 1.3473e7  # incl. ballast water; matches PR2 default

S2_DECK_DIR = REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/inputs/s2_pitch_decay"
S2_CSV = S2_DECK_DIR / "s2_pitch_decay.csv"
MARIN_SEMI = (
    REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/baseline/5MW_Baseline/HydroData/marin_semi.1"
)

# C_55 buoyancy-only contribution from Robertson C_55 - gravity at platform-only:
#   C_55_buoy = 1.078e9 - m_platform * g * |z_G_platform|
#             = 1.078e9 - 1.347e7 * 9.81 * 13.46
#             = 1.078e9 - 1.7793e9 = -7.013e8
# This is the BEM .hst contribution; same for both setups (no F1 effect).
G = 9.80665  # OpenFAST default Gravity
RHO_W = 1025.0  # OpenFAST default WtrDens

C55_BUOYANCY_ONLY = OC4_C55_PITCH_NM_PER_RAD - OC4_PLATFORM_MASS_KG * G * OC4_COG_BELOW_SWL_M
# numerical sanity print at start of main()


def _build_marin_semi_hdb(C: np.ndarray) -> HydroDatabase:
    omega, A, B, A_inf = read_added_mass_and_damping(MARIN_SEMI)
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


def _setup_a_mass_and_C() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Robertson platform-only mass + Robertson C_55."""
    M = _oc4_rigid_body_mass_matrix()
    C = np.zeros((6, 6))
    C[2, 2] = OC4_C33_HEAVE_N_PER_M
    C[3, 3] = OC4_C44_ROLL_NM_PER_RAD
    C[4, 4] = OC4_C55_PITCH_NM_PER_RAD
    diag = {
        "label": "A: platform-only Robertson",
        "m_total_kg": OC4_PLATFORM_MASS_KG,
        "z_G_m": -OC4_COG_BELOW_SWL_M,
        "x_G_m": 0.0,
        "I_55_at_SWL": float(M[4, 4]),
        "C_55": float(C[4, 4]),
    }
    return M, C, diag


def _setup_b_mass_and_C() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Combined mass + C_55 with combined CoG.

    Parses the OpenFAST S2 deck (post-Mod-1, drag-off) for the masses
    and CoG positions of platform-with-ballast / tower / hub / nacelle /
    yaw-bearing / blades; uses point-mass parallel-axis to assemble I_55
    at SWL, plus the platform's already-evaluated I_yy_about_CoG.
    """
    fst = next(S2_DECK_DIR.glob("*.fst"))
    elastodyn = _scan_named_path(fst, "EDFile")
    # NOTE: ed_ptfm_cmzt = -8.66 m is the OpenFAST PtfmMass (steel-only,
    # 3.85e6 kg) CoG. Robertson's total platform mass (1.347e7 kg, ballast
    # included) sits at -13.46 m per Table 3-1. We use the Robertson
    # (with-ballast) convention to match the M6 PR2 parser's mass total.
    # See docs/diagnostics/m6-pr3-period-gap-diagnostic.md.
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
        "platform_with_ballast": -OC4_COG_BELOW_SWL_M,  # Robertson Table 3-1
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

    m_total = sum(masses.values())
    z_G = sum(masses[k] * cogs_z[k] for k in masses) / m_total
    x_G = sum(masses[k] * cogs_x[k] for k in masses) / m_total

    # I_55 at SWL: parallel-axis sum.
    # Platform: use Robertson's I_yy_cog plus parallel-axis from PtfmCMzt
    # (treating the platform-with-ballast as a single rigid body with
    # I_yy_cog ≈ Robertson's platform-only value -- modest approximation,
    # the ballast water adds <10% to pitch inertia).
    I_55 = OC4_IYY_COG + masses["platform_with_ballast"] * cogs_z["platform_with_ballast"] ** 2
    # Other components: point-mass approximation.
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_55 += masses[k] * (cogs_x[k] ** 2 + cogs_z[k] ** 2)
    I_44 = OC4_IXX_COG + masses["platform_with_ballast"] * cogs_z["platform_with_ballast"] ** 2
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_44 += masses[k] * (cogs_z[k] ** 2)  # roll about x: x-coord doesn't matter
    I_66 = OC4_IZZ_COG
    for k in ("tower", "hub", "yaw_bearing", "nacelle", "blades_total"):
        I_66 += masses[k] * cogs_x[k] ** 2

    inertia_at_ref = np.diag([I_44, I_55, I_66])
    cog_offset = np.array([x_G, 0.0, z_G])
    M = rigid_body_mass_matrix(
        mass=m_total,
        inertia_at_reference=inertia_at_ref,
        cog_offset_body=cog_offset,
    )

    # C_55 with combined CoG (gravity contribution recomputed):
    #   C_55_combined = C_55_buoy + (-m_total * g * z_G_combined)
    C = np.zeros((6, 6))
    C[2, 2] = OC4_C33_HEAVE_N_PER_M
    C[3, 3] = (
        OC4_C44_ROLL_NM_PER_RAD
        - OC4_PLATFORM_MASS_KG * G * OC4_COG_BELOW_SWL_M
        + (-m_total * G * z_G)
    )
    C[4, 4] = C55_BUOYANCY_ONLY + (-m_total * G * z_G)

    diag = {
        "label": "B: combined deck",
        "m_total_kg": m_total,
        "z_G_m": z_G,
        "x_G_m": x_G,
        "I_55_at_SWL": I_55,
        "C_55": float(C[4, 4]),
        "C_55_buoy": C55_BUOYANCY_ONLY,
        "components": {k: (masses[k], cogs_z[k], cogs_x[k]) for k in masses},
    }
    return M, C, diag


def _fit_pitch_period(t: np.ndarray, pitch: np.ndarray, n_cycles: int = 10) -> float:
    """Period (s): mean inter-zero-crossing interval over the first n_cycles."""
    pitch_zm = pitch - float(np.mean(pitch[t >= t[-1] - 60.0]))
    signs = np.sign(pitch_zm)
    signs[signs == 0] = 1
    zero_idx = np.where(np.diff(signs) > 0)[0]
    if zero_idx.size < n_cycles + 1:
        raise AssertionError(f"need >= {n_cycles+1} zero crossings; got {zero_idx.size}")
    t_z = t[zero_idx] + (t[zero_idx + 1] - t[zero_idx]) * (
        -pitch_zm[zero_idx] / (pitch_zm[zero_idx + 1] - pitch_zm[zero_idx])
    )
    return float(np.mean(np.diff(t_z[: n_cycles + 1])))


def _run_floatsim(M: np.ndarray, C: np.ndarray, label: str) -> tuple[float, float]:
    hdb = _build_marin_semi_hdb(C)
    lhs = assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb)
    kernel = compute_retardation_kernel(hdb, t_max=200.0, dt=0.05)
    xi0 = np.zeros(6)
    xi0[4] = np.deg2rad(5.0)
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=600.0,
        rho_inf=1.0,
    )
    period = _fit_pitch_period(res.t, res.xi[:, 4], n_cycles=10)
    return period, float(np.max(np.abs(res.xi[:, 4])))


def main() -> None:
    print("M6 PR3 Pre-step -- pitch period gap diagnostic")
    print("=" * 64)
    print(f"OpenFAST reference: {S2_CSV.relative_to(REPO_ROOT)}")
    print("  (Morison drag DISABLED; PtfmSurge=0 IC; PtfmPitch=5 deg IC)")
    print()
    print("Hydrostatic decomposition:")
    print(f"  Robertson C_55 (full):         {OC4_C55_PITCH_NM_PER_RAD:.4e} N*m/rad")
    grav_platform_only = OC4_PLATFORM_MASS_KG * G * OC4_COG_BELOW_SWL_M
    print(f"  Gravity at platform-only:     +{grav_platform_only:.4e} N*m/rad")
    print(f"  Buoyancy-only (.hst-style):    {C55_BUOYANCY_ONLY:.4e} N*m/rad")
    print()

    # OpenFAST reference period.
    data = np.genfromtxt(S2_CSV, delimiter=",", skip_header=1)
    of_period = _fit_pitch_period(data[:, 0], data[:, 5], n_cycles=10)
    print(f"OpenFAST pitch period (drag-off): {of_period:.4f} s")
    print()

    # Setup A.
    M_a, C_a, diag_a = _setup_a_mass_and_C()
    print(f"Setup {diag_a['label']}:")
    print(
        f"  m_total = {diag_a['m_total_kg']:.4e} kg, "
        f"z_G = {diag_a['z_G_m']:+.3f} m, "
        f"x_G = {diag_a['x_G_m']:+.3f} m"
    )
    print(f"  I_55 at SWL = {diag_a['I_55_at_SWL']:.4e} kg*m^2")
    print(f"  C_55 = {diag_a['C_55']:.4e} N*m/rad")
    period_a, peak_a = _run_floatsim(M_a, C_a, "A")
    print(f"  FloatSim period: {period_a:.4f} s")
    print(f"  rel-err vs OpenFAST: {(period_a - of_period) / of_period:+.4f}")
    print(f"  pitch peak amplitude: {np.degrees(peak_a):.4f} deg")
    print()

    # Setup B.
    M_b, C_b, diag_b = _setup_b_mass_and_C()
    print(f"Setup {diag_b['label']}:")
    print(
        f"  m_total = {diag_b['m_total_kg']:.4e} kg, "
        f"z_G = {diag_b['z_G_m']:+.3f} m, "
        f"x_G = {diag_b['x_G_m']:+.3f} m"
    )
    print(f"  I_55 at SWL = {diag_b['I_55_at_SWL']:.4e} kg*m^2")
    print(f"  C_55_buoy = {diag_b['C_55_buoy']:.4e} N*m/rad")
    print(f"  C_55 (full, combined CoG) = {diag_b['C_55']:.4e} N*m/rad")
    print("  components (mass kg | z_G m | x_G m):")
    for k, (m, z, x) in diag_b["components"].items():
        print(f"    {k:>22s}: {m:>10.3e} | {z:+8.3f} | {x:+6.3f}")
    period_b, peak_b = _run_floatsim(M_b, C_b, "B")
    print(f"  FloatSim period: {period_b:.4f} s")
    print(f"  rel-err vs OpenFAST: {(period_b - of_period) / of_period:+.4f}")
    print(f"  pitch peak amplitude: {np.degrees(peak_b):.4f} deg")
    print()

    # Decision tree.
    rerr_b = abs(period_b - of_period) / of_period
    print("Decision tree:")
    if rerr_b < 2.0e-2:
        print(f"  Setup B within rtol=2e-2 ({rerr_b:.4f}) -- F1 fully explains gap.")
        print("  Use combined-deck for PR3 period assertion (fires GREEN; no xfail).")
    elif rerr_b < 5.0e-2:
        print(f"  Setup B within rtol=5e-2 ({rerr_b:.4f}) -- F1 mostly explains.")
        print("  Period assertion xfail-strict under 'F1-residual'; PR3 proceeds.")
    else:
        print(f"  Setup B beyond rtol=5e-2 ({rerr_b:.4f}) -- second deck-identity effect.")
        print("  STOP. Diagnose. Name as F2.")


if __name__ == "__main__":
    main()
