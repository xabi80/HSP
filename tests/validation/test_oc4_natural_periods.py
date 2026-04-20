"""Milestone 1 validation — OC4 DeepCwind uncoupled natural periods.

Exercises ``assemble_cummins_lhs`` and ``natural_periods_uncoupled`` on a
synthetic fixture whose inertial, added-mass, and restoring coefficients
are shaped after the OC4 DeepCwind semi-submersible (Robertson et al.
2014, NREL/TP-5000-60601, "Definition of the Semisubmersible Floating
System for Phase II of OC4"). The intent is a physical-realism check,
not a bit-exact reproduction of published free-decay periods.

Why not bit-exact? The published heave period of 17.3 s and pitch period
of ~26.8 s come from time-domain free-decay simulations in FAST/OrcaFlex.
Those match ``T = 2*pi*sqrt((M + A(omega_n)) / C)`` where ``A(omega_n)``
is the added mass at the natural frequency. The Milestone-1 helper
evaluates the cheaper ``T_inf = 2*pi*sqrt((M + A_inf) / C)`` using the
infinite-frequency limit. For a semi-submersible, ``A_inf`` is typically
10-30% smaller than ``A(omega_n)`` in heave, so ``T_inf`` is 5-15% below
the free-decay period — an expected, documented approximation.

The proper numerical reproduction belongs to Milestone 2, where the
retardation kernel and convolution evaluate the full frequency-dependent
response.

References
----------
Robertson, A. et al., 2014. "Definition of the Semisubmersible Floating
System for Phase II of OC4." NREL/TP-5000-60601.
Jonkman, J., 2009. "Dynamics of Offshore Floating Wind Turbines — Model
Development and Verification." Wind Energy 12(5):459-492.
"""

from __future__ import annotations

import numpy as np

from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.radiation import (
    assemble_cummins_lhs,
    natural_periods_uncoupled,
)
from tests.support.synthetic_bem import make_diagonal_hdb

# ---------------------------------------------------------------------------
# OC4 DeepCwind platform properties (hull + ballast, excluding tower/RNA).
# Values below are from Robertson 2014 Table 3-1 (mass, inertia) and
# Table 3-3 (hydrostatic stiffness diagonal entries). All in SI.
# ---------------------------------------------------------------------------

OC4_PLATFORM_MASS_KG = 1.3473e7
OC4_COG_BELOW_SWL_M = 13.46  # |z_CoG| in metres
OC4_IXX_COG = 6.827e9  # kg*m^2
OC4_IYY_COG = 6.827e9
OC4_IZZ_COG = 1.226e10

# Hydrostatic restoring, platform only at equilibrium about the body
# reference point (taken at SWL, CoG directly below). Only heave, roll,
# and pitch have non-zero diagonal entries for an unmoored free body.
OC4_C33_HEAVE_N_PER_M = 3.836e6
OC4_C44_ROLL_NM_PER_RAD = 1.078e9
OC4_C55_PITCH_NM_PER_RAD = 1.078e9

# Infinite-frequency added-mass diagonal entries (order-of-magnitude
# representatives in the range reported for OC4 DeepCwind BEM analyses).
# Surge/sway ~ 8-10e6 kg, heave ~ 1.1e7 kg, roll/pitch ~ 7.3e9 kg*m^2,
# yaw ~ 0 for this axisymmetric platform.
OC4_AINF_SURGE = 9.0e6
OC4_AINF_SWAY = 9.0e6
OC4_AINF_HEAVE = 1.09e7
OC4_AINF_ROLL = 7.3e9
OC4_AINF_PITCH = 7.3e9
OC4_AINF_YAW = 0.0

# Physically-plausible ranges for the uncoupled-at-A_inf formula. These
# are generous — the formula underestimates the free-decay period, so
# 14-20 s brackets both the approximation floor and the published 17.3 s
# upper anchor.
HEAVE_PERIOD_RANGE_S = (14.0, 20.0)
PITCH_PERIOD_RANGE_S = (22.0, 32.0)


def _oc4_rigid_body_mass_matrix() -> np.ndarray:
    """6x6 rigid-body mass matrix for the OC4 platform, reference at SWL."""
    inertia_at_cog = np.diag([OC4_IXX_COG, OC4_IYY_COG, OC4_IZZ_COG])
    # CoG is directly below the reference point at z = -OC4_COG_BELOW_SWL_M.
    # Parallel-axis: I_ref = I_cog + m * (|r|^2 * I_3 - r r^T) is handled by
    # rigid_body_mass_matrix internally through the skew-coupling blocks;
    # the diagonal inertia_at_reference captures only I_cog, and the
    # cog_offset_body arg inserts the cross-coupling -m*r_tilde. For the
    # uncoupled pitch natural period we need the full diagonal (M55) at the
    # reference, so we add the parallel-axis term explicitly here.
    z = -OC4_COG_BELOW_SWL_M
    I_ref = inertia_at_cog.copy()
    # Parallel-axis on a diagonal inertia with r = (0, 0, z):
    #   I_xx_ref = I_xx_cog + m * z^2
    #   I_yy_ref = I_yy_cog + m * z^2
    #   I_zz_ref = I_zz_cog (unchanged, rotation axis parallel to r)
    I_ref[0, 0] += OC4_PLATFORM_MASS_KG * z * z
    I_ref[1, 1] += OC4_PLATFORM_MASS_KG * z * z
    return rigid_body_mass_matrix(
        mass=OC4_PLATFORM_MASS_KG,
        inertia_at_reference=I_ref,
        # No cog_offset: we have already moved inertia to the reference point,
        # so the mass matrix is block-diagonal about that same point.
    )


def _oc4_hydro_database():
    """Diagonal synthetic HydroDatabase with OC4-shaped A_inf and C."""
    return make_diagonal_hdb(
        A_inf_diag=[
            OC4_AINF_SURGE,
            OC4_AINF_SWAY,
            OC4_AINF_HEAVE,
            OC4_AINF_ROLL,
            OC4_AINF_PITCH,
            OC4_AINF_YAW,
        ],
        C_diag=[
            0.0,
            0.0,
            OC4_C33_HEAVE_N_PER_M,
            OC4_C44_ROLL_NM_PER_RAD,
            OC4_C55_PITCH_NM_PER_RAD,
            0.0,
        ],
        metadata={"source": "OC4 DeepCwind (Robertson 2014, synthetic)"},
    )


# ---------- algebraic consistency on OC4-shaped inputs ----------


def test_oc4_natural_periods_match_analytical_formula() -> None:
    """Uncoupled helper must equal 2*pi*sqrt((M+A_inf)/C) on OC4 fixture."""
    M_body = _oc4_rigid_body_mass_matrix()
    hdb = _oc4_hydro_database()
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    periods = natural_periods_uncoupled(lhs)

    m_plus_a = np.diag(lhs.M_plus_Ainf)
    c = np.diag(lhs.C)
    # Heave (index 2), roll (3), pitch (4) have non-zero restoring.
    for i in (2, 3, 4):
        expected = 2.0 * np.pi * np.sqrt(m_plus_a[i] / c[i])
        np.testing.assert_allclose(periods[i], expected, rtol=1e-12)

    # Surge, sway, yaw have C_ii = 0 → NaN.
    for i in (0, 1, 5):
        assert np.isnan(periods[i])


# ---------- physical realism: heave and pitch in published ranges ----------


def test_oc4_heave_period_in_physical_range() -> None:
    """Computed heave period must fall inside the 14-20 s band for OC4 DeepCwind.

    Published free-decay heave period: 17.3 s (Robertson 2014, Table 3-3).
    The uncoupled-at-A_inf formula is expected to sit below this value by
    5-15% because A_inf < A(omega_heave) for a semi-submersible.
    """
    M_body = _oc4_rigid_body_mass_matrix()
    hdb = _oc4_hydro_database()
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    periods = natural_periods_uncoupled(lhs)

    t_heave = float(periods[2])
    lo, hi = HEAVE_PERIOD_RANGE_S
    assert lo <= t_heave <= hi, (
        f"heave period {t_heave:.2f} s outside physical range [{lo}, {hi}] s; "
        "check OC4 fixture M, A_inf, C_33 entries."
    )


def test_oc4_pitch_period_in_physical_range() -> None:
    """Computed pitch period must fall inside the 22-32 s band for OC4 DeepCwind.

    Published free-decay pitch period: ~26.8 s (Robertson 2014). The
    uncoupled-at-A_inf formula is expected to sit a few percent below this
    value; range is widened to account for variability in published
    A_inf(pitch) values across BEM implementations (7-9e9 kg*m^2).
    """
    M_body = _oc4_rigid_body_mass_matrix()
    hdb = _oc4_hydro_database()
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    periods = natural_periods_uncoupled(lhs)

    t_pitch = float(periods[4])
    lo, hi = PITCH_PERIOD_RANGE_S
    assert lo <= t_pitch <= hi, (
        f"pitch period {t_pitch:.2f} s outside physical range [{lo}, {hi}] s; "
        "check OC4 fixture I_yy_ref, A_inf, C_55 entries."
    )


def test_oc4_horizontal_and_yaw_periods_are_nan() -> None:
    """Surge, sway, yaw carry no restoring without mooring — periods must be NaN."""
    M_body = _oc4_rigid_body_mass_matrix()
    hdb = _oc4_hydro_database()
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    periods = natural_periods_uncoupled(lhs)
    for i in (0, 1, 5):
        assert np.isnan(periods[i])
