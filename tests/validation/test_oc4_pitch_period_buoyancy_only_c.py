"""Phase 1 latent-bug regression — assembly missing ``m*g*z_G`` gravity term.

Discovered during the M6 OpenFAST cross-check audit (April 2026). The
BEM readers (WAMIT, Capytaine, OrcaFlex VesselType) all produce a
**buoyancy-only** hydrostatic restoring matrix ``C`` and their
docstrings explicitly say "downstream ``floatsim.bodies.Body`` assembly
is expected to add the gravity contribution from the body's mass and
CoG." That downstream step was never implemented:
``floatsim/hydro/hydrostatics.py`` does not exist, and
``assemble_cummins_lhs`` consumes ``hdb.C`` verbatim.

For OC4 DeepCwind (Robertson 2014 NREL/TP-5000-60601 Table 3-3):

    Full C_55 (pitch) = rho*g*(I_yy_wp + V*z_B) - m*g*z_G ~= 1.078e9 N*m/rad
    Gravity contribution alone: -m*g*z_G = +1.78e9 N*m/rad
    Buoyancy-only C_55 (what the WAMIT/Capytaine readers produce)
        ~= 1.078e9 - 1.78e9 ~= -7.02e8 N*m/rad

The buoyancy-only value is **negative** for OC4 (the platform is
statically unstable in pitch without gravity). The M5 PR1 test
``test_marin_semi_trimmed_C_heave_is_positive`` explicitly notes
"Roll/pitch may be negative — that is expected, because the gravity
restoring contribution must be added by the body assembly downstream."

Why the bug stayed invisible through M5
---------------------------------------
No existing test exercised the combination
**(reader-supplied buoyancy-only C) ∧ (non-trivial mass/CoG) ∧
(pitch/roll DOFs)**:

- M1 ``test_oc4_natural_periods`` constructs a synthetic full-restoring
  ``C`` by hand (using Robertson 2014 Table 3-3 values). Bypasses the
  readers entirely.
- M2 ``test_oc4_heave_free_decay`` and ``test_cummins_free_decay_analytical``
  exercise heave only. Heave is ``C_33 = rho*g*A_wp``; gravity
  contributes zero. Bug invisible.
- M3 ``test_m3_regular_wave_steady_state`` uses a synthetic full-C
  fixture, heave-dominated.
- M4 multi-body validations use the OrcaFlex ``platform_small.yml``
  fixture with the same heave-only IC patterns.
- M5 reader unit tests (PR1, PR2) only test parsing — they don't go
  through ``assemble_cummins_lhs``.

This regression test closes the gap. It exercises the
buoyancy-only-C → assemble → physics path with a non-trivial CoG
offset and a pitch DOF. It MUST FAIL on the un-fixed main
(``natural_periods_uncoupled`` returns NaN for negative ``C_55``)
and pass after the fix lands.

Why hand-authored C rather than reading marin_semi.hst directly
---------------------------------------------------------------
``tests/fixtures/bem/wamit/marin_semi_trimmed.hst`` is the OpenFAST
distribution of the OC4 BEM data. Its values are non-dimensional
(WAMIT default with the OpenFAST setup), and the WAMIT reader does
NOT currently apply ULEN-based dimensional rescaling — that's a
separate latent bug, out of scope for this fix. To isolate the
gravity-coupling bug from the dimensional-scaling bug, this test
uses a **hand-authored** ``HydroDatabase`` whose ``C`` is
dimensional and equal to the buoyancy-only contribution for the OC4
case (i.e., what ``marin_semi.hst`` *would* report if dimensional).

After the gravity-coupling fix lands, a separate WAMIT-dimensional
fix can re-route this test through ``read_wamit(_MARIN_STEM)`` for
true end-to-end coverage.

Tolerance
---------
Pitch natural period ``T_55_pub = 26.8 s`` (Robertson 2014, free-decay
in FAST). The M1 sanity-check window is ``[22, 32] s`` — per
``test_oc4_natural_periods`` — to allow for the
``T_inf = 2*pi*sqrt((M+A_inf)/C)`` underestimate vs the
frequency-dependent free-decay period. We reuse the same window here.
"""

from __future__ import annotations

import numpy as np

from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.radiation import assemble_cummins_lhs, natural_periods_uncoupled
from tests.support.synthetic_bem import make_diagonal_hdb
from tests.validation.test_oc4_natural_periods import (
    OC4_AINF_HEAVE,
    OC4_AINF_PITCH,
    OC4_AINF_ROLL,
    OC4_AINF_SURGE,
    OC4_AINF_SWAY,
    OC4_AINF_YAW,
    OC4_C33_HEAVE_N_PER_M,
    OC4_C44_ROLL_NM_PER_RAD,
    OC4_C55_PITCH_NM_PER_RAD,
    OC4_COG_BELOW_SWL_M,
    OC4_IXX_COG,
    OC4_IYY_COG,
    OC4_IZZ_COG,
    OC4_PLATFORM_MASS_KG,
    PITCH_PERIOD_RANGE_S,
)

# Gravity per the deck default (CLAUDE.md §6).
_GRAVITY = 9.80665

# Gravity contribution to roll/pitch restoring (Faltinsen 1990 Eq. 2.104):
#     ΔC_44 = ΔC_55 = -m * g * z_G
# For OC4: z_G = -OC4_COG_BELOW_SWL_M (CoG below SWL reference), so
# ΔC = +m * g * |z_G| > 0 (stabilising).
_C_GRAV_DIAG = OC4_PLATFORM_MASS_KG * _GRAVITY * OC4_COG_BELOW_SWL_M  # +1.78e9 N*m/rad

# Robertson 2014 Table 3-3 reports the FULL restoring (buoyancy + gravity).
# The buoyancy-only contribution — what WAMIT/Capytaine/OrcaFlex readers
# return — is the residual after subtracting gravity:
_OC4_C44_BUOYANCY_ONLY = OC4_C44_ROLL_NM_PER_RAD - _C_GRAV_DIAG  # ≈ -7.0e8 (negative)
_OC4_C55_BUOYANCY_ONLY = OC4_C55_PITCH_NM_PER_RAD - _C_GRAV_DIAG  # ≈ -7.0e8 (negative)


def _oc4_rigid_body_mass_matrix() -> np.ndarray:
    """6x6 OC4 rigid-body mass matrix at the SWL reference point.

    Identical construction to ``test_oc4_natural_periods._oc4_rigid_body_mass_matrix``,
    inlined here so this test does not import private test helpers.
    """
    inertia_at_cog = np.diag([OC4_IXX_COG, OC4_IYY_COG, OC4_IZZ_COG])
    z = -OC4_COG_BELOW_SWL_M
    I_ref = inertia_at_cog.copy()
    # Parallel-axis for r = (0, 0, z): I_xx_ref = I_xx_cog + m * z^2, etc.
    I_ref[0, 0] += OC4_PLATFORM_MASS_KG * z * z
    I_ref[1, 1] += OC4_PLATFORM_MASS_KG * z * z
    return rigid_body_mass_matrix(
        mass=OC4_PLATFORM_MASS_KG,
        inertia_at_reference=I_ref,
    )


def _oc4_buoyancy_only_hdb():
    """Synthetic OC4 HydroDatabase whose ``C`` is buoyancy-only (no gravity).

    Diagonal A_inf and buoyancy-only C; everything else zero. This is what
    a working WAMIT/Capytaine reader would produce for the OC4 case if
    properly dimensional.
    """
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
            OC4_C33_HEAVE_N_PER_M,  # heave is buoyancy-only by definition
            _OC4_C44_BUOYANCY_ONLY,  # roll: full minus gravity
            _OC4_C55_BUOYANCY_ONLY,  # pitch: full minus gravity
            0.0,
        ],
        C_source="buoyancy_only",
    )


# CoG offset from the BEM hydrostatic origin (taken at SWL, axisymmetric):
# (x_G, y_G) = (0, 0); z_G = -OC4_COG_BELOW_SWL_M.
_OC4_COG_OFFSET = np.array([0.0, 0.0, -OC4_COG_BELOW_SWL_M])


def test_oc4_pitch_period_with_buoyancy_only_C_in_physical_range() -> None:
    """OC4 pitch period must fall in 22-32 s when assembling from buoyancy-only ``C``.

    On main (pre-fix), ``assemble_cummins_lhs`` returns ``C`` verbatim.
    Pitch ``C_55`` is negative, so ``natural_periods_uncoupled`` returns
    NaN (its ``c_diag > 0`` filter drops the entry). The assertion below
    fails.

    After the gravity-coupling fix, the assembly adds the
    ``-m*g*z_G`` term to ``C[3,3]`` and ``C[4,4]``, producing a
    stable positive pitch restoring (~1.078e9 N·m/rad). The pitch
    period falls in the published OC4 range.
    """
    M_body = _oc4_rigid_body_mass_matrix()
    hdb = _oc4_buoyancy_only_hdb()

    lhs = assemble_cummins_lhs(
        rigid_body_mass=M_body,
        hdb=hdb,
        mass=OC4_PLATFORM_MASS_KG,
        cog_offset_from_bem_origin=_OC4_COG_OFFSET,
        gravity=_GRAVITY,
    )

    periods = natural_periods_uncoupled(lhs)
    t_pitch = float(periods[4])
    lo, hi = PITCH_PERIOD_RANGE_S
    assert lo <= t_pitch <= hi, (
        f"pitch period {t_pitch!r} s is outside the published OC4 range "
        f"[{lo}, {hi}] s. If the result is NaN, the buoyancy-only C_55 "
        f"({_OC4_C55_BUOYANCY_ONLY:.3e} N*m/rad) is being passed through "
        f"assemble_cummins_lhs without the m*g*z_G gravity term added — "
        f"the Phase 1 latent bug this regression test was written to "
        f"catch. See docs/post-mortems/hydrostatic-gravity-bug.md."
    )


def test_oc4_heave_period_with_buoyancy_only_C_unaffected() -> None:
    """Heave period is unchanged because gravity contributes nothing to ``C_33``.

    This is the reason the bug stayed invisible through M5 — every
    existing dynamic test was heave-dominated. Confirming heave still
    works (with or without the fix) protects against accidentally
    breaking the heave path while wiring up the gravity term.
    """
    M_body = _oc4_rigid_body_mass_matrix()
    hdb = _oc4_buoyancy_only_hdb()
    lhs = assemble_cummins_lhs(
        rigid_body_mass=M_body,
        hdb=hdb,
        mass=OC4_PLATFORM_MASS_KG,
        cog_offset_from_bem_origin=_OC4_COG_OFFSET,
        gravity=_GRAVITY,
    )

    t_heave = float(natural_periods_uncoupled(lhs)[2])
    # Use the same heave window M1 uses (14-20 s).
    assert 14.0 <= t_heave <= 20.0, (
        f"heave period {t_heave:.2f} s outside published OC4 range; "
        f"unexpected — heave should be insensitive to the gravity fix."
    )
