"""M7-Foundation PR3 -- F3 catenary state-force composer pinning tests.

Per docs/m7-foundation-plan.md Q4 + Q6 PR3 row:

Step A (the hand prediction) lives in
``scripts/m7_pr3_catenary_prediction.py``. It replicates the
M6 PR5 OC4 fixture geometry verbatim (3 chains, anchors at 837 m
radius at 120 deg spacing, fairleads at 40.8 m on body) and
calls ``solve_catenary`` directly to compute the 6-DOF
generalised force on body 0 at two body poses:

  Point 1: xi_eq = (0, 0, heave_eq, 0, 0, 0)
           PR5 equilibrium. 3-fold symmetric -> Fx, Fy, Mx, Mz
           ~ 0; Fz dominates; My ~ 0 within float64 cancellation
           of large per-line values.

  Point 2: xi_offset = (5 m, 0, heave_eq, 0, 0, 0)
           Discriminator. Asymmetric load -> Fx, My non-trivial.
           A composer that drops the moment block at the fairlead
           arm would PASS Point 1 (symmetry hides the moment) and
           FAIL Point 2.

This file is Step C: the identity test. Numerical targets below
are copy-pasted verbatim from Step A's stdout at the locked
M6 PR5 geometry, rtol = 1e-12 element-wise per the plan.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.mooring.catenary_analytic import (
    CatenaryAttachment,
    CatenaryLine,
    make_catenary_state_force,
)

# ---------------------------------------------------------------------------
# Locked M6 PR5 OC4 geometry (verbatim from test_m6_openfast_moored_eq.py)
# ---------------------------------------------------------------------------

_RHO_KG_M3 = 1025.0
_G_M_S2 = 9.80665
_LINE_DIAM_M = 0.0766
_LINE_MASS_AIR_KG_PER_M = 113.35
_LINE_A_CROSS_M2 = float(np.pi * _LINE_DIAM_M**2 / 4.0)
_LINE_W_SUB_N_PER_M = (_LINE_MASS_AIR_KG_PER_M - _RHO_KG_M3 * _LINE_A_CROSS_M2) * _G_M_S2

_LINE_PROPS = CatenaryLine(
    length=835.35,
    weight_per_length=_LINE_W_SUB_N_PER_M,
    EA=7.536e8,
)
_SEABED_DEPTH_M = 200.0

_ANCHORS_3D = np.array(
    [
        [+418.80, +725.38, -200.0],
        [-837.60, 0.00, -200.0],
        [+418.80, -725.38, -200.0],
    ],
    dtype=np.float64,
)

_FAIRLEADS_BODY = np.array(
    [
        [+20.43, +35.39, -14.0],
        [-40.87, 0.00, -14.0],
        [+20.43, -35.39, -14.0],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Step A targets (verbatim from scripts/m7_pr3_catenary_prediction.py stdout)
# ---------------------------------------------------------------------------

_HEAVE_EQ_M = -0.005449883721945482

_F_TOTAL_AT_XI_EQ = np.array(
    [
        +1.852433035888243e02,
        +1.164153218269348e-10,
        -1.893240928303356e06,
        +1.862645149230957e-09,
        -6.623430609673262e03,
        -4.541107045952231e-09,
    ]
)

_F_TOTAL_AT_XI_OFFSET = np.array(
    [
        -3.887058293932072e05,
        +1.164153218269348e-10,
        -1.906336970537051e06,
        +1.862645149230957e-09,
        +7.355304752684440e05,
        -5.908077582716942e-09,
    ]
)

_F_PER_LINE_AT_XI_EQ = [
    np.array(
        [
            +4.537566315912333e05,
            +7.859214755921257e05,
            -6.310989128499081e05,
            -1.133168986746849e07,
            +6.540757947246356e06,
            -2.071445666618645e03,
        ]
    ),
    np.array(
        [
            -9.073280198788778e05,
            +1.111156355321374e-10,
            -6.310431026035399e05,
            +1.555618897449923e-09,
            -1.308813932510239e07,
            -4.541296024198454e-09,
        ]
    ),
    np.array(
        [
            +4.537566315912333e05,
            -7.859214755921257e05,
            -6.310989128499081e05,
            +1.133168986746849e07,
            +6.540757947246356e06,
            +2.071445666618645e03,
        ]
    ),
]

_F_PER_LINE_AT_XI_OFFSET = [
    np.array(
        [
            +397149.75841734017,
            +696619.8790207198,
            -597109.6317260426,
            -11379031.56049457,
            +6638853.158320288,
            +176814.17800363712,
        ]
    ),
    np.array(
        [
            -1.183005346227888e06,
            +1.448763710632189e-10,
            -7.121177070849660e05,
            +2.028269194885064e-09,
            -1.254217584137213e07,
            -5.921097285353755e-09,
        ]
    ),
    np.array(
        [
            +397149.75841734017,
            -696619.8790207198,
            -597109.6317260426,
            +11379031.56049457,
            +6638853.158320288,
            -176814.17800363712,
        ]
    ),
]


# ---------------------------------------------------------------------------
# Fixture: build the 3 attachments and the n_dof = 6 composer (single body)
# ---------------------------------------------------------------------------


def _build_oc4_attachments() -> list[CatenaryAttachment]:
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


# ---------------------------------------------------------------------------
# Identity tests at xi_eq (Point 1) and xi_offset (Point 2)
# ---------------------------------------------------------------------------


def test_step_C_total_force_at_xi_eq_matches_hand_prediction() -> None:
    """Point 1: at the M6 PR5 equilibrium, the composer must reproduce the
    hand-predicted total 6-vector force to rtol = 1e-12.
    """
    closure = make_catenary_state_force(_build_oc4_attachments(), n_dof=6)
    xi = np.array([0.0, 0.0, _HEAVE_EQ_M, 0.0, 0.0, 0.0])
    F = closure(0.0, xi, np.zeros(6))
    np.testing.assert_allclose(F, _F_TOTAL_AT_XI_EQ, rtol=1.0e-12, atol=1.0e-12)


def test_step_C_total_force_at_xi_offset_matches_hand_prediction() -> None:
    """Point 2 (DISCRIMINATOR): at xi = (5 m, 0, heave_eq, 0, 0, 0), the
    asymmetric load produces non-trivial Fx and My. A composer that
    drops the moment-arm cross-product at the fairlead would still
    pass Point 1 (by 3-fold symmetry) but fail here.
    """
    closure = make_catenary_state_force(_build_oc4_attachments(), n_dof=6)
    xi = np.array([5.0, 0.0, _HEAVE_EQ_M, 0.0, 0.0, 0.0])
    F = closure(0.0, xi, np.zeros(6))
    np.testing.assert_allclose(F, _F_TOTAL_AT_XI_OFFSET, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("line_idx", [0, 1, 2])
def test_step_C_per_line_force_at_xi_eq_matches_hand_prediction(line_idx: int) -> None:
    """Per-line identity at xi_eq -- check each of the 3 lines independently.

    Per-line tests catch a sign-flip on a single line that would be
    cancelled by symmetry in the total. The composer with a single
    attachment must reproduce that line's 6-vector exactly.
    """
    single = [
        CatenaryAttachment(
            body_index=0,
            fairlead_body=_FAIRLEADS_BODY[line_idx].copy(),
            anchor_global=_ANCHORS_3D[line_idx].copy(),
            line=_LINE_PROPS,
            seabed_depth=_SEABED_DEPTH_M,
        )
    ]
    closure = make_catenary_state_force(single, n_dof=6)
    xi = np.array([0.0, 0.0, _HEAVE_EQ_M, 0.0, 0.0, 0.0])
    F = closure(0.0, xi, np.zeros(6))
    np.testing.assert_allclose(F, _F_PER_LINE_AT_XI_EQ[line_idx], rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("line_idx", [0, 1, 2])
def test_step_C_per_line_force_at_xi_offset_matches_hand_prediction(
    line_idx: int,
) -> None:
    """Per-line identity at the discriminator pose."""
    single = [
        CatenaryAttachment(
            body_index=0,
            fairlead_body=_FAIRLEADS_BODY[line_idx].copy(),
            anchor_global=_ANCHORS_3D[line_idx].copy(),
            line=_LINE_PROPS,
            seabed_depth=_SEABED_DEPTH_M,
        )
    ]
    closure = make_catenary_state_force(single, n_dof=6)
    xi = np.array([5.0, 0.0, _HEAVE_EQ_M, 0.0, 0.0, 0.0])
    F = closure(0.0, xi, np.zeros(6))
    np.testing.assert_allclose(
        F, _F_PER_LINE_AT_XI_OFFSET[line_idx], rtol=1.0e-12, atol=1.0e-12
    )


# ---------------------------------------------------------------------------
# Discriminator sanity check (the moment-block-drop catcher)
# ---------------------------------------------------------------------------


def test_xi_offset_has_nonzero_Fx_and_My_distinguishing_from_xi_eq() -> None:
    """Sanity: confirm that Point 2's Fx and My are large enough that a
    composer dropping the moment-arm cross-product would fail at rtol = 1e-12.
    """
    closure = make_catenary_state_force(_build_oc4_attachments(), n_dof=6)
    xi_eq = np.array([0.0, 0.0, _HEAVE_EQ_M, 0.0, 0.0, 0.0])
    xi_off = np.array([5.0, 0.0, _HEAVE_EQ_M, 0.0, 0.0, 0.0])
    F_eq = closure(0.0, xi_eq, np.zeros(6))
    F_off = closure(0.0, xi_off, np.zeros(6))
    # Fx changes from ~ 0 (symmetric) to ~ -3.9e5 N at xi_offset.
    assert abs(F_off[0] - F_eq[0]) > 1.0e5, (
        f"Discriminator weak: Fx change between xi_eq and xi_offset is "
        f"{abs(F_off[0] - F_eq[0]):.3e} N -- expected >> 1e5. Reconsider the "
        "discriminator geometry."
    )
    # My changes from ~ -6.6e3 to ~ +7.4e5 N*m.
    assert abs(F_off[4] - F_eq[4]) > 1.0e5, (
        f"Discriminator weak: My change between xi_eq and xi_offset is "
        f"{abs(F_off[4] - F_eq[4]):.3e} N*m -- expected >> 1e5."
    )


# ---------------------------------------------------------------------------
# Locked-scope error paths
# ---------------------------------------------------------------------------


def test_body_to_body_attachment_raises_at_construction() -> None:
    """body_index < 0 is out of M7-Foundation PR3 scope."""
    with pytest.raises(ValueError, match="body-to-earth only"):
        CatenaryAttachment(
            body_index=-1,
            fairlead_body=np.zeros(3),
            anchor_global=np.array([100.0, 0.0, -200.0]),
            line=_LINE_PROPS,
            seabed_depth=200.0,
        )


def test_out_of_range_body_index_raises() -> None:
    """body_index outside [0, n_bodies) raises at composer construction."""
    att = CatenaryAttachment(
        body_index=2,  # asking for body 2 when n_dof = 6 (only body 0)
        fairlead_body=np.zeros(3),
        anchor_global=np.array([100.0, 0.0, -200.0]),
        line=_LINE_PROPS,
        seabed_depth=200.0,
    )
    with pytest.raises(ValueError, match="outside valid range"):
        make_catenary_state_force([att], n_dof=6)


def test_invalid_n_dof_raises() -> None:
    """n_dof not a positive multiple of 6 raises."""
    att = CatenaryAttachment(
        body_index=0,
        fairlead_body=np.zeros(3),
        anchor_global=np.array([100.0, 0.0, -200.0]),
        line=_LINE_PROPS,
        seabed_depth=200.0,
    )
    with pytest.raises(ValueError, match="multiple of 6"):
        make_catenary_state_force([att], n_dof=5)
    with pytest.raises(ValueError, match="multiple of 6"):
        make_catenary_state_force([att], n_dof=0)


# ---------------------------------------------------------------------------
# Multi-body composition (n_dof = 12)
# ---------------------------------------------------------------------------


def test_multi_body_composition_accumulates_per_body() -> None:
    """Two bodies, each with its own catenary at the OC4 line-1 geometry.
    The composer must write line-1's force on body 0 into xi[0:6] and
    line-1's force on body 1 into xi[6:12], independently.
    """
    att_a = CatenaryAttachment(
        body_index=0,
        fairlead_body=_FAIRLEADS_BODY[1].copy(),
        anchor_global=_ANCHORS_3D[1].copy(),
        line=_LINE_PROPS,
        seabed_depth=_SEABED_DEPTH_M,
    )
    att_b = CatenaryAttachment(
        body_index=1,
        fairlead_body=_FAIRLEADS_BODY[1].copy(),
        anchor_global=_ANCHORS_3D[1].copy(),
        line=_LINE_PROPS,
        seabed_depth=_SEABED_DEPTH_M,
    )
    closure = make_catenary_state_force([att_a, att_b], n_dof=12)
    xi = np.zeros(12)
    xi[2] = _HEAVE_EQ_M
    xi[2 + 6] = _HEAVE_EQ_M  # same heave on body 1
    F = closure(0.0, xi, np.zeros(12))
    np.testing.assert_allclose(F[0:6], _F_PER_LINE_AT_XI_EQ[1], rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(F[6:12], _F_PER_LINE_AT_XI_EQ[1], rtol=1.0e-12, atol=1.0e-12)
