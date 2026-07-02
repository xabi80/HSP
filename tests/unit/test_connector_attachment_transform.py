"""M7-Foundation PR2 -- F2 attachment-offset transform pinning tests.

Per docs/m7-foundation-plan.md Q3 + Q6 PR2 row:

  T = [ I_3   -r_tilde ]
      [ 0      I_3      ]

  K_ref       = T.T @ K_attach @ T
  rest_off_LC = T^{-1} @ rest_offset_attach     where T^{-1} = [I, +r_tilde; 0, I]

Identity test pins the closed-form transform at rtol = 1e-12 on
BOTH K_ref AND the 6-vector F_ref (force + moment). Property test
(hypothesis) verifies F_ref consistency for random SPD K_attach,
random small-angle xi (bounded to |theta| < 0.1 rad per Q3
validity), and random body-frame arms.

Framework constraint surfaced during derivation
-----------------------------------------------
``LinearConnector`` assumes symmetric Newton-III at reference
points (``F_b = -F_a`` exactly). With a non-zero attachment arm on
body A, the moment-arm cross-product gives ``F_a_ref`` a moment
block that ``F_b_ref`` (at its reference, no arm) does not see.
This makes the body-body-with-non-zero-offset case impossible to
express in the existing framework — F2 raises NotImplementedError
for it. F2's locked scope is therefore:

  (i)  both attach offsets zero -> degenerate to identity (M4 PR3
       reference-to-reference case).
  (ii) body-earth with single offset on the non-earth side -> F2
       transform applies (catenary fairlead use case).

This is consistent with Q3's locked singular-arm derivation; not a
Q3 re-open.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from floatsim.bodies.connector import (
    LinearConnector,
    assemble_attachment_transformed_connector,
)


# ---------------------------------------------------------------------------
# helpers -- closed-form T, T_inv, force pull-back
# ---------------------------------------------------------------------------


def _skew(r: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric cross-product matrix. r_tilde @ x = r x x."""
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    return np.array(
        [[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]], dtype=np.float64
    )


def _T_matrix(r: np.ndarray) -> np.ndarray:
    """T = [I_3, -r_tilde; 0, I_3]."""
    T = np.zeros((6, 6), dtype=np.float64)
    T[:3, :3] = np.eye(3)
    T[:3, 3:] = -_skew(r)
    T[3:, 3:] = np.eye(3)
    return T


def _T_inv_matrix(r: np.ndarray) -> np.ndarray:
    """T^{-1} = [I_3, +r_tilde; 0, I_3]."""
    T_inv = np.zeros((6, 6), dtype=np.float64)
    T_inv[:3, :3] = np.eye(3)
    T_inv[:3, 3:] = +_skew(r)
    T_inv[3:, 3:] = np.eye(3)
    return T_inv


# ---------------------------------------------------------------------------
# Identity tests (the locked-spec pre-flight pinning)
# ---------------------------------------------------------------------------


def test_q6_pr2_K_ref_matches_closed_form_unit_translation_at_1m_arm() -> None:
    """Q6 PR2 row: closed-form K_ref = T^T @ K_attach @ T for the
    canonical fixture (heave-only K at a 1 m arm along +Y).

    K_attach = diag(0, 0, k, 0, 0, 0); r = (0, 1, 0).
    The cross-product moment-arm couples heave-translation to
    pitch-moment via T's -r_tilde block.
    """
    k = 1.0e6
    K_attach = np.diag([0.0, 0.0, k, 0.0, 0.0, 0.0])
    B_attach = np.zeros((6, 6))
    r = np.array([0.0, 1.0, 0.0])  # 1 m arm along +Y

    conn = assemble_attachment_transformed_connector(
        body_a=0,
        body_b=-1,  # earth
        K_attach=K_attach,
        B_attach=B_attach,
        attach_a_body=r,
    )

    T = _T_matrix(r)
    K_ref_expected = T.T @ K_attach @ T
    np.testing.assert_allclose(conn.K, K_ref_expected, rtol=1.0e-12, atol=1.0e-12)


def test_q6_pr2_F_ref_translational_block_matches_hand_derivation() -> None:
    """Q3 discriminator: heave K at 1 m arm in +Y, under unit attachment
    translation in heave -> force on body's reference must have:

    * heave translational component  = -k * delta_z  (the spring force)
    * pitch-moment (around +Y, axis perpendicular to both heave and arm)

    Hand derivation: T @ delta_ref recovers attachment delta; F_attach
    = -K @ T @ delta_ref; F_a_ref = T^T @ F_attach. With r = +Y, the
    moment block is r x F_attach[:3].
    """
    k = 1.0e6
    K_attach = np.diag([0.0, 0.0, k, 0.0, 0.0, 0.0])
    r = np.array([0.0, 1.0, 0.0])
    conn = assemble_attachment_transformed_connector(
        body_a=0, body_b=-1, K_attach=K_attach, attach_a_body=r,
    )

    # Pure heave translation at the reference point: xi = [0, 0, 1, 0, 0, 0].
    delta_ref = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    F_a_ref = -conn.K @ delta_ref  # LinearConnector convention: F_a = -K @ delta

    # Translational components: spring pulls body BACK with k * delta_z.
    np.testing.assert_allclose(F_a_ref[0], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(F_a_ref[1], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(F_a_ref[2], -k * 1.0, rtol=1.0e-12)
    # Moment block: hand-derived via r x F_attach.
    # delta_attach = T @ delta_ref. r_tilde @ theta=0 = 0, so delta_attach
    # is the same as delta_ref (pure translation through the lever).
    # F_attach = -K @ delta_attach = [0, 0, -k, 0, 0, 0].
    # F_a_ref_moment = r x F_attach[:3] = (0,1,0) x (0,0,-k) = (-k, 0, 0).
    # (Compute by hand: e_y x e_z = e_x, with sign from -k: (-k)*e_x.)
    np.testing.assert_allclose(F_a_ref[3], -k, rtol=1.0e-12)
    np.testing.assert_allclose(F_a_ref[4], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(F_a_ref[5], 0.0, atol=1.0e-12)


def test_q6_pr2_F_ref_moment_block_NOT_dropped_by_T_T_transform() -> None:
    """**The whole point of F2.** A translational-only assertion would
    pass against a T that drops the -r_tilde block; the moment block
    is what F2 contributes. We assert here that the moment block is
    NON-ZERO for the canonical 1-m-arm fixture under translation.

    Per plan Q3 + PR2 description: a discriminator that catches the
    most likely implementation mistake (forgetting to apply T^T's
    bottom-left r_tilde block when transferring force).
    """
    k = 1.0e6
    K_attach = np.diag([k, 0.0, 0.0, 0.0, 0.0, 0.0])  # surge only
    r = np.array([0.0, 0.0, 2.0])  # 2 m arm along +Z (vertical)
    conn = assemble_attachment_transformed_connector(
        body_a=0, body_b=-1, K_attach=K_attach, attach_a_body=r,
    )

    # Surge translation at reference. Expected moment: r x F = e_z x (-k*e_x)
    # = -k * (e_z x e_x) = -k * e_y. So F_a_ref[4] = -k * 2 (pitch moment),
    # which must be NON-ZERO and equal -2k = -2e6 N*m.
    delta_ref = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    F_a_ref = -conn.K @ delta_ref
    assert abs(F_a_ref[4]) > 1.0e3, (
        f"pitch-moment block is zero ({F_a_ref[4]:.3e}); F2 dropped the -r_tilde "
        "block of T. This is the M7 PR2 moment-transfer discriminator -- the "
        "whole point of F2."
    )
    np.testing.assert_allclose(F_a_ref[4], -k * 2.0, rtol=1.0e-12)


def test_q6_pr2_rest_offset_transforms_through_T_inv() -> None:
    """rest_offset_LC = T^{-1} @ rest_offset_attach, derived in plan Q3.
    Verify the closed-form on a non-trivial rest_offset_attach.
    """
    K_attach = np.eye(6) * 1.0e5
    r = np.array([1.0, 0.5, -0.3])
    rest_attach = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03])
    conn = assemble_attachment_transformed_connector(
        body_a=0,
        body_b=-1,
        K_attach=K_attach,
        attach_a_body=r,
        rest_offset_attach=rest_attach,
    )
    T_inv = _T_inv_matrix(r)
    expected = T_inv @ rest_attach
    np.testing.assert_allclose(conn.rest_offset, expected, rtol=1.0e-12, atol=1.0e-12)


def test_q6_pr2_B_attach_transforms_through_same_T() -> None:
    """Damping transforms identically to stiffness: B_ref = T^T @ B_attach @ T."""
    K_attach = np.eye(6) * 1.0e5  # full-rank (required for clean K transform)
    B_attach = np.eye(6) * 1.0e3
    B_attach[2, 4] = 5.0e2  # off-diagonal to make the transform visible
    B_attach[4, 2] = 5.0e2
    r = np.array([0.0, 0.0, 1.5])
    conn = assemble_attachment_transformed_connector(
        body_a=0, body_b=-1, K_attach=K_attach, B_attach=B_attach, attach_a_body=r,
    )
    T = _T_matrix(r)
    np.testing.assert_allclose(conn.B, T.T @ B_attach @ T, rtol=1.0e-12, atol=1.0e-12)


def test_q6_pr2_zero_offset_degenerates_to_identity() -> None:
    """Both attach offsets None / zero -> F2 returns LinearConnector identical
    to input K_attach, B_attach, rest_offset_attach. Covers the M4 PR3
    reference-to-reference body-body case.
    """
    K_attach = np.diag([1.0e6, 2.0e6, 3.0e6, 4.0e8, 5.0e8, 6.0e8])
    B_attach = K_attach * 1.0e-3
    rest_attach = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    conn = assemble_attachment_transformed_connector(
        body_a=0,
        body_b=1,  # body-body OK when both offsets are zero
        K_attach=K_attach,
        B_attach=B_attach,
        rest_offset_attach=rest_attach,
    )
    np.testing.assert_array_equal(conn.K, K_attach)
    np.testing.assert_array_equal(conn.B, B_attach)
    np.testing.assert_array_equal(conn.rest_offset, rest_attach)


def test_q6_pr2_default_B_is_zero() -> None:
    """Optional B_attach defaults to a zero 6x6 (matches LinearConnector usage)."""
    K_attach = np.eye(6) * 1.0e5
    r = np.array([0.0, 0.0, 1.0])
    conn = assemble_attachment_transformed_connector(
        body_a=0, body_b=-1, K_attach=K_attach, attach_a_body=r,
    )
    np.testing.assert_array_equal(conn.B, np.zeros((6, 6)))


def test_q6_pr2_body_b_offset_with_body_a_earth_also_supported() -> None:
    """Symmetric configuration: body_a = earth, body_b = real, offset on b.
    Should produce the same K_ref / rest_offset as the a-real / b-earth
    case (the transform is per-arm; LinearConnector body_a/body_b just
    tags the slot).
    """
    K_attach = np.diag([1.0e6, 1.0e6, 1.0e6, 0.0, 0.0, 0.0])
    r = np.array([0.0, 1.0, 0.0])
    rest_attach = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])

    conn_a_real = assemble_attachment_transformed_connector(
        body_a=0,
        body_b=-1,
        K_attach=K_attach,
        attach_a_body=r,
        rest_offset_attach=rest_attach,
    )
    conn_b_real = assemble_attachment_transformed_connector(
        body_a=-1,
        body_b=0,
        K_attach=K_attach,
        attach_b_body=r,
        rest_offset_attach=rest_attach,
    )
    # Same K and rest_offset; only the body_a/body_b slot tags differ.
    np.testing.assert_allclose(conn_a_real.K, conn_b_real.K, rtol=1.0e-12)
    np.testing.assert_allclose(
        conn_a_real.rest_offset, conn_b_real.rest_offset, rtol=1.0e-12
    )


# ---------------------------------------------------------------------------
# Out-of-scope error paths
# ---------------------------------------------------------------------------


def test_body_body_with_nonzero_offset_raises_notimplemented() -> None:
    """Framework constraint: body-body LinearConnector with any non-zero
    attachment offset cannot be represented (Newton-III asymmetry).
    Raise NotImplementedError explicitly rather than silently returning
    a connector that drops physics.
    """
    K_attach = np.eye(6) * 1.0e5
    r = np.array([0.0, 0.0, 1.0])
    with pytest.raises(NotImplementedError, match="body-body"):
        assemble_attachment_transformed_connector(
            body_a=0, body_b=1, K_attach=K_attach, attach_a_body=r,
        )


def test_both_offsets_nonzero_raises_notimplemented() -> None:
    """Even body-earth with non-zero offset on BOTH attach_a_body and
    attach_b_body: not supported. Framework extension required.
    """
    K_attach = np.eye(6) * 1.0e5
    with pytest.raises(NotImplementedError, match="single non-zero attachment offset"):
        assemble_attachment_transformed_connector(
            body_a=0,
            body_b=1,
            K_attach=K_attach,
            attach_a_body=np.array([1.0, 0.0, 0.0]),
            attach_b_body=np.array([0.0, 1.0, 0.0]),
        )


def test_both_endpoints_earth_raises_valueerror() -> None:
    """body_a = body_b = -1 is meaningless."""
    K_attach = np.eye(6) * 1.0e5
    with pytest.raises(ValueError, match="both endpoints earth|requires distinct"):
        assemble_attachment_transformed_connector(
            body_a=-1, body_b=-1, K_attach=K_attach,
        )


def test_offset_on_earth_side_raises_valueerror() -> None:
    """attach_a_body non-zero but body_a = earth is a deck-config error."""
    K_attach = np.eye(6) * 1.0e5
    with pytest.raises(ValueError, match="earth"):
        assemble_attachment_transformed_connector(
            body_a=-1,
            body_b=0,
            K_attach=K_attach,
            attach_a_body=np.array([1.0, 0.0, 0.0]),
        )


# ---------------------------------------------------------------------------
# Property test (hypothesis) -- F_a_ref consistency for random SPD K,
# bounded arm, bounded small-angle xi
# ---------------------------------------------------------------------------


_K_DIAG_SCALE = 1.0e5  # representative stiffness magnitude
_ARM_MAX_M = 5.0  # bound the arm so r_tilde norm stays sane
_THETA_MAX_RAD = 0.05  # well inside Q3's 0.1 rad linearisation bound
_TRANS_MAX_M = 1.0  # bound translation for numerical hygiene


@st.composite
def _spd_6x6_K(draw) -> np.ndarray:
    """Symmetric positive-semidefinite 6x6 K via L @ L.T from random L.

    Property-test hygiene per Xabier (PR2 implementation note): generic
    random 6x6 will fail not because F2 is wrong but because physical
    stiffness matrices are SPD by construction. L lower-triangular with
    bounded entries gives K = L @ L.T which is SPD.
    """
    L = draw(
        arrays(
            dtype=np.float64,
            shape=(6, 6),
            elements=st.floats(
                min_value=-_K_DIAG_SCALE, max_value=_K_DIAG_SCALE,
                allow_nan=False, allow_infinity=False,
            ),
        )
    )
    L = np.tril(L)
    return L @ L.T


@st.composite
def _bounded_arm(draw) -> np.ndarray:
    """3-vector r with |r| <= _ARM_MAX_M."""
    r = draw(
        arrays(
            dtype=np.float64,
            shape=(3,),
            elements=st.floats(
                min_value=-_ARM_MAX_M, max_value=_ARM_MAX_M,
                allow_nan=False, allow_infinity=False,
            ),
        )
    )
    return r


@st.composite
def _bounded_xi(draw) -> np.ndarray:
    """6-vector xi with translations bounded and rotations within Q3 range."""
    trans = draw(
        arrays(
            dtype=np.float64,
            shape=(3,),
            elements=st.floats(
                min_value=-_TRANS_MAX_M, max_value=_TRANS_MAX_M,
                allow_nan=False, allow_infinity=False,
            ),
        )
    )
    rot = draw(
        arrays(
            dtype=np.float64,
            shape=(3,),
            elements=st.floats(
                min_value=-_THETA_MAX_RAD, max_value=_THETA_MAX_RAD,
                allow_nan=False, allow_infinity=False,
            ),
        )
    )
    return np.concatenate([trans, rot])


@given(K_attach=_spd_6x6_K(), r=_bounded_arm(), xi=_bounded_xi())
@settings(max_examples=100, deadline=2000)
def test_property_F_ref_equals_T_pullback_of_F_attach(
    K_attach: np.ndarray, r: np.ndarray, xi: np.ndarray
) -> None:
    """For random SPD K, random bounded arm, random bounded xi:
    the force on body A's reference produced by F2's connector
    equals T^T @ F_attach, where F_attach = -K_attach @ T @ xi.

    Body-earth configuration. Asserts BOTH translational and moment
    components of F_a_ref at rtol = 1e-8 (looser than the identity-test
    1e-12 to accommodate hypothesis-induced floating-point noise in
    K @ T @ ... compositions; the pre-fix 1e-9 gate was just barely
    exceeded by hypothesis-found corner cases combining |r| ~ 1e-8
    with small non-trivial rotations, per the M7.5 pre-milestone-audit
    baseline pytest surfacing).
    """
    conn = assemble_attachment_transformed_connector(
        body_a=0, body_b=-1, K_attach=K_attach, attach_a_body=r,
    )

    # F2's force: F_a_LC = -K_ref @ (xi_a - 0 - rest_offset) = -K_ref @ xi (rest=0)
    F_a_F2 = -conn.K @ xi

    # Hand-computed reference: T @ xi -> F_attach -> T^T @ F_attach.
    T = _T_matrix(r)
    delta_attach = T @ xi
    F_attach = -K_attach @ delta_attach
    F_a_hand = T.T @ F_attach

    np.testing.assert_allclose(F_a_F2, F_a_hand, rtol=1.0e-8, atol=1.0e-8)


@given(K_attach=_spd_6x6_K(), r=_bounded_arm())
@settings(max_examples=100, deadline=2000)
def test_property_K_ref_is_symmetric_positive_semidefinite(
    K_attach: np.ndarray, r: np.ndarray,
) -> None:
    """T^T @ K @ T preserves symmetry and positive-semidefiniteness:
    a physical stiffness in -> physical stiffness out.
    """
    conn = assemble_attachment_transformed_connector(
        body_a=0, body_b=-1, K_attach=K_attach, attach_a_body=r,
    )
    # Symmetry.
    np.testing.assert_allclose(conn.K, conn.K.T, rtol=1.0e-9, atol=1.0e-9)
    # SPD via eigenvalue non-negativity (LinearConnector requires
    # symmetric only, but physically K_ref should be SPSD too).
    eigs = np.linalg.eigvalsh(0.5 * (conn.K + conn.K.T))
    assert np.all(eigs > -1.0e-6 * (1.0 + np.max(np.abs(conn.K)))), (
        f"K_ref has a negative eigenvalue ({np.min(eigs):.3e}); T^T @ K_attach @ T "
        "should preserve positive-semidefiniteness."
    )


# ---------------------------------------------------------------------------
# Compatibility: F2-built connector composes with make_connector_state_force
# ---------------------------------------------------------------------------


def test_f2_connector_round_trips_through_make_connector_state_force() -> None:
    """F2 returns a regular LinearConnector. It must compose unmodified
    with floatsim.bodies.connector.make_connector_state_force at any
    valid n_dof; smoke test at n_dof = 6 (single body, earth-attached).
    """
    from floatsim.bodies.connector import make_connector_state_force

    K_attach = np.diag([1.0e5, 1.0e5, 1.0e5, 0.0, 0.0, 0.0])
    r = np.array([0.0, 1.0, 0.0])
    conn = assemble_attachment_transformed_connector(
        body_a=0, body_b=-1, K_attach=K_attach, attach_a_body=r,
    )
    assert isinstance(conn, LinearConnector)

    state_force = make_connector_state_force([conn], n_dof=6)
    xi = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    xi_dot = np.zeros(6)
    F = state_force(0.0, xi, xi_dot)
    assert F.shape == (6,)
    # Sanity: with a 1-m arm in Y and 1-m heave displacement, expect
    # non-zero pitch-moment per the discriminator (mirror of
    # test_q6_pr2_F_ref_moment_block_NOT_dropped: arm +Y, force +Z -> moment -X).
    # r x F[:3] = (0,1,0) x (0,0,-k) = (-k, 0, 0). Pitch moment block index 4 here.
    assert not np.allclose(F[3:], 0.0), (
        "F2-built connector lost moment block when composed through "
        "make_connector_state_force"
    )
