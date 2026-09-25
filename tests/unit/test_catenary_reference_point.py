"""Flume-mooring Phase C2: catenary anchors are TRUE inertial coordinates, whatever the body's
reference point.

The driver's state ``xi`` is the DISPLACEMENT from each body's deck ``reference_point``, but
``make_catenary_state_force`` placed the fairlead at ``xi + arm``: a body whose reference point
is R was moored as if it sat at the origin, i.e. against anchors displaced by -R. The flume
study worked around it by passing ``anchor - R``. With ``body_reference_points`` the fairlead
sits at ``R + xi + arm`` and is measured against the true anchor.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.mooring.catenary_analytic import (
    CatenaryAttachment,
    CatenaryLine,
    make_catenary_state_force,
    solve_catenary,
)

# A flume-like elastic line, a buoy of the 45-degree cluster, a pin-level fairlead and a wall
# anchor at the pin plane.
_LINE = CatenaryLine(length=3.864, weight_per_length=0.3, EA=32.07)
_REF = np.array([0.295, 0.295, -0.907])
_ARM = np.array([0.0, 0.0, 1.624])
_ANCHOR = np.array([5.0, 1.83, 0.717])
_XI = np.array([0.05, -0.02, 0.01, 0.004, -0.006, 0.01])
_ZERO6 = np.zeros(6)


def _att(anchor: np.ndarray) -> CatenaryAttachment:
    return CatenaryAttachment(
        body_index=0,
        fairlead_body=_ARM.copy(),
        anchor_global=np.asarray(anchor, dtype=np.float64),
        line=_LINE,
        seabed_depth=200.0,
    )


def test_fairlead_at_reference_plus_displacement_against_true_anchor() -> None:
    """Pin the geometry: an independent solve with the fairlead at R + xi + (arm + theta x arm)
    and the true anchor gives the same force and moment."""
    f = make_catenary_state_force([_att(_ANCHOR)], n_dof=6, body_reference_points=_REF[None, :])
    F = f(0.0, _XI, _ZERO6)
    r_arm = _ARM + np.cross(_XI[3:], _ARM)
    fair = _REF + _XI[:3] + r_arm
    d = _ANCHOR[:2] - fair[:2]
    span = float(np.hypot(*d))
    sol = solve_catenary(
        line=_LINE,
        anchor_pos=np.array([0.0, _ANCHOR[2]]),
        fairlead_pos=np.array([span, fair[2]]),
        seabed_depth=200.0,
    )
    F3 = np.array([sol.H * d[0] / span, sol.H * d[1] / span, -sol.V_fairlead])
    np.testing.assert_allclose(F[:3], F3, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(F[3:], np.cross(r_arm, F3), rtol=1e-12, atol=1e-14)


def test_equals_the_relative_anchor_workaround() -> None:
    """True anchor + reference points == the pre-C2 workaround (anchor - R, no reference)."""
    f_new = make_catenary_state_force([_att(_ANCHOR)], n_dof=6, body_reference_points=_REF[None, :])
    f_old = make_catenary_state_force([_att(_ANCHOR - _REF)], n_dof=6)
    np.testing.assert_allclose(
        f_new(0.0, _XI, _ZERO6), f_old(0.0, _XI, _ZERO6), rtol=1e-12, atol=1e-14
    )


def test_rigid_translation_of_the_whole_system_leaves_the_force_unchanged() -> None:
    shift = np.array([37.0, -12.5, 0.0])
    f0 = make_catenary_state_force([_att(_ANCHOR)], n_dof=6, body_reference_points=_REF[None, :])
    f1 = make_catenary_state_force(
        [_att(_ANCHOR + shift)], n_dof=6, body_reference_points=(_REF + shift)[None, :]
    )
    np.testing.assert_allclose(f1(0.0, _XI, _ZERO6), f0(0.0, _XI, _ZERO6), rtol=1e-9, atol=1e-12)


def test_origin_reference_points_are_a_bitwise_no_op() -> None:
    f_none = make_catenary_state_force([_att(_ANCHOR)], n_dof=6)
    f_zero = make_catenary_state_force(
        [_att(_ANCHOR)], n_dof=6, body_reference_points=np.zeros((1, 3))
    )
    assert np.array_equal(f_none(0.0, _XI, _ZERO6), f_zero(0.0, _XI, _ZERO6))


def test_the_old_default_misplaces_a_non_origin_body() -> None:
    """Documents the defect C2 removes: without reference points, a body at R with its TRUE
    anchor is moored as if it sat at the origin."""
    f_true = make_catenary_state_force(
        [_att(_ANCHOR)], n_dof=6, body_reference_points=_REF[None, :]
    )
    f_default = make_catenary_state_force([_att(_ANCHOR)], n_dof=6)
    diff = np.abs(f_true(0.0, _XI, _ZERO6)[:3] - f_default(0.0, _XI, _ZERO6)[:3])
    assert diff.max() > 0.1


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.zeros(3), np.full((1, 3), np.nan)])
def test_bad_reference_points_raise(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="body_reference_points"):
        make_catenary_state_force([_att(_ANCHOR)], n_dof=6, body_reference_points=bad)
