"""M10 PR0.75 -- reference-aware JointSet (plan Amendment A2).

The coupled Cummins system uses ``xi`` as displacement-from-reference
(``xi = 0`` is equilibrium), while the M9 joint layer treated
``xi[6k:6k+3]`` as an absolute world position. ``JointSet.body_references``
supplies each body's world reference so the absolute attach point is
``ref_k + xi_k[0:3] + R_k @ attach`` -- ``None`` (default) is the M9
absolute convention (all references at the origin), unchanged.

GATE 1: ``phi(rest) == 0`` for the real M10 topology.
GATE 2 (the one Option B would have destroyed): a perturbed attachment
point makes ``phi(rest)`` NONZERO at that magnitude -- a geometry error
is visible, not absorbed.
GATE 4: ``G`` is reference-independent (the fix touches ``phi`` only).
GATE 3 (M9 unchanged) lives in test_joints.py / test_m9_*.py, which
construct ``JointSet`` without ``body_references`` (ref=None) and pass
byte-identically.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.bodies.joints import JointSet, yaw_locked_joint

_R = 0.5
_ANG = np.deg2rad([0.0, 120.0, 240.0])
_Z_BUOY = -1.1956674320202696
_Z_ARM = 0.4933695679797303


def _m10_refs() -> tuple[np.ndarray, ...]:
    return tuple(
        [np.array([_R * np.cos(a), _R * np.sin(a), _Z_BUOY]) for a in _ANG]
        + [np.array([0.0, 0.0, _Z_ARM])]
    )


def _m10_joints(perturb_hub0: np.ndarray | None = None) -> tuple:
    js = []
    for i, a in enumerate(_ANG):
        ab = np.array([_R * np.cos(a), _R * np.sin(a), 0.0])
        if i == 0 and perturb_hub0 is not None:
            ab = ab + perturb_hub0
        js.append(
            yaw_locked_joint(
                i, 3, attach_a=[0.0, 0.0, _Z_ARM - _Z_BUOY], attach_b=ab, axis=[0.0, 0.0, 1.0]
            )
        )
    return tuple(js)


def _m10_jointset(perturb_hub0: np.ndarray | None = None) -> JointSet:
    return JointSet(joints=_m10_joints(perturb_hub0), n_bodies=4, body_references=_m10_refs())


# --- GATE 1: phi(rest) == 0 for the real M10 topology ---


def test_phi_rest_zero_m10_topology() -> None:
    js = _m10_jointset()
    phi = js.phi(np.zeros(24))
    assert np.max(np.abs(phi)) < 1e-14, np.max(np.abs(phi))


# --- GATE 2: a geometry error is VISIBLE, not absorbed (Option B would hide it) ---


def test_perturbed_attachment_is_visible() -> None:
    """Perturb one joint's hub attach by 0.1 m; phi(rest) must reflect it
    at that magnitude -- the check that Option A preserves and Option B
    (subtract-the-rest-value) would have zeroed."""
    js = _m10_jointset(perturb_hub0=np.array([0.1, 0.0, 0.0]))
    phi = js.phi(np.zeros(24))
    # the perturbed joint (rows 0:4) now has a 0.1 m translational residual
    assert np.max(np.abs(phi[0:3])) == pytest.approx(0.1, rel=1e-12)
    # the other two joints stay satisfied
    assert np.max(np.abs(phi[4:])) < 1e-14


# --- GATE 4: G is reference-independent (fix touches phi only) ---


def test_G_reference_independent() -> None:
    """The velocity Jacobian G is identical with and without references
    (adding a constant reference does not change dphi/dxi)."""
    joints = _m10_joints()
    js_ref = JointSet(joints=joints, n_bodies=4, body_references=_m10_refs())
    js_abs = JointSet(joints=joints, n_bodies=4)  # ref = None
    rng = np.random.default_rng(0)
    for _ in range(5):
        xi = 0.01 * rng.standard_normal(24)
        np.testing.assert_array_equal(js_ref.jacobian(xi), js_abs.jacobian(xi))


def test_ref_none_reproduces_absolute_convention() -> None:
    """ref=None is the M9 absolute convention: phi(rest) is the old
    nonzero reference-arm value (1.689037), unchanged."""
    js_abs = JointSet(joints=_m10_joints(), n_bodies=4)
    assert np.max(np.abs(js_abs.phi(np.zeros(24)))) == pytest.approx(1.689037, rel=1e-4)


def test_body_references_length_validated() -> None:
    with pytest.raises(ValueError, match=r"body_references has 3 entries; expected"):
        JointSet(joints=_m10_joints(), n_bodies=4, body_references=_m10_refs()[:3])
