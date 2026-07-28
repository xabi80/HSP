"""M9 PR4 -- two-body terminal gate: the double pendulum (Q4 BB-OFFSET closure).

The single-body terminal gates (hinge period, lambda static recovery,
energy, drift, unconstrained byte-identity, yaw_locked) live in
``test_m9_kkt_integrator.py`` (PR2). This module adds the **two-body**
gate the plan reserves for PR4:

* **Double pendulum modes** (plan Measurement MB, Terminal Gate 1
  two-body form): two point masses ``m`` on massless rods ``l``, body 0
  hinged to earth, body 1 hinged to body 0's CoM, both about ``y``.
  Small-oscillation normal modes ``omega^2 = (g/l)(2 -/+ sqrt2)`` give
  closed-form periods ``T_+ = 2.621052 s`` / ``T_- = 1.085675 s``
  (scipy ``solve_ivp`` cross-check 1.8e-6 / 2.1e-6, plan Phase 1). The
  reference is analytic, never FloatSim's own output.

* **BB-OFFSET-CONNECTOR closure cross-check** (plan Q4): the inter-body
  hinge attaches at an offset from body 1's CoG -- exactly the topology
  the penalty ``LinearConnector`` cannot represent
  (``docs/phase2-followups.md#bb-offset-connector``). The test asserts
  BOTH halves of the closure claim in one place: the penalty path still
  raises ``NotImplementedError``; the joint (multiplier) path holds the
  same offset constraint at machine precision.

Point-mass note: a point mass has a singular rotational inertia block,
which makes the bordered KKT solve singular. A tiny isotropic
regularization ``I_c = m l^2 * 1e-6`` restores well-posedness with **no
measurable effect on the periods** (identical to 5 sig figs at
``I_c = 1e-6`` and ``1e-5`` -- the mode inertia is carried by CoM
translation, not the body's own spin). Gate amplitude ``theta0 = 0.01``
(smaller than the single hinge's 0.02) keeps the faster mode's
finite-amplitude term well under the rtol-1e-3 gate: the second bob
swings ``sqrt2`` x larger, so at 0.02 the ``T_-`` finite-amplitude
offset sits at ~9e-4, right at the gate; 0.01 gives a 2x margin.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from floatsim.bodies.joints import JointSet, hinge_joint
from floatsim.hydro.radiation import CumminsLHS
from floatsim.hydro.retardation import RetardationKernel
from floatsim.solver.newmark import integrate_cummins

_G = 9.81
_M = 1.0
_L = 1.0
_IC_REG = _M * _L**2 * 1.0e-6  # KKT regularization; period-insensitive
_DT = 0.01
_THETA0 = 0.01

# Closed-form small-oscillation modes (angles from vertical, equal m/l).
_W_PLUS = np.sqrt(_G / _L * (2.0 - np.sqrt(2.0)))
_W_MINUS = np.sqrt(_G / _L * (2.0 + np.sqrt(2.0)))
_T_PLUS = 2.0 * np.pi / _W_PLUS  # 2.621052 s (in-phase, low freq)
_T_MINUS = 2.0 * np.pi / _W_MINUS  # 1.085675 s (out-of-phase, high freq)
# Mode shapes theta2/theta1 = +sqrt2 (low freq) / -sqrt2 (high freq).
_RATIO_PLUS = +np.sqrt(2.0)
_RATIO_MINUS = -np.sqrt(2.0)

_M6 = np.diag([_M, _M, _M, _IC_REG, _IC_REG, _IC_REG]).astype(np.float64)
_LHS = CumminsLHS(M_plus_Ainf=np.kron(np.eye(2), _M6), C=np.zeros((12, 12)))

# Hinge 1: body 0 -> earth at the origin (pivot at +l above body-0 CoM).
# Hinge 2: body 0 -> body 1; attaches at body 0's CoM ([0,0,0] in body-0
# frame) and at +l above body-1 CoM -- an offset from body 1's CoG.
_HINGES = JointSet(
    joints=(
        hinge_joint(0, -1, attach_a=[0, 0, _L], attach_b=[0, 0, 0], axis=[0, 1, 0]),
        hinge_joint(0, 1, attach_a=[0, 0, 0], attach_b=[0, 0, _L], axis=[0, 1, 0]),
    ),
    n_bodies=2,
)


def _zero_kernel(dt: float, n_k: int = 6001) -> RetardationKernel:
    return RetardationKernel(
        K=np.zeros((12, 12, n_k)), t=dt * np.arange(n_k, dtype=np.float64), dt=dt
    )


def _gravity(_t: float) -> np.ndarray:
    f = np.zeros(12)
    f[2] = -_M * _G  # body 0 weight
    f[8] = -_M * _G  # body 1 weight
    return f


def _mode_ic(theta1: float, ratio: float) -> np.ndarray:
    """Hanging config for absolute pitch angles (theta1, ratio*theta1)."""
    theta2 = ratio * theta1
    r1 = Rotation.from_rotvec([0, theta1, 0])
    r2 = Rotation.from_rotvec([0, theta2, 0])
    x0 = r1.apply([0, 0, -_L])
    x1 = x0 + r2.apply([0, 0, -_L])
    xi = np.zeros(12)
    xi[0:3], xi[3:6] = x0, [0, theta1, 0]
    xi[6:9], xi[9:12] = x1, [0, theta2, 0]
    return xi


def _period_zero_cross(theta: np.ndarray, t: np.ndarray) -> float:
    zc = np.where((theta[:-1] < 0) & (theta[1:] >= 0))[0]
    assert zc.size >= 3, f"too few zero-crossings ({zc.size}) to measure a period"
    return float(np.mean(np.diff(t[zc])))


@pytest.mark.parametrize(
    ("ratio", "t_ref", "label"),
    [(_RATIO_PLUS, _T_PLUS, "T_plus"), (_RATIO_MINUS, _T_MINUS, "T_minus")],
)
def test_double_pendulum_mode_period(ratio: float, t_ref: float, label: str) -> None:
    """Each normal mode, excited by its analytic eigenvector, reproduces the
    closed-form period at rtol 1e-3 -- the two-body BB-OFFSET gate. The
    inter-body constraint point is offset from body 1's CoG (finding ME's
    geometric restoring across a body-body joint)."""
    r = integrate_cummins(
        lhs=_LHS,
        kernel=_zero_kernel(_DT),
        xi0=_mode_ic(_THETA0, ratio),
        xi_dot0=np.zeros(12),
        duration=40.0 * t_ref,
        dt=_DT,
        external_force=_gravity,
        rho_inf=1.0,
        constraints=_HINGES,
        projection_interval=1,
    )
    drift = max(float(np.max(np.abs(_HINGES.phi(x)))) for x in r.xi[::200])
    assert drift < 1e-10, (label, drift)
    t_meas = _period_zero_cross(r.xi[:, 4], r.t)
    assert t_meas == pytest.approx(t_ref, rel=1e-3), (label, t_meas, t_ref)


def test_bb_offset_penalty_raises_but_joint_path_holds() -> None:
    """Q4 closure, tested not asserted: the body-body offset topology the
    penalty ``LinearConnector`` rejects, the multiplier joint path holds.

    Half 1 -- penalty path: a body-body ``LinearSpring`` with a non-zero
    attachment offset raises ``NotImplementedError`` (the framework's
    Newton-III-at-reference-points limit).
    Half 2 -- joint path: the same offset body-body constraint, expressed
    as a hinge, holds ``phi`` at machine precision and produces a
    multiplier (the force is solved, not prescribed)."""
    from floatsim.driver import _materialise_linear_spring
    from floatsim.io.deck import LinearSpring

    spring = LinearSpring(
        type="linear_spring",
        body_a="b0",
        body_b="b1",
        anchor_a_body=[0.5, 0.0, 0.0],  # non-zero offset -> the framework limit
        anchor_b_body=[-0.5, 0.0, 0.0],
        stiffness=1.0e5,
    )
    with pytest.raises(NotImplementedError, match="BB-OFFSET-CONNECTOR"):
        _materialise_linear_spring(spring, {"b0": 0, "b1": 1})

    # Joint path: a single body-body hinge whose attach point is offset from
    # body 1's CoG (attach_b = [0,0,L]); integrate briefly, constraint holds.
    js = JointSet(
        joints=(hinge_joint(0, 1, attach_a=[0, 0, 0], attach_b=[0, 0, _L], axis=[0, 1, 0]),),
        n_bodies=2,
    )
    xi0 = np.zeros(12)
    xi0[6:9] = [0.0, 0.0, -_L]  # body 1 CoM hangs one rod below body 0
    r = integrate_cummins(
        lhs=_LHS,
        kernel=_zero_kernel(_DT),
        xi0=xi0,
        xi_dot0=np.zeros(12),
        duration=5.0,
        dt=_DT,
        external_force=_gravity,
        rho_inf=1.0,
        constraints=js,
        projection_interval=1,
    )
    drift = max(float(np.max(np.abs(js.phi(x)))) for x in r.xi[::50])
    assert drift < 1e-10, drift
    assert r.lam is not None and r.lam.shape[1] == 5  # a hinge locks 5 DOFs
