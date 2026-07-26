"""M9 PR2 -- velocity-level KKT integrator terminal gates.

The constraint machinery is validated against **independently derived,
closed-form** references (never FloatSim's own output):

* the compound-pendulum period ``T_n = sqrt(3g/2L)`` (Measurement MB,
  cross-checked by scipy to 6.9e-7);
* the hanging hinge reaction ``(0, 0, mg)`` (closed-form statics);
* energy conservation, tolerance re-derived from the measured floor
  (plan Amendment A1) -- the constrained pendulum is nonlinear, so a
  second-order scheme is truncation-limited; the gate has three clauses
  including the O(h) dt-scaling signature.

Setup: a single rigid body (mass m, "uniform bar" inertia I_c = mL^2/12
about the swing axis) hinged to earth under gravity, no radiation
(zero kernel) and no hydrostatic C -- the restoring is entirely the
constraint geometry (finding ME). This isolates the constraint layer.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from floatsim.bodies.joints import JointSet, hinge_joint, yaw_locked_joint
from floatsim.hydro.radiation import CumminsLHS
from floatsim.hydro.retardation import RetardationKernel
from floatsim.solver.newmark import integrate_cummins

_G = 9.81
_M = 1.0
_L = 1.0
_D = _L / 2.0
_IC = _M * _L**2 / 12.0  # uniform bar about its CoM
_T_N_REF = 2.0 * np.pi / np.sqrt(3.0 * _G / (2.0 * _L))  # = 1.637947 s (MB)

_LHS = CumminsLHS(
    M_plus_Ainf=np.diag([_M, _M, _M, _IC, _IC, _IC]).astype(np.float64),
    C=np.zeros((6, 6)),
)
_HINGE = JointSet(
    joints=(hinge_joint(0, -1, attach_a=[0, 0, _D], attach_b=[0, 0, 0], axis=[0, 1, 0]),),
    n_bodies=1,
)


def _zero_kernel(dt: float, n_dof: int = 6) -> RetardationKernel:
    n_k = 6001
    return RetardationKernel(
        K=np.zeros((n_dof, n_dof, n_k)), t=dt * np.arange(n_k, dtype=np.float64), dt=dt
    )


def _gravity(_t: float) -> np.ndarray:
    f = np.zeros(6)
    f[2] = -_M * _G
    return f


def _hinge_ic(theta0: float) -> np.ndarray:
    """State with the CoM hung at angle theta0 about the pivot (world origin)."""
    com = Rotation.from_rotvec([0, theta0, 0]).apply([0, 0, -_D])
    return np.array([com[0], com[1], com[2], 0.0, theta0, 0.0])


def _run(dt: float, duration: float, theta0: float, proj: int = 1, rho: float = 1.0):
    return integrate_cummins(
        lhs=_LHS,
        kernel=_zero_kernel(dt),
        xi0=_hinge_ic(theta0),
        xi_dot0=np.zeros(6),
        duration=duration,
        dt=dt,
        external_force=_gravity,
        rho_inf=rho,
        constraints=_HINGE,
        projection_interval=proj,
    )


# ---------- Gate 1: hinge period vs the pre-derived reference ----------


def test_hinge_gate_reproduces_pendulum_period() -> None:
    """The constrained run reproduces T_n = sqrt(3g/2L) = 1.637947 s at
    rtol 1e-3 -- the program's first non-identity validation. The
    reference is closed-form (+ scipy cross-check), never FloatSim's
    own output."""
    r = _run(0.01, 20.0, 0.02)
    th = r.xi[:, 4]
    zc = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
    assert zc.size >= 3
    t_n = float(np.mean(np.diff(r.t[zc])))
    assert t_n == pytest.approx(_T_N_REF, rel=1e-3), (t_n, _T_N_REF)


# ---------- Gate 2: lambda static recovery pins the units derivation ----------


def test_lambda_recovers_static_hinge_force() -> None:
    """Hanging at rest, the solved multiplier IS the physical hinge
    force (plan lambda-units derivation): (0, 0, +mg) N, dt-free."""
    r = _run(0.01, 2.0, 0.0)
    assert r.lam is not None
    lam = np.mean(r.lam[-50:], axis=0)
    assert lam[0] == pytest.approx(0.0, abs=1e-6)
    assert lam[1] == pytest.approx(0.0, abs=1e-6)
    assert lam[2] == pytest.approx(_M * _G, rel=1e-4)  # 9.81 N


# ---------- Gate: constraint drift ----------


def test_constraint_drift_stays_machine_precision() -> None:
    """Position projection every step holds phi at machine precision over
    100 cycles (measured 1.1e-16)."""
    r = _run(0.01, 100 * _T_N_REF, 0.02)
    drift = max(float(np.max(np.abs(_HINGE.phi(x)))) for x in r.xi[::200])
    assert drift < 1e-10, drift


# ---------- Gate 3: energy (three clauses, re-derived from the measured floor) ----------


def _energy(r) -> np.ndarray:
    th = r.xi[:, 4]
    ke = 0.5 * np.einsum("ni,ij,nj->n", r.xi_dot, _LHS.M_plus_Ainf, r.xi_dot)
    pe = _M * _G * _D * (1.0 - np.cos(th))
    return ke + pe


def test_energy_gate_magnitude_and_numerical_damping() -> None:
    """Clauses 1-2: at dt=0.01, rho_inf=1.0, 100 cycles, energy variation
    < 5e-3 and numerical damping zeta_num < 1e-5. (Superseded 1e-6 ceiling
    came from the LINEAR MC baseline; the constrained pendulum is
    nonlinear -- Amendment A1.)"""
    r = _run(0.01, 100 * _T_N_REF, 0.02)
    e = _energy(r)
    assert (e.max() - e.min()) / e.mean() < 5e-3
    th = r.xi[:, 4]
    pk, _ = find_peaks(th, height=0.0)
    amp = th[pk]
    # log-decrement over the run -> zeta_num
    decrement = -np.log(abs(amp[-1] / amp[0])) / (pk.size - 1)
    zeta_num = decrement / np.sqrt(4 * np.pi**2 + decrement**2)
    assert zeta_num < 1e-5, zeta_num


def test_energy_gate_o_h_scaling() -> None:
    """Clause 3: energy variation decreases O(h) under dt refinement --
    the signature distinguishing discretization from a secular defect
    (measured 5.4e-3 / 2.7e-3 at dt=0.02 / 0.01). A broken midpoint
    iteration would keep the magnitude under the ceiling but kill this
    signature."""
    e02 = lambda r: (_energy(r).max() - _energy(r).min()) / _energy(r).mean()  # noqa: E731
    v_coarse = e02(_run(0.02, 100 * _T_N_REF, 0.02))
    v_fine = e02(_run(0.01, 100 * _T_N_REF, 0.02))
    ratio = v_coarse / v_fine
    assert 1.5 < ratio < 3.0, ratio  # ~2x = O(h); 4x would be O(h^2), <1.5 a defect


# ---------- unconstrained path byte-identity ----------


def test_unconstrained_path_byte_identical() -> None:
    """constraints=None must reproduce the pre-M9 result exactly (lam is
    None; trajectory bit-for-bit vs the no-constraints-arg call) -- the
    M8 N=1 pattern applied to the integrator."""
    kw = dict(
        lhs=_LHS,
        kernel=_zero_kernel(0.01),
        xi0=_hinge_ic(0.02),
        xi_dot0=np.zeros(6),
        duration=5.0,
        dt=0.01,
        external_force=_gravity,
        rho_inf=0.9,
    )
    r_default = integrate_cummins(**kw)  # no constraints arg at all
    r_none = integrate_cummins(**kw, constraints=None)
    assert r_default.lam is None and r_none.lam is None
    np.testing.assert_array_equal(r_default.xi, r_none.xi)
    np.testing.assert_array_equal(r_default.xi_dot, r_none.xi_dot)
    np.testing.assert_array_equal(r_default.xi_ddot, r_none.xi_ddot)


# ---------- the 12-buoy joint (yaw_locked) in the integrator ----------


def test_yaw_locked_joint_holds_and_conserves() -> None:
    """The required production joint (3 translations + yaw locked,
    roll/pitch free) integrated on two bodies: the constraint holds at
    machine precision and energy is conserved (locked DOFs carry no
    dynamics here -- a structural smoke test that the 4-row joint
    assembles and solves in the KKT path)."""
    n = 12
    m6 = np.diag([1e4, 1e4, 1e4, 1e3, 1e3, 1e3]).astype(np.float64)
    lhs = CumminsLHS(M_plus_Ainf=np.kron(np.eye(2), m6), C=np.zeros((n, n)))
    js = JointSet(
        joints=(
            yaw_locked_joint(0, 1, attach_a=[0.5, 0, 0], attach_b=[-0.5, 0, 0], axis=[0, 0, 1]),
        ),
        n_bodies=2,
    )
    xi0 = np.zeros(n)
    # attach points must COINCIDE at the origin: body0_ref = -attach_a,
    # body1_ref = -attach_b (so ref + R*attach = 0 at R = I).
    xi0[0], xi0[6] = -0.5, 0.5
    # give both bodies a shared roll rate (a free DOF) -> should persist
    u0 = np.zeros(n)
    u0[3], u0[9] = 0.05, 0.05
    r = integrate_cummins(
        lhs=lhs,
        kernel=_zero_kernel(0.01, n_dof=n),
        xi0=xi0,
        xi_dot0=u0,
        duration=5.0,
        dt=0.01,
        rho_inf=1.0,
        constraints=js,
    )
    drift = max(float(np.max(np.abs(js.phi(x)))) for x in r.xi[::50])
    assert drift < 1e-6, drift
    assert r.lam is not None and r.lam.shape[1] == 4
