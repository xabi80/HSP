"""PR1 -- rigid (weld) joint: the ARCHITECTURE.md §7 reference case
"two bodies, rigid connector -> combined-mass equivalent body", realised in
the velocity-level KKT integrator.

Two equal bodies (mass ``m``, own pitch inertia ``I_c``) are welded rigidly at
the midpoint between them, offset ``+-d/2`` along x, each carrying a heave
spring ``k``. The pair then behaves as ONE rigid body. The references are
closed-form (independent of FloatSim's output):

* **Pitch (the rigid discriminator).** A rigid tilt lifts one body and drops
  the other on their heave springs -> a single pitch mode with

      I_eff = 2 I_c + m d^2 / 2   (own inertia + parallel axis),
      K_eff = k d^2 / 2           (two springs at +-d/2),
      T_pitch = 2 pi sqrt(I_eff / K_eff).

  This period appears ONLY if the weld locks the relative pitch -- a
  ``yaw_locked`` or ``hinge`` joint (roll/pitch free) gives an entirely
  different mode structure. It is the combined-*inertia* equivalence.

* **Heave (translational).** Welded, the pair heaves as mass ``2m`` on
  stiffness ``2k``: ``T_heave = 2 pi sqrt(m / k)`` -- the combined-*mass*
  equivalence.

Plus the KKT invariants every joint must pass: constraint drift at machine
precision, energy conservation in undamped free response, and a 6-row
multiplier (3 force + 3 moment) reaction vector.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from floatsim.bodies.joints import JointSet, rigid_joint
from floatsim.hydro.radiation import CumminsLHS
from floatsim.hydro.retardation import RetardationKernel
from floatsim.solver.newmark import integrate_cummins

_M = 1.0
_D = 1.0
_IC = 0.5  # own pitch inertia per body (well-separated from the heave mode)
_K = 200.0  # heave stiffness per body (N/m)

_I_EFF = 2.0 * _IC + _M * _D**2 / 2.0  # = 1.5
_K_EFF = _K * _D**2 / 2.0  # = 100
_T_PITCH_REF = 2.0 * np.pi * np.sqrt(_I_EFF / _K_EFF)  # 0.76953 s
_T_HEAVE_REF = 2.0 * np.pi * np.sqrt(_M / _K)  # 0.44429 s

_LHS = CumminsLHS(
    M_plus_Ainf=np.kron(np.eye(2), np.diag([_M, _M, _M, _IC, _IC, _IC])).astype(np.float64),
    C=np.kron(np.eye(2), np.diag([0.0, 0.0, _K, 0.0, 0.0, 0.0])).astype(np.float64),
)
# body0 at (-d/2, 0, 0), body1 at (+d/2, 0, 0); weld at the origin (midpoint).
_WELD = JointSet(
    joints=(rigid_joint(0, 1, attach_a=[_D / 2, 0, 0], attach_b=[-_D / 2, 0, 0]),),
    n_bodies=2,
)


def _zero_kernel(dt: float, n_dof: int = 12) -> RetardationKernel:
    n_k = 6001
    return RetardationKernel(
        K=np.zeros((n_dof, n_dof, n_k)), t=dt * np.arange(n_k, dtype=np.float64), dt=dt
    )


def _pitch_ic(theta0: float) -> np.ndarray:
    """Rigid tilt of the whole pair by theta0 about the origin (satisfies phi=0)."""
    xi = np.zeros(12)
    for k, x0 in ((0, -_D / 2), (1, _D / 2)):
        pos = Rotation.from_rotvec([0, theta0, 0]).apply([x0, 0.0, 0.0])
        xi[6 * k : 6 * k + 3] = pos
        xi[6 * k + 3 : 6 * k + 6] = [0.0, theta0, 0.0]
    return xi


def _heave_ic(a0: float) -> np.ndarray:
    """Both bodies displaced +a0 in heave (a rigid translation; phi=0)."""
    xi = np.zeros(12)
    xi[0], xi[6] = -_D / 2, _D / 2
    xi[2], xi[8] = a0, a0
    return xi


def _run(xi0: np.ndarray, dt: float, duration: float, rho: float = 1.0):
    return integrate_cummins(
        lhs=_LHS,
        kernel=_zero_kernel(dt),
        xi0=xi0,
        xi_dot0=np.zeros(12),
        duration=duration,
        dt=dt,
        rho_inf=rho,
        constraints=_WELD,
    )


def _period(signal: np.ndarray, t: np.ndarray) -> float:
    zc = np.where((signal[:-1] < 0) & (signal[1:] >= 0))[0]
    assert zc.size >= 4, zc.size
    return float(np.mean(np.diff(t[zc])))


# ---------- combined inertia: the rigid-weld discriminator ----------


def test_rigid_seesaw_reproduces_combined_inertia_pitch_period() -> None:
    """The welded pair pitches at T = 2 pi sqrt(I_eff / K_eff) = 0.7695 s --
    the combined rigid-body inertia. Only a joint that locks the relative
    pitch (rigid) can produce this mode."""
    r = _run(_pitch_ic(0.02), dt=0.005, duration=8.0)
    t_pitch = _period(r.xi[:, 4], r.t)
    assert t_pitch == pytest.approx(_T_PITCH_REF, rel=1e-3), (t_pitch, _T_PITCH_REF)


# ---------- combined mass: translational lock ----------


def test_rigid_reproduces_combined_mass_heave_period() -> None:
    """Welded, the pair heaves as 2m on 2k: T = 2 pi sqrt(m / k) = 0.4443 s."""
    r = _run(_heave_ic(0.05), dt=0.005, duration=6.0)
    # heave about the equilibrium z = 0 (body 0 heave channel)
    t_heave = _period(r.xi[:, 2], r.t)
    assert t_heave == pytest.approx(_T_HEAVE_REF, rel=1e-3), (t_heave, _T_HEAVE_REF)


# ---------- KKT invariants ----------


def test_rigid_constraint_drift_stays_machine_precision() -> None:
    """Position projection every step holds all 6 weld constraints at machine
    precision over the run."""
    r = _run(_pitch_ic(0.02), dt=0.005, duration=8.0)
    drift = max(float(np.max(np.abs(_WELD.phi(x)))) for x in r.xi[::50])
    assert drift < 1e-10, drift


def test_rigid_energy_conserved_in_free_response() -> None:
    """Undamped (rho_inf=1) free response conserves energy -- the weld does no
    work. The mode is linear, so a 2nd-order scheme holds it tightly."""
    r = _run(_pitch_ic(0.02), dt=0.005, duration=8.0, rho=1.0)
    ke = 0.5 * np.einsum("ni,ij,nj->n", r.xi_dot, _LHS.M_plus_Ainf, r.xi_dot)
    pe = 0.5 * np.einsum("ni,ij,nj->n", r.xi, _LHS.C, r.xi)
    e = ke + pe
    assert (e.max() - e.min()) / e.mean() < 2e-3, (e.max() - e.min()) / e.mean()


def test_rigid_multiplier_is_six_rows() -> None:
    """The weld returns a 6-component reaction (3 force + 3 moment)."""
    r = _run(_pitch_ic(0.01), dt=0.01, duration=1.0)
    assert r.lam is not None and r.lam.shape[1] == 6
