"""M11b PR8 -- build-time restoring-PSD gate (PLATFORM-HYDROSTATIC-C-INDEFINITE).

The assembled restoring matrix ``C`` must be positive semi-definite on the
constraint-feasible subspace ``null(G)``: free rigid modes may sit at zero, but
a genuinely negative generalized eigenvalue ``omega^2 < 0`` is an unstable
negative-stiffness mode that diverges under any dynamics. The platform's
indefinite hydrostatic C (six negative feasible ``omega^2``, min -1.60) reached
the integrator with nothing catching it; this gate is the missing invariant
(cf. M8's PSD gate on ``B(omega)``).

These unit tests exercise the GATE MECHANISM with synthetic matrices -- not the
transient platform BEM file (which STEP 4 may regenerate). The M10-passes /
platform-fails end-to-end behaviour is covered by the M10 PR1 suite (which
still passes with the gate active) and reported in the PR8 diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.bodies.joints import JointSet, yaw_locked_joint
from floatsim.driver import _feasible_restoring_eigvals, _gate_restoring_psd
from floatsim.hydro.radiation import CumminsLHS


def _lhs(c: np.ndarray) -> CumminsLHS:
    """CumminsLHS with an SPD mass and the given restoring ``C``."""
    n = c.shape[0]
    return CumminsLHS(M_plus_Ainf=np.eye(n) * 10.0, C=c)


def _psd_single_body_c() -> np.ndarray:
    """6x6 buoyancy-only restoring: zero surge/sway/yaw (free), positive
    heave/roll/pitch. PSD with three legitimate zero modes."""
    return np.diag([0.0, 0.0, 200.0, 150.0, 150.0, 0.0])


def _indefinite_single_body_c() -> np.ndarray:
    """The platform signature: a surge-pitch coupling that exceeds the diagonal
    (surge stiffness 0), making the [surge, pitch] 2x2 block indefinite."""
    c = np.diag([0.0, 0.0, 200.0, 150.0, 150.0, 0.0])
    c[0, 4] = c[4, 0] = 164.0  # surge-pitch coupling >> sqrt(0 * 150) -> indefinite
    return c


# --- gate pass / fail on the full space (no constraints) --------------------


def test_gate_passes_psd_restoring() -> None:
    _gate_restoring_psd(_lhs(_psd_single_body_c()), None)  # must NOT raise


def test_gate_raises_on_indefinite_restoring() -> None:
    with pytest.raises(ValueError, match=r"indefinite on the constraint-feasible"):
        _gate_restoring_psd(_lhs(_indefinite_single_body_c()), None)


def test_gate_message_reports_count_and_min() -> None:
    with pytest.raises(ValueError, match=r"negative generalized eigenvalue"):
        _gate_restoring_psd(_lhs(_indefinite_single_body_c()), None)


def test_gate_allows_legitimate_zero_free_modes() -> None:
    """A restoring with three exact zero eigenvalues (surge/sway/yaw of a free
    body) passes -- the gate permits PSD-with-zeros, not just PD."""
    w = _feasible_restoring_eigvals(_lhs(_psd_single_body_c()), None)
    assert (w >= -1e-9).all()
    assert np.count_nonzero(np.abs(w) < 1e-9) == 3  # three free modes at zero
    _gate_restoring_psd(_lhs(_psd_single_body_c()), None)


def test_gate_tolerance_admits_numerical_zero() -> None:
    """A tiny negative eigenvalue at the numerical-zero floor must NOT trip the
    gate (the M10 case sits at ~ -5e-16)."""
    c = _psd_single_body_c()
    # inject a -1e-12 perturbation on an otherwise-zero free mode
    c[0, 0] = -1.0e-12
    _gate_restoring_psd(_lhs(c), None)  # must NOT raise (below the rtol floor)


# --- feasible-subspace qualifier --------------------------------------------


def _two_body_jointset() -> JointSet:
    """Two bodies, one yaw_locked joint body0<->body1 (4 constraint rows)."""
    j = yaw_locked_joint(
        0, 1, attach_a=np.zeros(3), attach_b=np.zeros(3), axis=np.array([0.0, 0.0, 1.0])
    )
    return JointSet(joints=(j,), n_bodies=2)


def test_feasible_dim_reduced_by_constraints() -> None:
    """The feasible spectrum has ``n_dof - rank(G)`` eigenvalues -- the joint
    removes its constrained directions before the PSD test (the subspace
    qualifier)."""
    c = np.zeros((12, 12))
    c[2, 2] = c[8, 8] = 200.0  # heave of each body
    js = _two_body_jointset()
    w = _feasible_restoring_eigvals(_lhs(c), js)
    g = js.jacobian(np.zeros(12))
    assert w.size == 12 - np.linalg.matrix_rank(g)  # 12 - 4 = 8 free modes


def test_gate_uses_feasible_not_full_space() -> None:
    """A restoring that is indefinite in the FULL space but whose negative
    direction is removed by a constraint passes the gate. Body-0 carries the
    indefinite surge-pitch block; a yaw_locked joint to earth locks body-0's
    surge (one of the two coupled DOFs), lifting the 2x2 indefiniteness."""
    c = np.zeros((6, 6))
    c[2, 2] = 200.0
    c[0, 4] = c[4, 0] = 164.0
    c[4, 4] = 150.0  # [surge,pitch] block [[0,164],[164,150]] -> indefinite in full space
    full = _feasible_restoring_eigvals(_lhs(c), None)
    assert full.min() < -1.0  # indefinite unconstrained
    # Lock body-0 surge (and the other translations + yaw) to earth.
    j = yaw_locked_joint(
        0, -1, attach_a=np.zeros(3), attach_b=np.zeros(3), axis=np.array([0.0, 0.0, 1.0])
    )
    js = JointSet(joints=(j,), n_bodies=1)
    _gate_restoring_psd(_lhs(c), js)  # surge constrained out -> must NOT raise
