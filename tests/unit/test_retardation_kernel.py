"""Milestone 2 — retardation kernel unit tests.

Covers the discrete cosine transform ``K(t) = (2/pi) * int_0^inf B(omega)
cos(omega t) domega`` (ARCHITECTURE.md §2.3), computed on a finite BEM
grid via trapezoidal quadrature with a B(omega=0)=0 prepend when the
grid does not already start at zero.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.retardation import (
    RetardationKernel,
    compute_retardation_kernel,
)
from tests.support.synthetic_bem import make_diagonal_hdb


def _hdb_with_diagonal_damping(
    *,
    omega: np.ndarray,
    B_diag_per_omega: np.ndarray,
) -> HydroDatabase:
    """Diagonal HDB with prescribed B(omega) per DOF, A_inf=0, C=0."""
    n_w = omega.size
    return make_diagonal_hdb(
        A_inf_diag=[0.0] * 6,
        C_diag=[0.0] * 6,
        A_diag_per_omega=[[0.0] * 6] * n_w,
        B_diag_per_omega=[list(row) for row in B_diag_per_omega],
        omega=list(omega),
    )


# ---------- basic contract ----------


def test_compute_retardation_kernel_returns_frozen_dataclass() -> None:
    omega = np.linspace(0.1, 3.0, 30)
    hdb = _hdb_with_diagonal_damping(
        omega=omega,
        B_diag_per_omega=np.zeros((omega.size, 6)),
    )
    k = compute_retardation_kernel(hdb, t_max=5.0, dt=0.1)
    assert isinstance(k, RetardationKernel)


def test_kernel_shape_matches_time_grid() -> None:
    omega = np.linspace(0.1, 3.0, 30)
    hdb = _hdb_with_diagonal_damping(
        omega=omega,
        B_diag_per_omega=np.zeros((omega.size, 6)),
    )
    k = compute_retardation_kernel(hdb, t_max=10.0, dt=0.1)
    # t grid spans 0..t_max inclusive in steps of dt -> 101 samples
    assert k.t.shape == (101,)
    assert k.K.shape == (6, 6, 101)
    assert k.dt == pytest.approx(0.1)


def test_kernel_time_grid_starts_at_zero_and_is_uniform() -> None:
    omega = np.linspace(0.1, 3.0, 30)
    hdb = _hdb_with_diagonal_damping(
        omega=omega,
        B_diag_per_omega=np.zeros((omega.size, 6)),
    )
    k = compute_retardation_kernel(hdb, t_max=2.0, dt=0.05)
    assert k.t[0] == 0.0
    np.testing.assert_allclose(np.diff(k.t), 0.05, rtol=1e-12)
    assert k.t[-1] == pytest.approx(2.0, rel=1e-12)


def test_zero_damping_gives_zero_kernel() -> None:
    omega = np.linspace(0.1, 3.0, 30)
    hdb = _hdb_with_diagonal_damping(
        omega=omega,
        B_diag_per_omega=np.zeros((omega.size, 6)),
    )
    k = compute_retardation_kernel(hdb, t_max=5.0, dt=0.1)
    np.testing.assert_allclose(k.K, 0.0, atol=1e-15)


def test_kernel_is_symmetric_at_each_lag() -> None:
    # Build a non-diagonal B(omega) by stacking full symmetric matrices
    omega = np.linspace(0.05, 2.5, 40)
    n_w = omega.size
    rng = np.random.default_rng(seed=42)
    B_stack = np.empty((6, 6, n_w), dtype=np.float64)
    for j in range(n_w):
        m = rng.standard_normal((6, 6))
        B_stack[:, :, j] = 0.5 * (m + m.T)
    hdb = make_diagonal_hdb(
        A_inf_diag=[0.0] * 6,
        C_diag=[0.0] * 6,
        omega=list(omega),
    )
    # Replace the diagonal-only B stack in the built hdb with a full symmetric one
    # by constructing a fresh HydroDatabase.
    hdb_full = HydroDatabase(
        omega=hdb.omega,
        heading_deg=hdb.heading_deg,
        A=hdb.A,
        B=B_stack,
        A_inf=hdb.A_inf,
        C=hdb.C,
        RAO=hdb.RAO,
        reference_point=hdb.reference_point,
        metadata=dict(hdb.metadata),
    )
    k = compute_retardation_kernel(hdb_full, t_max=3.0, dt=0.1)
    for i in range(k.K.shape[2]):
        np.testing.assert_allclose(k.K[:, :, i], k.K[:, :, i].T, atol=1e-10)


# ---------- analytical sanity: box-damping DCT ----------


def test_kernel_matches_analytical_box_damping_on_fine_grid() -> None:
    """For B_33(omega) = B0 on [0, omega_max], else 0:

        K_33(t) = (2 B0/pi) * sin(omega_max t) / t    (t > 0)
        K_33(0) = 2 B0 omega_max / pi

    The reader+transform combo is trapezoidal on a dense uniform grid
    starting at 0 — we prepend B(0)=0 when needed — so the match is
    trapezoidal-error-limited: rtol ~ 1e-2 on a 400-point grid is safe.
    """
    B0 = 1000.0
    omega_max = 3.0
    omega = np.linspace(0.0, omega_max, 401)
    B_diag = np.zeros((omega.size, 6))
    B_diag[:, 2] = B0  # heave
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=B_diag)

    k = compute_retardation_kernel(hdb, t_max=8.0, dt=0.05)
    t = k.t
    analytical = np.empty_like(t)
    analytical[0] = 2.0 * B0 * omega_max / np.pi
    analytical[1:] = (2.0 * B0 / np.pi) * np.sin(omega_max * t[1:]) / t[1:]

    np.testing.assert_allclose(k.K[2, 2, :], analytical, rtol=2e-2, atol=1e-2)


def test_kernel_off_diagonal_dofs_stay_zero_for_diagonal_damping() -> None:
    """Pure-diagonal B(omega) produces a pure-diagonal K(t)."""
    omega = np.linspace(0.0, 3.0, 101)
    B_diag = np.zeros((omega.size, 6))
    B_diag[:, 2] = 500.0  # heave only
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=B_diag)
    k = compute_retardation_kernel(hdb, t_max=5.0, dt=0.1)
    # All non-(2,2) entries must be exactly zero.
    mask = np.ones((6, 6), dtype=bool)
    mask[2, 2] = False
    assert np.max(np.abs(k.K[mask, :])) == 0.0


# ---------- diagnostic: slow-decay warning per §9.1 ----------


def test_kernel_warns_when_decay_is_too_slow() -> None:
    """Narrow-band B(omega) yields a slowly-decaying K(t). The diagnostic
    must fire when |K(t_max)| > 0.01 * max|K(t)| for any diagonal DOF."""
    omega = np.linspace(0.0, 3.0, 301)
    # Narrow Gaussian B(omega) centered at 1 rad/s -> K(t) decays slowly.
    B0 = 1.0e4
    B_diag = np.zeros((omega.size, 6))
    B_diag[:, 2] = B0 * np.exp(-((omega - 1.0) ** 2) / (2.0 * 0.05**2))
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=B_diag)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_retardation_kernel(hdb, t_max=5.0, dt=0.1)
    msgs = [str(w.message).lower() for w in caught]
    assert any("retardation" in m and "decay" in m for m in msgs), msgs


def test_kernel_does_not_warn_on_fast_decay() -> None:
    """Broad B(omega) gives a tight K(t) that decays well before t_max."""
    omega = np.linspace(0.0, 5.0, 501)
    B_diag = np.zeros((omega.size, 6))
    B_diag[:, 2] = 1.0  # flat -> sinc-shaped K, decays as 1/t
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=B_diag)
    # Use a large t_max so K has decayed below 1% of its peak.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_retardation_kernel(hdb, t_max=60.0, dt=0.1)
    decay_msgs = [
        w
        for w in caught
        if "retardation" in str(w.message).lower() and "decay" in str(w.message).lower()
    ]
    assert decay_msgs == []


# ---------- input validation ----------


def test_rejects_non_positive_t_max() -> None:
    omega = np.linspace(0.1, 3.0, 30)
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=np.zeros((omega.size, 6)))
    with pytest.raises(ValueError, match="t_max"):
        compute_retardation_kernel(hdb, t_max=0.0, dt=0.1)
    with pytest.raises(ValueError, match="t_max"):
        compute_retardation_kernel(hdb, t_max=-1.0, dt=0.1)


def test_rejects_non_positive_dt() -> None:
    omega = np.linspace(0.1, 3.0, 30)
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=np.zeros((omega.size, 6)))
    with pytest.raises(ValueError, match="dt"):
        compute_retardation_kernel(hdb, t_max=5.0, dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        compute_retardation_kernel(hdb, t_max=5.0, dt=-0.01)


def test_rejects_dt_larger_than_t_max() -> None:
    omega = np.linspace(0.1, 3.0, 30)
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=np.zeros((omega.size, 6)))
    with pytest.raises(ValueError, match="dt"):
        compute_retardation_kernel(hdb, t_max=0.1, dt=0.5)


def test_retardation_kernel_dataclass_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="K must have shape"):
        RetardationKernel(
            K=np.zeros((5, 6, 4), dtype=np.float64),
            t=np.array([0.0, 0.1, 0.2, 0.3]),
            dt=0.1,
        )


def test_retardation_kernel_dataclass_rejects_t_length_mismatch() -> None:
    with pytest.raises(ValueError, match="t"):
        RetardationKernel(
            K=np.zeros((6, 6, 4), dtype=np.float64),
            t=np.array([0.0, 0.1, 0.2]),  # length 3, K has 4 lags
            dt=0.1,
        )
