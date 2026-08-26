"""Milestone 2 — radiation convolution buffer unit tests.

Covers the circular-buffer discrete convolution, discretised with the
**trapezoidal** rule (endpoints half-weighted; ARCHITECTURE.md §2.4):

    mu_n = dt * ( sum_{k=0}^{N_K-1} K_k @ xi_dot_{n-k}
                  - 1/2 K_0 @ xi_dot_n - 1/2 K_{N_K-1} @ xi_dot_{n-(N_K-1)} )

where the newest pushed velocity carries lag 0. The lag-0 half-weight is the
fix for the FloatFEA-reported defect: a full-dt lag-0 weight (rectangle) over-
applies radiation damping by dt*K_0/2 (see :meth:`RadiationConvolution.evaluate`).
Pure algebraic tests; no ODE integration here.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.retardation import (
    RadiationConvolution,
    RetardationKernel,
)


def _make_kernel(K: np.ndarray, dt: float) -> RetardationKernel:
    assert K.ndim == 3 and K.shape[:2] == (6, 6)
    n_t = K.shape[2]
    t = dt * np.arange(n_t, dtype=np.float64)
    return RetardationKernel(K=K.astype(np.float64), t=t, dt=float(dt))


def _random_symmetric_kernel(n_t: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    K = np.empty((6, 6, n_t), dtype=np.float64)
    for i in range(n_t):
        m = rng.standard_normal((6, 6))
        K[:, :, i] = 0.5 * (m + m.T)
    return K


def _trap_ref(K: np.ndarray, buf: list[np.ndarray], dt: float) -> np.ndarray:
    """Independent trapezoidal reference: ``buf[k]`` is the sample at lag ``k``
    (newest = lag 0). Endpoints lag 0 and lag N-1 carry half weight."""
    n = K.shape[2]
    mu = dt * sum(K[:, :, k] @ buf[k] for k in range(n))
    mu -= 0.5 * dt * (K[:, :, 0] @ buf[0] + K[:, :, n - 1] @ buf[n - 1])
    return mu


# ---------- empty / freshly-reset buffer ----------


def test_empty_buffer_evaluates_to_zero() -> None:
    K = _random_symmetric_kernel(n_t=10)
    conv = RadiationConvolution(_make_kernel(K, dt=0.1))
    mu = conv.evaluate()
    assert mu.shape == (6,)
    np.testing.assert_allclose(mu, 0.0, atol=0.0)


def test_reset_clears_history() -> None:
    K = _random_symmetric_kernel(n_t=10)
    conv = RadiationConvolution(_make_kernel(K, dt=0.1))
    for _ in range(5):
        conv.push(np.arange(6, dtype=np.float64) + 1.0)
    conv.reset()
    np.testing.assert_allclose(conv.evaluate(), 0.0, atol=0.0)


# ---------- lag mapping (trapezoid: lag-0 endpoint carries HALF weight) ----------


def test_single_push_applies_half_weight_lag_zero_kernel() -> None:
    """One push: buffer is [v, 0, ...]; trapezoid halves the lag-0 endpoint
    (and the zero last lag), so mu = 0.5 * dt * K_0 @ v -- NOT dt * K_0 @ v
    (that full weight was the rectangular-rule defect)."""
    K = _random_symmetric_kernel(n_t=10, seed=1)
    dt = 0.1
    conv = RadiationConvolution(_make_kernel(K, dt))
    xi_dot = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    conv.push(xi_dot)
    expected = 0.5 * dt * (K[:, :, 0] @ xi_dot)
    np.testing.assert_allclose(conv.evaluate(), expected, rtol=1e-12)


def test_two_pushes_map_most_recent_to_lag_zero() -> None:
    """Newer push -> lag 0 (half-weighted endpoint); earlier push -> lag 1 (full)."""
    K = _random_symmetric_kernel(n_t=10, seed=2)
    dt = 0.05
    conv = RadiationConvolution(_make_kernel(K, dt))
    v_old = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_new = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    conv.push(v_old)
    conv.push(v_new)
    expected = dt * (0.5 * K[:, :, 0] @ v_new + K[:, :, 1] @ v_old)
    np.testing.assert_allclose(conv.evaluate(), expected, rtol=1e-12)


def test_n_pushes_map_to_full_history() -> None:
    """Push distinct velocities filling the buffer; verify against the
    independent trapezoidal reference (both endpoints half-weighted)."""
    n_t = 5
    K = _random_symmetric_kernel(n_t=n_t, seed=3)
    dt = 0.1
    conv = RadiationConvolution(_make_kernel(K, dt))
    vels = [np.eye(6)[i % 6] for i in range(n_t)]
    for v in vels:
        conv.push(v)
    buf = [vels[n_t - 1 - k] for k in range(n_t)]  # lag k -> (n_t-1-k)-th pushed
    np.testing.assert_allclose(conv.evaluate(), _trap_ref(K, buf, dt), rtol=1e-12)


# ---------- circular wrap-around ----------


def test_buffer_drops_oldest_sample_after_wrap() -> None:
    """After N_K + 1 pushes, the very first sample must no longer contribute."""
    n_t = 4
    K = _random_symmetric_kernel(n_t=n_t, seed=4)
    dt = 0.1
    conv = RadiationConvolution(_make_kernel(K, dt))

    first = np.array([999.0, -999.0, 0.0, 0.0, 0.0, 0.0])
    conv.push(first)
    follow = [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) for _ in range(n_t)]
    for v in follow:
        conv.push(v)
    buf = [follow[n_t - 1 - k] for k in range(n_t)]  # `first` has wrapped out
    np.testing.assert_allclose(conv.evaluate(), _trap_ref(K, buf, dt), rtol=1e-12)


# ---------- steady state ----------


def test_constant_velocity_reaches_trapezoid_sum_of_kernel() -> None:
    """Push the same velocity N_K times -> mu = dt*(sum_k K_k - 1/2 K_0 - 1/2 K_{N-1}) @ v."""
    n_t = 20
    K = _random_symmetric_kernel(n_t=n_t, seed=5)
    dt = 0.05
    conv = RadiationConvolution(_make_kernel(K, dt))
    v = np.array([0.2, -0.1, 0.5, 0.0, -0.3, 0.1])
    for _ in range(n_t):
        conv.push(v)
    trap_weight = K.sum(axis=2) - 0.5 * (K[:, :, 0] + K[:, :, -1])
    expected = (trap_weight * dt) @ v
    np.testing.assert_allclose(conv.evaluate(), expected, rtol=1e-12)


# ---------- DOF independence ----------


def test_diagonal_kernel_decouples_dofs() -> None:
    """For a diagonal K_k at every lag, exciting only DOF 2 yields mu in DOF 2 only;
    a single push gives the half-weighted lag-0 value."""
    n_t = 8
    K = np.zeros((6, 6, n_t))
    for i in range(6):
        K[i, i, :] = np.linspace(1.0, 0.1, n_t) * (i + 1)
    dt = 0.1
    conv = RadiationConvolution(_make_kernel(K, dt))
    v = np.zeros(6)
    v[2] = 1.0
    conv.push(v)
    mu = conv.evaluate()
    mask = np.ones(6, dtype=bool)
    mask[2] = False
    assert np.max(np.abs(mu[mask])) == 0.0
    assert mu[2] == pytest.approx(0.5 * K[2, 2, 0] * dt, rel=1e-12)


# ---------- input validation ----------


def test_push_rejects_wrong_shape() -> None:
    K = _random_symmetric_kernel(n_t=4)
    conv = RadiationConvolution(_make_kernel(K, dt=0.1))
    with pytest.raises(ValueError, match="shape"):
        conv.push(np.zeros(5))
    with pytest.raises(ValueError, match="shape"):
        conv.push(np.zeros((6, 1)))
