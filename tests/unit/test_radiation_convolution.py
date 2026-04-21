"""Milestone 2 — radiation convolution buffer unit tests.

Covers the circular-buffer discrete convolution

    mu_n = sum_{k=0}^{N_K-1} K_k @ xi_dot_{n-k} * dt   (ARCHITECTURE.md §2.4)

where the newest pushed velocity carries lag 0. Pure algebraic tests;
no ODE integration yet — that lands in the M2 integrator PR.
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


# ---------- lag mapping ----------


def test_single_push_applies_lag_zero_kernel() -> None:
    K = _random_symmetric_kernel(n_t=10, seed=1)
    dt = 0.1
    conv = RadiationConvolution(_make_kernel(K, dt))
    xi_dot = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    conv.push(xi_dot)
    expected = K[:, :, 0] @ xi_dot * dt
    np.testing.assert_allclose(conv.evaluate(), expected, rtol=1e-12)


def test_two_pushes_map_most_recent_to_lag_zero() -> None:
    """Newer push -> lag 0; earlier push -> lag 1."""
    K = _random_symmetric_kernel(n_t=10, seed=2)
    dt = 0.05
    conv = RadiationConvolution(_make_kernel(K, dt))
    v_old = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_new = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    conv.push(v_old)
    conv.push(v_new)
    expected = (K[:, :, 0] @ v_new + K[:, :, 1] @ v_old) * dt
    np.testing.assert_allclose(conv.evaluate(), expected, rtol=1e-12)


def test_n_pushes_map_to_full_history() -> None:
    """Push distinct velocities and verify each lag receives the right sample."""
    n_t = 5
    K = _random_symmetric_kernel(n_t=n_t, seed=3)
    dt = 0.1
    conv = RadiationConvolution(_make_kernel(K, dt))
    # Push e_1, e_2, ..., e_n in order. After push: e_n is lag 0, e_1 is lag n-1.
    vels = [np.eye(6)[i % 6] for i in range(n_t)]
    for v in vels:
        conv.push(v)
    expected = sum(K[:, :, k] @ vels[n_t - 1 - k] for k in range(n_t)) * dt
    np.testing.assert_allclose(conv.evaluate(), expected, rtol=1e-12)


# ---------- circular wrap-around ----------


def test_buffer_drops_oldest_sample_after_wrap() -> None:
    """After N_K + 1 pushes, the very first sample must no longer contribute."""
    n_t = 4
    K = _random_symmetric_kernel(n_t=n_t, seed=4)
    dt = 0.1
    conv = RadiationConvolution(_make_kernel(K, dt))

    first = np.array([999.0, -999.0, 0.0, 0.0, 0.0, 0.0])
    conv.push(first)
    # Push n_t more samples — the "first" one must wrap out.
    follow = [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) for _ in range(n_t)]
    for v in follow:
        conv.push(v)
    got = conv.evaluate()

    # Expected: only the most recent n_t samples contribute; those are all `follow[k]`.
    expected = sum(K[:, :, k] @ follow[n_t - 1 - k] for k in range(n_t)) * dt
    np.testing.assert_allclose(got, expected, rtol=1e-12)


# ---------- steady state ----------


def test_constant_velocity_reaches_sum_of_kernel() -> None:
    """Push the same velocity N_K times -> mu = (sum_k K_k * dt) @ v_const."""
    n_t = 20
    K = _random_symmetric_kernel(n_t=n_t, seed=5)
    dt = 0.05
    conv = RadiationConvolution(_make_kernel(K, dt))
    v = np.array([0.2, -0.1, 0.5, 0.0, -0.3, 0.1])
    for _ in range(n_t):
        conv.push(v)
    expected = (K.sum(axis=2) * dt) @ v
    np.testing.assert_allclose(conv.evaluate(), expected, rtol=1e-12)


# ---------- DOF independence ----------


def test_diagonal_kernel_decouples_dofs() -> None:
    """For a diagonal K_k at every lag, exciting only DOF 2 yields mu in DOF 2 only."""
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
    assert mu[2] == pytest.approx(K[2, 2, 0] * dt, rel=1e-12)


# ---------- input validation ----------


def test_push_rejects_wrong_shape() -> None:
    K = _random_symmetric_kernel(n_t=4)
    conv = RadiationConvolution(_make_kernel(K, dt=0.1))
    with pytest.raises(ValueError, match="shape"):
        conv.push(np.zeros(5))
    with pytest.raises(ValueError, match="shape"):
        conv.push(np.zeros((6, 1)))
