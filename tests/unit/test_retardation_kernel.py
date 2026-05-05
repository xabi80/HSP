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
from tests.support.synthetic_bem import make_diagonal_hdb, well_behaved_b


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
    # Build a non-diagonal B(omega) by stacking the SAME symmetric matrix
    # at every frequency, scaled by a well-behaved ω⁻⁴ roll-off. This is
    # symmetric at every ω and has a clean ω⁻⁴ asymptote (so the M6 PR3
    # Refinement-2 input gates pass).
    omega = np.linspace(0.05, 20.0, 200)
    rng = np.random.default_rng(seed=42)
    m = rng.standard_normal((6, 6))
    sym_matrix = 0.5 * (m + m.T)
    # Ensure diagonals are positive (radiation damping is passive).
    np.fill_diagonal(sym_matrix, np.abs(np.diag(sym_matrix)))
    rolloff = well_behaved_b(omega, band_value=1.0, cutoff_omega=3.0)
    B_stack = sym_matrix[:, :, None] * rolloff[None, None, :]
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
        C_source=hdb.C_source,
        metadata=dict(hdb.metadata),
    )
    k = compute_retardation_kernel(hdb_full, t_max=3.0, dt=0.1)
    for i in range(k.K.shape[2]):
        np.testing.assert_allclose(k.K[:, :, i], k.K[:, :, i].T, atol=1e-10)


# ---------- analytical sanity: box-damping DCT ----------


def test_kernel_matches_analytical_lorentzian_damping_on_fine_grid() -> None:
    """For B_33(ω) = B0 · exp(-ω/τ):

        K_33(t) = (2 B0 / π) · a / (a² + t²)    with a = 1/τ

    Filon-trapezoidal computes the integral of (piecewise-linear B)·cos(ωt)
    exactly per segment; the only discretisation error is the linear
    interpolation of the smooth exponential. On a dense grid (200 pts on
    [0, 20]) the residual is below 1e-2 of the peak.

    (Replaces the pre-M6-PR3 sharp-box test, whose B(ω_max)/peak = 100% is
    exactly what Refinement-2 Check 1 is designed to prevent. The
    smooth-box analogue with a Hann taper is covered separately in
    test_retardation_kernel_extension.test_synthetic_smooth_box_kernel_matches_analytical.)
    """
    B0 = 1000.0
    tau = 2.0
    omega = np.linspace(0.0, 20.0, 401)
    B_diag = np.zeros((omega.size, 6))
    B_diag[:, 2] = B0 * np.exp(-omega / tau)  # heave
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=B_diag)

    k = compute_retardation_kernel(hdb, t_max=8.0, dt=0.05)
    t = k.t
    a = 1.0 / tau
    analytical = (2.0 * B0 / np.pi) * a / (a * a + t * t)

    np.testing.assert_allclose(k.K[2, 2, :], analytical, rtol=2e-2, atol=1e-2)


def test_kernel_off_diagonal_dofs_stay_zero_for_diagonal_damping() -> None:
    """Pure-diagonal B(omega) produces a pure-diagonal K(t)."""
    omega = np.linspace(0.0, 20.0, 401)
    B_diag = np.zeros((omega.size, 6))
    # Heave-only damping with ω⁻⁴ roll-off so the gates pass.
    B_diag[:, 2] = well_behaved_b(omega, band_value=500.0, cutoff_omega=5.0)
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
    omega = np.linspace(0.0, 20.0, 501)
    B_diag = np.zeros((omega.size, 6))
    # Well-behaved B with ω⁻⁴ tail -- gates pass; K decays as 1/t².
    B_diag[:, 2] = well_behaved_b(omega, band_value=1.0, cutoff_omega=5.0)
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
