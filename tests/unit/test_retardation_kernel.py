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
    # t_max = 200 s gives the well-behaved-rolloff kernel room to drop
    # below the Check 3 0.1 % decay threshold (the symmetric synthetic
    # is otherwise as fast-decaying as a typical Lorentzian; 200 s
    # is comfortable headroom).
    k = compute_retardation_kernel(hdb_full, t_max=200.0, dt=0.1)
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

    # The Lorentzian has 1/t^2 long-tail decay; for tau=2 (a=0.5),
    # K(t)/K(0) drops to 1e-3 around t ~ 28 s. Use t_max = 60 s so
    # the post-extension Check 3 (kernel decay < 0.1 % of peak) clears
    # cleanly.
    k = compute_retardation_kernel(hdb, t_max=60.0, dt=0.05)
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


def test_kernel_raises_check_3_when_decay_is_too_slow() -> None:
    """Check 3 (post-extension kernel decay) must raise when
    ``|K_ii(t_max)| / max|K_ii(t)|`` exceeds the 0.1 % gate.

    Narrow-band B(omega) yields a slowly-decaying K(t). With t_max
    too short to capture the decay, Check 3 fires as a hard error
    (post-fix-wamit-dimensionalisation; previously this was the
    ``_emit_decay_diagnostic`` warning at 1 %). See
    ``floatsim/hydro/retardation.py``'s ``_validate_kernel_decay``
    docstring for the threshold rationale.
    """
    omega = np.linspace(0.0, 3.0, 301)
    # Narrow Gaussian B(omega) centered at 1 rad/s -> K(t) decays
    # slowly (its envelope decays as ~ 1/t since the bandwidth is
    # very narrow); at t_max=5 s, K(t_max)/peak is well above 0.1 %.
    B0 = 1.0e4
    B_diag = np.zeros((omega.size, 6))
    B_diag[:, 2] = B0 * np.exp(-((omega - 1.0) ** 2) / (2.0 * 0.05**2))
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=B_diag)

    with pytest.raises(ValueError, match=r"Check 3.*kernel decay"):
        compute_retardation_kernel(hdb, t_max=5.0, dt=0.1)


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


def test_kernel_check1_warns_on_high_b_at_omega_max_but_does_not_raise() -> None:
    """Check 1 (advisory) fires when ``|B_ii(omega_max)|/peak`` > 5 %.

    Three-check structure (locked at fix-wamit-dimensionalisation,
    Decision 3): Check 1 is a SOFT WARNING. It indicates the BEM grid
    is under-resolved relative to the typical asymptotic-regime
    cutoff, but the 1/omega^4 tail extension on
    ``[omega_max, 5*omega_max]`` typically rescues the kernel —
    Check 3 (post-extension decay) is the authoritative gate.

    Synthetic: ``well_behaved_b`` with ``cutoff_omega=2.5`` on a grid
    extending only to ``omega_max=3``. At omega_max the value is
    ``cutoff^4 / (cutoff^4 + omega_max^4) ≈ 32.5 %`` of peak, well
    above the 5 % advisory threshold but with a clean omega^-4 tail
    that satisfies Check 2 and a kernel that decays cleanly before
    t_max so Check 3 also passes.
    """
    omega = np.linspace(0.0, 3.0, 301)
    B_diag = np.zeros((omega.size, 6))
    # cutoff=2.5 at omega_max=3 puts B at ~32% of peak -- well above 5%.
    B_diag[:, 2] = well_behaved_b(omega, band_value=1.0, cutoff_omega=2.5)
    hdb = _hdb_with_diagonal_damping(omega=omega, B_diag_per_omega=B_diag)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Must NOT raise (Check 1 is advisory, Check 2 + 3 must pass).
        # t_max=60 gives 1/t^2-decaying kernel ample headroom.
        k = compute_retardation_kernel(hdb, t_max=60.0, dt=0.1)
    check1_msgs = [
        w for w in caught if "Check 1" in str(w.message) and "B(omega_max)" in str(w.message)
    ]
    assert len(check1_msgs) == 1, (
        f"expected exactly one Check 1 warning; got {len(check1_msgs)}: "
        f"{[str(w.message)[:80] for w in caught]}"
    )
    # And the kernel must actually be returned (Check 3 must pass).
    assert k.K[2, 2, -1] / np.max(np.abs(k.K[2, 2, :])) < 1.0e-3


def test_marin_semi_passes_all_three_checks_cleanly() -> None:
    """Regression gate: marin_semi.1 (the M6 OC4 BEM database) must
    pass all three checks of the kernel gate without raising and
    without emitting Check 1 advisories at the canonical t_max=200 s.

    Pins the Pre-3 finding (post-WAMIT-dimensionalisation +
    three-check refactor): per
    ``docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png``, every
    diagonal DOF of marin_semi has B(omega_max)/peak below 5 %
    (heave/roll/pitch under 1 %; surge/sway/yaw at ~1.7 %), a clean
    omega^-4 asymptote, and a post-extension kernel that decays to
    < 6e-5 of peak by t = 200 s. Any future regression that changes
    the WAMIT reader, BEM normalisation, or the gate thresholds
    must keep this test green.
    """
    from pathlib import Path

    from floatsim.hydro.database import HydroDatabase
    from floatsim.hydro.readers.wamit import read_added_mass_and_damping

    repo_root = Path(__file__).resolve().parents[2]
    marin_path = (
        repo_root
        / "tests"
        / "fixtures"
        / "openfast"
        / "oc4_deepcwind"
        / "baseline"
        / "5MW_Baseline"
        / "HydroData"
        / "marin_semi.1"
    )
    omega, A, B, A_inf = read_added_mass_and_damping(marin_path)

    # Build a minimal HDB shell. Hydrostatics are not gate-relevant
    # (the gates only consume B, omega, A_inf); zero them here.
    hdb = HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0]),
        A=A,
        B=B,
        A_inf=A_inf,
        C=np.zeros((6, 6), dtype=np.float64),
        RAO=np.zeros((6, omega.size, 1), dtype=np.complex128),
        reference_point=np.array([0.0, 0.0, 0.0]),
        C_source="full",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Should NOT raise (all three checks pass) AND NOT warn
        # (Check 1 advisory should not fire on marin_semi).
        k = compute_retardation_kernel(hdb, t_max=200.0, dt=0.05)
    check_msgs = [w for w in caught if "Check 1" in str(w.message) or "Check 3" in str(w.message)]
    assert check_msgs == [], (
        f"marin_semi must pass all three checks cleanly; got "
        f"{[str(w.message)[:120] for w in check_msgs]}"
    )
    # Cross-check: every diagonal must decay below the Check 3 ratio.
    for i in range(6):
        peak = float(np.max(np.abs(k.K[i, i, :])))
        if peak < 1.0e-12:
            continue
        end = float(np.abs(k.K[i, i, -1]))
        assert end / peak < 1.0e-3, (
            f"DOF {i}: |K(t_max)|/peak = {end / peak:.2e} >= 1e-3 "
            "(Check 3 should have raised; missed regression)"
        )


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
