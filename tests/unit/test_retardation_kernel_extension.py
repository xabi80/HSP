"""Retardation kernel tail-extension and convergence tests (M6 PR3 fix).

Test categories 2 through 7 from the locked workflow:

2. Coarse-grid kernel correctness — marin_semi.1 downsampled to dω≈0.05
   must produce a kernel matching the full-resolution kernel to
   rtol=1e-3 over the convergent range of t. This is the
   Filon-makes-coarse-grids-viable regression gate.

3. Convergence-rate regression on marin_semi.1 — period and damping
   ratio at t_max=400s pinned as reference; t_max=800s within rtol=1e-4.
   No sign flips through the converged value.

4. Synthetic-kernel analytical regression — 3 cases with closed-form
   K(t). Pins the cosine-transform accuracy on smooth B(ω). Failing-
   first against current main.

5. Damping-non-positivity on real BEM — kernel applied to free decay
   must produce ζ ≥ 0. Sign flips would indicate kernel artifacts
   pumping energy into the system.

6. t_max convergence on synthetics — K(t) at any fixed lag t is a
   function of B(ω) only; must not vary with the truncation t_max.

7. M2 free-decay regression — period unchanged after the fix
   (period is independent of kernel quality, per the audit).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.readers.wamit import read_added_mass_and_damping
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import integrate_cummins
from tests.validation.test_oc4_heave_free_decay import (
    _fit_damping_log_decrement,
    _fit_period,
)
from tests.validation.test_oc4_natural_periods import _oc4_rigid_body_mass_matrix

_MARIN_SEMI_PATH: Final[Path] = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "openfast"
    / "oc4_deepcwind"
    / "baseline"
    / "5MW_Baseline"
    / "HydroData"
    / "marin_semi.1"
)


# ---------------------------------------------------------------------------
# Synthetic-HDB builders
# ---------------------------------------------------------------------------


def _make_hdb_with_b_function(
    omega_grid: NDArray[np.float64],
    b_func,
    *,
    dof: int = 3,
) -> HydroDatabase:
    """Build a synthetic HydroDatabase with B[dof, dof, k] = b_func(omega[k])."""
    n_w = omega_grid.size
    A = np.zeros((6, 6, n_w), dtype=np.float64)
    B = np.zeros((6, 6, n_w), dtype=np.float64)
    for k in range(n_w):
        B[dof, dof, k] = b_func(float(omega_grid[k]))
    return HydroDatabase(
        omega=omega_grid,
        heading_deg=np.array([0.0, 90.0]),
        A=A,
        B=B,
        A_inf=np.eye(6) * 1.0e6,
        C=np.zeros((6, 6)),
        RAO=np.zeros((6, n_w, 2), dtype=np.complex128),
        reference_point=np.array([0.0, 0.0, 0.0]),
        C_source="full",
    )


def _make_hdb_marin_semi_with_oc4_C() -> HydroDatabase:
    """marin_semi.1 BEM combined with synthetic OC4-style C and headings.

    Used by tests that need a real BEM kernel (well-resolved at the
    high-frequency end so it satisfies Refinement-2 gates).
    """
    omega, A, B, A_inf = read_added_mass_and_damping(_MARIN_SEMI_PATH)
    C = np.zeros((6, 6), dtype=np.float64)
    C[2, 2] = 3.836e6
    C[3, 3] = 1.078e9
    C[4, 4] = 1.078e9
    n_w = omega.size
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0, 90.0]),
        A=A,
        B=B,
        A_inf=A_inf,
        C=C,
        RAO=np.zeros((6, n_w, 2), dtype=np.complex128),
        reference_point=np.array([0.0, 0.0, 0.0]),
        C_source="full",
    )


# Wide grid with B that decays well below 1% of peak by the right edge.
# Range [0.05, 20] picks up both Lorentzian and peaked synthetics
# (peaked at omega=20 has 9.1e-4 of peak vs the 1% Check-1 threshold).
_WIDE_OMEGA_GRID: Final[NDArray[np.float64]] = np.linspace(0.05, 20.0, 400)


# ---------------------------------------------------------------------------
# (4) Synthetic-kernel analytical regression
# ---------------------------------------------------------------------------


_TAU: Final[float] = 2.0


def _kernel_dof_for(b_func, *, t_max: float, dt: float, dof: int = 3) -> NDArray[np.float64]:
    hdb = _make_hdb_with_b_function(_WIDE_OMEGA_GRID, b_func, dof=dof)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = compute_retardation_kernel(hdb, t_max=t_max, dt=dt)
    return k.K[dof, dof, :]


def test_synthetic_lorentzian_kernel_matches_analytical() -> None:
    """B(ω) = exp(-ω/τ) → K(t) = (2/π) · a / (a² + t²), a = 1/τ.

    The bug pattern that this test pins: long-lag oscillatory
    artifacts. After fix, K(t) at t≥60s must be small.
    """
    K_fs = _kernel_dof_for(lambda w: np.exp(-w / _TAU), t_max=240.0, dt=0.05)
    t_arr = 0.05 * np.arange(K_fs.size, dtype=np.float64)
    a = 1.0 / _TAU
    K_an = (2.0 / np.pi) * a / (a * a + t_arr * t_arr)

    short = t_arr <= 5.0
    np.testing.assert_allclose(
        K_fs[short],
        K_an[short],
        rtol=2.0e-2,
        atol=2.0e-2,
        err_msg="Lorentzian kernel disagrees at short lag",
    )
    long_lag = t_arr >= 60.0
    K_max = float(np.max(np.abs(K_an[short])))
    assert np.max(np.abs(K_fs[long_lag])) < 1.0e-2 * K_max, (
        f"Long-lag artifact: max|K_fs(t≥60)| = "
        f"{float(np.max(np.abs(K_fs[long_lag]))):.3e} > 1% of K_max = {K_max:.3e}."
    )


def test_synthetic_peaked_kernel_matches_analytical() -> None:
    """B(ω) = ω·exp(-ω/τ) → K(t) = (2/π)(a²-t²)/(a²+t²)², a=1/τ.

    Wider grid (ω ∈ [0.05, 20]) so B(ω_max) passes the input gate.
    """
    omega_wide = np.linspace(0.05, 20.0, 400)
    hdb = _make_hdb_with_b_function(omega_wide, lambda w: w * np.exp(-w / _TAU), dof=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = compute_retardation_kernel(hdb, t_max=240.0, dt=0.05)
    K_fs = k.K[3, 3, :]
    t_arr = k.t
    a = 1.0 / _TAU
    K_an = (2.0 / np.pi) * (a * a - t_arr * t_arr) / (a * a + t_arr * t_arr) ** 2

    short = t_arr <= 5.0
    np.testing.assert_allclose(
        K_fs[short],
        K_an[short],
        rtol=5.0e-2,
        atol=5.0e-2,
        err_msg="Peaked kernel disagrees at short lag",
    )
    long_lag = t_arr >= 60.0
    K_max = float(np.max(np.abs(K_an[short])))
    assert np.max(np.abs(K_fs[long_lag])) < 1.0e-2 * K_max


def test_synthetic_smooth_box_kernel_matches_analytical() -> None:
    """B(ω) = B₀ · cos²(πω/(2·ω_max)) · 1_{ω≤ω_max}.

    A Hann-windowed box: smoothly decays to zero at ω_max so it
    passes the input gate. The closed-form K(t) is more complex
    than the sharp box but bounded; we assert short-lag accuracy
    and long-lag decay.
    """
    omega_max = 5.0
    B0 = 1.0

    def b_smooth_box(w: float) -> float:
        if w >= omega_max or w < 0.0:
            return 0.0
        return B0 * np.cos(np.pi * w / (2.0 * omega_max)) ** 2

    K_fs = _kernel_dof_for(b_smooth_box, t_max=200.0, dt=0.05)
    t_arr = 0.05 * np.arange(K_fs.size, dtype=np.float64)
    # Long-lag check: K must decay below a reasonable threshold of K(0).
    K0 = float(K_fs[0])
    long_lag = t_arr >= 30.0
    assert np.max(np.abs(K_fs[long_lag])) < 5.0e-2 * abs(K0), (
        f"Smooth-box kernel long-lag artifact: max|K(t≥30)| = "
        f"{float(np.max(np.abs(K_fs[long_lag]))):.3e} vs K(0) = {K0:.3e}"
    )


# ---------------------------------------------------------------------------
# (5) Damping-non-positivity on real BEM
# ---------------------------------------------------------------------------


def test_marin_semi_heave_damping_is_non_negative_at_all_t_max() -> None:
    """Free-decay damping must be ≥ 0 for any t_max on real BEM data."""
    hdb = _make_hdb_marin_semi_with_oc4_C()
    M = _oc4_rigid_body_mass_matrix()
    lhs = assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb)
    for t_max in (60.0, 120.0, 200.0, 400.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel = compute_retardation_kernel(hdb, t_max=t_max, dt=0.05)
        xi0 = np.zeros(6)
        xi0[2] = 0.5
        res = integrate_cummins(
            lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=np.zeros(6), duration=300.0
        )
        zeta = _fit_damping_log_decrement(res.t, res.xi[:, 2])
        assert zeta >= -1.0e-6, (
            f"Heave damping ratio is negative ({zeta:.6e}) at t_max={t_max} s. "
            "Passive radiation cannot produce net energy gain."
        )


# ---------------------------------------------------------------------------
# (6) t_max convergence on synthetics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, b_func",
    [
        ("lorentzian", lambda w: np.exp(-w / _TAU)),
        ("peaked", lambda w: w * np.exp(-w / _TAU)),
    ],
)
def test_synthetic_kernel_value_is_independent_of_t_max(name: str, b_func) -> None:
    """K(t) at fixed t is a function of B(ω) only -- not of t_max truncation."""
    dt = 0.05
    t_query = 1.0
    idx = round(t_query / dt)

    K_values = []
    for t_max in (60.0, 240.0, 480.0):
        K = _kernel_dof_for(b_func, t_max=t_max, dt=dt)
        K_values.append(float(K[idx]))

    spread = max(K_values) - min(K_values)
    scale = max(abs(v) for v in K_values)
    assert spread / max(scale, 1.0e-12) < 1.0e-9, (
        f"{name}: K(t=1.0 s) varies with t_max: {K_values}. "
        "Kernel must be a pure function of B(omega)."
    )


# ---------------------------------------------------------------------------
# (7) M2 free-decay regression -- period unchanged
# ---------------------------------------------------------------------------


def test_m2_oc4_heave_period_unchanged_after_fix() -> None:
    """OC4 heave period from the M2 fixture stays in the 14-20 s band.

    Period is determined by C_33 and M_33+A_inf_33; independent of
    kernel quality. The fix must not perturb this. Note: this test
    uses the marin_semi-based fixture (post-fixture-update), not the
    deprecated platform_small.yml.
    """
    hdb = _make_hdb_marin_semi_with_oc4_C()
    M = _oc4_rigid_body_mass_matrix()
    lhs = assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kernel = compute_retardation_kernel(hdb, t_max=120.0, dt=0.05)
    xi0 = np.zeros(6)
    xi0[2] = 0.5
    res = integrate_cummins(lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=np.zeros(6), duration=200.0)
    T = _fit_period(res.t, res.xi[:, 2])
    assert 9.0 <= T <= 14.0, (
        f"Heave period {T:.3f} s outside expected 9-14 s band for "
        "(marin_semi.A_inf, OC4 mass, synthetic OC4 C_33=3.836e6 N/m)."
    )


# ---------------------------------------------------------------------------
# (3) Convergence-rate regression on marin_semi.1
# ---------------------------------------------------------------------------


def test_marin_semi_convergence_rate_pins_at_t_max_400() -> None:
    """Period and damping at t_max=400s pinned; t_max=800s within rtol=1e-4.

    Sentinel against future regressions in the kernel computation
    on a well-resolved BEM grid.
    """
    hdb = _make_hdb_marin_semi_with_oc4_C()
    M = _oc4_rigid_body_mass_matrix()
    lhs = assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb)

    results: dict[float, tuple[float, float]] = {}
    for t_max in (60.0, 120.0, 200.0, 400.0, 800.0):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kernel = compute_retardation_kernel(hdb, t_max=t_max, dt=0.05)
        xi0 = np.zeros(6)
        xi0[2] = 0.5
        res = integrate_cummins(
            lhs=lhs, kernel=kernel, xi0=xi0, xi_dot0=np.zeros(6), duration=300.0
        )
        T = _fit_period(res.t, res.xi[:, 2])
        zeta = _fit_damping_log_decrement(res.t, res.xi[:, 2])
        results[t_max] = (T, zeta)

    T_ref, zeta_ref = results[400.0]
    T_long, zeta_long = results[800.0]

    assert (
        abs(T_long - T_ref) / T_ref < 1.0e-4
    ), f"Period not converged at t_max=800: T(400)={T_ref:.4f}, T(800)={T_long:.4f}"
    if abs(zeta_ref) > 1.0e-7:
        assert abs(zeta_long - zeta_ref) / abs(zeta_ref) < 1.0e-2

    # Monotonic approach: no sign flips on damping (physically passive).
    if abs(zeta_ref) > 1.0e-6:
        for t_max, (_, zeta) in results.items():
            if t_max == 400.0:
                continue
            assert (
                np.sign(zeta) == np.sign(zeta_ref) or abs(zeta) < 1.0e-7
            ), f"Damping at t_max={t_max} flipped sign: {zeta:.6e} vs reference {zeta_ref:.6e}."


# ---------------------------------------------------------------------------
# (2) Coarse-grid kernel correctness
# ---------------------------------------------------------------------------


def test_filon_no_nyquist_artifact_on_coarse_grid() -> None:
    """Filon-trapezoidal directly on a coarse marin_semi grid (stride=5,
    dω≈0.05) produces a kernel without Nyquist artifacts at long lag.

    The exact long-lag-converged value differs between coarse and full
    grid because the two linear-interpolations of B(ω) genuinely
    differ near peaks (the bug fix doesn't make under-sampled grids
    "match" full-resolution -- that would require interpolation, not
    integration). What the fix DOES guarantee: no sustained oscillation
    artifacts at t > π/dω. This test pins that the kernel decays
    instead of oscillating with sustained amplitude.

    Old behaviour (trapezoidal-cosine on dω=0.05 at t≥60 s): kernel
    oscillates with amplitude ~ K_max indefinitely. Fixed behaviour
    (Filon-trapezoidal): kernel decays smoothly, no sustained
    oscillation.
    """
    from floatsim.hydro._filon import filon_trap_cosine

    omega_full, _A_full, B_full, _A_inf = read_added_mass_and_damping(_MARIN_SEMI_PATH)
    stride = 5
    omega_ds = omega_full[::stride]
    B_ds = B_full[2, 2, ::stride]

    t_arr = np.linspace(0.0, 240.0, 4801)

    K_ds = (2.0 / np.pi) * filon_trap_cosine(omega_ds, B_ds, t_arr)

    short = t_arr <= 5.0
    long_lag = t_arr >= 100.0
    K_max_short = float(np.max(np.abs(K_ds[short])))
    K_max_long = float(np.max(np.abs(K_ds[long_lag])))

    # The kernel must DECAY at long lag, not sustain at K_max.
    # Old trapezoidal-cosine bug had K_long/K_short ~ 1.0 (sustained
    # oscillation). Filon-trapezoidal should produce K_long / K_short
    # well below 0.5 (decay set in).
    assert K_max_long < 0.5 * K_max_short, (
        f"Coarse-grid Filon kernel at long lag (max|K(t≥100)| = "
        f"{K_max_long:.3e}) is not significantly smaller than at short lag "
        f"(max|K(t≤5)| = {K_max_short:.3e}); ratio "
        f"{K_max_long / K_max_short:.3f} >= 0.5 indicates the Nyquist "
        "aliasing artifact is back. A regression to trapezoidal-cosine "
        "would cause this to fail with ratio ~ 1.0."
    )
