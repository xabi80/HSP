"""Milestone 2 — generalized-alpha integrator unit tests.

Drives ``floatsim.solver.newmark.integrate_cummins`` through the linear
Cummins ODE

    [M + A_inf] xi_ddot + mu(t) + C xi = F_ext(t)

on synthetic diagonal fixtures where the analytical response is known.
Validates:

* state bookkeeping (initial conditions, output shapes)
* undamped harmonic oscillator against ``xi(t) = cos(omega_n t)``
* damped oscillator against exponential-envelope decay
* constant-force steady offset
* input validation (``dt`` must match ``kernel.dt``, positive ``duration``).
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import IntegrationResult, integrate_cummins
from tests.support.synthetic_bem import make_diagonal_hdb, well_behaved_b

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _identity_mass() -> np.ndarray:
    """6x6 identity mass matrix — diagonal, unit mass/inertia."""
    return np.eye(6, dtype=np.float64)


def _build_setup(
    *,
    M_diag: list[float] | None = None,
    A_inf_diag: list[float] | None = None,
    C_diag: list[float] | None = None,
    B_diag_per_omega: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    t_max_kernel: float = 8.0,
    dt: float = 0.02,
) -> tuple:
    """Assemble (lhs, kernel) for an integrator test on a diagonal fixture."""
    if omega is None:
        # M6 PR3: extend grid into the asymptotic decay band so the
        # Refinement-2 input gates pass.
        omega = np.linspace(0.0, 20.0, 401)
    if A_inf_diag is None:
        A_inf_diag = [0.0] * 6
    if C_diag is None:
        C_diag = [0.0] * 6
    if B_diag_per_omega is None:
        B_diag_per_omega = np.zeros((omega.size, 6))
    hdb = make_diagonal_hdb(
        A_inf_diag=A_inf_diag,
        C_diag=C_diag,
        A_diag_per_omega=[A_inf_diag] * omega.size,
        B_diag_per_omega=[list(row) for row in B_diag_per_omega],
        omega=list(omega),
    )
    M = _identity_mass() if M_diag is None else np.diag(M_diag).astype(np.float64)
    lhs = assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb)
    kernel = compute_retardation_kernel(hdb, t_max=t_max_kernel, dt=dt)
    return lhs, kernel


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_integrate_cummins_returns_result_dataclass() -> None:
    lhs, kernel = _build_setup(C_diag=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=np.zeros(6),
        xi_dot0=np.zeros(6),
        duration=1.0,
    )
    assert isinstance(res, IntegrationResult)


def test_output_time_grid_and_shape() -> None:
    lhs, kernel = _build_setup(C_diag=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dt=0.1)
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=np.zeros(6),
        xi_dot0=np.zeros(6),
        duration=1.0,
    )
    # 1.0 s with dt=0.1 → 11 samples (including t=0)
    assert res.t.shape == (11,)
    np.testing.assert_allclose(res.t, np.linspace(0.0, 1.0, 11))
    assert res.xi.shape == (11, 6)
    assert res.xi_dot.shape == (11, 6)
    assert res.xi_ddot.shape == (11, 6)


def test_initial_conditions_are_respected() -> None:
    lhs, kernel = _build_setup(C_diag=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    xi0 = np.array([0.1, -0.2, 0.3, 0.0, 0.05, -0.02])
    xi_dot0 = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=xi_dot0,
        duration=0.5,
    )
    np.testing.assert_allclose(res.xi[0], xi0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(res.xi_dot[0], xi_dot0, rtol=0.0, atol=0.0)


def test_initial_acceleration_satisfies_eom() -> None:
    """At t=0 with xi_dot0=0 and empty buffer, xi_ddot0 = M^-1 (F_0 - C xi_0)."""
    M_diag = [2.0] * 6
    C_diag = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0]  # heave-only restoring
    lhs, kernel = _build_setup(M_diag=M_diag, C_diag=C_diag)
    xi0 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=0.1,
    )
    expected_xi_ddot = -np.array(C_diag) * xi0 / np.array(M_diag)
    np.testing.assert_allclose(res.xi_ddot[0], expected_xi_ddot, rtol=1e-12)


# ---------------------------------------------------------------------------
# zero-everything sanity
# ---------------------------------------------------------------------------


def test_zero_ic_and_zero_force_stays_at_rest() -> None:
    omega = np.linspace(0.0, 20.0, 401)
    # Well-behaved B(ω): flat ≈ 1.0 in the band, decays as ω⁻⁴ above
    # the cutoff so the Refinement-2 input gates pass.
    b_per_omega = np.tile(
        well_behaved_b(omega, band_value=1.0, cutoff_omega=5.0)[:, None],
        (1, 6),
    )
    lhs, kernel = _build_setup(
        C_diag=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        B_diag_per_omega=b_per_omega,
        omega=omega,
    )
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=np.zeros(6),
        xi_dot0=np.zeros(6),
        duration=5.0,
    )
    assert np.max(np.abs(res.xi)) == 0.0
    assert np.max(np.abs(res.xi_dot)) == 0.0
    assert np.max(np.abs(res.xi_ddot)) == 0.0


# ---------------------------------------------------------------------------
# analytical oscillator responses
# ---------------------------------------------------------------------------


def test_undamped_oscillator_matches_cos_at_one_period() -> None:
    """1-DOF heave: M=1, C=1, B=0 -> xi_heave(t) = cos(t). After one period
    (t = 2*pi) we must return to xi=1 with xi_dot=0 to high accuracy."""
    lhs, kernel = _build_setup(
        C_diag=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dt=0.005,
        t_max_kernel=1.0,  # B=0 so kernel is zero anyway
    )
    xi0 = np.zeros(6)
    xi0[2] = 1.0
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=2.0 * np.pi,
    )
    # Generalized-alpha with rho_inf=0.9 has ~ O(dt^2) period error.
    # At dt=0.005 over one period (~1257 steps), expect rtol ~ 1e-3.
    assert res.xi[-1, 2] == pytest.approx(1.0, rel=5e-3)
    assert abs(res.xi_dot[-1, 2]) < 5e-3


def test_undamped_oscillator_matches_cos_along_full_trajectory() -> None:
    lhs, kernel = _build_setup(
        C_diag=[0.0, 0.0, 4.0, 0.0, 0.0, 0.0],  # omega_n = 2
        dt=0.002,
        t_max_kernel=1.0,
    )
    xi0 = np.zeros(6)
    xi0[2] = 0.5
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=np.pi,  # one period for omega=2
    )
    analytical = 0.5 * np.cos(2.0 * res.t)
    np.testing.assert_allclose(res.xi[:, 2], analytical, atol=5e-3)


def test_constant_force_drives_toward_static_offset() -> None:
    """With damping + restoring + constant force, the system must settle
    at xi_ss = C^-1 F_ext (within a small residual oscillation)."""
    M_diag = [1.0] * 6
    C_diag = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # Well-behaved B(omega) near omega_n=1 with ω⁻⁴ tail -- gates pass.
    omega = np.linspace(0.0, 20.0, 501)
    B_diag = np.zeros((omega.size, 6))
    B_diag[:, 2] = well_behaved_b(omega, band_value=0.8, cutoff_omega=5.0)
    lhs, kernel = _build_setup(
        M_diag=M_diag,
        C_diag=C_diag,
        B_diag_per_omega=B_diag,
        omega=omega,
        dt=0.01,
        t_max_kernel=20.0,
    )
    F0 = 0.3
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=np.zeros(6),
        xi_dot0=np.zeros(6),
        duration=60.0,  # many natural periods
        external_force=lambda _t: np.array([0.0, 0.0, F0, 0.0, 0.0, 0.0]),
    )
    steady = F0 / C_diag[2]  # = 0.3
    # Last-quarter mean should be close to the steady offset.
    tail = res.xi[-res.xi.shape[0] // 4 :, 2]
    assert tail.mean() == pytest.approx(steady, rel=0.1)


# ---------------------------------------------------------------------------
# damping: positive B(omega) -> energy decreases
# ---------------------------------------------------------------------------


def test_radiation_damping_causes_decay() -> None:
    """Free release with positive B in heave -> heave amplitude strictly
    smaller several periods later than at the first peak."""
    M_diag = [1.0] * 6
    C_diag = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    omega = np.linspace(0.0, 20.0, 501)
    B_diag = np.zeros((omega.size, 6))
    B_diag[:, 2] = well_behaved_b(omega, band_value=0.5, cutoff_omega=5.0)
    lhs, kernel = _build_setup(
        M_diag=M_diag,
        C_diag=C_diag,
        B_diag_per_omega=B_diag,
        omega=omega,
        dt=0.01,
        t_max_kernel=20.0,
    )
    xi0 = np.zeros(6)
    xi0[2] = 1.0
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=30.0,
    )
    # Envelope: compare early peak to late peak.
    heave = res.xi[:, 2]
    early_peak = np.max(np.abs(heave[: heave.size // 4]))
    late_peak = np.max(np.abs(heave[-heave.size // 4 :]))
    assert late_peak < 0.5 * early_peak


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_rejects_dt_mismatched_to_kernel() -> None:
    lhs, kernel = _build_setup(C_diag=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dt=0.01)
    with pytest.raises(ValueError, match="dt"):
        integrate_cummins(
            lhs=lhs,
            kernel=kernel,
            xi0=np.zeros(6),
            xi_dot0=np.zeros(6),
            duration=1.0,
            dt=0.02,  # kernel.dt == 0.01
        )


def test_rejects_non_positive_duration() -> None:
    lhs, kernel = _build_setup(C_diag=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="duration"):
        integrate_cummins(
            lhs=lhs,
            kernel=kernel,
            xi0=np.zeros(6),
            xi_dot0=np.zeros(6),
            duration=0.0,
        )
    with pytest.raises(ValueError, match="duration"):
        integrate_cummins(
            lhs=lhs,
            kernel=kernel,
            xi0=np.zeros(6),
            xi_dot0=np.zeros(6),
            duration=-1.0,
        )


def test_rejects_rho_inf_out_of_range() -> None:
    lhs, kernel = _build_setup(C_diag=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="rho_inf"):
        integrate_cummins(
            lhs=lhs,
            kernel=kernel,
            xi0=np.zeros(6),
            xi_dot0=np.zeros(6),
            duration=1.0,
            rho_inf=-0.1,
        )
    with pytest.raises(ValueError, match="rho_inf"):
        integrate_cummins(
            lhs=lhs,
            kernel=kernel,
            xi0=np.zeros(6),
            xi_dot0=np.zeros(6),
            duration=1.0,
            rho_inf=1.1,
        )
