"""Milestone 3 gate — single body in regular waves matches steady-state RAO.

ARCHITECTURE.md §8 M3 requires: *"Validate: single body in regular waves
matches steady-state RAO response."*

Governing equation (frequency domain, ``exp(-i omega t)`` convention)::

    [-omega^2 (M + A(omega)) - i omega B(omega) + C] xi_hat = F_hat_wave
    F_hat_wave = RAO(omega, beta) * eta_hat(body)

With Ogilvie's relation ``int_0^inf K(tau) exp(+i omega tau) dtau =
B(omega) + i omega (A_inf - A(omega))``, the Cummins equation's
frequency-domain image at a harmonic drive is equivalently::

    impedance(omega) = -omega^2 (M + A_inf) + (-i omega) I(omega) + C
    I(omega) = int_0^inf K(tau) exp(+i omega tau) dtau

This is the form the time-domain code realizes. We predict ``xi_hat``
from the **discrete kernel's own FT at ``omega_wave``** (self-consistent
prediction), which isolates the wave-to-motion pipeline from the BEM
fixture's physical consistency. The DCT that builds ``K(t)`` from
``B(omega)`` is validated separately in
``tests/unit/test_retardation_kernel.py``.

Fixture choice
--------------
A physically plausible BEM has a smooth ``B(omega)`` profile. We use a
gaussian bump centred at the wave frequency with narrow support, well
inside the frequency grid, on a 1200-point grid (``d_omega = 0.005``)
so both the forward and inverse cosine transforms converge. The stored ``A(omega) = A_inf`` is
not causally consistent with this ``B``, but the self-consistent
prediction above sidesteps that: we compare against what the kernel
*actually* encodes, not what the fixture metadata claims.

Tolerance
---------
* **Amplitude**: ``rtol = 5.0e-3`` (0.5 %).
* **Phase**:     ``atol = 5.0e-3 rad`` (~ 0.29 deg).

Residual error budget: generalized-alpha is 2nd-order in ``h``
(``(omega h)^2 ~ 1e-4`` at ``h = 0.01``, ``omega ~ 1``); the explicit-mu
treatment ``mu_{n+1-alpha_f} ~= mu_n`` induces a phase lag
``alpha_f h omega ~ 5e-3 rad`` on the damping term at ``rho_inf = 1.0``.
Tightening further would cost 10x per test for no physical gain.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.excitation import interpolate_rao, make_regular_wave_force
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.retardation import RetardationKernel, compute_retardation_kernel
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave

# ---------------------------------------------------------------------------
# fixture constants
# ---------------------------------------------------------------------------

_OMEGA_GRID = np.arange(0.005, 6.0 + 1.0e-9, 0.005)  # contains 0.6 and 1.0 exactly
_HEADING = np.array([0.0, 90.0])

_M_PER_DOF = [1.0e7, 1.0e7, 1.0e7, 1.0e9, 1.0e9, 1.0e9]
_A_INF_PER_DOF = [1.0e7, 1.0e7, 1.0e7, 1.0e9, 1.0e9, 1.0e9]
_C_PER_DOF = [0.0, 0.0, 1.28e7, 3.0e9, 3.0e9, 0.0]

# Per-DOF gaussian B(omega) peak amplitudes and centres.
_B_PEAK_PER_DOF = [0.0, 0.0, 1.6e6, 1.0e8, 1.0e8, 0.0]
_B_SIGMA = 0.5  # gaussian width in rad/s


def _build_hdb(
    rao_values: dict[tuple[int, float], complex],
    b_centres: list[float],
) -> HydroDatabase:
    """Build a diagonal HDB with gaussian-shaped ``B(omega)`` per DOF.

    Parameters
    ----------
    rao_values
        Mapping ``(dof_index, omega_on_grid) -> complex RAO``, same at
        both headings.
    b_centres
        Per-DOF centre of the gaussian ``B(omega)`` peak in rad/s. Use 0
        for DOFs whose peak is irrelevant (combined with ``_B_PEAK_PER_DOF
        = 0`` to make ``B`` identically zero).
    """
    n_w = _OMEGA_GRID.size
    A = np.zeros((6, 6, n_w), dtype=np.float64)
    B = np.zeros((6, 6, n_w), dtype=np.float64)
    for dof in range(6):
        peak = _B_PEAK_PER_DOF[dof]
        if peak > 0.0:
            B[dof, dof, :] = peak * np.exp(-(((_OMEGA_GRID - b_centres[dof]) / _B_SIGMA) ** 2))
        A[dof, dof, :] = _A_INF_PER_DOF[dof]

    A_inf = np.diag(_A_INF_PER_DOF).astype(np.float64)
    C = np.diag(_C_PER_DOF).astype(np.float64)

    RAO = np.zeros((6, n_w, _HEADING.size), dtype=np.complex128)
    for (dof, omega_value), rao_complex in rao_values.items():
        diffs = np.abs(_OMEGA_GRID - omega_value)
        i = int(np.argmin(diffs))
        assert (
            diffs[i] < 1.0e-10
        ), f"omega={omega_value} is not on the grid; closest = {_OMEGA_GRID[i]}"
        RAO[dof, i, :] = rao_complex

    return HydroDatabase(
        omega=_OMEGA_GRID.astype(np.float64),
        heading_deg=_HEADING.astype(np.float64),
        A=A,
        B=B,
        A_inf=A_inf,
        C=C,
        RAO=RAO,
        reference_point=np.zeros(3, dtype=np.float64),
    )


def _rigid_body_mass_matrix() -> np.ndarray:
    return np.diag(_M_PER_DOF).astype(np.float64)


# ---------------------------------------------------------------------------
# helpers: kernel FT, analytical H, LS fit
# ---------------------------------------------------------------------------


def _kernel_ft_at(kernel: RetardationKernel, omega: float) -> np.ndarray:
    """Evaluate ``I(omega) = int_0^inf K(tau) exp(+i omega tau) dtau`` at a single omega.

    Uses trapezoidal quadrature on the kernel's time grid, returning a
    ``(6, 6)`` complex matrix.
    """
    t = kernel.t
    dt = kernel.dt
    w = np.exp(1j * omega * t) * dt
    w[0] *= 0.5
    w[-1] *= 0.5
    return np.einsum("ijn,n->ij", kernel.K, w).astype(np.complex128)


def _analytical_phasor_from_kernel(
    hdb: HydroDatabase,
    kernel: RetardationKernel,
    M_rigid: np.ndarray,
    wave: RegularWave,
    body_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Steady-state ``xi_hat`` predicted by the freq-domain image of the discrete kernel."""
    omega = wave.omega
    A_inf = hdb.A_inf
    I_w = _kernel_ft_at(kernel, omega)
    impedance = -(omega**2) * (M_rigid + A_inf) + (-1j * omega) * I_w + hdb.C
    H = np.linalg.inv(impedance)

    rao = interpolate_rao(hdb, omega, wave.heading_deg)
    beta = np.radians(wave.heading_deg)
    k = wave.wavenumber
    x_b, y_b, _ = body_position
    eta_hat = wave.amplitude * np.exp(
        1j * (k * (x_b * np.cos(beta) + y_b * np.sin(beta)) + wave.phase)
    )
    return H @ (rao * eta_hat)


def _fit_complex_amplitude(t: np.ndarray, x: np.ndarray, omega: float) -> complex:
    """Least-squares fit of ``x(t) ~= Re{z exp(-i omega t)} = a cos + b sin``."""
    basis = np.column_stack([np.cos(omega * t), np.sin(omega * t)])
    coeffs, *_ = np.linalg.lstsq(basis, x, rcond=None)
    a, b = coeffs
    return complex(a, b)


def _phase_diff(z1: complex, z2: complex) -> float:
    """Wrap-safe absolute phase difference in ``[0, pi]``."""
    d = np.angle(z1) - np.angle(z2)
    return float(abs((d + np.pi) % (2.0 * np.pi) - np.pi))


# ---------------------------------------------------------------------------
# shared time-domain run
# ---------------------------------------------------------------------------


def _run_time_domain(
    hdb: HydroDatabase,
    wave: RegularWave,
    *,
    dt: float,
    duration: float,
    t_max_kernel: float = 50.0,
    ramp_duration: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, RetardationKernel]:
    """Integrate Cummins with the given wave forcing; return (t, xi, kernel)."""
    lhs = assemble_cummins_lhs(rigid_body_mass=_rigid_body_mass_matrix(), hdb=hdb)
    kernel = compute_retardation_kernel(hdb, t_max=t_max_kernel, dt=dt)
    ramp = HalfCosineRamp(duration=ramp_duration)
    force = make_regular_wave_force(hdb=hdb, wave=wave, ramp=ramp)
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=np.zeros(6),
        xi_dot0=np.zeros(6),
        duration=duration,
        external_force=force,
        rho_inf=1.0,
    )
    return res.t, res.xi, kernel


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


_AMP_RTOL: float = 5.0e-3
_PHASE_ATOL: float = 5.0e-3


def test_m3_heave_steady_state_matches_kernel_consistent_rao_response() -> None:
    """Heave excitation at omega_wave = 1 rad/s."""
    rao_heave = 1.0e6 + 0.5e6j
    b_centres = [0.0, 0.0, 1.0, 1.2, 1.2, 0.0]
    hdb = _build_hdb({(2, 1.0): rao_heave}, b_centres=b_centres)
    wave = RegularWave(amplitude=1.0, omega=1.0, heading_deg=0.0)

    t, xi, kernel = _run_time_domain(hdb, wave, dt=0.01, duration=250.0)
    late = t >= 150.0
    z_fit = _fit_complex_amplitude(t[late], xi[late, 2], wave.omega)
    z_true = _analytical_phasor_from_kernel(hdb, kernel, _rigid_body_mass_matrix(), wave)[2]

    mag_err = abs(abs(z_fit) - abs(z_true)) / abs(z_true)
    phase_err = _phase_diff(z_fit, z_true)
    assert mag_err < _AMP_RTOL, (
        f"heave amplitude relative error {mag_err:.3%} exceeds {_AMP_RTOL:.1%} "
        f"(|z_fit|={abs(z_fit):.5e}, |z_true|={abs(z_true):.5e})"
    )
    assert phase_err < _PHASE_ATOL, (
        f"heave phase error {phase_err:.3e} rad exceeds {_PHASE_ATOL:.1e} "
        f"(arg z_fit={np.angle(z_fit):.5f}, arg z_true={np.angle(z_true):.5f})"
    )


def test_m3_pitch_steady_state_at_different_frequency() -> None:
    """Pitch excitation at omega_wave = 0.6 rad/s."""
    rao_pitch = 8.0e7 - 2.0e7j
    # Pitch B peak centred near wave freq for a good KK-consistent fixture; heave
    # peak at 0.8 to ensure no accidental coupling at omega=0.6.
    b_centres = [0.0, 0.0, 0.8, 0.6, 0.6, 0.0]
    hdb = _build_hdb({(4, 0.6): rao_pitch}, b_centres=b_centres)
    wave = RegularWave(amplitude=1.5, omega=0.6, heading_deg=0.0, phase=0.3)

    t, xi, kernel = _run_time_domain(hdb, wave, dt=0.01, duration=350.0)
    late = t >= 200.0
    z_fit = _fit_complex_amplitude(t[late], xi[late, 4], wave.omega)
    z_true = _analytical_phasor_from_kernel(hdb, kernel, _rigid_body_mass_matrix(), wave)[4]

    mag_err = abs(abs(z_fit) - abs(z_true)) / abs(z_true)
    phase_err = _phase_diff(z_fit, z_true)
    assert mag_err < _AMP_RTOL, (
        f"pitch amplitude relative error {mag_err:.3%} exceeds {_AMP_RTOL:.1%} "
        f"(|z_fit|={abs(z_fit):.5e}, |z_true|={abs(z_true):.5e})"
    )
    assert (
        phase_err < _PHASE_ATOL
    ), f"pitch phase error {phase_err:.3e} rad exceeds {_PHASE_ATOL:.1e}"


def test_m3_non_excited_dofs_remain_quiescent() -> None:
    """With only the heave RAO nonzero, other DOFs must stay near zero."""
    rao_heave = 1.0e6 + 0.5e6j
    b_centres = [0.0, 0.0, 1.0, 1.2, 1.2, 0.0]
    hdb = _build_hdb({(2, 1.0): rao_heave}, b_centres=b_centres)
    wave = RegularWave(amplitude=1.0, omega=1.0, heading_deg=0.0)

    t, xi, _kernel = _run_time_domain(hdb, wave, dt=0.01, duration=150.0)
    late = t >= 100.0
    other_dofs = np.delete(xi[late], 2, axis=1)
    max_other = float(np.max(np.abs(other_dofs)))
    assert max_other < 1.0e-6, (
        f"decoupled DOFs saw {max_other:.3e} m|rad of motion (expected 0 "
        f"under diagonal hydro + heave-only RAO)"
    )


def test_m3_heading_90_selects_second_heading_slice() -> None:
    """RAO that differs between heading 0 and heading 90 — verify routing."""
    n_w = _OMEGA_GRID.size
    A = np.zeros((6, 6, n_w), dtype=np.float64)
    B = np.zeros((6, 6, n_w), dtype=np.float64)
    for dof in range(6):
        A[dof, dof, :] = _A_INF_PER_DOF[dof]
        if _B_PEAK_PER_DOF[dof] > 0.0:
            centre = 1.0 if dof == 2 else 1.2
            B[dof, dof, :] = _B_PEAK_PER_DOF[dof] * np.exp(
                -(((_OMEGA_GRID - centre) / _B_SIGMA) ** 2)
            )

    RAO = np.zeros((6, n_w, 2), dtype=np.complex128)
    i_omega = int(np.argmin(np.abs(_OMEGA_GRID - 1.0)))
    RAO[2, i_omega, 0] = 2.0e6 + 0.0j
    RAO[2, i_omega, 1] = -1.0e6 + 0.0j

    hdb = HydroDatabase(
        omega=_OMEGA_GRID.astype(np.float64),
        heading_deg=_HEADING.astype(np.float64),
        A=A,
        B=B,
        A_inf=np.diag(_A_INF_PER_DOF).astype(np.float64),
        C=np.diag(_C_PER_DOF).astype(np.float64),
        RAO=RAO,
        reference_point=np.zeros(3, dtype=np.float64),
    )
    wave = RegularWave(amplitude=1.0, omega=1.0, heading_deg=90.0)

    t, xi, kernel = _run_time_domain(hdb, wave, dt=0.01, duration=250.0)
    late = t >= 150.0
    z_fit = _fit_complex_amplitude(t[late], xi[late, 2], wave.omega)
    z_true = _analytical_phasor_from_kernel(hdb, kernel, _rigid_body_mass_matrix(), wave)[2]

    mag_err = abs(abs(z_fit) - abs(z_true)) / abs(z_true)
    phase_err = _phase_diff(z_fit, z_true)
    assert mag_err < _AMP_RTOL, f"heading-90 heave amplitude error {mag_err:.3%}"
    assert phase_err < _PHASE_ATOL, f"heading-90 heave phase error {phase_err:.3e} rad"
    # Cross-check against heading-0 RAO: if the heading axis were mis-indexed
    # the time series would match the heading-0 response, which has twice the
    # amplitude and opposite sign from the heading-90 prediction z_true.
    wave_zero_heading = RegularWave(amplitude=1.0, omega=1.0, heading_deg=0.0)
    z_wrong = _analytical_phasor_from_kernel(
        hdb, kernel, _rigid_body_mass_matrix(), wave_zero_heading
    )[2]
    assert abs(z_fit - z_wrong) > abs(z_fit - z_true), (
        "time-domain response is closer to the heading-0 prediction than "
        "heading-90 — heading axis may be mis-indexed"
    )


# Full-length Cummins simulations; keep out of the default fast PR suite.
pytestmark = pytest.mark.slow
