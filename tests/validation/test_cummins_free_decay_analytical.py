"""Milestone 2 gate validation — analytical damping on a synthetic BEM.

The OC4 fixture :mod:`tests.validation.test_oc4_heave_free_decay` validates
**period** agreement on realistic frequency-dependent BEM data. Its
damping comparison is not a clean analytical gate: the OrcaWave-computed
``B_33(omega)`` has local structure (a near-zero dip at
``omega = 0.4 rad/s``) that makes the single-DOF formula
``zeta = B(omega_n) / (2 omega_n (M + A(omega_n)))`` sensitive to
linear-interpolation artefacts at the heave natural frequency.

This file carries the **damping** gate. It isolates the integrator +
discrete convolution by feeding the code a BEM whose ``B(omega)`` is
flat in the natural-frequency band and decays cleanly as ``omega^-4``
at high frequency (per the M6 PR3 Refinement-2 input gates: the BEM
grid must reach the asymptotic regime). At the heave natural
frequency ``omega_n = 0.8 rad/s`` the value is essentially ``B_0`` —
the smooth roll-off has ``cutoff_omega = 5 rad/s`` so
``B(0.8)/B_0 = 5^4 / (5^4 + 0.8^4) ≈ 0.999``. In the narrowband
free-decay limit ``B(omega_n) = B_0`` is unambiguous and the
single-DOF formula is a closed-form reference.

Why no period assertion here
----------------------------
Cummins theory links ``A(omega)`` and ``B(omega)`` through a
Kramers-Kronig dispersion relation — they cannot be specified
independently. The synthetic fixture here holds ``A(omega) = A_inf``
while setting ``B(omega) = B_0 > 0``, which is *not* causally
consistent. As a result the true kernel ``K(t)`` (correctly computed by
:func:`floatsim.hydro.retardation.compute_retardation_kernel`) induces
a small (~0.5 %) effective-mass correction at ``omega_n``, and the
observed period drifts from ``sqrt(C/(M+A_inf))`` by that amount.
Period fidelity on physically consistent BEM data is the job of the OC4
file; here we only want a clean analytical damping reference.

Parameters
----------
Heave-only, all other DOFs decoupled::

    M_33 = A_inf,33 = 1.0e7 kg   -> M + A_inf = 2.0e7 kg
    C_33             = 1.28e7 N/m
    -> omega_n = sqrt(C/(M+A_inf)) = 0.8 rad/s   (T_n = 7.854 s)
    B_33(omega) = 1.6e6 N*s/m    (constant on the grid)
    -> zeta_n = B / (2 omega_n (M+A_inf)) = 0.05   (5 % of critical)

Tolerance
---------
* **Damping ratio**: ``rtol = 5e-2`` — 5 %. The explicit-convolution
  treatment (``mu_{n+1-alpha_f} ~= mu_n``) is O(h) in the radiation
  term; with ``dt = 0.01 s`` and decay timescale ``1/(zeta omega_n)
  = 25 s`` the leading per-cycle error is small but accumulates over
  the 10-cycle log-decrement fit. This is consistent with the
  implicit-vs-explicit convolution gap reported in Fossen (2011) §5.
  We run ``rho_inf = 1.0`` (trapezoidal) to remove generalized-alpha
  numerical dissipation from the budget — the residual is pure
  physical + convolution-discretization error.
"""

from __future__ import annotations

import numpy as np

from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import integrate_cummins
from tests.support.synthetic_bem import make_diagonal_hdb, well_behaved_b
from tests.validation.test_oc4_heave_free_decay import (
    _fit_damping_log_decrement,
)

# ---------------------------------------------------------------------------
# system definition
# ---------------------------------------------------------------------------

_M_33 = 1.0e7  # rigid-body heave mass                    [kg]
_A_INF_33 = 1.0e7  # infinite-frequency heave added mass      [kg]
_C_33 = 1.28e7  # heave hydrostatic stiffness              [N/m]
_B_33 = 1.6e6  # constant heave radiation damping         [N*s/m]

# Non-heave DOFs are fully decoupled and irrelevant to the heave free decay.
# They only need positive (M+A_inf)_ii so the integrator's linear solve is
# well-conditioned. Translations get the same mass, rotations a larger one.
_M_OTHER = 1.0e7  # surge/sway mass                          [kg]
_I_OTHER = 1.0e9  # roll/pitch/yaw inertia                   [kg*m^2]

# Grid extends to 20 rad/s so that |B(omega_max)| < 1% of peak per the
# Refinement-2 amplitude gate.
_OMEGA_GRID = np.linspace(0.05, 20.0, 401)
_HEADING = np.array([0.0, 90.0])
# Roll-off above this cutoff; chosen well above heave omega_n = 0.8 rad/s
# so B(omega_n) ≈ B_0 (within 0.1 %).
_CUTOFF_OMEGA = 5.0


def _build_hdb():
    A_inf_diag = [_M_OTHER, _M_OTHER, _A_INF_33, _I_OTHER, _I_OTHER, _I_OTHER]
    # A(omega) = A_inf across the grid (frequency-flat added mass). Note
    # this is intentionally not Kramers-Kronig-consistent with the chosen
    # B(omega) -- see module docstring for why period is not asserted here.
    A_diag_per_omega = [list(A_inf_diag) for _ in range(_OMEGA_GRID.size)]
    # B with the well-behaved profile per DOF: flat in the band and
    # ω⁻⁴ at high frequency. The DOF band-values reproduce the
    # pre-fix [1.0e3, 1.0e3, _B_33, 1.0e4, 1.0e4, 1.0e4] plateau.
    band_values = [1.0e3, 1.0e3, _B_33, 1.0e4, 1.0e4, 1.0e4]
    rolloff = well_behaved_b(_OMEGA_GRID, band_value=1.0, cutoff_omega=_CUTOFF_OMEGA)
    B_diag_per_omega = [[bv * float(r) for bv in band_values] for r in rolloff]
    C_diag = [0.0, 0.0, _C_33, 0.0, 0.0, 0.0]  # heave-only restoring
    return make_diagonal_hdb(
        A_inf_diag=A_inf_diag,
        C_diag=C_diag,
        A_diag_per_omega=A_diag_per_omega,
        B_diag_per_omega=B_diag_per_omega,
        omega=_OMEGA_GRID.tolist(),
        heading_deg=_HEADING.tolist(),
    )


def _rigid_body_mass_matrix():
    return np.diag([_M_OTHER, _M_OTHER, _M_33, _I_OTHER, _I_OTHER, _I_OTHER]).astype(np.float64)


# ---------------------------------------------------------------------------
# analytical references (closed-form under frequency-flat A, constant B)
# ---------------------------------------------------------------------------

_OMEGA_N = float(np.sqrt(_C_33 / (_M_33 + _A_INF_33)))
_ZETA_N = _B_33 / (2.0 * _OMEGA_N * (_M_33 + _A_INF_33))


# ---------------------------------------------------------------------------
# the actual validation
# ---------------------------------------------------------------------------


def _run_heave_free_decay():
    hdb = _build_hdb()
    lhs = assemble_cummins_lhs(rigid_body_mass=_rigid_body_mass_matrix(), hdb=hdb)
    dt = 0.01  # ~ 785 samples per heave period; shrinks explicit-mu O(h) error
    # Long kernel window — the constant-B / band-limited kernel envelope
    # falls off slowly (~ 1/t), so 200 s is needed to drive |K(t_max)| well
    # below the 1 % §9.1 decay gate.
    kernel = compute_retardation_kernel(hdb, t_max=200.0, dt=dt)

    xi0 = np.zeros(6)
    xi0[2] = 1.0  # 1.0 m heave displacement
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=100.0,  # ~ 12.7 heave periods at T ~ 7.85 s
        rho_inf=1.0,  # trapezoidal limit: zero numerical damping isolates physical zeta
    )
    return res


def test_analytical_heave_damping_ratio_matches_closed_form() -> None:
    res = _run_heave_free_decay()
    zeta_fit = _fit_damping_log_decrement(res.t, res.xi[:, 2])
    rel_err = abs(zeta_fit - _ZETA_N) / abs(_ZETA_N)
    assert rel_err < 5.0e-2, (
        f"heave damping ratio {zeta_fit:.5f} deviates from analytical "
        f"{_ZETA_N:.5f} by {rel_err:.3%} (limit 5%)"
    )


def test_other_dofs_stay_at_rest_under_heave_only_ic() -> None:
    """DOFs are decoupled -> heave IC must not excite any other DOF."""
    res = _run_heave_free_decay()
    other = np.delete(res.xi, 2, axis=1)
    assert np.max(np.abs(other)) < 1.0e-10, (
        f"coupled motion detected on non-heave DOFs: max|xi_other| = "
        f"{np.max(np.abs(other)):.3e} m (expected 0 under diagonal hydro)"
    )
