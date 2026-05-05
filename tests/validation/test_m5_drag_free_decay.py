"""Milestone 5 gate validation — Morison-drag free-decay hyperbolic envelope.

Quadratic damping has a closed-form free-decay envelope distinct from
linear damping (Faltinsen 1990, *Sea Loads on Ships and Offshore
Structures*, Ch. 4). For the heave equation::

    (m + A_inf) * xi_ddot + C * xi = -0.5 * rho * Cd * D * L * |xi_dot| * xi_dot

the successive peak amplitudes obey the **hyperbolic** recursion::

    xi_n = xi_0 / (1 + n * xi_0 * delta),    delta = (4/3) * rho * Cd * D * L
                                                     ----------------------------
                                                            m + A_inf_zz

Linear damping gives ``xi_n = xi_0 * exp(-n * zeta)`` — a different
shape. Over one or two cycles the two are visually similar; over five
they diverge sharply. **The envelope shape is the test.** Do not fit
an exponential.

System
------
- Heave-only restoring with all other DOFs decoupled.
- Calm sea (``u_fluid = 0``) — drag is driven entirely by the body's
  own velocity, isolating the quadratic damping mechanism from wave
  forcing.
- Radiation damping ``B(omega) = 0`` (a tiny non-zero diagonal is used
  only so the kernel routine has well-defined output; the linear
  damping ratio is < 1e-9 and irrelevant).
- One horizontal Morison element centred at the body reference point,
  oriented along ``X`` so that vertical body motion produces
  fully-normal flow ``u_n = -v_z * z_hat`` (no axial-flow ambiguity).

Numerical setup
---------------
- ``dt = 0.01 s`` (~ 555 samples per period at ``T_n = 5.55 s``); small
  enough that the explicit ``state_force`` lag (drag evaluated at the
  previous step) contributes a per-cycle error well below ``rtol``.
- ``rho_inf = 1.0`` (trapezoidal limit, zero numerical damping) so the
  peak match measures *physical* hyperbolic agreement, not the
  generalized-alpha damping that would also bleed amplitude.

Tolerance — ``rtol = 1e-4``
---------------------------
The plan target is ``rtol = 1e-3``. With ``xi_0 * delta = 0.027``
(lightly damped, ~2.7 % drop per cycle) the asymptotic envelope theory
is essentially exact, and the explicit-state-force lag at
``dt = 0.01`` contributes a per-peak error of ~ 1e-5. The actual
agreement comes in at ``rel-err ~ 1e-5`` per peak — an order of
magnitude better than the plan target. Tightening to ``rtol = 1e-4``
(10x safety margin over observed) keeps the test sensitive to a
silent regression in the drag formula, the projected-area calculation,
or the body-velocity-at-midpoint kinematics, while staying clear of
machine-noise false failures.

Bonus discrimination test
-------------------------
We assert that an *exponential extrapolation* (calibrated to match the
first observed peak) over-predicts ``xi_5`` by **more** than the
hyperbolic-envelope error — confirming the test genuinely
discriminates the two damping models, addressing the
"envelope-vs-exponential trap" risk noted in M5 PR5.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.morison import MorisonElement, make_morison_state_force
from floatsim.hydro.radiation import assemble_cummins_lhs
from floatsim.hydro.retardation import compute_retardation_kernel
from floatsim.solver.newmark import integrate_cummins
from tests.support.synthetic_bem import make_diagonal_hdb, well_behaved_b

# Synthetic ``B(omega) = 1e-3`` (chosen so the radiation-induced linear
# damping ratio is ~ 4e-10 and physically irrelevant) is constant across
# the band; the resulting retardation kernel decays as ~ 1/t and never
# reaches the 1 % gate. The diagnostic is informational here — the kernel
# magnitude is tiny in absolute terms — so we silence the module-level
# warning at the test boundary rather than fight it with an unrealistic
# t_max.
pytestmark = pytest.mark.filterwarnings("ignore:retardation kernel decay is too slow:UserWarning")


# ---------------------------------------------------------------------------
# system parameters
# ---------------------------------------------------------------------------

_RHO = 1025.0  # seawater density                     [kg/m^3]

_M_33 = 5.0e5  # rigid-body heave mass                [kg]
_A_INF_33 = 5.0e5  # infinite-frequency added mass    [kg]
_C_33 = 1.28e6  # heave hydrostatic stiffness         [N/m]

# Drag plate: a horizontal cylinder of diameter D, length L.
_CD = 2.0  # drag coefficient                          [-]
_D = 2.0  # hydrodynamic diameter                       [m]
_L = 10.0  # member length                              [m]

# Other DOFs are decoupled (they contribute zero coupling to heave).
_M_OTHER = 1.0e6  # surge/sway mass                    [kg]
_I_OTHER = 1.0e8  # roll/pitch/yaw inertia             [kg*m^2]

# Tiny non-zero radiation damping so the retardation-kernel diagnostic
# has a well-defined max|K|. The induced linear damping ratio is
# B/(2 omega_n (M+A_inf)) ~ 4e-10 — utterly negligible against the
# hyperbolic-envelope test.
_B_TINY = 1.0e-3

_OMEGA_GRID = np.linspace(0.05, 20.0, 401)  # extended for M6 PR3 input gates
_HEADING = np.array([0.0, 90.0])
_CUTOFF_OMEGA = 5.0  # ω⁻⁴ roll-off cutoff

_XI0_HEAVE = 0.5  # initial heave displacement         [m]


# Closed-form references (module-level so the discrimination test can
# import them too).
_OMEGA_N = float(np.sqrt(_C_33 / (_M_33 + _A_INF_33)))
_T_N = 2.0 * np.pi / _OMEGA_N
_DELTA = (4.0 / 3.0) * _RHO * _CD * _D * _L / (_M_33 + _A_INF_33)


def _expected_peak(xi_0: float, n: int) -> float:
    """Hyperbolic envelope ``xi_n = xi_0 / (1 + n * xi_0 * delta)``."""
    return float(xi_0 / (1.0 + n * xi_0 * _DELTA))


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _build_hdb():
    A_inf_diag = [_M_OTHER, _M_OTHER, _A_INF_33, _I_OTHER, _I_OTHER, _I_OTHER]
    # A(omega) = A_inf across the grid (frequency-flat added mass).
    A_diag_per_omega = [list(A_inf_diag) for _ in range(_OMEGA_GRID.size)]
    # Tiny well-behaved B per DOF: ω⁻⁴ roll-off above the cutoff so the
    # M6 PR3 Refinement-2 input gates pass.
    rolloff = well_behaved_b(_OMEGA_GRID, band_value=_B_TINY, cutoff_omega=_CUTOFF_OMEGA)
    B_diag_per_omega = [[float(r)] * 6 for r in rolloff]
    C_diag = [0.0, 0.0, _C_33, 0.0, 0.0, 0.0]  # heave-only restoring
    return make_diagonal_hdb(
        A_inf_diag=A_inf_diag,
        C_diag=C_diag,
        A_diag_per_omega=A_diag_per_omega,
        B_diag_per_omega=B_diag_per_omega,
        omega=_OMEGA_GRID.tolist(),
        heading_deg=_HEADING.tolist(),
    )


def _rigid_body_mass_matrix() -> np.ndarray:
    return np.diag([_M_OTHER, _M_OTHER, _M_33, _I_OTHER, _I_OTHER, _I_OTHER]).astype(np.float64)


def _calm_fluid(_point: np.ndarray, _t: float) -> np.ndarray:
    """Calm sea: zero fluid velocity at every point and time."""
    return np.zeros(3, dtype=np.float64)


def _build_state_force():
    # Single horizontal member along body-frame X, centred at the
    # reference point. Length L, diameter D.
    elements = [
        MorisonElement(
            body_index=0,
            node_a_body=np.array([-0.5 * _L, 0.0, 0.0]),
            node_b_body=np.array([+0.5 * _L, 0.0, 0.0]),
            diameter=_D,
            Cd=_CD,
            include_inertia=False,
        )
    ]
    return make_morison_state_force(
        elements,
        n_dof=6,
        fluid_velocity_fn=_calm_fluid,
        rho=_RHO,
    )


# ---------------------------------------------------------------------------
# integration driver and peak extraction
# ---------------------------------------------------------------------------


def _run_decay():
    hdb = _build_hdb()
    lhs = assemble_cummins_lhs(rigid_body_mass=_rigid_body_mass_matrix(), hdb=hdb)
    dt = 0.01
    # Long kernel window because B is near-zero everywhere; the kernel
    # is essentially zero already, but we still need t_max comfortably
    # past the first few periods to silence the decay diagnostic.
    kernel = compute_retardation_kernel(hdb, t_max=120.0, dt=dt)

    xi0 = np.zeros(6)
    xi0[2] = _XI0_HEAVE
    res = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=6.0 * _T_N,  # ~ 6 periods -> 5 interior peaks
        rho_inf=1.0,  # trapezoidal: zero numerical dissipation
        state_force=_build_state_force(),
    )
    return res


def _positive_peaks(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_peaks, x_peaks)`` for positive interior local maxima of ``x``."""
    is_peak = (x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] > 0)
    idx = np.where(is_peak)[0] + 1
    return t[idx], x[idx]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_first_five_peaks_match_hyperbolic_envelope() -> None:
    """Primary M5 PR5 gate: peaks 1..5 match ``xi_n = xi_0 / (1 + n xi_0 delta)``."""
    res = _run_decay()
    _, peaks = _positive_peaks(res.t, res.xi[:, 2])
    assert peaks.size >= 5, f"need >= 5 positive peaks; got {peaks.size}"
    expected = np.array([_expected_peak(_XI0_HEAVE, n) for n in range(1, 6)])
    np.testing.assert_allclose(
        peaks[:5],
        expected,
        rtol=1.0e-4,
        err_msg=(
            f"first 5 peaks {peaks[:5]} deviate from hyperbolic "
            f"envelope {expected} by more than rtol=1e-4 "
            f"(delta={_DELTA:.4e} 1/m, xi_0={_XI0_HEAVE} m, "
            f"omega_n={_OMEGA_N:.4f} rad/s)."
        ),
    )


def test_other_dofs_stay_at_rest_under_heave_only_ic() -> None:
    """Heave-only IC + horizontal drag plate at the reference point ->
    no coupling into surge/sway/roll/pitch/yaw.

    The plate is centred at body origin (zero arm), and the drag force
    is purely vertical (``u_n = -v_z z_hat``). The arm-cross-force
    moment is therefore zero, and no other DOF is excited.
    """
    res = _run_decay()
    other = np.delete(res.xi, 2, axis=1)
    assert np.max(np.abs(other)) < 1.0e-10, (
        f"coupled motion detected on non-heave DOFs: max|xi_other| = "
        f"{np.max(np.abs(other)):.3e} (expected 0)."
    )


def test_envelope_distinguishes_hyperbolic_from_exponential() -> None:
    """The test must be sensitive to the *shape* of the decay envelope.

    Calibrate an exponential ``xi_n = xi_0 * exp(-n * zeta)`` to match
    the first observed peak (``zeta = ln(xi_0 / peak_1)``), then
    extrapolate to peak 5. The exponential extrapolation must
    over-predict the observed peak 5 by **at least an order of
    magnitude** more than the hyperbolic envelope's residual — otherwise
    this test reduces to a generic "decays smoothly" check rather than
    a hyperbolic-vs-exponential discriminator.
    """
    res = _run_decay()
    _, peaks = _positive_peaks(res.t, res.xi[:, 2])
    assert peaks.size >= 5

    # Calibrate exponential to peak[0] (n=1).
    zeta = float(np.log(_XI0_HEAVE / peaks[0]))
    xi_5_expo = _XI0_HEAVE * np.exp(-5.0 * zeta)
    xi_5_hyper = _expected_peak(_XI0_HEAVE, 5)
    xi_5_obs = float(peaks[4])

    expo_resid = abs(xi_5_expo - xi_5_obs) / xi_5_obs
    hyper_resid = abs(xi_5_hyper - xi_5_obs) / xi_5_obs

    assert expo_resid > 10.0 * hyper_resid, (
        f"exponential extrapolation matches as well as the hyperbolic "
        f"envelope (exp resid {expo_resid:.4e} vs hyperbolic "
        f"{hyper_resid:.4e}); the test does not discriminate the two "
        f"decay models. Increase xi_0 * delta to widen the envelope gap."
    )
