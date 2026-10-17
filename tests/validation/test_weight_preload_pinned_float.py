"""Weight preload, Gate A dynamics (docs/weight-preload-plan.md §3).

The pinned float's free pitch decay about its pin against the closed form
``T = 2 pi sqrt((I_yy + m L^2) / (C55 - P L))`` (tests/support/pinned_float.py). The closed form
was derived before any code and is never calibrated on FloatSim. Tolerance: rtol 1e-3, the
analytic tolerance of CLAUDE.md §5. The integrator's own period error at dt = 0.01 is
~(omega dt)^2 / 12 < 2e-5, and the synthetic radiation damping shifts the period by
~zeta^2 / 2 < 1e-4.

The preload-off run documents the gap the preload closes: it has no -P L term, so it gives the
P = 0 period whatever the body's displaced volume.

The closed form is the small-amplitude limit. At amplitude theta0 the body also rises against
C33, which stiffens the swing by Duffing's ``3 eps theta0^2 / (2 k)`` (pinned_float docstring):
measured 3.53-3.63 theta0^2 against the derived 3.58 at P = +30 N, scaling exactly as theta0^2.
So the closed-form runs start at theta0 = 0.005 (shift < 1e-4), and a fourth test checks the
finite-amplitude shift itself at theta0 = 0.02, at 5 % on the leading-order coefficient.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.solver.newmark import integrate_cummins
from tests.support import pinned_float as pf

pytestmark = pytest.mark.slow

THETA0 = 0.005  # rad: initial pitch about the pin (Duffing shift < 1e-4)


def _pitch_period(
    net_up: float | None, *, preload: bool, duration: float, theta0: float = THETA0
) -> float:
    s = pf.setup(net_up, preload=preload)
    xi0 = s.xi0.copy()
    xi0[4] += theta0
    xi0[0] += -pf.L_PIN * theta0  # rotation about the pin: the CoG swings by -L theta
    r = integrate_cummins(
        lhs=s.lhs,
        kernel=s.kernel,
        xi0=xi0,
        xi_dot0=s.xi_dot0,
        duration=duration,
        dt=pf.DT,
        rho_inf=0.8,
        constraints=s.constraints,
        state_force=s.state_force,
        projection_interval=1,
    )
    x = r.xi[:, 4] - s.xi0[4]
    up = np.where((x[:-1] < 0.0) & (x[1:] >= 0.0))[0]
    t_up = r.t[up] - x[up] * (r.t[up + 1] - r.t[up]) / (x[up + 1] - x[up])
    assert t_up.size >= 3, "fewer than two full cycles"
    return float(np.mean(np.diff(t_up)))


@pytest.mark.parametrize("net_up", [30.0, -30.0])
def test_pitch_period_matches_closed_form(net_up: float) -> None:
    expected = pf.analytic_period(net_up)
    measured = _pitch_period(net_up, preload=True, duration=5.0 * expected)
    assert measured == pytest.approx(expected, rel=1e-3)


def test_preload_off_has_no_pin_term() -> None:
    expected = pf.analytic_period(0.0)
    measured = _pitch_period(30.0, preload=False, duration=5.0 * expected)
    assert measured == pytest.approx(expected, rel=1e-3)


def test_finite_amplitude_shift_matches_duffing() -> None:
    """The body rising against C33 at theta0 = 0.02 rad: the exact constraint geometry."""
    theta0 = 0.02
    expected = pf.analytic_period(30.0)
    measured = _pitch_period(30.0, preload=True, duration=5.0 * expected, theta0=theta0)
    assert measured / expected - 1.0 == pytest.approx(pf.duffing_shift(30.0, theta0), rel=0.05)
