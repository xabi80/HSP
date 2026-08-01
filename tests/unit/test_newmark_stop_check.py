"""M11b PR8 STEP 1 -- opt-in early-stop (``stop_check``) for integrate_cummins.

The adaptive-settle feature must be STRICTLY ADDITIVE: with ``stop_check=None``
(or ``stop_check_interval=0``) the run is byte-identical to the fixed-duration
path; when the check fires, the result is the exact truncated prefix of the
full run (no re-solve, no drift)."""

from __future__ import annotations

import numpy as np

from floatsim.hydro.radiation import CumminsLHS
from floatsim.hydro.retardation import RetardationKernel
from floatsim.solver.newmark import integrate_cummins


def _system(nt: int = 400, dt: float = 0.01):  # type: ignore[no-untyped-def]
    """Trivial single-body 6-DOF (M = C = I) with a zero radiation kernel."""
    lhs = CumminsLHS(M_plus_Ainf=np.eye(6), C=np.eye(6))
    kernel = RetardationKernel(K=np.zeros((6, 6, nt)), t=np.arange(nt) * dt, dt=dt)
    return lhs, kernel


def _run(duration: float, dt: float, **kw):  # type: ignore[no-untyped-def]
    lhs, kernel = _system(dt=dt)
    xi0 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # unit heave IC -> damped decay
    return integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=duration,
        dt=dt,
        rho_inf=0.8,
        **kw,
    )


def test_stop_check_none_is_byte_identical() -> None:
    """The default (no stop_check) is unchanged: N+1 samples, same values as a
    plain run."""
    dt = 0.01
    full = _run(2.0, dt)
    again = _run(2.0, dt, stop_check=None, stop_check_interval=25)
    assert full.xi.shape[0] == round(2.0 / dt) + 1
    assert np.array_equal(full.xi, again.xi)
    assert np.array_equal(full.xi_dot, again.xi_dot)
    assert np.array_equal(full.xi_ddot, again.xi_ddot)


def test_stop_check_interval_zero_disables() -> None:
    """A stop_check that always returns True does nothing when the interval is
    0 (the guard is interval > 0)."""
    dt = 0.01
    res = _run(1.0, dt, stop_check=lambda t, x: True, stop_check_interval=0)
    assert res.xi.shape[0] == round(1.0 / dt) + 1


def test_stop_check_truncates_to_exact_prefix() -> None:
    """When the check fires, the result is the exact prefix of the full run --
    same steps, same values (no re-solve, no discontinuity)."""
    dt = 0.01
    full = _run(3.0, dt)

    def stop_at_len(t: np.ndarray, x: np.ndarray) -> bool:
        return t.shape[0] >= 120  # fire once >= 120 samples accumulated

    early = _run(3.0, dt, stop_check=stop_at_len, stop_check_interval=10)
    # fires at the first interval multiple with len >= 120 -> step 120 (t index 120)
    assert early.xi.shape[0] < full.xi.shape[0]
    n = early.xi.shape[0]
    assert np.array_equal(early.xi, full.xi[:n])
    assert np.array_equal(early.xi_dot, full.xi_dot[:n])
    assert np.array_equal(early.t, full.t[:n])


def test_stop_check_receives_history_so_far() -> None:
    """The callback sees exactly the accumulated (t, xi) prefix at call time."""
    dt = 0.01
    seen: list[int] = []

    def spy(t: np.ndarray, x: np.ndarray) -> bool:
        assert t.shape[0] == x.shape[0]
        seen.append(t.shape[0])
        return False

    _run(1.0, dt, stop_check=spy, stop_check_interval=50)
    # called at samples 51, 101 (every 50 steps, m = n+2)
    assert seen == [51, 101]
