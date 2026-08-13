"""The exported ``mu`` channel must satisfy the steady-state radiation identity.

``mu`` is the newest and most error-prone channel in the FloatFEA `.flr` schema:
it is a loop local in ``newmark.py`` that appears in no return value, and
:func:`floatsim.io.flr_export.recompute_mu` recovers it by replaying the
convolution offline. That replay has to be verified, not assumed.

The identity
------------
In steady periodic motion at ``omega`` the total radiation force is
``A(omega) xi_ddot + B(omega) xi_dot``, while the Cummins split is
``A_inf xi_ddot + mu(t)``. Equating::

    mu  =  [A(omega) - A_inf] xi_ddot  +  B(omega) xi_dot

Derivation for a harmonic ``xi_dot = V cos(wt)``, with
``K`` the retardation kernel and the Ogilvie relations
``B(w) = int K(tau) cos(w tau) dtau`` and
``w [A_inf - A(w)] = int K(tau) sin(w tau) dtau``::

    mu(t) = int K(tau) V cos(w(t-tau)) dtau
          = V [ cos(wt) int K cos(w tau) + sin(wt) int K sin(w tau) ]
          = V B(w) cos(wt)  +  V w [A_inf - A(w)] sin(wt)
          = B(w) xi_dot     +  [A(w) - A_inf] xi_ddot        since xi_ddot = -V w sin(wt)

so the identity is exact, not approximate, for harmonic motion.

This test is **independent of FloatFEA gate G1.6**: it verifies the export and
the convolution implementation, where G1.6 verifies the panel reconstruction. A
fault in either would otherwise be attributed to the other.

Corollary worth noting where it is used: because ``K(t)`` and ``B(omega)`` are a
Fourier pair and convolution is linear, harmonic motion produces radiation at
``omega`` and nowhere else. **Fundamental-only reconstruction of radiation is
exact in steady periodic motion** -- radiation is the cleanest of the load
sources, not the riskiest.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.retardation import RetardationKernel
from floatsim.io.flr_export import integrator_block, recompute_mu

_DT = 0.01
_T_MAX = 12.0          # kernel support; must decay well inside this
_DECAY = 0.8
_KERNEL_OMEGA = 1.3


def _kernel(n_dof: int = 6) -> RetardationKernel:
    """A decaying-oscillatory kernel, 6-DOF diagonal.

    RetardationKernel requires a 6N DOF dimension, so the fixture is 6-DOF and
    only DOF 0 is driven; a diagonal kernel keeps DOF 0's mu a function of DOF
    0's velocity alone. The kernel decays to ~7e-5 inside its support, so
    truncation does not pollute the identity.
    """
    t = np.arange(0.0, _T_MAX, _DT)
    k = np.exp(-_DECAY * t) * np.cos(_KERNEL_OMEGA * t)
    K = np.zeros((n_dof, n_dof, t.size), dtype=np.float64)
    for i in range(n_dof):
        K[i, i, :] = k
    return RetardationKernel(K=K, t=t, dt=_DT)


def _ogilvie(kernel: RetardationKernel, omega: float, *, rule: str) -> tuple[float, float]:
    """``(B(omega), S(omega))`` by quadrature over the same kernel samples.

    ``S = int K sin(w tau) dtau = w [A_inf - A(w)]``, so
    ``A(w) - A_inf = -S / w``. Computed from the kernel itself, so the test does
    not depend on a second data source that could disagree with it.

    ``rule`` selects the quadrature deliberately, because the choice decides
    *what is being tested*:

    ``"rectangle"``
        Matches ``RadiationConvolution.evaluate``, which computes
        ``sum_k K_k @ v_{n-k} * dt`` (retardation.py:811-812) -- a left-endpoint
        sum. Comparing like with like isolates **the identity and the replay**
        from the quadrature.
    ``"trapezoid"``
        The continuous-limit reference. Comparing against it measures identity
        *plus* quadrature error, which is O(dt) and is pinned separately by
        :func:`test_convolution_quadrature_is_first_order`.
    """
    tau = kernel.t
    k = kernel.K[0, 0, :]
    if rule == "rectangle":
        return float(np.sum(k * np.cos(omega * tau)) * kernel.dt), float(
            np.sum(k * np.sin(omega * tau)) * kernel.dt
        )
    if rule == "trapezoid":
        return float(np.trapezoid(k * np.cos(omega * tau), tau)), float(
            np.trapezoid(k * np.sin(omega * tau), tau)
        )
    raise ValueError(f"unknown quadrature rule {rule!r}")


@pytest.mark.parametrize("omega", [0.7, 1.3, 2.1])
def test_mu_satisfies_the_steady_state_radiation_identity(omega: float) -> None:
    """``mu`` from the replay equals ``[A(w)-A_inf] xi_ddot + B(w) xi_dot``."""
    kernel = _kernel()
    b, s = _ogilvie(kernel, omega, rule="rectangle")

    # Drive long enough that the convolution buffer is full before comparing.
    t = np.arange(0.0, 60.0, _DT)
    amp = 0.37
    xi_dot = np.zeros((t.size, 6), dtype=np.float64)
    xi_ddot = np.zeros((t.size, 6), dtype=np.float64)
    xi_dot[:, 0] = amp * np.cos(omega * t)
    xi_ddot[:, 0] = -amp * omega * np.sin(omega * t)

    mu, _ = recompute_mu(kernel, xi_dot, from_run_start=True)

    a_minus_ainf = -s / omega
    predicted = a_minus_ainf * xi_ddot + b * xi_dot

    # Compare only where the buffer is fully loaded: before t = T_MAX the
    # convolution is still seeing the zero-padded startup, which is the
    # integrator's documented behaviour and not part of the identity.
    settled = t > _T_MAX + 1.0
    err = np.abs(mu[settled, 0] - predicted[settled, 0]).max()
    scale = np.abs(predicted[settled, 0]).max()
    assert scale > 1e-6, "degenerate test: predicted amplitude is ~0"
    # Matched quadrature, so what remains is float accumulation over 1200 lags
    # and 6000 steps -- not a modelling discrepancy.
    assert err / scale < 1.0e-10, (
        f"mu identity violated at omega={omega}: "
        f"max |mu - predicted| = {err:.3e}, relative {err / scale:.3e}"
    )


def test_convolution_quadrature_is_first_order() -> None:
    """Against the CONTINUOUS reference the error is O(dt). Pin the rate.

    RadiationConvolution uses a left-endpoint sum, so it approaches the
    continuous Ogilvie relation at first order. Measured ratios are 2.00 at
    every halving. Asserting the *rate* rather than a magnitude is what would
    catch a regression to a zeroth-order rule, whose absolute error at any one
    dt could still look acceptable -- the same reasoning as FloatFEA gate G4.5.
    """
    omega, amp = 1.3, 0.37
    errors = []
    for dt in (0.02, 0.01, 0.005):
        tau = np.arange(0.0, _T_MAX, dt)
        k = np.exp(-_DECAY * tau) * np.cos(_KERNEL_OMEGA * tau)
        K = np.zeros((6, 6, tau.size), dtype=np.float64)
        for i in range(6):
            K[i, i, :] = k
        kernel = RetardationKernel(K=K, t=tau, dt=dt)
        b, s = _ogilvie(kernel, omega, rule="trapezoid")

        t = np.arange(0.0, 60.0, dt)
        xi_dot = np.zeros((t.size, 6), dtype=np.float64)
        xi_ddot = np.zeros((t.size, 6), dtype=np.float64)
        xi_dot[:, 0] = amp * np.cos(omega * t)
        xi_ddot[:, 0] = -amp * omega * np.sin(omega * t)
        mu, _ = recompute_mu(kernel, xi_dot, from_run_start=True)
        predicted = (-s / omega) * xi_ddot + b * xi_dot
        settled = t > _T_MAX + 1.0
        errors.append(
            float(
                np.abs(mu[settled, 0] - predicted[settled, 0]).max()
                / np.abs(predicted[settled, 0]).max()
            )
        )

    for coarse, fine in zip(errors[:-1], errors[1:], strict=True):
        ratio = coarse / fine
        assert 1.8 < ratio < 2.2, f"expected first-order convergence, got ratio {ratio:.3f}"


def test_mu_startup_is_zero_matching_the_integrator() -> None:
    """``mu[0]`` is zero by the startup convention, not the buffer artifact.

    ARCHITECTURE.md §9.3 gives ``mu(0) = 0``; the integrator skips the
    O(dt) buffer-evaluated value at the first RHS (``newmark.py:388-391``). The
    replay must match that, or every exported record is off by one step at the
    start.
    """
    kernel = _kernel()
    xi_dot = np.ones((50, 6), dtype=np.float64)
    mu, valid_from = recompute_mu(kernel, xi_dot, from_run_start=True)
    assert mu[0, 0] == 0.0
    assert valid_from == 0


def test_recompute_mu_rejects_a_dof_mismatch() -> None:
    """A velocity history from a different run must not be silently accepted."""
    kernel = _kernel()
    with pytest.raises(ValueError, match="same run"):
        recompute_mu(kernel, np.zeros((10, 3)), from_run_start=True)


def test_integrator_block_carries_alpha_m_as_well_as_alpha_f() -> None:
    """Both blend parameters are exported, and they differ.

    Generalized-alpha blends inertia with ``alpha_m`` and force/stiffness with
    ``alpha_f``. A consumer given only ``alpha_f`` would form the right force
    blend against the wrong acceleration blend, and the residual would look
    exactly like an FE mapping error.
    """
    block = integrator_block(rho_inf=0.9, dt=0.01)
    for key in ("alpha_m", "alpha_f", "beta", "gamma", "rho_inf", "dt", "mu_treatment"):
        assert key in block, f"integrator block missing {key!r}"
    assert block["alpha_m"] != block["alpha_f"]
    assert block["mu_treatment"] == "lagged_unblended"
    # Chung & Hulbert 1993 at rho_inf = 0.9.
    assert block["alpha_m"] == pytest.approx(0.8 / 1.9)
    assert block["alpha_f"] == pytest.approx(0.9 / 1.9)


def test_truncated_window_marks_the_whole_mu_history_invalid() -> None:
    """A window without pre-history is INVALID for a kernel-memory length.

    ``mu[0] = 0`` is right at a true run start and wrong at the start of a
    window with prior history, and it stays wrong until the buffer refills. For
    the 12-buoy platform this bites hard: a 60 s kernel is 6000 lags while
    ``run_case`` returns ~1955 samples, so the entire window is invalid.
    """
    kernel = _kernel()
    xi_dot = np.ones((50, 6), dtype=np.float64)
    _, valid_from = recompute_mu(kernel, xi_dot, from_run_start=False)
    assert valid_from == kernel.K.shape[2]
    assert valid_from > 50, "this fixture should demonstrate a wholly-invalid window"


def test_from_run_start_has_no_default() -> None:
    """It must be stated, because a wrong default yields plausible numbers."""
    import inspect

    sig = inspect.signature(recompute_mu)
    param = sig.parameters["from_run_start"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
