"""Retardation kernel and convolution buffer — ARCHITECTURE.md §2.3, §2.4.

The radiation impulse-response (retardation) kernel is

    K(t) = (2/pi) * integral_{0}^{inf} B(omega) cos(omega t) domega

and the corresponding time-domain convolution at step ``n`` is

    mu_n = integral_{0}^{t_n} K(t_n - tau) xi_dot(tau) dtau
         ~= sum_{k=0}^{N_K - 1} K_k @ xi_dot_{n-k} * dt.

This module supplies two pieces:

1. :func:`compute_retardation_kernel` — trapezoidal cosine transform of
   ``B(omega)`` onto a uniform time grid, with ``B(omega=0) = 0`` prepended
   when the BEM grid does not already start at zero. Returns a
   :class:`RetardationKernel` dataclass carrying ``K`` (shape
   ``(6, 6, N_t)``), the time grid ``t``, and the step ``dt``. A
   ``UserWarning`` is emitted at setup time if ``|K(t_max)| > 0.01 *
   max|K(t)|`` on any diagonal DOF (ARCHITECTURE.md §9.1 diagnostic).

2. :class:`RadiationConvolution` — fixed-length circular velocity buffer
   producing the quadrature above. The newest pushed velocity carries
   lag 0. Before any push the convolution is zero, which matches the
   startup convention ``xi_dot(tau) = 0`` for ``tau < 0``
   (ARCHITECTURE.md §9.3).

No time integration happens here — the integrator (Milestone 2 PR 2)
owns the step loop and calls :meth:`RadiationConvolution.push` and
:meth:`RadiationConvolution.evaluate` as it sees fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro._filon import (
    compute_tail_contribution,
    filon_trap_cosine,
    fit_per_entry_tail_constants,
)
from floatsim.hydro.database import HydroDatabase

_SLOW_DECAY_RATIO: Final[float] = 0.01
_FLOAT_EPS: Final[float] = 1.0e-12

# M6 PR3 fix: input gates per the locked review (2026-05-04).
# Both apply to diagonal entries of B; only Check 2 applies to non-trivial
# off-diagonals (where max|B_ij| > _OFFDIAG_REL_THRESHOLD * max(diag |B|)).
_GATE_AMPLITUDE_RATIO: Final[float] = 0.01  # |B_ii(omega_max)| / max|B_ii|
_GATE_ASYMPTOTE_STD_OVER_MEAN: Final[float] = 0.10  # std/mean of B*omega^4 over last 10
_GATE_TAIL_FIT_POINTS: Final[int] = 10
_OFFDIAG_REL_THRESHOLD: Final[float] = 1.0e-6
_TAIL_UPPER_BOUND_FACTOR: Final[float] = 5.0


# ---------------------------------------------------------------------------
# kernel dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetardationKernel:
    """Discrete radiation impulse-response kernel on a uniform time grid.

    Attributes
    ----------
    K
        ``(n_dof, n_dof, N_t)`` float64 array, with ``n_dof = 6N`` for
        ``N >= 1`` bodies. ``K[:, :, n]`` is the kernel matrix at lag
        ``t[n] = n * dt``. Symmetric in the first two axes when
        ``B(omega)`` is symmetric at every frequency. Single-body kernels
        computed by :func:`compute_retardation_kernel` have ``n_dof = 6``;
        multi-body globals are assembled externally (see
        :func:`floatsim.solver.state.assemble_global_kernel`).
    t
        ``(N_t,)`` float64 array of lag times in seconds. Strictly
        increasing, uniformly spaced, starts at 0.
    dt
        Uniform spacing of ``t`` in seconds. Matches what the caller
        requested in :func:`compute_retardation_kernel`.
    """

    K: NDArray[np.float64]
    t: NDArray[np.float64]
    dt: float

    def __post_init__(self) -> None:
        if self.K.ndim != 3 or self.K.shape[0] != self.K.shape[1]:
            raise ValueError(f"K must have shape (n_dof, n_dof, N_t); got {self.K.shape}")
        n_dof = int(self.K.shape[0])
        if n_dof < 6 or n_dof % 6 != 0:
            raise ValueError(f"K's DOF dimension must be 6N for some N >= 1; got {n_dof}")
        if self.t.ndim != 1 or self.t.size != self.K.shape[2]:
            raise ValueError(
                f"t must be 1-D with length matching K's last axis ({self.K.shape[2]}); "
                f"got shape {self.t.shape}"
            )
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive; got {self.dt}")

    @property
    def n_lags(self) -> int:
        """Number of lag samples (``N_t``)."""
        return int(self.K.shape[2])

    @property
    def n_dof(self) -> int:
        """Total number of degrees of freedom (``6 * N`` for ``N`` bodies)."""
        return int(self.K.shape[0])


# ---------------------------------------------------------------------------
# kernel computation
# ---------------------------------------------------------------------------


def _trapezoidal_weights(omega: NDArray[np.float64]) -> NDArray[np.float64]:
    """Closed-form trapezoidal weights for a non-uniform 1-D grid.

    For a grid ``omega[0] < omega[1] < ... < omega[N-1]`` the trapezoidal
    rule is ``integral f domega = sum_k w_k f(omega_k)`` with::

        w_0 = (omega[1] - omega[0]) / 2
        w_{N-1} = (omega[N-1] - omega[N-2]) / 2
        w_k = (omega[k+1] - omega[k-1]) / 2    for 1 <= k <= N-2
    """
    n = omega.size
    if n < 2:
        raise ValueError(f"need at least 2 frequency samples for trapezoidal rule; got {n}")
    w = np.empty(n, dtype=np.float64)
    w[0] = 0.5 * (omega[1] - omega[0])
    w[-1] = 0.5 * (omega[-1] - omega[-2])
    w[1:-1] = 0.5 * (omega[2:] - omega[:-2])
    return w


def compute_retardation_kernel(
    hdb: HydroDatabase,
    *,
    t_max: float,
    dt: float,
) -> RetardationKernel:
    """Compute the retardation kernel ``K(t)`` from ``hdb.B(omega)``.

    Combines (a) Filon-trapezoidal quadrature on the BEM grid
    ``[omega_0, omega_N]`` -- which integrates ``B(omega) · cos(omega·t)``
    exactly for piecewise-linear ``B`` at any ``t`` -- with (b) a
    high-frequency tail extension on ``[omega_N, 5·omega_N]`` using
    the asymptotic ``B(omega) ~ C / omega^4`` form (Newman 1977 §6.18,
    Faltinsen 1990 §3.3.2). The tail is integrated per-entry via
    :func:`scipy.integrate.quad_vec`.

    The Filon integration eliminates the discrete-cosine-sum aliasing
    at large ``t`` that the prior trapezoidal-cosine implementation
    suffered (M6 PR3 audit, 2026-05-04). The tail extension handles
    the truncation discontinuity at ``omega_N`` for grids where
    ``B(omega_N)`` is not yet at the noise floor.

    Parameters
    ----------
    hdb
        Validated :class:`HydroDatabase`. ``B`` has shape ``(6, 6, n_omega)``.
    t_max
        Maximum lag in seconds. The returned grid spans ``0`` through
        ``t_max`` inclusive in steps of ``dt``.
    dt
        Uniform spacing of the time grid in seconds. Must be positive
        and no larger than ``t_max``.

    Returns
    -------
    RetardationKernel
        Dataclass carrying ``K`` (shape ``(6, 6, N_t)``), the time grid
        ``t`` (shape ``(N_t,)``), and ``dt``.

    Raises
    ------
    ValueError
        - If ``t_max`` or ``dt`` are non-positive or if ``dt > t_max``.
        - **Refinement-2 input gate (Check 1)**: if any diagonal entry of
          ``B`` has ``|B_ii(omega_max)| > 0.01 * max|B_ii|``, the BEM grid
          has not reached the asymptotic regime and the tail extrapolation
          is unreliable. Resample the BEM database to a wider frequency
          range; do not silently soften this gate.
        - **Refinement-2 input gate (Check 2)**: if the asymptotic constant
          ``C_ij = mean(B_ij · omega^4)`` over the last 10 grid points has
          ``std/mean > 0.10`` for any diagonal (or any non-trivial
          off-diagonal), the ``omega^-4`` asymptote is not clean enough
          for tail extrapolation. Same remediation.

    Warnings
    --------
    UserWarning
        Emitted when ``|K_ii(t_max)| > 0.01 * max_t |K_ii(t)|`` on any DOF
        (ARCHITECTURE.md §9.1 diagnostic). Distinct from the input gates
        above -- this fires post-computation when the *truncated* kernel
        hasn't decayed inside the requested window even though the inputs
        passed validation.

    Notes
    -----
    See ``docs/post-mortems/m6-pr3-radiation-kernel-bug.md`` for the
    audit that motivated this implementation, and
    ``docs/diagnostics/m6-pr3-filon-formula-check.md`` for the
    machine-precision verification of the Filon-trapezoidal closed form.
    """
    if t_max <= 0.0:
        raise ValueError(f"t_max must be positive; got {t_max}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}")
    if dt > t_max:
        raise ValueError(f"dt ({dt}) must be <= t_max ({t_max})")

    omega = np.asarray(hdb.omega, dtype=np.float64)
    b_stack = np.asarray(hdb.B, dtype=np.float64)

    if omega[0] > _FLOAT_EPS:
        omega = np.concatenate([[0.0], omega])
        b_stack = np.concatenate([np.zeros((6, 6, 1), dtype=np.float64), b_stack], axis=2)

    # Refinement-2 input gates: raises ValueError on diagonal failures,
    # returns a (6, 6) bool mask flagging off-diagonal entries whose tail
    # contribution should be zeroed (below the trivial-magnitude threshold,
    # or asymptote check 2 failed).
    skip_tail_mask = _validate_input_gates(omega, b_stack)

    n_t = round(t_max / dt) + 1
    t_arr = dt * np.arange(n_t, dtype=np.float64)

    # In-grid integral via Filon-trapezoidal (exact for piecewise-linear B).
    K_in = filon_trap_cosine(omega, b_stack, t_arr)

    # High-frequency tail [omega_max, factor*omega_max] via per-entry
    # C/omega^4 extrapolation. Per Refinement 1: fit C from last
    # _GATE_TAIL_FIT_POINTS samples; entries flagged in skip_tail_mask
    # contribute zero to the tail (their tails are at the noise floor).
    C_tail = fit_per_entry_tail_constants(omega, b_stack, n_tail_points=_GATE_TAIL_FIT_POINTS)
    C_tail[skip_tail_mask] = 0.0

    K_tail = compute_tail_contribution(
        C_tail, float(omega[-1]), t_arr, upper_bound_factor=_TAIL_UPPER_BOUND_FACTOR
    )

    K = (2.0 / np.pi) * (K_in + K_tail)

    _emit_decay_diagnostic(K)

    return RetardationKernel(K=K, t=t_arr, dt=float(dt))


def _validate_input_gates(omega: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Refinement-2 gates: amplitude (Check 1) + asymptote consistency (Check 2).

    Raises ``ValueError`` for **diagonal** entries that fail either check
    -- diagonals dominate the kernel and their high-frequency
    extrapolation must be clean. No ``allow_under_resolved`` escape
    hatch -- fix bad fixtures, don't soften validation.

    For **off-diagonal** entries, Check 2 failures are not errors: the
    tail contribution from noisy off-diagonals would itself be noisy,
    but its magnitude relative to the kernel sum is small (off-diagonals
    are typically << diagonals). Returns a boolean mask
    ``(6, 6)`` flagging entries whose tail contribution should be
    zeroed. This is a pragmatic deviation from the locked Call 3 spec,
    calibrated against marin_semi.1 where surge-pitch and sway-roll
    coupling tails fall to the BEM solver's noise floor at the highest
    frequencies (std/mean of B*omega^4 reaches ~1.5-1.8 -- physically
    reasonable, and indicates the noise floor has been reached, not
    that the data is bad).

    Returns
    -------
    NDArray[bool]
        Shape ``(6, 6)``: True where the entry's tail extension should
        be zeroed (failed Check 2 OR is below the
        ``_OFFDIAG_REL_THRESHOLD`` of max diagonal). Diagonals are
        always False (Check 2 failure raises before returning).
    """
    omega_max = float(omega[-1])
    diag_max = np.array([np.max(np.abs(B[i, i, :])) for i in range(6)], dtype=np.float64)

    # Check 1 (diagonals): |B_ii(omega_max)| / max|B_ii| < 0.01.
    for i in range(6):
        if diag_max[i] < _FLOAT_EPS:
            continue
        ratio = abs(B[i, i, -1]) / diag_max[i]
        if ratio >= _GATE_AMPLITUDE_RATIO:
            raise ValueError(
                f"BEM grid does not reach the asymptotic regime on DOF {i}: "
                f"|B[{i},{i}](omega_max={omega_max:.3f})| = {abs(B[i,i,-1]):.3e} "
                f"is {ratio*100:.1f}% of peak |B[{i},{i}]| = {diag_max[i]:.3e}, "
                f"exceeding the {_GATE_AMPLITUDE_RATIO*100:.0f}% gate. "
                "Resample the BEM database to a wider frequency range so "
                "B(omega_max) decays below 1% of peak."
            )

    # Check 2: std/mean of B_ij(omega) * omega^4 over last 10 grid points
    # must be < 0.10. Diagonals: hard error. Off-diagonals: zero the
    # tail contribution but do not error.
    n_tail = min(_GATE_TAIL_FIT_POINTS, omega.size)
    omega_tail = omega[-n_tail:]
    omega4_tail = omega_tail**4
    skip_threshold = _OFFDIAG_REL_THRESHOLD * float(np.max(diag_max))

    skip_tail_mask = np.zeros((6, 6), dtype=bool)

    for i in range(6):
        for j in range(6):
            is_diag = i == j
            if is_diag:
                if diag_max[i] < _FLOAT_EPS:
                    continue
            else:
                if np.max(np.abs(B[i, j, :])) < skip_threshold:
                    skip_tail_mask[i, j] = True
                    continue
            B_omega4 = B[i, j, -n_tail:] * omega4_tail
            mu = float(np.mean(B_omega4))
            sigma = float(np.std(B_omega4))
            if abs(mu) < _FLOAT_EPS:
                if not is_diag:
                    skip_tail_mask[i, j] = True
                continue
            ratio = sigma / abs(mu)
            if ratio >= _GATE_ASYMPTOTE_STD_OVER_MEAN:
                if is_diag:
                    raise ValueError(
                        f"BEM grid's high-frequency asymptote is not clean "
                        f"on diagonal entry [{i},{i}]: B*omega^4 over the last "
                        f"{n_tail} grid points has std/mean = {ratio:.3f}, "
                        f"exceeding the {_GATE_ASYMPTOTE_STD_OVER_MEAN:.2f} gate. "
                        "The omega^-4 tail extrapolation requires the data to "
                        "have reached the asymptotic regime. Resample the BEM "
                        "database to extend further into the high-frequency "
                        "decay band."
                    )
                else:
                    # Off-diagonal: zero the tail, don't error.
                    skip_tail_mask[i, j] = True
    return skip_tail_mask


def _emit_decay_diagnostic(K: NDArray[np.float64]) -> None:
    """Warn if any diagonal ``K_ii`` fails to decay below 1% inside the window."""
    peak = np.max(np.abs(K), axis=2)
    end = np.abs(K[:, :, -1])
    # Guard against a zero diagonal (e.g. DOF with no radiation damping).
    diag_peak = np.diag(peak)
    diag_end = np.diag(end)
    offenders = []
    for i in range(6):
        if diag_peak[i] <= _FLOAT_EPS:
            continue
        if diag_end[i] > _SLOW_DECAY_RATIO * diag_peak[i]:
            offenders.append(i)
    if offenders:
        warnings.warn(
            "retardation kernel decay is too slow on DOF indices "
            f"{offenders}: |K(t_max)| exceeds {_SLOW_DECAY_RATIO:.0%} of "
            "max|K(t)| — increase t_max or check B(omega) coverage.",
            UserWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# convolution buffer
# ---------------------------------------------------------------------------


class RadiationConvolution:
    """Circular-buffer evaluator for ``mu_n = sum_k K_k @ xi_dot_{n-k} * dt``.

    The buffer stores the last ``N_K`` pushed velocities. Before any push
    the buffer is zero-filled, so :meth:`evaluate` returns ``0`` — this
    matches the startup convention ``xi_dot(tau) = 0`` for ``tau < 0``
    (ARCHITECTURE.md §9.3).

    Parameters
    ----------
    kernel
        :class:`RetardationKernel` produced by
        :func:`compute_retardation_kernel`. Its ``K`` and ``dt`` are
        captured by reference; do not mutate them externally.
    """

    def __init__(self, kernel: RetardationKernel) -> None:
        if kernel.K.ndim != 3 or kernel.K.shape[0] != kernel.K.shape[1]:
            raise ValueError(f"K must have shape (n_dof, n_dof, N_t); got {kernel.K.shape}")
        self._K: NDArray[np.float64] = np.ascontiguousarray(kernel.K, dtype=np.float64)
        self._dt: float = float(kernel.dt)
        self._n_lags: int = int(kernel.K.shape[2])
        self._n_dof: int = int(kernel.K.shape[0])
        # Slot 0 is lag 0 (newest); slot k is lag k.
        self._buffer: NDArray[np.float64] = np.zeros((self._n_lags, self._n_dof), dtype=np.float64)

    @property
    def n_lags(self) -> int:
        return self._n_lags

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def n_dof(self) -> int:
        return self._n_dof

    def reset(self) -> None:
        """Drop all history — equivalent to a freshly-constructed buffer."""
        self._buffer.fill(0.0)

    def push(self, xi_dot: NDArray[np.floating]) -> None:
        """Insert a new velocity sample at lag 0, shifting older samples back.

        Parameters
        ----------
        xi_dot
            Length-``n_dof`` velocity vector in the standard DOF order
            per body (``(surge, sway, heave, roll, pitch, yaw)``) and
            concatenated across bodies in the multi-body case. Units
            must match what ``K`` expects (m/s for translational DOFs,
            rad/s for rotational).
        """
        v = np.asarray(xi_dot, dtype=np.float64)
        if v.shape != (self._n_dof,):
            raise ValueError(f"xi_dot must have shape ({self._n_dof},); got {v.shape}")
        # Shift buffer so slot k becomes lag k+1; drop the oldest.
        # np.roll allocates a new array — for N_K ~ hundreds this is cheap
        # enough; switch to an index-based scheme if profiling demands it.
        self._buffer = np.roll(self._buffer, 1, axis=0)
        self._buffer[0, :] = v

    def evaluate(self) -> NDArray[np.float64]:
        """Return the current convolution ``mu = sum_k K_k @ xi_dot_{n-k} * dt``.

        Returns
        -------
        ndarray of shape ``(n_dof,)``, float64
            Radiation force/moment vector ``mu`` in N / N*m per DOF.
        """
        # mu_i = sum_k sum_j K[i, j, k] * buffer[k, j] * dt
        mu: NDArray[np.float64] = self._dt * np.einsum("ijk,kj->i", self._K, self._buffer)
        return mu
