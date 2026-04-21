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

from floatsim.hydro.database import HydroDatabase

_SLOW_DECAY_RATIO: Final[float] = 0.01
_FLOAT_EPS: Final[float] = 1.0e-12


# ---------------------------------------------------------------------------
# kernel dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetardationKernel:
    """Discrete radiation impulse-response kernel on a uniform time grid.

    Attributes
    ----------
    K
        ``(6, 6, N_t)`` float64 array. ``K[:, :, n]`` is the 6x6 kernel
        matrix at lag ``t[n] = n * dt``. Symmetric in the first two axes
        when ``B(omega)`` is symmetric at every frequency.
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
        if self.K.ndim != 3 or self.K.shape[0] != 6 or self.K.shape[1] != 6:
            raise ValueError(f"K must have shape (6, 6, N_t); got {self.K.shape}")
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
        If ``t_max`` or ``dt`` are non-positive or if ``dt > t_max``.

    Warnings
    --------
    UserWarning
        Emitted when ``|K_ii(t_max)| > 0.01 * max_t |K_ii(t)|`` on any DOF,
        indicating the kernel has not decayed inside the truncation window
        (ARCHITECTURE.md §9.1 diagnostic).

    Notes
    -----
    Implementation uses trapezoidal quadrature of ``(2/pi) B(omega) cos(omega t)``
    over the BEM frequency grid. When ``hdb.omega[0] > 0`` a virtual
    ``B(omega=0) = 0`` sample is prepended, which is exact in linear
    potential flow (radiation damping vanishes at ``omega = 0``).
    High-frequency truncation error is left to the caller — standard BEM
    grids span the dominant radiation band and ``B`` decays rapidly
    beyond it.
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

    weights = _trapezoidal_weights(omega)
    n_t = round(t_max / dt) + 1
    t_arr = dt * np.arange(n_t, dtype=np.float64)

    # (n_omega, n_t) cosine matrix; avoid building a full 4-D tensor.
    cos_mat = np.cos(np.outer(omega, t_arr))
    weighted_cos = weights[:, None] * cos_mat
    # K[i, j, n] = (2/pi) * sum_k B[i, j, k] * weights[k] * cos(omega[k] * t[n])
    K = (2.0 / np.pi) * np.einsum("ijk,kn->ijn", b_stack, weighted_cos)

    _emit_decay_diagnostic(K)

    return RetardationKernel(K=K, t=t_arr, dt=float(dt))


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
        if kernel.K.ndim != 3 or kernel.K.shape[:2] != (6, 6):
            raise ValueError(f"K must have shape (6, 6, N_t); got {kernel.K.shape}")
        self._K: NDArray[np.float64] = np.ascontiguousarray(kernel.K, dtype=np.float64)
        self._dt: float = float(kernel.dt)
        self._n_lags: int = int(kernel.K.shape[2])
        # Slot 0 is lag 0 (newest); slot k is lag k.
        self._buffer: NDArray[np.float64] = np.zeros((self._n_lags, 6), dtype=np.float64)

    @property
    def n_lags(self) -> int:
        return self._n_lags

    @property
    def dt(self) -> float:
        return self._dt

    def reset(self) -> None:
        """Drop all history — equivalent to a freshly-constructed buffer."""
        self._buffer.fill(0.0)

    def push(self, xi_dot: NDArray[np.floating]) -> None:
        """Insert a new velocity sample at lag 0, shifting older samples back.

        Parameters
        ----------
        xi_dot
            Length-6 velocity vector ``(xi_dot_0, ..., xi_dot_5)`` in the
            standard DOF order. Units must match what ``K`` expects (m/s
            for translational DOFs, rad/s for rotational).
        """
        v = np.asarray(xi_dot, dtype=np.float64)
        if v.shape != (6,):
            raise ValueError(f"xi_dot must have shape (6,); got {v.shape}")
        # Shift buffer so slot k becomes lag k+1; drop the oldest.
        # np.roll allocates a new array — for N_K ~ hundreds this is cheap
        # enough; switch to an index-based scheme if profiling demands it.
        self._buffer = np.roll(self._buffer, 1, axis=0)
        self._buffer[0, :] = v

    def evaluate(self) -> NDArray[np.float64]:
        """Return the current convolution ``mu = sum_k K_k @ xi_dot_{n-k} * dt``.

        Returns
        -------
        ndarray of shape (6,), float64
            Radiation force/moment vector ``mu`` in N / N*m per DOF.
        """
        # mu_i = sum_k sum_j K[i, j, k] * buffer[k, j] * dt
        mu: NDArray[np.float64] = self._dt * np.einsum("ijk,kj->i", self._K, self._buffer)
        return mu
