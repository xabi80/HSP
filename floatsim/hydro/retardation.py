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

# Three-check gate structure (M6 PR4 Pre-3 / fix-wamit-dimensionalisation,
# locked 2026-05-07). Each check tests something different:
#
#   Check 1 -- input proxy: is the BEM grid wide enough?
#       Computes |B_ii(omega_max)| / max|B_ii|. SOFT WARNING at >5%.
#       Below 5%: "fine but worth noting"; above: BEM is genuinely
#       under-resolved -- but the 1/omega^4 tail extension may still
#       compensate. The post-extension Check 3 is what determines
#       whether the kernel is usable; Check 1 is an early heads-up.
#       (Pre-fix this was a hard error at 1%; per Option E disposition
#       it became advisory because the Pre-3 diagnostic showed the
#       tail extension cleanly recovered marin_semi's surge/sway/yaw
#       kernels even at ~1.7% B(omega_max)/peak.)
#
#   Check 2 -- asymptote consistency: is the tail fit well-defined?
#       std/mean of B*omega^4 over the last GATE_TAIL_FIT_POINTS
#       samples must be < 0.10 on diagonals (HARD ERROR; the per-entry
#       1/omega^4 fit is unreliable otherwise). Off-diagonal failures
#       fall back to zero-tail-contribution -- their kernel impact is
#       small relative to the diagonal-driven sum.
#
#   Check 3 -- post-extension kernel decay: does the resulting K(t)
#       decay? |K_ii(t_max)| / max|K_ii(t)| must be < 0.001 on each
#       diagonal (HARD ERROR). Threshold rationale: 0.1% is 20x the
#       measured marin_semi ratio (~0.005%); conservative for future
#       fixtures with potentially worse decay properties. This is the
#       check that actually matters dynamically -- a kernel that
#       doesn't decay produces sustained oscillation in the
#       convolution sum and corrupts the simulation.
#
# Calibration evidence:
#   docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png shows that
#   marin_semi's blocked-by-pre-fix-Check-1 DOFs (surge, sway, yaw at
#   ~1.7% B/peak) all decay cleanly post-extension to <6e-5 of peak
#   by t=200 s.

# Check 1 (advisory).
_GATE_AMPLITUDE_RATIO_WARN: Final[float] = 0.05  # 5% B(omega_max)/peak

# Check 2 (hard error).
_GATE_ASYMPTOTE_STD_OVER_MEAN: Final[float] = 0.10  # std/mean of B*omega^4 over last 10
_GATE_TAIL_FIT_POINTS: Final[int] = 10
_OFFDIAG_REL_THRESHOLD: Final[float] = 1.0e-6

# Check 3 (post-extension, hard error).
_GATE_KERNEL_DECAY_RATIO: Final[float] = 1.0e-3  # |K(t_max)| / max|K| < 0.1%

# Tail extension upper bound (independent of the gates).
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
    asymptote_check_override: str | None = None,
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
    asymptote_check_override
        Optional non-empty rationale string that bypasses Item 25's
        asymptote gate (M7.5 PR1; see conventions doc Item 25
        applicability sub-item at
        ``docs/openfast-cross-check-conventions.md``). When supplied,
        ``_validate_input_gates`` is skipped entirely (both the
        Check 1 advisory warning and the Check 2 hard error) and the
        1/omega^4 tail extension is zero-filled. Check 3
        (post-extension kernel decay) still runs -- the authoritative
        gate is not bypassable. Intended for wave-tank-scale bodies
        (characteristic length ~ 1-2 m) where the 1/omega^4
        asymptotic regime is not reached on the available BEM grid;
        the rationale string is the forcing-function acknowledgment
        that the caller has judged Item 25 inapplicable. Empty
        strings (``None``, ``""``, or whitespace-only via
        ``.strip() == ""``) raise ``ValueError``.

    Returns
    -------
    RetardationKernel
        Dataclass carrying ``K`` (shape ``(6, 6, N_t)``), the time grid
        ``t`` (shape ``(N_t,)``), and ``dt``.

    Raises
    ------
    ValueError
        - If ``t_max`` or ``dt`` are non-positive or if ``dt > t_max``.
        - If ``asymptote_check_override`` is supplied but is not a
          non-empty rationale string (M7.5 PR1).
        - **Check 2** (asymptote consistency; standard path only): if
          the asymptotic constant ``C_ij = mean(B_ij · omega^4)`` over
          the last 10 grid points has ``std/mean > 0.10`` for any
          diagonal entry (or any non-trivial off-diagonal), the
          ``omega^-4`` asymptote is not clean enough for tail
          extrapolation. Resample the BEM database with a wider
          frequency range, OR use ``asymptote_check_override`` if the
          small-body regime applies.
        - **Check 3** (post-extension kernel decay; both paths): if
          any diagonal ``|K_ii(t_max)| / max|K_ii(t)| > 0.001``, the
          kernel fails to decay -- sustained kernel oscillation would
          corrupt the convolution sum. Increase ``t_max``, OR widen
          the BEM grid if Check 1 also warned. Check 3 is not
          bypassable by ``asymptote_check_override``.

    Warnings
    --------
    UserWarning
        - **Check 1** (BEM amplitude proxy; standard path only): if
          any diagonal has ``|B_ii(omega_max)| / max|B_ii| > 0.05``,
          the BEM grid is under-resolved relative to a "comfortably
          decayed" target (~ 1 %). The 1/omega^4 tail extension
          typically compensates; Check 3 is the authoritative gate.
          The warning is an early heads-up that a future contributor
          may want to widen the BEM grid for tighter cross-check
          tolerances. Not emitted on the override path.
        - **Item 25 override** (M7.5 PR1): emitted once when
          ``asymptote_check_override`` is supplied, echoing the
          rationale string and noting that high-frequency response
          in overridden DOFs is not analytically bounded.

    Notes
    -----
    Three-check gate structure (see module-level constants ``_GATE_*``):

    - Check 1 is an advisory input proxy ("is the grid wide enough?").
    - Check 2 is a hard input gate ("is the tail fit well-defined?").
    - Check 3 is a hard post-computation gate ("does the kernel
      actually decay?") and is authoritative.

    The M7.5 PR1 ``asymptote_check_override`` parameter bypasses
    ``_validate_input_gates`` in its entirety (both Check 1 and
    Check 2) for the small-body regime where the 1/omega^4 asymptote
    is nowhere reached on the available BEM grid. Check 3 remains
    authoritative and continues to fire on both paths.

    See ``docs/post-mortems/m6-pr3-radiation-kernel-bug.md`` for the
    audit that motivated the original implementation,
    ``docs/diagnostics/m6-pr3-filon-formula-check.md`` for the
    machine-precision verification of the Filon-trapezoidal closed
    form, and ``docs/post-mortems/m6-pr4-wamit-dim-bug.md`` /
    ``docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png`` for the
    Option E gate refactor (locked at fix-wamit-dimensionalisation).
    The Item 25 applicability sub-item at
    ``docs/openfast-cross-check-conventions.md`` and Phase 2 tracker
    entry ``ITEM25-SMALL-BODY-APPLICABILITY`` at
    ``docs/phase2-followups.md`` carry the M7.5 PR1 override
    rationale.
    """
    if t_max <= 0.0:
        raise ValueError(f"t_max must be positive; got {t_max}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive; got {dt}")
    if dt > t_max:
        raise ValueError(f"dt ({dt}) must be <= t_max ({t_max})")

    # Item 25 override (M7.5 PR1). Inline rationale validation +
    # UserWarning emission per plan §I3; no separate helper (over-
    # abstracted for 5 lines per plan Phase 2 fix #6).
    if asymptote_check_override is not None:
        if not isinstance(asymptote_check_override, str):
            raise ValueError(
                "asymptote_check_override must be a non-empty rationale string; got "
                f"{type(asymptote_check_override).__name__}"
            )
        if asymptote_check_override.strip() == "":
            raise ValueError(
                "asymptote_check_override rationale is empty or whitespace-only; "
                "the override requires an explicit rationale (see "
                "docs/openfast-cross-check-conventions.md Item 25 applicability "
                "sub-item)."
            )
        warnings.warn(
            "Item 25 asymptote check bypassed via asymptote_check_override; "
            f"rationale: {asymptote_check_override!r}. Kernel computed via "
            "zero-fill tail -- high-frequency response in overridden DOFs is "
            "not analytically bounded. See docs/openfast-cross-check-"
            "conventions.md Item 25 applicability sub-item for the full "
            "applicability envelope.",
            UserWarning,
            stacklevel=2,
        )

    omega = np.asarray(hdb.omega, dtype=np.float64)
    b_stack = np.asarray(hdb.B, dtype=np.float64)

    if omega[0] > _FLOAT_EPS:
        omega = np.concatenate([[0.0], omega])
        b_stack = np.concatenate([np.zeros((6, 6, 1), dtype=np.float64), b_stack], axis=2)

    if asymptote_check_override is None:
        # Standard path -- bit-identical to pre-PR1 behavior.
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
    else:
        # Override path per Q3 lock (docs/m7.5-reader-hygiene-plan.md §I3):
        # skip _validate_input_gates entirely and zero-fill the tail extension.
        # Check 3 still runs below via _validate_kernel_decay -- the
        # post-extension gate is not bypassable per plan Q3 lock.
        n_t = round(t_max / dt) + 1
        t_arr = dt * np.arange(n_t, dtype=np.float64)

        K_in = filon_trap_cosine(omega, b_stack, t_arr)
        K_tail = np.zeros_like(K_in)

    K = (2.0 / np.pi) * (K_in + K_tail)

    # Check 3 (post-extension hard error). Authoritative gate -- the
    # BEM input proxies in _validate_input_gates only screen for likely
    # problems; this measures the actual decay we care about. Runs on
    # both the standard and override paths per plan Q3 lock.
    _validate_kernel_decay(K)

    return RetardationKernel(K=K, t=t_arr, dt=float(dt))


def _validate_input_gates(omega: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Pre-computation gates: Check 1 (advisory) + Check 2 (hard).

    See module-level constants ``_GATE_*`` for the rationale and
    threshold values. Three-check structure locked at fix-wamit-
    dimensionalisation; this function carries Checks 1 and 2.
    Check 3 (post-extension kernel decay) runs separately on the
    computed kernel via :func:`_validate_kernel_decay`.

    Returns
    -------
    NDArray[bool]
        Shape ``(6, 6)``: True where the entry's tail extension should
        be zeroed (Check 2 failed OR entry is below the
        ``_OFFDIAG_REL_THRESHOLD`` of max diagonal). Diagonals are
        always False (Check 2 failure on a diagonal raises).
    """
    omega_max = float(omega[-1])
    diag_max = np.array([np.max(np.abs(B[i, i, :])) for i in range(6)], dtype=np.float64)

    # Check 1 (advisory): |B_ii(omega_max)| / max|B_ii| > 5% warns that
    # the BEM grid is under-resolved. The 1/omega^4 tail extension may
    # still recover a clean kernel -- Check 3 is the authoritative
    # gate -- but values this high are worth flagging early so a future
    # contributor knows to widen the BEM grid if they need a tighter
    # cross-check tolerance.
    check1_offenders: list[tuple[int, float]] = []
    for i in range(6):
        if diag_max[i] < _FLOAT_EPS:
            continue
        ratio = abs(B[i, i, -1]) / diag_max[i]
        if ratio >= _GATE_AMPLITUDE_RATIO_WARN:
            check1_offenders.append((i, ratio))
    if check1_offenders:
        offender_lines = "\n".join(
            f"  DOF {i}: |B[{i},{i}](omega_max={omega_max:.3f})|/peak = " f"{ratio * 100:.1f}%"
            for i, ratio in check1_offenders
        )
        warnings.warn(
            "Check 1 (BEM grid amplitude proxy): high B(omega_max)/peak "
            f"on diagonal entries:\n{offender_lines}\n"
            f"The {_GATE_AMPLITUDE_RATIO_WARN * 100:.0f}% advisory threshold is "
            "for early heads-up; the 1/omega^4 tail extension on "
            "[omega_max, 5*omega_max] typically compensates and the "
            "post-extension kernel-decay check (Check 3) is authoritative. "
            "However, values this high suggest the BEM grid would benefit "
            "from being widened (re-run WAMIT with a higher omega_max or "
            "larger PER list). See "
            "docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png for the "
            "marin_semi reference (1.7%/peak passes Check 3 cleanly).",
            UserWarning,
            stacklevel=2,
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


def _validate_kernel_decay(K: NDArray[np.float64]) -> None:
    """Check 3 (post-extension hard error): ``|K_ii(t_max)| / max|K_ii(t)|``
    must be < ``_GATE_KERNEL_DECAY_RATIO`` (= 0.001) on every diagonal.

    Threshold rationale: 0.1 % is 20x the measured marin_semi ratio
    (~ 0.005 %); conservative for future fixtures with potentially
    worse decay properties. Beyond that, the un-decayed kernel
    produces sustained oscillation in the convolution sum and
    corrupts the simulation -- not a heuristic, a real failure mode.

    This is the authoritative gate: the BEM input proxies (Check 1
    advisory, Check 2 hard) screen for likely pathologies, but the
    actual condition we care about is whether the resulting K(t)
    decays. Check 3 measures that directly.

    See ``docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png`` for
    the marin_semi reference: surge / sway / yaw decay to < 6e-5 of
    peak by t = 200 s even with B(omega_max)/peak ~ 1.7 % (above the
    Check 1 advisory threshold).
    """
    peak = np.max(np.abs(K), axis=2)
    end = np.abs(K[:, :, -1])
    diag_peak = np.diag(peak)
    diag_end = np.diag(end)
    offenders: list[tuple[int, float, float]] = []
    for i in range(6):
        if diag_peak[i] <= _FLOAT_EPS:
            continue
        ratio = float(diag_end[i] / diag_peak[i])
        if ratio > _GATE_KERNEL_DECAY_RATIO:
            offenders.append((i, ratio, float(diag_peak[i])))
    if offenders:
        offender_lines = "\n".join(
            f"  DOF {i}: |K[{i},{i}](t_max)|/peak = {ratio * 100:.4f}%, "
            f"max|K[{i},{i}]| = {peak_val:.3e}"
            for i, ratio, peak_val in offenders
        )
        raise ValueError(
            "Check 3 (post-extension kernel decay): the radiation kernel "
            "fails to decay below the "
            f"{_GATE_KERNEL_DECAY_RATIO * 100:.2f}% threshold on the "
            f"following diagonal entries:\n{offender_lines}\n"
            "An un-decayed kernel produces sustained oscillation in the "
            "Cummins convolution and corrupts the simulation. Likely "
            "remedies: increase t_max so the kernel has more room to "
            "decay, OR widen the BEM grid (re-run WAMIT with higher "
            "omega_max) -- if Check 1 also warned, the second is the "
            "more reliable fix. See "
            "docs/diagnostics/m6-pr4-pre3-surge-kernel-quality.png for "
            "the marin_semi reference (passes at ~5e-5)."
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
