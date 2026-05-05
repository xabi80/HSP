"""Filon-trapezoidal cosine quadrature -- ARCHITECTURE.md §2.3 / M6 PR3 fix.

Computes ``∫_{ω_0}^{ω_N} B(ω) cos(ω·t) dω`` exactly for piecewise-linear
``B(ω)`` at any lag ``t``, by integrating each segment's
``(B_k + (ω - ω_k)·m_k) · cos(ω·t)`` analytically.

This replaces the trapezoidal-cosine approximation in
:func:`floatsim.hydro.retardation.compute_retardation_kernel` whose
discrete cosine sum has irreducible Nyquist aliasing at large ``t``
on grids where ``dω · t > π/2``. Filon-trapezoidal has no such
constraint -- the cosine factor is integrated analytically per
segment rather than sampled.

Per-segment closed form (Davis & Rabinowitz 1984 §2.10.3, Tuck 1967)::

    I_k(t) = [B_{k+1} sin(ω_{k+1} t) - B_k sin(ω_k t)] / t
           + m_k [cos(ω_{k+1} t) - cos(ω_k t)] / t²        (t ≠ 0)
    I_k(0) = ½ (B_k + B_{k+1}) (ω_{k+1} - ω_k)

with ``m_k = (B_{k+1} - B_k) / (ω_{k+1} - ω_k)``. The full integral
sums these; the sin endpoints telescope across segments to leave
``B_N sin(ω_N t) - B_0 sin(ω_0 t)`` plus a cosine-difference sum.

Verified to machine precision in
``docs/diagnostics/m6-pr3-filon-formula-check.md``.

References
----------
- Filon, L.N.G., 1928. "On a quadrature formula for trigonometric
  integrals." Proc. Royal Soc. Edinburgh 49:38-47 (original Simpson form).
- Tuck, E.O., 1967. "A simple Filon-trapezoidal rule." Math. Comp.
  21:239-241 (the trapezoidal case used here).
- Davis, P.J. and Rabinowitz, P., 1984. *Methods of Numerical
  Integration*, 2nd ed., Academic Press, §2.10.3 "Filon-type rules".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def filon_trap_cosine(
    omega: NDArray[np.floating],
    B: NDArray[np.floating],
    t: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Filon-trapezoidal quadrature of ``B(ω) · cos(ω·t)`` on ``[ω_0, ω_N]``.

    The integrand is approximated as piecewise-linear in ``ω`` between
    grid points ``(ω_k, B_k)``; the cosine factor is integrated
    analytically per segment. The resulting integral is exact for
    piecewise-linear ``B`` at any ``t`` (no Nyquist constraint on
    ``dω · t``).

    Parameters
    ----------
    omega
        ``(n_omega,)`` strictly increasing frequency grid in rad/s.
    B
        ``(..., n_omega)`` values to integrate. Broadcasts over leading
        axes. Typical use: ``B`` has shape ``(6, 6, n_omega)`` for the
        radiation damping matrix.
    t
        ``(n_t,)`` lag values in seconds. Non-negative.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(*B.shape[:-1], n_t)``. ``out[..., j]`` is the integral
        evaluated at ``t[j]``.

    Raises
    ------
    ValueError
        If ``omega`` or ``t`` are not 1-D, or if ``B``'s last dimension
        does not match ``omega.size``.
    """
    omega_arr = np.asarray(omega, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)

    if omega_arr.ndim != 1:
        raise ValueError(f"omega must be 1D, got shape {omega_arr.shape}")
    if t_arr.ndim != 1:
        raise ValueError(f"t must be 1D, got shape {t_arr.shape}")
    if B_arr.shape[-1] != omega_arr.size:
        raise ValueError(f"B last dim ({B_arr.shape[-1]}) must match omega size ({omega_arr.size})")

    out_shape = B_arr.shape[:-1] + t_arr.shape
    out = np.empty(out_shape, dtype=np.float64)

    # Per-segment slope m_k.
    domega = np.diff(omega_arr)
    dB = B_arr[..., 1:] - B_arr[..., :-1]
    m_seg = dB / domega  # shape (..., n_omega - 1)

    # Split t into zero / non-zero.
    t_zero_mask = t_arr == 0.0

    if t_zero_mask.any():
        # Trapezoidal rule reduces to ½(B_k + B_{k+1}) · domega summed.
        trap = 0.5 * np.sum((B_arr[..., :-1] + B_arr[..., 1:]) * domega, axis=-1)  # shape (...,)
        # Broadcast trap over all t==0 entries.
        out[..., t_zero_mask] = trap[..., None]

    t_nonzero = t_arr[~t_zero_mask]
    if t_nonzero.size > 0:
        # Cosine and sine of (ω_k, t_j) for k in [0, n_omega), j in [0, n_t_nz).
        wt = np.outer(omega_arr, t_nonzero)  # (n_omega, n_t_nz)
        sin_wt = np.sin(wt)
        cos_wt = np.cos(wt)

        # First (telescoped) term: [B_N sin(ω_N t) - B_0 sin(ω_0 t)] / t.
        first = (
            B_arr[..., -1, None] * sin_wt[-1, :] - B_arr[..., 0, None] * sin_wt[0, :]
        ) / t_nonzero  # shape (..., n_t_nz)

        # Second term: Σ_k m_k [cos(ω_{k+1} t) - cos(ω_k t)] / t².
        cos_diff = cos_wt[1:, :] - cos_wt[:-1, :]  # (n_omega-1, n_t_nz)
        # Sum over k axis: m_seg has (..., n_omega-1), cos_diff has (n_omega-1, n_t_nz).
        second = np.einsum("...k,kn->...n", m_seg, cos_diff) / (t_nonzero * t_nonzero)

        out[..., ~t_zero_mask] = first + second

    return out


def fit_per_entry_tail_constants(
    omega: NDArray[np.floating],
    B: NDArray[np.floating],
    *,
    n_tail_points: int = 10,
) -> NDArray[np.float64]:
    """Per-entry constant ``C_ij`` for the ``B_ij(ω) ≈ C_ij / ω⁴`` tail.

    Per Newman (1977) §6.18 / Faltinsen (1990) §3.3.2: 3D surface-piercing
    bodies in deep water have radiation damping decaying as ``ω⁻⁴`` at high
    frequency. Constant of proportionality depends on body geometry near
    the waterline.

    Fit ``C_ij = mean(B_ij(ω_k) · ω_k⁴)`` over the last ``n_tail_points``
    grid samples (per Refinement 1, M6 PR3 audit).

    Parameters
    ----------
    omega
        ``(n_omega,)`` strictly increasing grid.
    B
        ``(..., n_omega)`` values.
    n_tail_points
        Number of trailing grid samples to fit over.

    Returns
    -------
    NDArray[np.float64]
        Shape ``B.shape[:-1]``: per-entry constant ``C_ij``. Off-diagonal
        sign is preserved.
    """
    omega_arr = np.asarray(omega, dtype=np.float64)
    B_arr = np.asarray(B, dtype=np.float64)
    n = min(int(n_tail_points), omega_arr.size)
    omega_tail = omega_arr[-n:]
    B_tail = B_arr[..., -n:]
    omega4 = omega_tail**4
    result: NDArray[np.float64] = np.mean(B_tail * omega4, axis=-1)
    return result


def compute_tail_contribution(
    C: NDArray[np.floating],
    omega_max: float,
    t: NDArray[np.floating],
    *,
    upper_bound_factor: float = 5.0,
    epsrel: float = 1.0e-8,
) -> NDArray[np.float64]:
    """Integrate ``(C / ω⁴) · cos(ω·t)`` on ``[ω_max, factor·ω_max]``.

    Per Refinement 1 (M6 PR3 audit, Newman 1977 §6.18 high-frequency
    asymptote). Uses :func:`scipy.integrate.quad_vec` per ``(i, j)``
    entry, looping internally to avoid the memory blow-up of a fully
    vectorised ``(6, 6, n_t)`` integrand.

    The 5× upper bound captures > 99% of the tail integral for a ``ω⁻⁴``
    decay (the ``[5·ω_max, ∞)`` remainder is < ``(1/5)³ ≈ 0.008`` of the
    ``[ω_max, 5·ω_max]`` contribution, integrated over a smooth cosine).

    Parameters
    ----------
    C
        ``(..., 1)`` or ``(...,)`` per-entry constants from
        :func:`fit_per_entry_tail_constants`. Entries that are zero
        (or numerically negligible per the caller's threshold) skip
        the integration.
    omega_max
        Lower limit (the BEM grid's highest sample).
    t
        ``(n_t,)`` lag values.
    upper_bound_factor
        Sets upper limit ``factor · omega_max``. Default 5.
    epsrel
        Relative tolerance forwarded to ``quad_vec``.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(*C.shape, n_t)``: the tail contribution per entry, per
        time sample. Already includes the ``(2/π)`` Cummins-kernel
        prefactor consumed by the caller -- no, actually the prefactor
        is applied by the caller, so this returns just the integral.
    """
    from scipy.integrate import quad_vec  # local import keeps top-level lean

    C_arr = np.asarray(C, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)

    omega_lo = float(omega_max)
    omega_hi = upper_bound_factor * omega_lo

    out_shape = C_arr.shape + t_arr.shape
    out = np.zeros(out_shape, dtype=np.float64)

    # Iterate over flat indices into C; skip zero entries.
    flat_C = C_arr.reshape(-1)
    flat_out = out.reshape(-1, t_arr.size)
    for k, c_val in enumerate(flat_C):
        if c_val == 0.0:
            continue

        def integrand(omega_val: float, c_val: float = float(c_val)) -> NDArray[np.float64]:
            return (c_val / (omega_val**4)) * np.cos(omega_val * t_arr)

        result, _ = quad_vec(integrand, omega_lo, omega_hi, epsrel=epsrel)
        flat_out[k, :] = result
    return out
