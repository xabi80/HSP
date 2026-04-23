"""Global state pack/unpack and block-diagonal assembly for N-body Cummins.

ARCHITECTURE.md §2.2 defines the multi-body generalization of the
single-body Cummins equation as::

    [M + A_inf]_global Xi_ddot(t)
      + integral_{0}^{t} K_global(t - tau) Xi_dot(tau) dtau
      + C_global Xi(t)
      = F_global(t)

where the global state ``Xi = [xi_1, xi_2, ..., xi_N]`` has size ``6N``
and the global matrices are ``6N x 6N`` (or ``6N x 6N x N_t`` for the
retardation kernel). When the underlying BEM database carries no
hydrodynamic interaction between bodies, those matrices are
block-diagonal — each 6x6 block is the single-body quantity — and the
assembly is a plain per-body stack.

This module supplies that plumbing. It is intentionally thin: no new
physics, no global state, just two pack/unpack helpers for the state
vector and two block-diagonal stackers for the Cummins LHS and
retardation kernel. The integrator (:mod:`floatsim.solver.newmark`) and
the retardation buffer (:class:`floatsim.hydro.retardation.RadiationConvolution`)
both accept ``6N``-DOF inputs transparently — they size their internal
buffers from the matrices they receive.

Hydrodynamic cross-coupling between bodies (off-block-diagonal entries
from a multi-body BEM run) will be plugged in later by assembling the
global matrices directly rather than via these helpers. The helpers
here target the M4 PR1 path: N independent bodies, each backed by its
own single-body BEM database.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.radiation import CumminsLHS
from floatsim.hydro.retardation import RetardationKernel


def pack_state(per_body: Sequence[NDArray[np.floating]]) -> NDArray[np.float64]:
    """Concatenate per-body length-6 state vectors into a single length-6N vector.

    Parameters
    ----------
    per_body
        Sequence of ``N`` arrays, each of shape ``(6,)``. Order is
        preserved: body ``k`` occupies slots ``[6k, 6k+6)`` of the
        returned vector.

    Returns
    -------
    ndarray of shape ``(6N,)``, float64.

    Raises
    ------
    ValueError
        If the input sequence is empty, or any entry is not length 6.
    """
    if len(per_body) == 0:
        raise ValueError("pack_state requires at least one per-body vector")
    out = np.empty(6 * len(per_body), dtype=np.float64)
    for k, v in enumerate(per_body):
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != (6,):
            raise ValueError(f"per_body[{k}] must have shape (6,); got {arr.shape}")
        out[6 * k : 6 * (k + 1)] = arr
    return out


def unpack_state(xi: NDArray[np.floating]) -> NDArray[np.float64]:
    """Split a length-6N global state vector into an ``(N, 6)`` per-body view.

    Parameters
    ----------
    xi
        Global state of shape ``(6N,)`` for some ``N >= 1``.

    Returns
    -------
    ndarray of shape ``(N, 6)``, float64. ``out[k, :]`` is body ``k``'s
    length-6 slice.

    Raises
    ------
    ValueError
        If ``xi`` is not 1-D with a length that is a positive multiple of 6.
    """
    arr = np.asarray(xi, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or arr.size % 6 != 0:
        raise ValueError(
            f"xi must be 1-D with length a positive multiple of 6; got shape {arr.shape}"
        )
    return arr.reshape(-1, 6).copy()


def _block_diagonal(blocks: Sequence[NDArray[np.floating]]) -> NDArray[np.float64]:
    """Stack square matrix blocks block-diagonally into one larger matrix."""
    if len(blocks) == 0:
        raise ValueError("block-diagonal assembly requires at least one block")
    sizes = [int(b.shape[0]) for b in blocks]
    for k, b in enumerate(blocks):
        if b.ndim != 2 or b.shape[0] != b.shape[1]:
            raise ValueError(f"block {k} must be square 2-D; got shape {b.shape}")
    total = sum(sizes)
    out = np.zeros((total, total), dtype=np.float64)
    offset = 0
    for b, n in zip(blocks, sizes, strict=True):
        out[offset : offset + n, offset : offset + n] = b
        offset += n
    return out


def assemble_global_lhs(per_body: Sequence[CumminsLHS]) -> CumminsLHS:
    """Stack per-body :class:`CumminsLHS` block-diagonally into a 6N-DOF global LHS.

    Parameters
    ----------
    per_body
        Sequence of ``N`` single-body :class:`CumminsLHS` instances, each
        carrying 6x6 ``M_plus_Ainf`` and ``C``. Block-diagonal stacking
        assumes no hydrodynamic coupling between bodies — appropriate for
        the common case of ``N`` independent BEM runs.

    Returns
    -------
    CumminsLHS
        New instance whose ``M_plus_Ainf`` and ``C`` are 6N x 6N
        block-diagonal matrices, with body ``k`` occupying the
        ``[6k:6k+6, 6k:6k+6]`` block.

    Raises
    ------
    ValueError
        If ``per_body`` is empty.
    """
    if len(per_body) == 0:
        raise ValueError("assemble_global_lhs requires at least one body")
    m_global = _block_diagonal([lhs.M_plus_Ainf for lhs in per_body])
    c_global = _block_diagonal([lhs.C for lhs in per_body])
    return CumminsLHS(M_plus_Ainf=m_global, C=c_global)


def assemble_global_kernel(per_body: Sequence[RetardationKernel]) -> RetardationKernel:
    """Stack per-body :class:`RetardationKernel` block-diagonally along the DOF axes.

    Parameters
    ----------
    per_body
        Sequence of ``N`` single-body :class:`RetardationKernel` instances.
        Every kernel must share the same ``dt`` and number of lag samples
        (``n_lags``) — the integrator advances them on a common time grid.

    Returns
    -------
    RetardationKernel
        New kernel whose ``K`` has shape ``(6N, 6N, N_t)`` with body ``k``
        occupying the ``[6k:6k+6, 6k:6k+6, :]`` block, same ``t`` and
        ``dt`` as the inputs.

    Raises
    ------
    ValueError
        If ``per_body`` is empty, or the input kernels disagree on ``dt``
        or ``n_lags``.
    """
    if len(per_body) == 0:
        raise ValueError("assemble_global_kernel requires at least one body")
    dt0 = per_body[0].dt
    n_lags0 = per_body[0].n_lags
    for k, ker in enumerate(per_body[1:], start=1):
        if ker.dt != dt0:
            raise ValueError(
                f"per_body[{k}].dt = {ker.dt} does not match per_body[0].dt = {dt0}; "
                "resample to a common time grid before stacking"
            )
        if ker.n_lags != n_lags0:
            raise ValueError(
                f"per_body[{k}].n_lags = {ker.n_lags} does not match "
                f"per_body[0].n_lags = {n_lags0}"
            )

    n = len(per_body)
    K_global = np.zeros((6 * n, 6 * n, n_lags0), dtype=np.float64)
    for k, ker in enumerate(per_body):
        K_global[6 * k : 6 * (k + 1), 6 * k : 6 * (k + 1), :] = ker.K
    return RetardationKernel(K=K_global, t=per_body[0].t.copy(), dt=dt0)
