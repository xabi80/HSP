"""Programmatic builders for synthetic HydroDatabase instances used in tests.

Rationale: M1 has no need for serializing BEM data to disk, so we skip a YAML
fixture format. When a real BEM reader lands (M1.5, M5), production-scale
fixtures will come from that reader.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase


def diagonal_6x6(diagonal: Sequence[float]) -> NDArray[np.float64]:
    """Build a 6x6 float64 matrix with the given diagonal and zero elsewhere."""
    arr = np.asarray(diagonal, dtype=np.float64)
    assert arr.shape == (6,), f"expected 6-vector; got {arr.shape}"
    return np.diag(arr)


def make_diagonal_hdb(
    *,
    A_inf_diag: Sequence[float],
    C_diag: Sequence[float],
    A_diag_per_omega: Sequence[Sequence[float]] | None = None,
    B_diag_per_omega: Sequence[Sequence[float]] | None = None,
    omega: Sequence[float] | None = None,
    heading_deg: Sequence[float] | None = None,
    reference_point: Sequence[float] = (0.0, 0.0, 0.0),
    metadata: dict[str, str] | None = None,
) -> HydroDatabase:
    """Build a HydroDatabase whose 6x6 blocks are diagonal.

    Convenient for frequency-domain sanity tests where only the diagonal
    entries affect natural periods. RAO defaults to zeros.
    """
    if omega is None:
        omega = [0.1, 0.5, 1.0, 2.0, 3.0]
    if heading_deg is None:
        heading_deg = [0.0, 90.0]
    omega_arr = np.asarray(omega, dtype=np.float64)
    heading_arr = np.asarray(heading_deg, dtype=np.float64)
    n_w = omega_arr.size
    n_h = heading_arr.size

    if A_diag_per_omega is None:
        A_diag_per_omega = [list(A_inf_diag)] * n_w
    if B_diag_per_omega is None:
        B_diag_per_omega = [[0.0] * 6] * n_w

    assert len(A_diag_per_omega) == n_w
    assert len(B_diag_per_omega) == n_w

    A = np.stack([diagonal_6x6(d) for d in A_diag_per_omega], axis=-1)
    B = np.stack([diagonal_6x6(d) for d in B_diag_per_omega], axis=-1)

    return HydroDatabase(
        omega=omega_arr,
        heading_deg=heading_arr,
        A=A,
        B=B,
        A_inf=diagonal_6x6(A_inf_diag),
        C=diagonal_6x6(C_diag),
        RAO=np.zeros((6, n_w, n_h), dtype=np.complex128),
        reference_point=np.asarray(reference_point, dtype=np.float64),
        metadata=metadata or {"source": "tests.support.synthetic_bem"},
    )
