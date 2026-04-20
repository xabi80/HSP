"""Cummins LHS assembly and natural-period helpers — ARCHITECTURE.md §2.1, §2.2.

The Cummins equation for a single rigid body in the inertial frame is::

    [M + A_inf] * xi_ddot(t)
      + integral_{0}^{t} K(t - tau) * xi_dot(tau) dtau
      + C * xi(t)
      = F_exc(t) + F_visc + F_moor + F_ext

In Milestone 1 we assemble only the frequency-domain left-hand-side pieces
that do not depend on time history: the generalized inertia ``M + A_inf``
and the hydrostatic restoring matrix ``C``. The retardation kernel ``K(t)``
and the convolution buffer arrive in Milestone 2.

DOF ordering matches :mod:`floatsim.hydro.database`: ``(surge, sway,
heave, roll, pitch, yaw)`` — see ARCHITECTURE.md §3.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase

_SYMMETRY_RTOL: Final[float] = 1.0e-6
_SYMMETRY_ATOL: Final[float] = 1.0e-10


def _require_symmetric(m: NDArray[np.floating], label: str) -> None:
    if not np.allclose(m, m.T, rtol=_SYMMETRY_RTOL, atol=_SYMMETRY_ATOL):
        raise ValueError(f"{label} must be symmetric (within rtol={_SYMMETRY_RTOL:.0e})")


@dataclass(frozen=True)
class CumminsLHS:
    """Time-independent left-hand-side matrices of the Cummins equation.

    Attributes
    ----------
    M_plus_Ainf
        6x6 symmetric positive-definite generalized inertia ``M + A_inf``
        in kg / kg*m / kg*m^2 per block.
    C
        6x6 symmetric hydrostatic restoring matrix in N/m / N / N*m per block.
    """

    M_plus_Ainf: NDArray[np.float64]
    C: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.M_plus_Ainf.shape != (6, 6):
            raise ValueError(f"M_plus_Ainf must have shape (6, 6); got {self.M_plus_Ainf.shape}")
        if self.C.shape != (6, 6):
            raise ValueError(f"C must have shape (6, 6); got {self.C.shape}")
        _require_symmetric(self.M_plus_Ainf, "M_plus_Ainf")
        _require_symmetric(self.C, "C")


def assemble_cummins_lhs(
    *,
    rigid_body_mass: NDArray[np.floating],
    hdb: HydroDatabase,
) -> CumminsLHS:
    """Assemble ``M + A_inf`` and ``C`` for a single body — ARCHITECTURE.md §2.1.

    Parameters
    ----------
    rigid_body_mass
        6x6 rigid-body mass/inertia matrix from
        :func:`floatsim.bodies.mass_properties.rigid_body_mass_matrix`,
        expressed at the body reference point in the body frame.
    hdb
        Validated :class:`HydroDatabase` for this body. Must be referred
        to the same reference point as ``rigid_body_mass``.

    Returns
    -------
    CumminsLHS
        Frozen dataclass carrying ``M_plus_Ainf`` (``M + A_inf``) and ``C``
        (the hydrostatic restoring matrix straight from the database).

    Notes
    -----
    No frame rotation is applied here: in Phase-1 Level-2 fidelity
    (ARCHITECTURE.md §9.2), BEM coefficients are evaluated in the body
    frame and rotated to inertial at each time step by the solver, not
    at assembly time. The quantities returned here are body-frame.
    """
    m_body = np.asarray(rigid_body_mass, dtype=np.float64)
    if m_body.shape != (6, 6):
        raise ValueError(f"rigid_body_mass must have shape (6, 6); got {m_body.shape}")
    _require_symmetric(m_body, "rigid_body_mass")

    m_plus_ainf = m_body + np.asarray(hdb.A_inf, dtype=np.float64)
    c_matrix = np.asarray(hdb.C, dtype=np.float64).copy()
    return CumminsLHS(M_plus_Ainf=m_plus_ainf, C=c_matrix)


def natural_periods_uncoupled(lhs: CumminsLHS) -> NDArray[np.float64]:
    """Uncoupled natural period per DOF: ``T_i = 2*pi * sqrt((M+A_inf)_ii / C_ii)``.

    This ignores off-diagonal coupling in ``M + A_inf`` and ``C`` — it is a
    sanity check, not the true coupled eigenperiod. DOFs whose diagonal
    restoring is non-positive (e.g. surge, sway, yaw for an unmoored body)
    have no natural period and are returned as NaN.

    Parameters
    ----------
    lhs
        Assembled Cummins LHS.

    Returns
    -------
    ndarray of shape (6,), float64
        Natural periods in seconds, per DOF, in the ``(surge, sway, heave,
        roll, pitch, yaw)`` order. NaN where ``C_ii <= 0`` or where
        ``(M+A_inf)_ii <= 0`` (the latter should never happen for physical
        inputs, but we guard against it to avoid silent bad output).
    """
    m_diag = np.diag(lhs.M_plus_Ainf).astype(np.float64)
    c_diag = np.diag(lhs.C).astype(np.float64)
    periods = np.full(6, np.nan, dtype=np.float64)
    valid = (c_diag > 0.0) & (m_diag > 0.0)
    periods[valid] = 2.0 * np.pi * np.sqrt(m_diag[valid] / c_diag[valid])
    return periods
