"""Cummins LHS assembly — ARCHITECTURE.md §2.1, §2.2.

We assemble (M + A_inf) and C for a single body from a rigid-body mass
matrix and a HydroDatabase. No time stepping in M1 — this test suite
validates the frequency-domain sanity of the assembled matrices.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.radiation import (
    CumminsLHS,
    assemble_cummins_lhs,
    natural_periods_uncoupled,
)
from tests.support.synthetic_bem import make_diagonal_hdb


def _rigid_mass(m: float, Ixx: float, Iyy: float, Izz: float) -> np.ndarray:
    return rigid_body_mass_matrix(mass=m, inertia_at_reference=np.diag([Ixx, Iyy, Izz]))


# ---------- basic assembly ----------


def test_diagonal_assembly_readback() -> None:
    m = 1.4091e7
    Ixx, Iyy, Izz = 1.1254e10, 1.1254e10, 1.2571e10
    M_body = _rigid_mass(m, Ixx, Iyy, Izz)

    A_inf_diag = [5.0e5, 5.0e5, 1.9e7, 8.0e9, 8.0e9, 0.0]
    C_diag = [0.0, 0.0, 4.27e6, 1.5e9, 1.5e9, 0.0]
    hdb = make_diagonal_hdb(A_inf_diag=A_inf_diag, C_diag=C_diag)

    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)

    expected_diag = np.array(
        [
            m + A_inf_diag[0],
            m + A_inf_diag[1],
            m + A_inf_diag[2],
            Ixx + A_inf_diag[3],
            Iyy + A_inf_diag[4],
            Izz + A_inf_diag[5],
        ]
    )
    np.testing.assert_allclose(np.diag(lhs.M_plus_Ainf), expected_diag)
    np.testing.assert_allclose(lhs.C, hdb.C)


def test_returns_cummins_lhs_dataclass() -> None:
    M_body = _rigid_mass(1.0, 1.0, 1.0, 1.0)
    hdb = make_diagonal_hdb(A_inf_diag=[1.0] * 6, C_diag=[1.0] * 6)
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    assert isinstance(lhs, CumminsLHS)
    assert lhs.M_plus_Ainf.shape == (6, 6)
    assert lhs.C.shape == (6, 6)


# ---------- invariants ----------


def test_lhs_matrices_are_symmetric() -> None:
    M_body = _rigid_mass(1.0e6, 1.0e9, 2.0e9, 3.0e9)
    hdb = make_diagonal_hdb(
        A_inf_diag=[1.0e5] * 3 + [1.0e8] * 3,
        C_diag=[0.0, 0.0, 1.0e7, 1.0e8, 1.0e8, 0.0],
    )
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    np.testing.assert_allclose(lhs.M_plus_Ainf, lhs.M_plus_Ainf.T, atol=1e-8)
    np.testing.assert_allclose(lhs.C, lhs.C.T, atol=1e-8)


def test_m_plus_ainf_is_positive_definite() -> None:
    M_body = _rigid_mass(1.0, 1.0, 1.0, 1.0)
    hdb = make_diagonal_hdb(A_inf_diag=[0.5] * 6, C_diag=[1.0] * 6)
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    eigvals = np.linalg.eigvalsh(lhs.M_plus_Ainf)
    assert float(eigvals.min()) > 0.0


# ---------- input validation ----------


def test_wrong_mass_matrix_shape_rejected() -> None:
    hdb = make_diagonal_hdb(A_inf_diag=[1.0] * 6, C_diag=[1.0] * 6)
    with pytest.raises(ValueError, match="shape"):
        assemble_cummins_lhs(rigid_body_mass=np.eye(5), hdb=hdb)


def test_non_symmetric_mass_matrix_rejected() -> None:
    hdb = make_diagonal_hdb(A_inf_diag=[1.0] * 6, C_diag=[1.0] * 6)
    M = np.eye(6)
    M[0, 1] = 0.5  # break symmetry
    with pytest.raises(ValueError, match="symmetric"):
        assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb)


# ---------- natural periods helper ----------


def test_natural_periods_uncoupled_matches_analytical() -> None:
    """Diagonal case: T_i = 2*pi * sqrt((M+A_inf)_ii / C_ii)."""
    M_body = _rigid_mass(1.0e7, 1.0e10, 1.0e10, 1.0e10)
    A_inf_diag = [5.0e5, 5.0e5, 2.0e7, 0.0, 0.0, 0.0]
    C_diag = [0.0, 0.0, 4.27e6, 1.5e9, 1.5e9, 0.0]
    hdb = make_diagonal_hdb(A_inf_diag=A_inf_diag, C_diag=C_diag)
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    periods = natural_periods_uncoupled(lhs)

    expected_heave = 2.0 * np.pi * np.sqrt((1.0e7 + 2.0e7) / 4.27e6)
    expected_pitch = 2.0 * np.pi * np.sqrt((1.0e10 + 0.0) / 1.5e9)

    np.testing.assert_allclose(periods[2], expected_heave, rtol=1e-12)
    np.testing.assert_allclose(periods[4], expected_pitch, rtol=1e-12)


def test_natural_periods_return_nan_where_restoring_is_zero() -> None:
    """DOFs with C_ii <= 0 have no natural period (surge/sway/yaw in unmoored body)."""
    M_body = _rigid_mass(1.0, 1.0, 1.0, 1.0)
    hdb = make_diagonal_hdb(
        A_inf_diag=[1.0] * 6,
        C_diag=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # surge, sway, yaw zero
    )
    lhs = assemble_cummins_lhs(rigid_body_mass=M_body, hdb=hdb)
    periods = natural_periods_uncoupled(lhs)
    assert np.isnan(periods[0])
    assert np.isnan(periods[1])
    assert np.isnan(periods[5])
    assert np.isfinite(periods[2])
