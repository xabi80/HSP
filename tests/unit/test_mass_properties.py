"""Rigid-body 6x6 mass matrix assembly — ARCHITECTURE.md §4 (bodies/mass_properties).

Convention: velocity vector is [v_x, v_y, v_z, omega_x, omega_y, omega_z] — linear
first, angular second (ARCHITECTURE.md §3.3). The matrix is expressed at the body
reference point, in the body frame.

Formula (with r_CoG = vector from reference point to CoG, body frame;
I_ref = inertia tensor about the reference point):

    M = [ m * I_3        -m * r_tilde        ]
        [  m * r_tilde    I_ref              ]

where r_tilde is the skew-symmetric cross-product matrix of r_CoG.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from floatsim.bodies.mass_properties import rigid_body_mass_matrix


def _diag_inertia(Ixx: float, Iyy: float, Izz: float) -> np.ndarray:
    return np.diag([Ixx, Iyy, Izz])


# ---------- block-diagonal case (CoG at reference point) ----------


def test_zero_cog_offset_yields_block_diagonal_matrix() -> None:
    m = 2.0e7
    I = _diag_inertia(3.0e10, 3.0e10, 5.0e10)
    M = rigid_body_mass_matrix(mass=m, inertia_at_reference=I)

    assert M.shape == (6, 6)
    expected_diag = np.array([m, m, m, 3.0e10, 3.0e10, 5.0e10])
    np.testing.assert_allclose(np.diag(M), expected_diag)

    # Off-diagonal 3x3 blocks must be zero when r_CoG = 0.
    np.testing.assert_allclose(M[:3, 3:], 0.0)
    np.testing.assert_allclose(M[3:, :3], 0.0)


# ---------- CoG offset (parallel-axis coupling) ----------


def test_cog_offset_produces_expected_skew_block() -> None:
    m = 1000.0
    r = np.array([1.0, 2.0, 3.0])
    I = _diag_inertia(100.0, 200.0, 300.0)
    M = rigid_body_mass_matrix(mass=m, inertia_at_reference=I, cog_offset_body=r)

    # r_tilde = [[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]]
    expected_r_tilde = np.array(
        [
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ]
    )
    # M[:3, 3:] = -m * r_tilde; M[3:, :3] = m * r_tilde.
    np.testing.assert_allclose(M[:3, 3:], -m * expected_r_tilde)
    np.testing.assert_allclose(M[3:, :3], m * expected_r_tilde)


# ---------- universal invariants ----------


def test_mass_matrix_is_symmetric() -> None:
    m = 5.0e6
    I = _diag_inertia(1.0e9, 4.0e9, 4.0e9)
    r = np.array([0.5, -0.3, 1.2])
    M = rigid_body_mass_matrix(mass=m, inertia_at_reference=I, cog_offset_body=r)
    np.testing.assert_allclose(M, M.T, atol=1e-8)


def test_mass_matrix_is_positive_definite_for_physical_input() -> None:
    m = 1.0
    I = _diag_inertia(1.0, 1.0, 1.0)
    M = rigid_body_mass_matrix(
        mass=m, inertia_at_reference=I, cog_offset_body=np.array([0.1, 0.2, 0.0])
    )
    eigvals = np.linalg.eigvalsh(M)
    assert float(eigvals.min()) > 0.0


# ---------- input validation ----------


def test_negative_mass_rejected() -> None:
    with pytest.raises(ValueError, match="mass"):
        rigid_body_mass_matrix(mass=-1.0, inertia_at_reference=np.eye(3))


def test_zero_mass_rejected() -> None:
    with pytest.raises(ValueError, match="mass"):
        rigid_body_mass_matrix(mass=0.0, inertia_at_reference=np.eye(3))


def test_non_symmetric_inertia_rejected() -> None:
    I = np.array([[1.0, 0.2, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        rigid_body_mass_matrix(mass=1.0, inertia_at_reference=I)


def test_wrong_inertia_shape_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        rigid_body_mass_matrix(mass=1.0, inertia_at_reference=np.eye(4))


def test_wrong_cog_offset_shape_rejected() -> None:
    with pytest.raises(ValueError, match="cog_offset"):
        rigid_body_mass_matrix(
            mass=1.0, inertia_at_reference=np.eye(3), cog_offset_body=np.zeros(2)
        )


# ---------- property-based ----------


@st.composite
def _physical_input(draw: st.DrawFn) -> tuple[float, np.ndarray, np.ndarray]:
    m = draw(st.floats(min_value=1.0, max_value=1e8, allow_nan=False))
    # Diagonal inertia with dominant diagonal to guarantee PD.
    Ixx, Iyy, Izz = (
        draw(st.floats(min_value=1.0, max_value=1e10)) for _ in range(3)
    )
    I = _diag_inertia(Ixx, Iyy, Izz)
    # Keep CoG offset modest so |I_ref| still dominates m*|r|^2 and M stays PD.
    bound = float(np.sqrt(min(Ixx, Iyy, Izz) / m)) * 0.3
    r = np.array(
        [draw(st.floats(min_value=-bound, max_value=bound)) for _ in range(3)]
    )
    return m, I, r


@given(triple=_physical_input())
@settings(max_examples=40, deadline=None)
def test_property_symmetry_and_positive_definite(
    triple: tuple[float, np.ndarray, np.ndarray],
) -> None:
    m, I, r = triple
    M = rigid_body_mass_matrix(mass=m, inertia_at_reference=I, cog_offset_body=r)
    np.testing.assert_allclose(M, M.T, atol=1e-6 * abs(M).max())
    eigvals = np.linalg.eigvalsh(M)
    assert float(eigvals.min()) > 0.0
