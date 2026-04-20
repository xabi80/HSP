"""Sanity tests for the synthetic HydroDatabase builder used elsewhere in the suite."""

from __future__ import annotations

import numpy as np

from tests.support.synthetic_bem import diagonal_6x6, make_diagonal_hdb


def test_diagonal_6x6_shape_and_values() -> None:
    m = diagonal_6x6([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert m.shape == (6, 6)
    np.testing.assert_array_equal(np.diag(m), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.count_nonzero(m - np.diag(np.diag(m))) == 0


def test_make_diagonal_hdb_defaults_pass_validation() -> None:
    hdb = make_diagonal_hdb(
        A_inf_diag=[1.0, 1.0, 2.0, 10.0, 10.0, 10.0],
        C_diag=[0.0, 0.0, 4.0e5, 1.0e6, 1.0e6, 0.0],
    )
    assert hdb.n_frequencies == 5
    assert hdb.n_headings == 2
    np.testing.assert_array_equal(np.diag(hdb.A_inf), [1.0, 1.0, 2.0, 10.0, 10.0, 10.0])
    np.testing.assert_array_equal(np.diag(hdb.C), [0.0, 0.0, 4.0e5, 1.0e6, 1.0e6, 0.0])
    # A defaults to A_inf at every omega when A_diag_per_omega not given.
    for k in range(hdb.n_frequencies):
        np.testing.assert_array_equal(np.diag(hdb.A[..., k]), np.diag(hdb.A_inf))


def test_make_diagonal_hdb_custom_grids() -> None:
    hdb = make_diagonal_hdb(
        A_inf_diag=[1.0] * 6,
        C_diag=[1.0] * 6,
        omega=[0.2, 0.4, 0.6],
        heading_deg=[0.0],
        A_diag_per_omega=[[2.0] * 6, [3.0] * 6, [4.0] * 6],
        B_diag_per_omega=[[0.1] * 6, [0.2] * 6, [0.3] * 6],
    )
    assert hdb.n_frequencies == 3
    assert hdb.n_headings == 1
    np.testing.assert_allclose(np.diag(hdb.A[..., 1]), [3.0] * 6)
    np.testing.assert_allclose(np.diag(hdb.B[..., 2]), [0.3] * 6)
