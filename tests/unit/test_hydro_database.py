"""HydroDatabase dataclass — ARCHITECTURE.md §2, §3.3, §6.2.

Pluggable-reader contract: every BEM reader produces this same dataclass.
These tests encode the invariants that downstream code (Cummins assembly,
RAO evaluation) is allowed to assume.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from floatsim.hydro.database import DOF_ORDER, HydroDatabase

# ---------- fixture helpers ----------


def _valid_kwargs(n_w: int = 5, n_h: int = 2) -> dict:
    """Construct a minimal, valid set of constructor arguments.

    Values are physically benign (not OC4 DeepCwind) — purpose is to pass
    validation, not to model any real body.
    """
    rng = np.random.default_rng(0)
    omega = np.linspace(0.1, 3.0, n_w)
    heading = np.linspace(0.0, 180.0, n_h)

    # Symmetric A, B at each frequency (6x6).
    def _sym6(scale: float) -> np.ndarray:
        m = rng.standard_normal((6, 6)) * scale
        return 0.5 * (m + m.T)

    A = np.stack([_sym6(1.0) + 10.0 * np.eye(6) for _ in range(n_w)], axis=-1)
    B = np.stack([_sym6(0.1) + 1.0 * np.eye(6) for _ in range(n_w)], axis=-1)
    A_inf = _sym6(1.0) + 10.0 * np.eye(6)
    C = _sym6(100.0) + 1.0e3 * np.eye(6)
    RAO = rng.standard_normal((6, n_w, n_h)) + 1j * rng.standard_normal((6, n_w, n_h))

    return {
        "omega": omega,
        "heading_deg": heading,
        "A": A,
        "B": B,
        "A_inf": A_inf,
        "C": C,
        "RAO": RAO,
        "reference_point": np.zeros(3),
        "metadata": {"source": "test"},
    }


# ---------- happy path ----------


def test_valid_construction() -> None:
    kw = _valid_kwargs()
    hdb = HydroDatabase(**kw)
    assert hdb.n_frequencies == 5
    assert hdb.n_headings == 2
    assert hdb.A.shape == (6, 6, 5)
    assert hdb.RAO.dtype == np.complex128
    assert tuple(hdb.dof_order) == DOF_ORDER


def test_dof_order_is_surge_first_yaw_last() -> None:
    assert DOF_ORDER == ("surge", "sway", "heave", "roll", "pitch", "yaw")


# ---------- shape checks ----------


@pytest.mark.parametrize(
    "field,bad_shape",
    [
        ("A", (6, 6, 4)),  # wrong n_w
        ("B", (6, 6, 4)),
        ("A_inf", (5, 6)),
        ("C", (5, 5)),
        ("RAO", (6, 5, 3)),  # wrong n_h
        ("reference_point", (4,)),
    ],
)
def test_wrong_shape_rejected(field: str, bad_shape: tuple[int, ...]) -> None:
    kw = _valid_kwargs()
    dtype = complex if field == "RAO" else float
    kw[field] = np.zeros(bad_shape, dtype=dtype)
    with pytest.raises(ValueError, match=field):
        HydroDatabase(**kw)


# ---------- symmetry checks ----------


def test_non_symmetric_C_rejected() -> None:
    kw = _valid_kwargs()
    kw["C"][0, 1] += 1.0  # break symmetry by a lot
    with pytest.raises(ValueError, match="symmetric"):
        HydroDatabase(**kw)


def test_non_symmetric_Ainf_rejected() -> None:
    kw = _valid_kwargs()
    kw["A_inf"][2, 3] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        HydroDatabase(**kw)


def test_non_symmetric_A_at_any_omega_rejected() -> None:
    kw = _valid_kwargs()
    kw["A"][0, 1, 2] += 1.0  # break symmetry at one frequency slice
    with pytest.raises(ValueError, match="symmetric"):
        HydroDatabase(**kw)


# ---------- monotonicity / sign / finiteness checks ----------


def test_omega_must_be_monotonically_increasing() -> None:
    kw = _valid_kwargs()
    kw["omega"] = np.array([0.1, 0.5, 0.3, 1.0, 2.0])
    with pytest.raises(ValueError, match="monotonic"):
        HydroDatabase(**kw)


def test_negative_omega_rejected() -> None:
    kw = _valid_kwargs()
    kw["omega"] = np.linspace(-0.5, 2.0, 5)
    with pytest.raises(ValueError, match="non-negative"):
        HydroDatabase(**kw)


def test_nan_values_rejected() -> None:
    kw = _valid_kwargs()
    kw["A"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        HydroDatabase(**kw)


def test_inf_values_rejected() -> None:
    kw = _valid_kwargs()
    kw["C"][0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        HydroDatabase(**kw)


# ---------- dtype checks ----------


def test_rao_must_be_complex() -> None:
    kw = _valid_kwargs()
    kw["RAO"] = np.real(kw["RAO"]).astype(np.float64)
    with pytest.raises(ValueError, match="complex"):
        HydroDatabase(**kw)


# ---------- dimensional consistency ----------


def test_omega_length_must_be_at_least_two() -> None:
    """A single-point omega grid is degenerate — radiation convolution needs a band."""
    kw = _valid_kwargs(n_w=1)
    with pytest.raises(ValueError, match="at least 2"):
        HydroDatabase(**kw)


# ---------- property-based ----------


@st.composite
def _symmetric_6x6(draw: st.DrawFn) -> np.ndarray:
    vals = draw(
        st.lists(
            st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
            min_size=21,
            max_size=21,
        )
    )
    m = np.zeros((6, 6))
    idx = 0
    for i in range(6):
        for j in range(i, 6):
            m[i, j] = vals[idx]
            m[j, i] = vals[idx]
            idx += 1
    return m


@given(C=_symmetric_6x6(), Ainf=_symmetric_6x6())
@settings(max_examples=25, deadline=None)
def test_symmetric_C_and_Ainf_always_accepted(C: np.ndarray, Ainf: np.ndarray) -> None:
    kw = _valid_kwargs()
    kw["C"] = C
    kw["A_inf"] = Ainf
    hdb = HydroDatabase(**kw)
    np.testing.assert_allclose(hdb.C, C)
    np.testing.assert_allclose(hdb.A_inf, Ainf)
