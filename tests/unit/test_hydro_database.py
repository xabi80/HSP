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
        "C_source": "full",
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


# ---------- symmetrization checks (M7.5 PR2 Q1 lock) ----------
#
# Refactored from the pre-M7.5 raise-on-perturbation tests (per
# docs/audits/m7.5-reader-audit.md §Item 5 Class-C disposition):
# under Q1's HydroDatabase-level symmetrization, perturbations
# get averaged out by 0.5 * (M + M.T) BEFORE _require_symmetric
# runs. The tests now assert the new audit-trail behavior:
#   (i)  hdb.metadata["symmetrization_max_residual_*"] captures
#        the pre-symmetrization asymmetry
#   (ii) hdb.M bit-equals 0.5 * (M_input + M_input.T)
# rather than asserting a raise that no longer fires.
#
# See docs/m7.5-reader-hygiene-plan.md §Q1, §I1.


def test_asymmetric_input_symmetrized_with_residual_captured_C() -> None:
    """Perturbing C[0, 1] by delta=1.0 gets symmetrized to the average
    of the perturbed entry and its transpose partner; the pre-symmetrization
    residual `max|C - C.T|` is captured on metadata as "1.000000e+00".
    """
    kw = _valid_kwargs()
    C_input = kw["C"].copy()
    C_input[0, 1] += 1.0  # asymmetric by exactly 1.0
    kw["C"] = C_input
    hdb = HydroDatabase(**kw)
    # (i) residual reported as delta=1.0
    assert hdb.metadata["symmetrization_max_residual_C"] == f"{1.0:.6e}"
    # (ii) stored C is bit-identical to 0.5 * (C_input + C_input.T)
    expected_C = 0.5 * (C_input + C_input.T)
    np.testing.assert_array_equal(hdb.C, expected_C)


def test_asymmetric_input_symmetrized_with_residual_captured_A_inf() -> None:
    """Perturbing A_inf[2, 3] by delta=1.0: analogous to the C case."""
    kw = _valid_kwargs()
    A_inf_input = kw["A_inf"].copy()
    A_inf_input[2, 3] += 1.0
    kw["A_inf"] = A_inf_input
    hdb = HydroDatabase(**kw)
    assert hdb.metadata["symmetrization_max_residual_A_inf"] == f"{1.0:.6e}"
    expected_A_inf = 0.5 * (A_inf_input + A_inf_input.T)
    np.testing.assert_array_equal(hdb.A_inf, expected_A_inf)


def test_asymmetric_input_symmetrized_with_residual_captured_A_at_omega() -> None:
    """Perturbing A[0, 1, 2] by delta=1.0 (at omega index 2, DOF pair
    (0, 1)): residual is a scalar max across ALL omega slices AND all
    off-diagonals, so it still reports 1.0 (all other slices are
    symmetric by construction so contribute 0.0 to the max).
    """
    kw = _valid_kwargs()
    A_input = kw["A"].copy()
    A_input[0, 1, 2] += 1.0  # asymmetric at one frequency slice
    kw["A"] = A_input
    hdb = HydroDatabase(**kw)
    assert hdb.metadata["symmetrization_max_residual_A"] == f"{1.0:.6e}"
    # Only the perturbed slice is affected; all other omega slices
    # are already symmetric so their symmetrized versions are unchanged.
    A_transpose = A_input.swapaxes(0, 1)
    expected_A = 0.5 * (A_input + A_transpose)
    np.testing.assert_array_equal(hdb.A, expected_A)


# Delta values must be exactly representable in float64
# so the metadata's f"{delta:.6e}" format string matches
# the computed max|M - M.T| residual bit-exactly. The
# chosen values 1e-4, 1e-8, and 1.0 all satisfy this.
# Non-representable values (e.g., 0.1) would produce a
# residual that differs from the formatted delta by
# accumulated float error.
#
# Note on base-value control: the test explicitly zeros
# C[0, 1] and C[1, 0] before applying delta. This gives a
# clean base of 0.0 where (0.0 + delta) - 0.0 == delta
# bit-exactly for any float64-representable delta. Without
# this control, _valid_kwargs()'s randomised C entries at
# |base|~O(1e2) push ULP noise (~1e-14) into (base + 1e-8)
# - base, giving 9.999994e-09 instead of 1.000000e-08 and
# breaking the 6th-decimal exact-string match. The
# metadata reporting IS bit-correct in either case; the
# zeroing here is a test-hygiene step to make the assertion
# express what it's supposed to test (the format-string
# roundtrip), not to hide any real issue with the code
# under test.
@pytest.mark.parametrize("delta", [1.0e-4, 1.0e-8, 1.0])
def test_symmetrization_residual_records_delta_across_magnitudes(delta: float) -> None:
    """Metadata residual reporting must be quantitatively correct
    across the panel-method-noise (~1e-4), near-float-precision
    (~1e-8), and egregious (1.0) magnitudes.
    """
    kw = _valid_kwargs()
    C_input = kw["C"].copy()
    # Zero out C[0, 1] / C[1, 0] first so the perturbation sees a
    # clean base of 0.0 (see comment above the parametrize block).
    C_input[0, 1] = 0.0
    C_input[1, 0] = 0.0
    # Now apply the perturbation from the clean base.
    C_input[0, 1] = delta
    kw["C"] = C_input
    hdb = HydroDatabase(**kw)
    # (i) residual exact-string match to the perturbation magnitude
    assert hdb.metadata["symmetrization_max_residual_C"] == f"{delta:.6e}"
    # (ii) stored C[0, 1] is the average of the perturbed value
    # (delta) and its transpose partner (0.0): 0.5 * delta.
    assert hdb.C[0, 1] == 0.5 * delta
    # And the symmetric partner C[1, 0] equals the same average.
    assert hdb.C[1, 0] == hdb.C[0, 1]


def test_symmetric_input_preserved_bit_identical() -> None:
    """Q1 idempotency: symmetrization on already-symmetric input is a
    bit-exact no-op. Residual is 0.0; stored arrays are bit-identical
    to input arrays (via 0.5 * (M + M) = M for IEEE 754 finite floats).
    """
    kw = _valid_kwargs()
    # _valid_kwargs() already builds A, B, A_inf, C via _sym6 which
    # constructs them as 0.5 * (m + m.T) -- bit-exactly symmetric.
    C_input = kw["C"].copy()
    A_inf_input = kw["A_inf"].copy()
    A_input = kw["A"].copy()
    B_input = kw["B"].copy()
    hdb = HydroDatabase(**kw)
    # (i) All four residuals report exact zero.
    assert hdb.metadata["symmetrization_max_residual_A"] == f"{0.0:.6e}"
    assert hdb.metadata["symmetrization_max_residual_B"] == f"{0.0:.6e}"
    assert hdb.metadata["symmetrization_max_residual_A_inf"] == f"{0.0:.6e}"
    assert hdb.metadata["symmetrization_max_residual_C"] == f"{0.0:.6e}"
    # (ii) Stored arrays bit-identical to input arrays.
    np.testing.assert_array_equal(hdb.C, C_input)
    np.testing.assert_array_equal(hdb.A_inf, A_inf_input)
    np.testing.assert_array_equal(hdb.A, A_input)
    np.testing.assert_array_equal(hdb.B, B_input)


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


# ---------- M8 PR1: N-body extension (body_labels) ----------
#
# Q1 lock (docs/m8-coupled-bem-plan.md): HydroDatabase gains
# `body_labels: tuple[str, ...] | None = None`.
#   - None      -> LEGACY path, shapes (6,6,n_w) / (6,6). Existing code
#                  untouched; bit-identical by construction.
#   - provided  -> N = len(body_labels), shapes (6N,6N,n_w) / (6N,6N).
# `n_bodies` derives from LABELS, never from shape arithmetic.


def _valid_kwargs_nbody(n_bodies: int = 3, n_w: int = 5, n_h: int = 2) -> dict:
    """Valid constructor args for an N-body (6N x 6N) database."""
    rng = np.random.default_rng(1)
    nd = 6 * n_bodies
    omega = np.linspace(0.1, 3.0, n_w)
    heading = np.linspace(0.0, 180.0, n_h)

    def _sym(scale: float) -> np.ndarray:
        m = rng.standard_normal((nd, nd)) * scale
        return 0.5 * (m + m.T)

    A = np.stack([_sym(1.0) + 10.0 * np.eye(nd) for _ in range(n_w)], axis=-1)
    B = np.stack([_sym(0.1) + 1.0 * np.eye(nd) for _ in range(n_w)], axis=-1)
    return {
        "omega": omega,
        "heading_deg": heading,
        "A": A,
        "B": B,
        "A_inf": _sym(1.0) + 10.0 * np.eye(nd),
        "C": _sym(100.0) + 1.0e3 * np.eye(nd),
        "RAO": rng.standard_normal((nd, n_w, n_h)) + 1j * rng.standard_normal((nd, n_w, n_h)),
        "reference_point": np.zeros(3),
        "C_source": "full",
        "metadata": {"source": "test-nbody"},
        "body_labels": tuple(f"buoy{i + 1}" for i in range(n_bodies)),
    }


def test_nbody_construction_accepts_6N_shapes() -> None:
    kw = _valid_kwargs_nbody(n_bodies=3)
    hdb = HydroDatabase(**kw)
    assert hdb.n_bodies == 3
    assert hdb.A.shape == (18, 18, 5)
    assert hdb.A_inf.shape == (18, 18)
    assert hdb.RAO.shape == (18, 5, 2)
    assert hdb.body_labels == ("buoy1", "buoy2", "buoy3")


def test_n_bodies_derives_from_labels_not_shape() -> None:
    """n_bodies must read len(body_labels), never infer from array shape."""
    kw = _valid_kwargs_nbody(n_bodies=2)
    hdb = HydroDatabase(**kw)
    assert hdb.n_bodies == 2
    assert len(hdb.body_labels) == 2


def test_legacy_path_has_none_labels_and_one_body() -> None:
    """No labels -> legacy single-body database."""
    hdb = HydroDatabase(**_valid_kwargs())
    assert hdb.body_labels is None
    assert hdb.n_bodies == 1
    assert hdb.A.shape == (6, 6, 5)


def test_legacy_path_still_rejects_non_6x6() -> None:
    """Without labels the (6,6,n_w) requirement is unchanged."""
    kw = _valid_kwargs_nbody(n_bodies=2)
    kw.pop("body_labels")
    with pytest.raises(ValueError, match=r"A must have shape \(6, 6, "):
        HydroDatabase(**kw)


def test_label_count_must_match_array_shape() -> None:
    kw = _valid_kwargs_nbody(n_bodies=3)
    kw["body_labels"] = ("buoy1", "buoy2")  # 2 labels vs 18x18 arrays
    with pytest.raises(ValueError, match="A must have shape"):
        HydroDatabase(**kw)


def test_empty_body_labels_rejected() -> None:
    kw = _valid_kwargs_nbody(n_bodies=1)
    kw["body_labels"] = ()
    with pytest.raises(ValueError, match="body_labels must be non-empty"):
        HydroDatabase(**kw)


def test_duplicate_body_labels_rejected() -> None:
    kw = _valid_kwargs_nbody(n_bodies=2)
    kw["body_labels"] = ("buoy1", "buoy1")
    with pytest.raises(ValueError, match="body_labels must be unique"):
        HydroDatabase(**kw)


def test_nbody_full_matrix_symmetrization_records_residuals() -> None:
    """Full-matrix symmetrization generalizes to 6N x 6N (program Q2)."""
    kw = _valid_kwargs_nbody(n_bodies=3)
    kw["A"] = kw["A"].copy()
    kw["A"][0, 5, :] += 1.0e-3  # inject a known asymmetry
    hdb = HydroDatabase(**kw)
    resid = float(hdb.metadata["symmetrization_max_residual_A"])
    assert resid == pytest.approx(1.0e-3, rel=1e-6)
    for k in range(hdb.n_frequencies):  # symmetric post-construction
        np.testing.assert_allclose(hdb.A[..., k], hdb.A[..., k].T, atol=1e-12)


def test_nbody_single_body_with_label_is_n1() -> None:
    """N=1 WITH a label is legal and reports one body (Q2 detection case)."""
    kw = _valid_kwargs_nbody(n_bodies=1)
    hdb = HydroDatabase(**kw)
    assert hdb.n_bodies == 1
    assert hdb.body_labels == ("buoy1",)
    assert hdb.A.shape == (6, 6, 5)
