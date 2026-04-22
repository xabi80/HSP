"""Unit tests for the half-cosine excitation ramp — ARCHITECTURE.md §9.3."""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.solver.ramp import HalfCosineRamp

# ---------------------------------------------------------------------------
# boundary values and clamp behavior
# ---------------------------------------------------------------------------


def test_value_is_zero_at_origin() -> None:
    ramp = HalfCosineRamp(duration=20.0)
    assert ramp.value(0.0) == 0.0


def test_value_is_one_at_and_after_duration() -> None:
    ramp = HalfCosineRamp(duration=20.0)
    assert ramp.value(20.0) == 1.0
    assert ramp.value(20.001) == 1.0
    assert ramp.value(1.0e6) == 1.0


def test_value_is_zero_for_negative_time() -> None:
    ramp = HalfCosineRamp(duration=20.0)
    assert ramp.value(-1.0) == 0.0


def test_half_point_is_exactly_half() -> None:
    """r(T_ramp/2) = 0.5 * (1 - cos(pi/2)) = 0.5."""
    ramp = HalfCosineRamp(duration=20.0)
    assert ramp.value(10.0) == pytest.approx(0.5, rel=1e-15)


# ---------------------------------------------------------------------------
# shape: formula, monotonicity, smooth endpoints
# ---------------------------------------------------------------------------


def test_matches_formula_on_dense_grid() -> None:
    ramp = HalfCosineRamp(duration=20.0)
    t = np.linspace(0.0, 20.0, 201)
    expected = 0.5 * (1.0 - np.cos(np.pi * t / 20.0))
    got = ramp(t)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_monotonically_increasing_on_ramp_interval() -> None:
    ramp = HalfCosineRamp(duration=15.0)
    t = np.linspace(0.0, 15.0, 401)
    r = ramp(t)
    assert np.all(np.diff(r) >= 0.0)


def test_derivative_vanishes_at_both_endpoints() -> None:
    """Finite-difference derivative must vanish at t=0 and t=T_ramp (smooth kick-on)."""
    ramp = HalfCosineRamp(duration=20.0)
    h = 1.0e-4
    r0 = ramp.value(h) - ramp.value(0.0)
    r1 = ramp.value(20.0) - ramp.value(20.0 - h)
    # derivative is O(h^2) near endpoints of a half-cosine, so it should be
    # well below h itself.
    assert abs(r0) < h
    assert abs(r1) < h


# ---------------------------------------------------------------------------
# vectorization
# ---------------------------------------------------------------------------


def test_call_accepts_scalar_and_matches_value() -> None:
    ramp = HalfCosineRamp(duration=10.0)
    for t in [-1.0, 0.0, 3.5, 10.0, 25.0]:
        assert ramp(t) == pytest.approx(ramp.value(t), rel=1e-15)


def test_call_vectorizes_over_ndarray() -> None:
    ramp = HalfCosineRamp(duration=5.0)
    t = np.array([-0.5, 0.0, 1.0, 2.5, 4.0, 5.0, 5.5, 100.0])
    r = ramp(t)
    assert r.shape == t.shape
    assert r[0] == 0.0  # negative
    assert r[1] == 0.0  # t=0
    assert r[2] == pytest.approx(0.5 * (1 - np.cos(np.pi * 1.0 / 5.0)))
    assert r[3] == pytest.approx(0.5, rel=1e-15)  # midpoint
    assert r[5] == 1.0  # t = duration
    assert r[-1] == 1.0  # long after duration


# ---------------------------------------------------------------------------
# zero-duration (ramp disabled)
# ---------------------------------------------------------------------------


def test_zero_duration_is_step_at_origin() -> None:
    ramp = HalfCosineRamp(duration=0.0)
    assert ramp.value(0.0) == 0.0
    assert ramp.value(1.0e-12) == 1.0
    assert ramp.value(10.0) == 1.0


def test_zero_duration_vectorizes() -> None:
    ramp = HalfCosineRamp(duration=0.0)
    t = np.array([-1.0, 0.0, 1.0e-9, 5.0])
    r = ramp(t)
    np.testing.assert_array_equal(r, [0.0, 0.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------


def test_negative_duration_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        HalfCosineRamp(duration=-1.0)
