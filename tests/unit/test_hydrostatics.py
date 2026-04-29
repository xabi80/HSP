"""Unit tests for :mod:`floatsim.hydro.hydrostatics`.

Focus: the gravity contribution to ``C`` is computed correctly for both
the dominant diagonal terms (``-m·g·z_G`` on roll-roll and pitch-pitch)
AND the cross-coupling terms that arise when the CoG is offset from the
BEM hydrostatic origin in the body's x-y plane.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.hydrostatics import gravity_restoring_contribution

# ---------------------------------------------------------------------------
# Hand-derived reference: axisymmetric (x_G = y_G = 0)
# ---------------------------------------------------------------------------


def test_axisymmetric_only_diagonal_terms_populated() -> None:
    """For a body with CoG directly below the BEM origin (x_G = y_G = 0),
    only the diagonal C[3,3] and C[4,4] entries are non-zero, with value
    ``-m·g·z_G``. All other entries vanish.
    """
    m, g, z_G = 1.0e6, 9.80665, -10.0
    dC = gravity_restoring_contribution(
        mass=m,
        cog_offset_from_bem_origin=np.array([0.0, 0.0, z_G]),
        gravity=g,
    )
    assert dC.shape == (6, 6)
    expected_diag = -m * g * z_G  # = +9.80665e7 since z_G is negative
    assert dC[3, 3] == pytest.approx(expected_diag, rel=1e-12)
    assert dC[4, 4] == pytest.approx(expected_diag, rel=1e-12)
    # Everything else zero.
    mask = np.ones((6, 6), dtype=bool)
    mask[3, 3] = False
    mask[4, 4] = False
    assert np.all(dC[mask] == 0.0), (
        f"unexpected non-zero entries for axisymmetric CoG: " f"{dC[mask][np.nonzero(dC[mask])]}"
    )


# ---------------------------------------------------------------------------
# Hand-derived reference: offset CoG (x_G, y_G non-zero) — cross-coupling
# ---------------------------------------------------------------------------


def test_offset_cog_produces_hand_derived_cross_coupling() -> None:
    """Hand-computed gravity restoring contribution for a non-symmetric CoG.

    Setup (round numbers chosen so the bookkeeping is unambiguous):
        m = 1000 kg,  g = 10 m/s^2  (so m*g = 1e4)
        r_G = (x_G, y_G, z_G) = (+2, -3, -4) m

    Per the rotation-vector convention derivation in
    ``floatsim.hydro.hydrostatics`` module docstring:

        ΔC[3, 3] = -m·g·z_G   = -1e4 · (-4)     = +4e4
        ΔC[4, 4] = -m·g·z_G   = -1e4 · (-4)     = +4e4
        ΔC[3, 5] = +½·m·g·x_G = +0.5 · 1e4 · 2  = +1e4
        ΔC[5, 3] = +½·m·g·x_G = +1e4   (symmetric)
        ΔC[4, 5] = +½·m·g·y_G = +0.5 · 1e4 · -3 = -1.5e4
        ΔC[5, 4] = +½·m·g·y_G = -1.5e4   (symmetric)
        all others = 0

    This test pins down the convention: any change to the gravity
    formula (e.g. switching to the Faltinsen/Euler m·g·x_G off-diagonal
    without ½) will trip this assertion and force re-evaluation of the
    parameterisation choice.
    """
    m, g = 1000.0, 10.0
    x_G, y_G, z_G = 2.0, -3.0, -4.0

    expected = np.zeros((6, 6))
    expected[3, 3] = -m * g * z_G  # +40_000
    expected[4, 4] = -m * g * z_G  # +40_000
    expected[3, 5] = 0.5 * m * g * x_G  # +10_000
    expected[5, 3] = 0.5 * m * g * x_G
    expected[4, 5] = 0.5 * m * g * y_G  # -15_000
    expected[5, 4] = 0.5 * m * g * y_G

    dC = gravity_restoring_contribution(
        mass=m,
        cog_offset_from_bem_origin=np.array([x_G, y_G, z_G]),
        gravity=g,
    )
    np.testing.assert_array_equal(dC, expected)


def test_returned_matrix_is_symmetric() -> None:
    """Restoring matrices must be symmetric (gravity is conservative)."""
    dC = gravity_restoring_contribution(
        mass=2.5e6,
        cog_offset_from_bem_origin=np.array([1.5, -0.7, -8.3]),
        gravity=9.80665,
    )
    np.testing.assert_array_equal(dC, dC.T)


def test_translation_block_and_yaw_diagonal_are_zero() -> None:
    """Gravity does not contribute to translation restoring (gravity is
    constant under translation) nor to yaw restoring (rotation about the
    vertical does not change CoG height in inertial frame)."""
    dC = gravity_restoring_contribution(
        mass=1.0e6,
        cog_offset_from_bem_origin=np.array([0.5, -0.5, -5.0]),
        gravity=9.80665,
    )
    # Translation block: rows/cols 0-2.
    assert np.all(dC[:3, :] == 0.0)
    assert np.all(dC[:, :3] == 0.0)
    # Yaw-yaw diagonal: gravity is invariant under yaw.
    assert dC[5, 5] == 0.0


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_zero_mass_raises() -> None:
    with pytest.raises(ValueError, match=r"mass must be finite and positive"):
        gravity_restoring_contribution(
            mass=0.0,
            cog_offset_from_bem_origin=np.zeros(3),
            gravity=9.80665,
        )


def test_negative_gravity_raises() -> None:
    with pytest.raises(ValueError, match=r"gravity must be finite and positive"):
        gravity_restoring_contribution(
            mass=1.0,
            cog_offset_from_bem_origin=np.zeros(3),
            gravity=-9.80665,
        )


def test_wrong_cog_shape_raises() -> None:
    with pytest.raises(ValueError, match=r"cog_offset_from_bem_origin must have shape"):
        gravity_restoring_contribution(
            mass=1.0,
            cog_offset_from_bem_origin=np.zeros(4),
            gravity=9.80665,
        )


def test_nonfinite_cog_raises() -> None:
    with pytest.raises(ValueError, match=r"cog_offset_from_bem_origin must be all-finite"):
        gravity_restoring_contribution(
            mass=1.0,
            cog_offset_from_bem_origin=np.array([0.0, np.nan, 0.0]),
            gravity=9.80665,
        )
