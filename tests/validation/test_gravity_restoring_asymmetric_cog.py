"""Convention-discriminator test for the gravity restoring matrix.

Settles the ½ vs 1 question raised in
``docs/post-mortems/hydrostatic-gravity-bug.md`` §"Convention notes":
does the cross-coupling between yaw and roll/pitch carry a factor of
½ (rotation-vector V-Hessian convention) or 1 (Faltinsen 1990
Eq. 2.104, the convention used by HydroDyn / AQWA / WAMIT and
implicit in the Newton-Euler-from-first-principles derivation)?

Method
------
For an asymmetric CoG (``x_G``, ``y_G``, ``z_G`` all non-zero), apply
small single-DOF perturbations and compute the linearised gravity
moment about the body reference point from **first principles** —
gravity force ``-m·g·ẑ`` applied at the inertial-frame CoG ``R(θ)·r_G``,
with ``R(θ)`` linearised to first order. The implied stiffness column
is ``C[i, j] = -δM_i / δξ_j`` (Cummins ``Cξ`` is the negative of the
restoring moment, so ``C·ξ = -M_grav``).

Three perturbations probe distinct, complementary aspects:

- **Surge perturbation** -- gravity is translation-invariant, so
  the entire column 0 of ``ΔC_grav`` must be exactly zero. This is
  agreement-by-construction across every convention; if it fails,
  there is a bookkeeping bug.

- **Pitch perturbation** -- the diagonal ``C[4, 4] = -m·g·z_G`` is
  unambiguous (single-axis rotations agree across rotation-vector
  and Euler conventions). Probes the dominant gravity term that the
  M5 OC4 cross-check needs.

- **Yaw perturbation** -- THE convention discriminator. From
  first principles ``δM_x = -m·g·x_G·θ_z``, giving ``C[3, 5] = m·g·x_G``
  (Newton-Euler / Faltinsen). The rotation-vector V-Hessian
  alternative gives ``½·m·g·x_G``. For an asymmetric body these
  differ by exactly a factor of 2. For an axisymmetric body
  (``x_G = y_G = 0``) both conventions give zero, so the OC4 case
  in :mod:`tests.validation.test_oc4_pitch_period_buoyancy_only_c`
  cannot discriminate -- this asymmetric test is the only Phase 1
  regression that does.

A separate **symmetry** test asserts ``ΔC_grav = ΔC_grav^T`` -- the
Cummins-equation linearised stiffness must be symmetric for any
conservative restoring (so the V = ½·ξ^T·C·ξ energy interpretation
is well-defined). This rules out asymmetric "Newton-Euler-only"
implementations and complements the Faltinsen-symmetric form.

Why the asymmetric companion entries are NOT individually probed
----------------------------------------------------------------
Newton-Euler-from-roll-perturbation gives ``δM_z = 0`` (rotation
about an axis through the reference point cannot generate a moment
about that axis from gravity at the displaced CoG, since ``r cross F``
has zero z-component when ``F`` is z-directed). Naively this would
assert ``C[5, 3] = 0``, contradicting the symmetric Faltinsen form
``C[5, 3] = C[3, 5] = m·g·x_G``.

The resolution: Newton-Euler-from-one-side gives one perspective on
the linearised force; the symmetric form imposed by Cummins-equation
theory is what the integrator actually uses (and what HydroDyn,
AQWA, etc. write). The off-diagonal entries are best probed via the
column that Newton-Euler unambiguously gives (yaw → roll/pitch),
with the symmetric companion implied by ``ΔC = ΔC^T``.

Reference body
--------------
- ``m = 1.347e7 kg`` (OC4 platform mass scale)
- ``r_G = (+5.0, -3.0, -13.46) m`` -- deliberately asymmetric in
  x and y to make the cross-coupling non-trivial; ``z_G`` matches
  OC4 so the diagonal magnitudes are representative.
- ``g = 9.80665 m/s²``

Tolerance
---------
``rtol = 1e-12`` -- the gravity contribution is a closed-form
algebraic expression with no numerical noise sources beyond
IEEE-754 float multiplication. Anything looser would mask a
factor-of-2 convention bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.hydrostatics import gravity_restoring_contribution

# ---------------------------------------------------------------------------
# Reference body (asymmetric CoG, OC4-scale magnitudes).
# ---------------------------------------------------------------------------

_M_KG = 1.347e7
_X_G = 5.0
_Y_G = -3.0
_Z_G = -13.46
_R_G = np.array([_X_G, _Y_G, _Z_G])
_G = 9.80665


def _gravity_restoring() -> np.ndarray:
    """Module-under-test output for the reference body."""
    return gravity_restoring_contribution(
        mass=_M_KG,
        cog_offset_from_bem_origin=_R_G,
        gravity=_G,
    )


def _first_principles_moment_perturbation(theta_axis: int, theta_value: float) -> np.ndarray:
    """Linearised gravity moment perturbation under a single-axis rotation.

    Returns ``δM = M(θ) - M(0)`` with ``R(θ)`` truncated to first
    order in θ. ``theta_axis`` is 0/1/2 for roll/pitch/yaw.
    """
    omega = np.zeros(3, dtype=np.float64)
    omega[theta_axis] = theta_value
    delta_r_G_inertial = np.cross(omega, _R_G)
    r_G_inertial = _R_G + delta_r_G_inertial
    F_grav = np.array([0.0, 0.0, -_M_KG * _G])
    M_total = np.cross(r_G_inertial, F_grav)
    M_static = np.cross(_R_G, F_grav)
    return M_total - M_static


# ---------------------------------------------------------------------------
# Surge perturbation: gravity is translation-invariant -> column 0 is zero.
# ---------------------------------------------------------------------------


def test_surge_perturbation_no_gravity_contribution() -> None:
    """A surge displacement does not change the gravity moment about
    the (translated) body reference point. The entire surge column of
    ``ΔC_grav`` must be exactly zero, even with asymmetric CoG.
    """
    dC = _gravity_restoring()
    np.testing.assert_array_equal(
        dC[:, 0],
        np.zeros(6),
        err_msg="ΔC_grav[:, 0] (surge column) must be identically zero",
    )


# ---------------------------------------------------------------------------
# Pitch perturbation: the diagonal C[4, 4] = -m·g·z_G is unambiguous.
# ---------------------------------------------------------------------------


def test_pitch_perturbation_diagonal_term_matches_first_principles() -> None:
    """Pitch perturbation produces ``δM_y = m·g·z_G·θ_y`` to first order;
    by ``C[4, 4]·θ_y = -δM_y`` this gives ``C[4, 4] = -m·g·z_G``.

    First-principles also gives ``δM_x = δM_z = 0`` for pure pitch, but
    we do NOT assert ``C[3, 4] = C[5, 4] = 0`` here -- the symmetric
    companions of any non-zero off-diagonal entries from yaw will appear
    in column 4 too, and that's correct under the symmetric
    Cummins-equation convention. The yaw test below probes the
    non-trivial cross-couplings; symmetry is asserted separately.

    The :func:`_first_principles_moment_perturbation` helper is used as
    a derivation crosscheck (loose tolerance to absorb FD round-off);
    the tight assertion compares the module-under-test directly to the
    closed-form Faltinsen value.
    """
    dC = _gravity_restoring()
    theta = 1.0e-6

    # Sanity-check the FD derivation against the closed-form formula.
    delta_M = _first_principles_moment_perturbation(theta_axis=1, theta_value=theta)
    np.testing.assert_allclose(delta_M, [0.0, _M_KG * _G * _Z_G * theta, 0.0], rtol=1e-9, atol=1e-9)

    # Tight assertion vs the closed-form Newton-Euler / Faltinsen value.
    expected_C_44 = -_M_KG * _G * _Z_G  # = +m*g*|z_G| since z_G < 0
    assert dC[4, 4] == pytest.approx(
        expected_C_44, rel=1e-12
    ), f"C[4, 4] = {dC[4, 4]}; expected -m*g*z_G = {expected_C_44}"


# ---------------------------------------------------------------------------
# Yaw perturbation: THE convention discriminator (½ vs 1 on cross-coupling).
# ---------------------------------------------------------------------------


def test_yaw_perturbation_cross_coupling_matches_first_principles() -> None:
    """Yaw cross-coupling: ``C[3, 5] = m·g·x_G``, ``C[4, 5] = m·g·y_G``.

    First-principles: under yaw ``θ_z`` the inertial-frame CoG translates
    horizontally by ``Δr_G = (-y_G·θ_z, +x_G·θ_z, 0)``. Gravity moment
    perturbation:

        δM_x = -m·g·x_G·θ_z
        δM_y = -m·g·y_G·θ_z
        δM_z = 0    (z-rotation preserves the z-coordinate of the CoG)

    Implied column of C:

        C[3, 5] = -δM_x / θ_z = +m·g·x_G    (Newton-Euler / Faltinsen)
        C[4, 5] = -δM_y / θ_z = +m·g·y_G    (same)
        C[5, 5] = 0

    Under the rotation-vector V-Hessian convention
    (``½·m·g·x_G``, ``½·m·g·y_G``) this test FAILS by exactly a factor
    of 2 -- making it the explicit discriminator between conventions.
    The OC4 axisymmetric test cannot distinguish the two (both give
    zero cross-coupling); only the asymmetric case does.
    """
    dC = _gravity_restoring()
    theta = 1.0e-6

    # Sanity-check FD vs closed-form (loose to absorb float64 round-off
    # in the chained cross products).
    delta_M = _first_principles_moment_perturbation(theta_axis=2, theta_value=theta)
    np.testing.assert_allclose(
        delta_M,
        [-_M_KG * _G * _X_G * theta, -_M_KG * _G * _Y_G * theta, 0.0],
        rtol=1e-9,
        atol=1e-9,
    )

    # Tight assertions vs the closed-form Newton-Euler / Faltinsen values.
    expected_C_35 = _M_KG * _G * _X_G  # +m*g*x_G
    expected_C_45 = _M_KG * _G * _Y_G  # +m*g*y_G
    cog_scale = abs(_M_KG * _G * max(abs(_X_G), abs(_Y_G)))

    assert dC[3, 5] == pytest.approx(expected_C_35, rel=1e-12), (
        f"C[3, 5] = {dC[3, 5]}; first-principles +m*g*x_G = {expected_C_35}. "
        f"A factor-of-2 mismatch indicates the rotation-vector V-Hessian "
        f"convention is in use; rollback to Faltinsen Eq. 2.104 (no ½) is "
        f"required. See docs/post-mortems/hydrostatic-gravity-bug.md "
        f'§"Asymmetric CoG verification".'
    )
    assert dC[4, 5] == pytest.approx(
        expected_C_45, rel=1e-12
    ), f"C[4, 5] = {dC[4, 5]}; first-principles +m*g*y_G = {expected_C_45}"
    assert dC[5, 5] == pytest.approx(0.0, abs=1e-12 * cog_scale), (
        f"C[5, 5] = {dC[5, 5]}; gravity produces no yaw-yaw restoring "
        f"(z-rotation preserves CoG height in inertial frame)"
    )


# ---------------------------------------------------------------------------
# Symmetry: gravity is conservative, so the linearised C must be symmetric.
# ---------------------------------------------------------------------------


def test_returned_matrix_is_symmetric_under_asymmetric_cog() -> None:
    """``ΔC_grav`` must equal its transpose. Conservation of energy
    requires the linearised stiffness of any conservative force to
    admit a scalar potential ``V = ½·ξ^T·C·ξ`` -- which forces
    symmetry. The symmetric companions of the yaw cross-coupling
    (``C[5, 3] = C[3, 5]``, ``C[5, 4] = C[4, 5]``) are populated by
    this requirement; they do not arise directly from any single
    Newton-Euler perturbation but are dictated by the conservation
    structure.
    """
    dC = _gravity_restoring()
    np.testing.assert_array_equal(
        dC, dC.T, err_msg="ΔC_grav must be symmetric (gravity is conservative)"
    )
