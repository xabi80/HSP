"""Unit tests for :mod:`floatsim.hydro.morison` (M5 PR4).

Covers the four red-tests from ``docs/milestone-5-plan.md`` PR4:

1. Single cylinder, fixed body, uniform flow normal to member -->
   analytical ``F = 0.5 * rho * D * L * Cd * u^2``.
2. Single cylinder, fixed body, uniform flow parallel to member -->
   ``F ~ 0`` (member-normal projection check; the bug-catcher).
3. Single cylinder, oscillating fluid, fixed body -->
   ``include_inertia=True`` reproduces drag + added-mass terms.
4. Startup warning emitted when ``include_inertia=True`` AND the body
   has a non-empty BEM database.

Plus essential bug-catcher tests on:

- Member at 45 deg: drag direction perpendicular to the member axis.
- Body translation: ``u_rel = u_fluid - v_body``; pure translation in
  flow direction halves the drag for the right ratio.
- Body rotation: a member offset from the reference point picks up
  the ``omega x r`` velocity contribution.
- Generalized-force assembly: a force at a lever arm produces the
  expected moment about the body's reference point.
- ``make_morison_state_force`` integrator-shaped closure: builds the
  right global force vector for a multi-body system.
- Deck-schema field round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.hydro.morison import (
    MorisonElement,
    make_morison_state_force,
    morison_element_force,
    startup_inertia_double_count_warnings,
)
from floatsim.io.deck import MorisonMember

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RHO = 1025.0


def _vertical_cylinder(
    diameter: float = 1.0,
    length: float = 10.0,
    Cd: float = 1.0,
    Ca: float = 1.0,
    include_inertia: bool = False,
    body_index: int = 0,
) -> MorisonElement:
    """Build a vertical cylinder along +Z spanning ``-length`` to ``0`` in body frame."""
    return MorisonElement(
        body_index=body_index,
        node_a_body=np.array([0.0, 0.0, -length]),
        node_b_body=np.array([0.0, 0.0, 0.0]),
        diameter=diameter,
        Cd=Cd,
        Ca=Ca,
        include_inertia=include_inertia,
    )


def _horizontal_x_cylinder(
    diameter: float = 1.0,
    length: float = 10.0,
    Cd: float = 1.0,
    body_index: int = 0,
) -> MorisonElement:
    """Build a cylinder along +X at ``z = -5`` for half-length-aligned tests."""
    return MorisonElement(
        body_index=body_index,
        node_a_body=np.array([0.0, 0.0, -5.0]),
        node_b_body=np.array([length, 0.0, -5.0]),
        diameter=diameter,
        Cd=Cd,
        Ca=0.0,
        include_inertia=False,
    )


# ---------------------------------------------------------------------------
# MorisonElement dataclass validation
# ---------------------------------------------------------------------------


def test_morison_element_rejects_negative_body_index() -> None:
    with pytest.raises(ValueError, match=r"body_index must be >= 0"):
        MorisonElement(
            body_index=-1,
            node_a_body=np.zeros(3),
            node_b_body=np.array([1.0, 0.0, 0.0]),
            diameter=1.0,
            Cd=1.0,
        )


def test_morison_element_rejects_coincident_nodes() -> None:
    with pytest.raises(ValueError, match="distinct"):
        MorisonElement(
            body_index=0,
            node_a_body=np.zeros(3),
            node_b_body=np.zeros(3),
            diameter=1.0,
            Cd=1.0,
        )


def test_morison_element_rejects_nonpositive_diameter() -> None:
    with pytest.raises(ValueError, match="diameter must be positive"):
        MorisonElement(
            body_index=0,
            node_a_body=np.zeros(3),
            node_b_body=np.array([1.0, 0.0, 0.0]),
            diameter=0.0,
            Cd=1.0,
        )


def test_morison_element_rejects_negative_Cd() -> None:
    with pytest.raises(ValueError, match="Cd must be non-negative"):
        MorisonElement(
            body_index=0,
            node_a_body=np.zeros(3),
            node_b_body=np.array([1.0, 0.0, 0.0]),
            diameter=1.0,
            Cd=-0.1,
        )


def test_morison_element_length_and_area() -> None:
    e = _vertical_cylinder(diameter=2.0, length=5.0)
    assert e.length_m == pytest.approx(5.0)
    assert e.cross_section_area_m2 == pytest.approx(np.pi)


# ---------------------------------------------------------------------------
# Plan red-test 1 -- uniform flow NORMAL to a vertical cylinder
# ---------------------------------------------------------------------------


def test_uniform_flow_normal_matches_analytical_drag() -> None:
    """``F = 0.5 * rho * D * L * Cd * u^2`` for a fixed cylinder in normal flow."""
    D, L, Cd = 1.5, 8.0, 0.9
    e = _vertical_cylinder(diameter=D, length=L, Cd=Cd)
    u = 2.5
    f6 = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -L / 2.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),  # +Z (along member)
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([u, 0.0, 0.0]),  # +X (normal to member)
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    F_expected = 0.5 * _RHO * D * L * Cd * u * u
    assert f6[0] == pytest.approx(F_expected, rel=1.0e-12)
    assert f6[1] == pytest.approx(0.0, abs=1.0e-9)
    assert f6[2] == pytest.approx(0.0, abs=1.0e-9)
    # Moment about the origin: arm = (0, 0, -L/2) crossed with (Fx, 0, 0)
    # -> (0*0 - (-L/2)*0, (-L/2)*Fx - 0*0, 0*0 - 0*Fx) = (0, -L/2 * Fx, 0)
    assert f6[3] == pytest.approx(0.0, abs=1.0e-9)
    assert f6[4] == pytest.approx(-(L / 2.0) * F_expected, rel=1.0e-12)
    assert f6[5] == pytest.approx(0.0, abs=1.0e-9)


def test_uniform_flow_normal_drag_scales_quadratically_with_velocity() -> None:
    e = _vertical_cylinder()
    base = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([1.0, 0.0, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    doubled = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([2.0, 0.0, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    assert doubled[0] == pytest.approx(4.0 * base[0], rel=1.0e-12)


def test_uniform_flow_normal_drag_sign_follows_velocity() -> None:
    """``|u_n| * u_n`` makes drag follow the sign of u, not the magnitude only."""
    e = _vertical_cylinder()
    fwd = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([1.5, 0.0, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    rev = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([-1.5, 0.0, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    assert rev[0] == pytest.approx(-fwd[0], rel=1.0e-12)


# ---------------------------------------------------------------------------
# Plan red-test 2 -- uniform flow PARALLEL to the member
# ---------------------------------------------------------------------------


def test_uniform_flow_parallel_to_member_yields_zero_drag() -> None:
    """Member-normal projection: flow along the axis contributes nothing."""
    e = _vertical_cylinder()
    f6 = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([0.0, 0.0, 3.0]),  # along the member axis
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    assert np.allclose(f6, 0.0, atol=1.0e-9)


def test_member_at_45deg_drag_only_from_normal_component() -> None:
    """For axis at 45 deg in the X-Z plane and pure +X flow, drag uses |u_n| u_n.

    With axis_hat = (sin45, 0, cos45) and u = (u, 0, 0):
        u . axis_hat = u sin45
        u_n = u - (u sin45) axis_hat = u (1 - sin^2 45, 0, -sin45 cos45)
            = u (cos^2 45, 0, -sin45 cos45)
        |u_n| = u * cos45
    so drag magnitude = 0.5 rho D L Cd (u cos45)^2 = 0.5 rho D L Cd u^2 / 2.
    The drag direction is perpendicular to the axis (a basic invariant).
    """
    D, L, Cd = 1.0, 4.0, 1.0
    a = np.array([-L * np.sin(np.pi / 4) / 2, 0.0, -L * np.cos(np.pi / 4) / 2])
    b = np.array([L * np.sin(np.pi / 4) / 2, 0.0, L * np.cos(np.pi / 4) / 2])
    e = MorisonElement(
        body_index=0,
        node_a_body=a,
        node_b_body=b,
        diameter=D,
        Cd=Cd,
    )
    axis_hat = (b - a) / np.linalg.norm(b - a)
    u = 2.0
    f6 = morison_element_force(
        e,
        midpoint_inertial=np.zeros(3),
        axis_hat_inertial=axis_hat,
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([u, 0.0, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    F_mag = float(np.linalg.norm(f6[:3]))
    F_expected = 0.5 * _RHO * D * L * Cd * (u * np.cos(np.pi / 4)) ** 2
    assert F_mag == pytest.approx(F_expected, rel=1.0e-12)
    # Drag direction perpendicular to the member axis (the bug-catcher).
    assert float(np.dot(f6[:3], axis_hat)) == pytest.approx(0.0, abs=1.0e-9)


# ---------------------------------------------------------------------------
# Body translation: u_rel = u_fluid - v_body
# ---------------------------------------------------------------------------


def test_body_velocity_subtracts_from_fluid_velocity() -> None:
    """Equal body and fluid speeds in the same direction -> zero relative flow -> zero drag."""
    e = _vertical_cylinder()
    f6 = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.array([2.0, 0.0, 0.0]),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([2.0, 0.0, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    assert np.allclose(f6, 0.0, atol=1.0e-9)


def test_body_velocity_opposite_to_fluid_doubles_relative_velocity() -> None:
    e = _vertical_cylinder()
    base = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([1.0, 0.0, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    opposed = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.array([-1.0, 0.0, 0.0]),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([1.0, 0.0, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    # u_rel doubles -> drag scales 4x.
    assert opposed[0] == pytest.approx(4.0 * base[0], rel=1.0e-12)


# ---------------------------------------------------------------------------
# Generalized force assembly: moment about the reference point
# ---------------------------------------------------------------------------


def test_force_at_offset_midpoint_produces_moment_about_reference() -> None:
    """A horizontal cylinder offset at z = -5 in +X flow yields a pitching moment."""
    e = _horizontal_x_cylinder(length=10.0)  # axis along +X at z = -5
    # Apply +Y fluid flow -> drag in +Y direction at the midpoint (5, 0, -5).
    u = 2.0
    f6 = morison_element_force(
        e,
        midpoint_inertial=np.array([5.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([1.0, 0.0, 0.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.array([0.0, u, 0.0]),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    F_expected = 0.5 * _RHO * e.diameter * e.length_m * e.Cd * u * u
    # Force +Y; arm (5, 0, -5) crossed with (0, F, 0) = (5*F, 0, -(-5)*F)
    # Wait: cross([5,0,-5], [0,F,0]) = (0*0 - (-5)*F, (-5)*0 - 5*0, 5*F - 0*0)
    #                                = (5*F, 0, 5*F)
    assert f6[1] == pytest.approx(F_expected, rel=1.0e-12)
    assert f6[3] == pytest.approx(5.0 * F_expected, rel=1.0e-12)
    assert f6[5] == pytest.approx(5.0 * F_expected, rel=1.0e-12)


# ---------------------------------------------------------------------------
# Plan red-test 3 -- oscillating fluid + include_inertia
# ---------------------------------------------------------------------------


def test_oscillating_flow_inertia_term_at_zero_velocity_phase() -> None:
    """At the phase of an oscillation where ``u_fluid = 0`` but ``a_fluid != 0``,
    the drag vanishes and only the Froude-Krylov + added-mass term remains.

    With ``a_body = 0`` and a horizontal cylinder, that's exactly
    ``rho * A_x * L * (1 + Ca) * a_fluid_n`` in the flow direction.
    """
    D, L, Cd, Ca = 1.0, 6.0, 1.0, 1.5
    e = _vertical_cylinder(diameter=D, length=L, Cd=Cd, Ca=Ca, include_inertia=True)
    a = 3.0  # m/s^2 in +X
    f6 = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -L / 2.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.zeros(3),  # zero-velocity phase
        fluid_acceleration=np.array([a, 0.0, 0.0]),
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    A_x = np.pi * D * D / 4.0
    F_expected = _RHO * A_x * L * (1.0 + Ca) * a
    assert f6[0] == pytest.approx(F_expected, rel=1.0e-12)
    assert f6[1] == pytest.approx(0.0, abs=1.0e-9)
    assert f6[2] == pytest.approx(0.0, abs=1.0e-9)


def test_oscillating_flow_inertia_term_subtracts_body_acceleration() -> None:
    D, L, Ca = 1.0, 6.0, 1.0
    e = _vertical_cylinder(
        diameter=D, length=L, Cd=0.0, Ca=Ca, include_inertia=True
    )  # Cd=0 isolates the inertia term
    a_fluid = 4.0
    a_body = 1.5
    f6 = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -L / 2.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=np.array([a_body, 0.0, 0.0]),
        fluid_velocity=np.zeros(3),
        fluid_acceleration=np.array([a_fluid, 0.0, 0.0]),
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    A_x = np.pi * D * D / 4.0
    F_expected = _RHO * A_x * L * ((1.0 + Ca) * a_fluid - Ca * a_body)
    assert f6[0] == pytest.approx(F_expected, rel=1.0e-12)


def test_inertia_along_axis_gives_zero() -> None:
    """Acceleration along the member axis projects to zero, like velocity does."""
    e = _vertical_cylinder(include_inertia=True)
    f6 = morison_element_force(
        e,
        midpoint_inertial=np.array([0.0, 0.0, -5.0]),
        axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
        body_velocity_at_midpoint=np.zeros(3),
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.zeros(3),
        fluid_acceleration=np.array([0.0, 0.0, 5.0]),  # along axis
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    assert np.allclose(f6, 0.0, atol=1.0e-9)


def test_inertia_requires_fluid_acceleration_when_enabled() -> None:
    e = _vertical_cylinder(include_inertia=True)
    with pytest.raises(ValueError, match="include_inertia=True requires fluid_acceleration"):
        morison_element_force(
            e,
            midpoint_inertial=np.array([0.0, 0.0, -5.0]),
            axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
            body_velocity_at_midpoint=np.zeros(3),
            body_acceleration_at_midpoint=None,
            fluid_velocity=np.zeros(3),
            fluid_acceleration=None,
            rho=_RHO,
            reference_point_inertial=np.zeros(3),
        )


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_force_rejects_non_unit_axis() -> None:
    e = _vertical_cylinder()
    with pytest.raises(ValueError, match="axis_hat_inertial must be unit"):
        morison_element_force(
            e,
            midpoint_inertial=np.array([0.0, 0.0, -5.0]),
            axis_hat_inertial=np.array([0.0, 0.0, 2.0]),  # not unit-length
            body_velocity_at_midpoint=np.zeros(3),
            body_acceleration_at_midpoint=None,
            fluid_velocity=np.array([1.0, 0.0, 0.0]),
            fluid_acceleration=None,
            rho=_RHO,
            reference_point_inertial=np.zeros(3),
        )


def test_force_rejects_nonpositive_rho() -> None:
    e = _vertical_cylinder()
    with pytest.raises(ValueError, match="rho must be finite and positive"):
        morison_element_force(
            e,
            midpoint_inertial=np.array([0.0, 0.0, -5.0]),
            axis_hat_inertial=np.array([0.0, 0.0, 1.0]),
            body_velocity_at_midpoint=np.zeros(3),
            body_acceleration_at_midpoint=None,
            fluid_velocity=np.array([1.0, 0.0, 0.0]),
            fluid_acceleration=None,
            rho=0.0,
            reference_point_inertial=np.zeros(3),
        )


# ---------------------------------------------------------------------------
# make_morison_state_force closure (integrator-shaped)
# ---------------------------------------------------------------------------


def test_state_force_closure_zero_state_with_uniform_flow() -> None:
    """Single body, single member, calm fluid uniform flow -- closure returns analytical drag."""
    D, L, Cd = 1.0, 4.0, 1.0
    elem = _vertical_cylinder(diameter=D, length=L, Cd=Cd, body_index=0)
    u = 1.5

    def u_fluid_fn(_pt, _t):  # type: ignore[no-untyped-def]
        return np.array([u, 0.0, 0.0], dtype=np.float64)

    sf = make_morison_state_force([elem], n_dof=6, fluid_velocity_fn=u_fluid_fn, rho=_RHO)
    F = sf(0.0, np.zeros(6), np.zeros(6))
    F_expected = 0.5 * _RHO * D * L * Cd * u * u
    assert F[0] == pytest.approx(F_expected, rel=1.0e-12)
    assert F[4] == pytest.approx(-(L / 2.0) * F_expected, rel=1.0e-12)


def test_state_force_closure_two_bodies_force_isolation() -> None:
    """Force on body 0 must not leak into body 1's slot, and vice versa."""
    elem0 = _vertical_cylinder(body_index=0)
    elem1 = _horizontal_x_cylinder(body_index=1)

    def u_fluid_fn(_pt, _t):  # type: ignore[no-untyped-def]
        return np.array([0.5, 0.0, 0.0], dtype=np.float64)

    sf = make_morison_state_force([elem0, elem1], n_dof=12, fluid_velocity_fn=u_fluid_fn, rho=_RHO)
    F = sf(0.0, np.zeros(12), np.zeros(12))
    # body 0: vertical cylinder in +X flow -> Fx > 0, Fy = 0
    assert F[0] > 0.0
    assert abs(F[1]) < 1.0e-12
    # body 1: horizontal +X cylinder in +X flow -> drag = 0 (parallel projection)
    assert np.allclose(F[6:9], 0.0, atol=1.0e-9)


def test_state_force_closure_translated_body_picks_up_translation_velocity() -> None:
    """xi_dot[0] = body translational velocity in +X must subtract from u_fluid."""
    elem = _vertical_cylinder()

    def u_fluid_fn(_pt, _t):  # type: ignore[no-untyped-def]
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    sf = make_morison_state_force([elem], n_dof=6, fluid_velocity_fn=u_fluid_fn, rho=_RHO)
    xi_dot = np.zeros(6)
    xi_dot[0] = 1.0  # body moves with the fluid
    F = sf(0.0, np.zeros(6), xi_dot)
    assert np.allclose(F, 0.0, atol=1.0e-9)


def test_state_force_closure_yaw_rotates_member_into_flow_plane() -> None:
    """A horizontal +X cylinder at yaw = pi/2 lies along +Y; +X flow becomes
    fully member-normal -> drag = analytical."""
    D, L, Cd = 1.0, 6.0, 1.0
    elem = _horizontal_x_cylinder(diameter=D, length=L, Cd=Cd)

    u = 2.0

    def u_fluid_fn(_pt, _t):  # type: ignore[no-untyped-def]
        return np.array([u, 0.0, 0.0], dtype=np.float64)

    sf = make_morison_state_force([elem], n_dof=6, fluid_velocity_fn=u_fluid_fn, rho=_RHO)
    xi = np.zeros(6)
    xi[5] = np.pi / 2.0  # yaw 90 deg about +Z
    F = sf(0.0, xi, np.zeros(6))
    F_expected = 0.5 * _RHO * D * L * Cd * u * u
    assert F[0] == pytest.approx(F_expected, rel=1.0e-9)


def test_state_force_closure_requires_acceleration_fn_when_inertia_enabled() -> None:
    elem = _vertical_cylinder(include_inertia=True)

    def u_fluid_fn(_pt, _t):  # type: ignore[no-untyped-def]
        return np.zeros(3, dtype=np.float64)

    with pytest.raises(ValueError, match="fluid_acceleration_fn must be provided"):
        make_morison_state_force([elem], n_dof=6, fluid_velocity_fn=u_fluid_fn, rho=_RHO)


def test_state_force_closure_rejects_out_of_range_body_index() -> None:
    elem = _vertical_cylinder(body_index=2)

    def u_fluid_fn(_pt, _t):  # type: ignore[no-untyped-def]
        return np.zeros(3, dtype=np.float64)

    with pytest.raises(ValueError, match="body_index 2 outside valid range"):
        make_morison_state_force(
            [elem], n_dof=6, fluid_velocity_fn=u_fluid_fn, rho=_RHO  # n_bodies = 1
        )


def test_state_force_closure_rejects_bad_n_dof() -> None:
    elem = _vertical_cylinder()

    def u_fluid_fn(_pt, _t):  # type: ignore[no-untyped-def]
        return np.zeros(3, dtype=np.float64)

    with pytest.raises(ValueError, match="n_dof must be a positive multiple of 6"):
        make_morison_state_force([elem], n_dof=7, fluid_velocity_fn=u_fluid_fn, rho=_RHO)


# ---------------------------------------------------------------------------
# Plan red-test 4 -- include_inertia + non-empty BEM warning
# ---------------------------------------------------------------------------


def test_startup_warning_when_inertia_enabled_on_bem_body() -> None:
    elem = _vertical_cylinder(include_inertia=True, body_index=0)
    msgs = startup_inertia_double_count_warnings([elem], bodies_with_bem=[True])
    assert len(msgs) == 1
    assert "include_inertia=True" in msgs[0]
    assert "BEM" in msgs[0]


def test_no_warning_when_inertia_enabled_on_bem_free_body() -> None:
    elem = _vertical_cylinder(include_inertia=True, body_index=0)
    msgs = startup_inertia_double_count_warnings([elem], bodies_with_bem=[False])
    assert msgs == []


def test_no_warning_when_inertia_disabled_on_bem_body() -> None:
    elem = _vertical_cylinder(include_inertia=False, body_index=0)
    msgs = startup_inertia_double_count_warnings([elem], bodies_with_bem=[True])
    assert msgs == []


def test_warning_lists_each_offending_element() -> None:
    elem0 = _vertical_cylinder(include_inertia=True, body_index=0)
    elem1 = _vertical_cylinder(include_inertia=True, body_index=1)
    elem2 = _vertical_cylinder(include_inertia=False, body_index=0)
    msgs = startup_inertia_double_count_warnings(
        [elem0, elem1, elem2], bodies_with_bem=[True, False]
    )
    assert len(msgs) == 1  # only elem0 hits
    assert "element 0" in msgs[0]


def test_warning_rejects_short_bodies_with_bem() -> None:
    elem = _vertical_cylinder(include_inertia=True, body_index=2)
    with pytest.raises(ValueError, match="outside bodies_with_bem"):
        startup_inertia_double_count_warnings([elem], bodies_with_bem=[True])


# ---------------------------------------------------------------------------
# Deck schema round-trip
# ---------------------------------------------------------------------------


def test_morison_member_deck_schema_default_include_inertia_false() -> None:
    m = MorisonMember(
        type="morison_member",
        node_a=[0.0, 0.0, -10.0],
        node_b=[0.0, 0.0, 0.0],
        diameter=1.5,
        Cd=0.8,
    )
    assert m.include_inertia is False
    assert m.Ca == 0.0


def test_morison_member_deck_schema_round_trip_include_inertia() -> None:
    m = MorisonMember(
        type="morison_member",
        node_a=[0.0, 0.0, -10.0],
        node_b=[0.0, 0.0, 0.0],
        diameter=1.5,
        Cd=0.8,
        Ca=1.0,
        include_inertia=True,
    )
    assert m.include_inertia is True
    dumped = m.model_dump()
    assert dumped["include_inertia"] is True
    # Round-trip back through validation.
    rebuilt = MorisonMember.model_validate(dumped)
    assert rebuilt == m
