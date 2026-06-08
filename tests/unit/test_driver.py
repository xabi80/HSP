"""M7-Foundation PR4 -- F1 build_system driver tests.

Three test classes per the user-locked pre-flight contract:

  Step C round-trip identity: build_system(M4_PR6_deck) reproduces
  the hand-wired M4 PR6 setup at rtol = 1e-12 on every component
  (lhs.M_plus_Ainf, lhs.C, kernel.K, state_force(0, xi, 0),
  xi0, xi_dot0).

  Pre-flight item 1: single-body deck (OC4-style synthetic) round-
  trips through build_system. M2-M6's validation history is
  single-body; the driver must not break that path.

  Pre-flight item 2: "earth" sentinel both directions. The
  framework is symmetric under Newton III but the deck-loader is
  where order-sensitivity bugs hide.

  Pre-flight item 3: BB-OFFSET-CONNECTOR NotImplementedError
  message content asserts the message includes
  "phase2-followups.md" AND "BB-OFFSET-CONNECTOR" (or the section
  anchor). A generic message is half the value of the pinned
  disposition.

Plus error-path coverage for the locked-scope edges.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

from floatsim.driver import SimulationSetup, build_system

# This module exercises the build_system driver against the M4 PR6 fixture,
# which involves a 60001-lag retardation kernel per body and a 12-DOF
# equilibrium solve. The expensive operations are cached at module scope
# via lru_cache so all 8 Step-C round-trip tests share one build_system
# call + one hand-wired-target call. Marked slow because the un-cached
# parts (kernel computation, equilibrium iteration) still take ~minute
# total even with caching.
pytestmark = pytest.mark.slow
from floatsim.io.deck import (
    Body,
    Catenary,
    CatenaryLine,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    LinearSpring,
    Output,
    RegularWave,
    RigidLink,
    Simulation,
)
from scripts.m7_pr4_driver_prediction import (
    _DT,
    _PENALTY_FACTOR,
    _SEABED_DEPTH,
    _ANCHOR_PLUS_GLOBAL,
    _ANCHOR_MINUS_GLOBAL,
    _CATENARY_LINE,
    _single_body_hdb,
    build_m4_pr6_hand_wired,
)
from tests.validation.test_cummins_free_decay_analytical import (
    _A_INF_33,
    _C_33,
    _I_OTHER,
    _M_33,
    _M_OTHER,
)


# ---------------------------------------------------------------------------
# Deck builders -- programmatic, no YAML round-trip
# ---------------------------------------------------------------------------


def _dummy_simulation() -> Simulation:
    return Simulation(duration=10.0, dt=_DT)


def _dummy_environment() -> Environment:
    return Environment(water_depth=200.0, water_density=1025.0)


def _dummy_waves() -> RegularWave:
    return RegularWave(type="regular", height=1.0, period=10.0, heading=0.0)


def _dummy_output() -> Output:
    return Output(file="out.h5", channels=["surge", "heave"], sample_rate=10.0)


def _m2_body(name: str) -> Body:
    """A deck-Body matching the M2 synthetic-fixture body (no offset CoG)."""
    return Body(
        name=name,
        reference_point=[0.0, 0.0, 0.0],
        mass=_M_OTHER,  # = M_33 = 1e7
        inertia=Inertia(Ixx=_I_OTHER, Iyy=_I_OTHER, Izz=_I_OTHER),
        hydro_database=HydroDatabaseRef(format="wamit", path="synthetic_simple"),
        initial_conditions=InitialConditions(),
    )


def _m4_pr6_deck() -> Deck:
    """Build the deck-level equivalent of the M4 PR6 fixture.

    2 bodies, 1 heave RigidLink, 2 Catenaries from body 0 to earth.
    """
    return Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("body_0"), _m2_body("body_1")],
        connections=[
            RigidLink(
                type="rigid_link",
                body_a="body_0",
                body_b="body_1",
                penalty_stiffness_factor=_PENALTY_FACTOR,
            ),
            Catenary(
                type="catenary",
                body_a="body_0",
                body_b="earth",
                attach_a_body=[0.0, 0.0, 0.0],
                attach_b_body=_ANCHOR_PLUS_GLOBAL.tolist(),
                line=CatenaryLine(
                    length=_CATENARY_LINE.length,
                    weight_per_length=_CATENARY_LINE.weight_per_length,
                    EA=_CATENARY_LINE.EA,
                ),
            ),
            Catenary(
                type="catenary",
                body_a="body_0",
                body_b="earth",
                attach_a_body=[0.0, 0.0, 0.0],
                attach_b_body=_ANCHOR_MINUS_GLOBAL.tolist(),
                line=CatenaryLine(
                    length=_CATENARY_LINE.length,
                    weight_per_length=_CATENARY_LINE.weight_per_length,
                    EA=_CATENARY_LINE.EA,
                ),
            ),
        ],
        output=_dummy_output(),
    )


# ---------------------------------------------------------------------------
# Step C: round-trip identity vs Step A hand-wired M4 PR6
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _m4_pr6_setup_cached() -> tuple[SimulationSetup, dict]:
    deck = _m4_pr6_deck()
    bem = {"body_0": _single_body_hdb(), "body_1": _single_body_hdb()}
    setup = build_system(
        deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0, solve_equilibrium=True
    )
    targets = build_m4_pr6_hand_wired()
    return setup, targets


def _m4_pr6_setup() -> tuple[SimulationSetup, dict]:
    return _m4_pr6_setup_cached()


def test_step_C_M_plus_Ainf_matches_hand_wired() -> None:
    setup, targets = _m4_pr6_setup()
    np.testing.assert_allclose(
        setup.lhs.M_plus_Ainf, targets["lhs_global"].M_plus_Ainf,
        rtol=1.0e-12, atol=1.0e-12,
    )


def test_step_C_C_matches_hand_wired() -> None:
    setup, targets = _m4_pr6_setup()
    np.testing.assert_allclose(
        setup.lhs.C, targets["lhs_global"].C, rtol=1.0e-12, atol=1.0e-12
    )


def test_step_C_kernel_K_matches_hand_wired() -> None:
    setup, targets = _m4_pr6_setup()
    np.testing.assert_allclose(
        setup.kernel.K, targets["kernel_global"].K, rtol=1.0e-12, atol=1.0e-12
    )


def test_step_C_state_force_at_xi_zero_matches_hand_wired() -> None:
    setup, targets = _m4_pr6_setup()
    xi = np.zeros(12)
    F_driver = setup.state_force(0.0, xi, np.zeros(12))
    F_hand = targets["state_force"](0.0, xi, np.zeros(12))
    np.testing.assert_allclose(F_driver, F_hand, rtol=1.0e-12, atol=1.0e-12)


def test_step_C_state_force_at_xi_both_surge_matches_hand_wired() -> None:
    """Discriminator pose: both bodies displaced in surge (rigid link sees
    zero relative surge, catenaries see body-0 surge asymmetry)."""
    setup, targets = _m4_pr6_setup()
    xi = np.zeros(12)
    xi[0] = 0.5
    xi[6] = 0.5
    F_driver = setup.state_force(0.0, xi, np.zeros(12))
    F_hand = targets["state_force"](0.0, xi, np.zeros(12))
    np.testing.assert_allclose(F_driver, F_hand, rtol=1.0e-12, atol=1.0e-12)


def test_step_C_xi0_post_equilibrium_matches_hand_wired() -> None:
    setup, targets = _m4_pr6_setup()
    np.testing.assert_allclose(
        setup.xi0, targets["xi0_post_equilibrium"], rtol=1.0e-12, atol=1.0e-9
    )


def test_step_C_xi_dot0_is_zero() -> None:
    setup, _ = _m4_pr6_setup()
    np.testing.assert_array_equal(setup.xi_dot0, np.zeros(12))


def test_step_C_body_name_to_index_matches_deck_order() -> None:
    setup, _ = _m4_pr6_setup()
    assert setup.body_name_to_index == {"body_0": 0, "body_1": 1}


# ---------------------------------------------------------------------------
# Pre-flight item 1: single-body deck (M2-M6 validation surface)
# ---------------------------------------------------------------------------


def _single_body_deck(no_connections: bool = True) -> Deck:
    return Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("body_0")],
        connections=([] if no_connections else []),
        output=_dummy_output(),
    )


def test_preflight_1_single_body_deck_builds_clean() -> None:
    """Single-body deck (analog of OC4 unmoored M6 PR2/S1) round-trips
    through build_system without breaking the M2-M6 validation path.
    """
    deck = _single_body_deck()
    bem = {"body_0": _single_body_hdb()}
    setup = build_system(
        deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0, solve_equilibrium=True
    )
    assert setup.lhs.n_dof == 6
    assert setup.lhs.n_bodies == 1
    assert setup.kernel.K.shape[0] == 6
    assert setup.xi0.shape == (6,)
    assert setup.xi_dot0.shape == (6,)
    # No connections -> state_force returns zero.
    F = setup.state_force(0.0, np.zeros(6), np.zeros(6))
    np.testing.assert_array_equal(F, np.zeros(6))


def test_preflight_1_single_body_no_connections_matches_hand_wired() -> None:
    """Hand-wired single-body LHS = driver-built single-body LHS at rtol = 1e-12."""
    from floatsim.hydro.radiation import assemble_cummins_lhs
    from floatsim.hydro.retardation import compute_retardation_kernel
    from scripts.m7_pr4_driver_prediction import _single_body_rigid_mass

    hdb = _single_body_hdb()
    lhs_hand = assemble_cummins_lhs(rigid_body_mass=_single_body_rigid_mass(), hdb=hdb)
    kernel_hand = compute_retardation_kernel(hdb, t_max=120.0, dt=_DT)

    deck = _single_body_deck()
    setup = build_system(
        deck, bem_databases={"body_0": _single_body_hdb()},
        dt=_DT, t_max_kernel=120.0, solve_equilibrium=True,
    )
    np.testing.assert_allclose(setup.lhs.M_plus_Ainf, lhs_hand.M_plus_Ainf, rtol=1.0e-12)
    np.testing.assert_allclose(setup.lhs.C, lhs_hand.C, rtol=1.0e-12)
    np.testing.assert_allclose(setup.kernel.K, kernel_hand.K, rtol=1.0e-12)


# ---------------------------------------------------------------------------
# Pre-flight item 2: "earth" sentinel both directions
# ---------------------------------------------------------------------------


def _earth_spring_deck(earth_first: bool) -> Deck:
    """One body, one LinearSpring connecting body_0 to earth.

    If earth_first: LinearSpring(body_a="earth", body_b="body_0").
    Else:           LinearSpring(body_a="body_0", body_b="earth").
    """
    if earth_first:
        spring = LinearSpring(
            type="linear_spring",
            body_a="earth",
            body_b="body_0",
            anchor_a_body=[0.0, 0.0, -10.0],  # treated as inertial when body_a=earth
            anchor_b_body=[0.0, 0.0, 0.0],
            stiffness=1.0e6,
            rest_length=0.0,
        )
    else:
        spring = LinearSpring(
            type="linear_spring",
            body_a="body_0",
            body_b="earth",
            anchor_a_body=[0.0, 0.0, 0.0],
            anchor_b_body=[0.0, 0.0, -10.0],  # treated as inertial when body_b=earth
            stiffness=1.0e6,
            rest_length=0.0,
        )
    return Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("body_0")],
        connections=[spring],
        output=_dummy_output(),
    )


def test_preflight_2_earth_as_body_a_resolves_cleanly() -> None:
    """LinearSpring(body_a='earth', body_b='body_0') must build without error
    and produce a state_force that exercises body_0's slot.
    """
    deck = _earth_spring_deck(earth_first=True)
    bem = {"body_0": _single_body_hdb()}
    setup = build_system(
        deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0, solve_equilibrium=False
    )
    assert setup.lhs.n_dof == 6
    # state_force should be non-trivial at non-zero xi.
    F = setup.state_force(0.0, np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6))
    assert not np.allclose(F, 0.0), (
        "earth-as-body_a spring produced zero state_force at non-zero xi -- "
        "deck-loader order-sensitivity bug?"
    )


def test_preflight_2_earth_as_body_b_resolves_cleanly() -> None:
    """Symmetric direction: LinearSpring(body_a='body_0', body_b='earth')."""
    deck = _earth_spring_deck(earth_first=False)
    bem = {"body_0": _single_body_hdb()}
    setup = build_system(
        deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0, solve_equilibrium=False
    )
    assert setup.lhs.n_dof == 6
    F = setup.state_force(0.0, np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6))
    assert not np.allclose(F, 0.0), (
        "earth-as-body_b spring produced zero state_force at non-zero xi -- "
        "deck-loader order-sensitivity bug?"
    )


# ---------------------------------------------------------------------------
# Pre-flight item 3: BB-OFFSET-CONNECTOR error-message content
# ---------------------------------------------------------------------------


def test_preflight_3_bb_offset_raises_with_tracker_citation() -> None:
    """Per the Xabier-pinned Q9 disposition, a deck-driven body-body
    LinearSpring with any non-zero attach offset must raise
    NotImplementedError with a message citing the tracker entry.

    The message must include BOTH "phase2-followups.md" AND
    "BB-OFFSET-CONNECTOR" (case-insensitive on the latter to allow
    section-anchor formatting). A generic 'not supported' message
    would be half the value of the pinned disposition.
    """
    spring = LinearSpring(
        type="linear_spring",
        body_a="body_0",
        body_b="body_1",
        anchor_a_body=[1.0, 0.0, 0.0],  # non-zero offset on body_a side
        anchor_b_body=[0.0, 0.0, 0.0],
        stiffness=1.0e6,
        rest_length=0.0,
    )
    deck = Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("body_0"), _m2_body("body_1")],
        connections=[spring],
        output=_dummy_output(),
    )
    bem = {"body_0": _single_body_hdb(), "body_1": _single_body_hdb()}
    with pytest.raises(NotImplementedError) as exc_info:
        build_system(deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0)
    msg = str(exc_info.value)
    assert "phase2-followups.md" in msg, (
        f"BB-OFFSET-CONNECTOR NotImplementedError must cite "
        f"phase2-followups.md; got: {msg}"
    )
    assert "BB-OFFSET-CONNECTOR" in msg or "bb-offset-connector" in msg, (
        f"BB-OFFSET-CONNECTOR NotImplementedError must name the tracker "
        f"entry by ID; got: {msg}"
    )


# ---------------------------------------------------------------------------
# Locked-scope error paths
# ---------------------------------------------------------------------------


def test_duplicate_body_names_raise() -> None:
    deck = Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("body_0"), _m2_body("body_0")],
        connections=[],
        output=_dummy_output(),
    )
    bem = {"body_0": _single_body_hdb()}
    with pytest.raises(ValueError, match="duplicate body names"):
        build_system(deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0)


def test_body_name_earth_collides_with_sentinel() -> None:
    deck = Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("earth")],
        connections=[],
        output=_dummy_output(),
    )
    bem = {"earth": _single_body_hdb()}
    with pytest.raises(ValueError, match="collides with the earth sentinel"):
        build_system(deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0)


def test_missing_bem_database_raises() -> None:
    deck = _single_body_deck()
    bem: dict = {}  # missing body_0
    with pytest.raises(ValueError, match="bem_databases missing entry for body"):
        build_system(deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0)


def test_unknown_connection_endpoint_raises() -> None:
    deck = Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("body_0")],
        connections=[
            RigidLink(
                type="rigid_link",
                body_a="body_0",
                body_b="body_missing",
                penalty_stiffness_factor=1.0e3,
            ),
        ],
        output=_dummy_output(),
    )
    bem = {"body_0": _single_body_hdb()}
    with pytest.raises(ValueError, match="unknown body name"):
        build_system(deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0)


def test_body_body_catenary_raises() -> None:
    cat = Catenary(
        type="catenary",
        body_a="body_0",
        body_b="body_1",
        attach_a_body=[0.0, 0.0, 0.0],
        attach_b_body=[0.0, 0.0, 0.0],
        line=CatenaryLine(length=500.0, weight_per_length=1000.0, EA=5.0e8),
    )
    deck = Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("body_0"), _m2_body("body_1")],
        connections=[cat],
        output=_dummy_output(),
    )
    bem = {"body_0": _single_body_hdb(), "body_1": _single_body_hdb()}
    with pytest.raises(NotImplementedError, match="body-to-body"):
        build_system(deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0)


def test_linear_spring_with_nonzero_rest_length_raises() -> None:
    spring = LinearSpring(
        type="linear_spring",
        body_a="body_0",
        body_b="earth",
        anchor_a_body=[0.0, 0.0, 0.0],
        anchor_b_body=[0.0, 0.0, -10.0],
        stiffness=1.0e6,
        rest_length=5.0,  # non-zero
    )
    deck = Deck(
        simulation=_dummy_simulation(),
        environment=_dummy_environment(),
        waves=_dummy_waves(),
        bodies=[_m2_body("body_0")],
        connections=[spring],
        output=_dummy_output(),
    )
    bem = {"body_0": _single_body_hdb()}
    with pytest.raises(NotImplementedError, match="rest_length"):
        build_system(deck, bem_databases=bem, dt=_DT, t_max_kernel=120.0)
