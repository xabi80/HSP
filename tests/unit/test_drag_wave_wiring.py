"""STEP 5 PR1 commit (c): wave kinematics wired into the deck's Morison drag.

``build_system(..., drag_wave=, drag_wave_ramp=)`` makes the deck's drag act on the
velocity RELATIVE to the regular-wave orbital field (and so adds the drag excitation),
instead of on the body velocity in calm water. The caller passes the same wave and ramp
it uses for the BEM excitation; ``None`` keeps the calm-water path byte-identical.

The fluid is sampled at each element's ABSOLUTE position (reference point + displacement
+ R @ arm, commit b) with the corrected Airy field (commit a). Tracker
DRAG-WAVE-KINEMATICS-UNWIRED.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.driver import _build_drag_state_force, build_system
from floatsim.hydro.morison import MorisonElement, PlateDragElement, make_morison_state_force
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    MorisonMember,
    Output,
    PlateMember,
    Simulation,
    distributed_cylinder_drag,
)
from floatsim.io.deck import (
    RegularWave as DeckWave,
)
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.kinematics import airy_velocity
from floatsim.waves.regular import RegularWave
from scripts.m7_pr4_driver_prediction import _DT, _single_body_hdb

_RHO = 1025.0
# body 0 referenced at the origin, body 1 like an M11b platform buoy (away from it)
_REFS = np.array([[0.0, 0.0, 0.0], [2.5, -1.0, -1.19567]])
_WAVE = RegularWave(amplitude=0.15, omega=2.0 * np.pi / 3.141, heading_deg=20.0, phase=0.3)
_RAMP = HalfCosineRamp(duration=20.0)


def _drag() -> list:  # type: ignore[type-arg]
    spar = distributed_cylinder_drag(
        z_bottom=-0.2617, z_top=1.19567, diameter=0.1682, cd=1.2, n_segments=10
    )
    plate = PlateMember(
        type="plate",
        center=[0.0, 0.0, -0.2617],
        normal=[0.0, 0.0, 1.0],
        radius=0.215,
        thickness=0.0039,
        Cd_n=5.0,
        Cd_t=1.5,
    )
    return [*spar, plate]


def _deck(drag: list | None = None) -> Deck:  # type: ignore[type-arg]
    bodies = [
        Body(
            name=f"b{k}",
            reference_point=list(_REFS[k]),
            mass=1.0e3,
            inertia=Inertia(Ixx=1.0e3, Iyy=1.0e3, Izz=1.0e3),
            hydro_database=HydroDatabaseRef(format="wamit", path="synthetic_simple"),
            initial_conditions=InitialConditions(),
            drag_elements=_drag() if drag is None else drag,
        )
        for k in range(2)
    ]
    return Deck(
        simulation=Simulation(duration=10.0, dt=_DT),
        environment=Environment(water_depth=200.0, water_density=_RHO),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=bodies,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def _states(n: int = 20):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(11)
    for _ in range(n):
        yield (float(rng.uniform(0.0, 40.0)), rng.normal(0, 0.05, 12), rng.normal(0, 0.3, 12))


def _reference_elements(dk: Deck) -> dict[int, list]:  # type: ignore[type-arg]
    """The deck's drag elements, built independently of the driver."""
    out: dict[int, list] = {}  # type: ignore[type-arg]
    for k, b in enumerate(dk.bodies):
        for e in b.drag_elements:
            if isinstance(e, PlateMember):
                el = PlateDragElement(
                    body_index=k,
                    center_body=np.asarray(e.center, float),
                    normal_body=np.asarray(e.normal, float),
                    radius=e.radius,
                    thickness=e.thickness,
                    Cd_n=e.Cd_n,
                    Cd_t=e.Cd_t,
                    n_radial=e.n_radial,
                    n_azimuthal=e.n_azimuthal,
                )
            else:
                el = MorisonElement(
                    body_index=k,
                    node_a_body=np.asarray(e.node_a, float),
                    node_b_body=np.asarray(e.node_b, float),
                    diameter=e.diameter,
                    Cd=e.Cd,
                    Ca=e.Ca,
                    include_inertia=False,
                )
            out.setdefault(k, []).append(el)
    return out


def test_no_wave_is_the_calm_path_byte_identical() -> None:
    """drag_wave=None: the driver's drag equals calm-water Morison (zero fluid,
    displacement-frame sampling), byte for byte."""
    dk = _deck()
    drv = _build_drag_state_force(dk, 12, rho=_RHO)
    flat = [e for es in _reference_elements(dk).values() for e in es]
    ref = make_morison_state_force(
        flat, n_dof=12, rho=_RHO, fluid_velocity_fn=lambda p, t: np.zeros(3)
    )
    assert drv is not None
    for t, xi, xd in _states():
        assert drv(t, xi, xd).tobytes() == ref(t, xi, xd).tobytes()


def test_wave_path_equals_an_independent_per_body_composition() -> None:
    """The wired path equals the STEP 1 study-side construction, byte for byte: one closure
    per body, fluid = ramp(t) * airy_velocity(wave, point + reference_point, t)."""
    dk = _deck()
    drv = _build_drag_state_force(dk, 12, rho=_RHO, wave=_WAVE, ramp=_RAMP)
    parts = []
    for k, es in _reference_elements(dk).items():
        refk = _REFS[k]

        def fluid(p, t, refk=refk):  # type: ignore[no-untyped-def]
            return _RAMP.value(t) * airy_velocity(_WAVE, p + refk, t)

        parts.append(make_morison_state_force(es, n_dof=12, fluid_velocity_fn=fluid, rho=_RHO))
    assert drv is not None
    for t, xi, xd in _states():
        tot = parts[0](t, xi, xd) + parts[1](t, xi, xd)
        np.testing.assert_array_equal(drv(t, xi, xd), tot)


def _single_cylinder_deck() -> tuple[Deck, np.ndarray]:
    member = MorisonMember(
        type="morison_member",
        node_a=[0.0, 0.0, -0.4],
        node_b=[0.0, 0.0, -0.2],
        diameter=0.2,
        Cd=1.2,
    )
    mid_body = np.array([0.0, 0.0, -0.3])
    return _deck(drag=[member]), mid_body


def test_body_riding_the_flow_at_its_absolute_position_feels_no_drag() -> None:
    """After the ramp, move body 1 (reference point away from the origin) with the fluid
    velocity at its element's ABSOLUTE midpoint: relative velocity 0, drag 0. With the
    displacement-frame point (the pre-fix sampling) the same motion leaves drag."""
    dk, mid = _single_cylinder_deck()
    drv = _build_drag_state_force(dk, 12, rho=_RHO, wave=_WAVE, ramp=_RAMP)
    assert drv is not None
    t = 31.7
    xd = np.zeros(12)
    xd[6:9] = airy_velocity(_WAVE, _REFS[1] + mid, t)
    np.testing.assert_allclose(drv(t, np.zeros(12), xd)[6:12], 0.0, atol=1e-12)
    xd_wrong = np.zeros(12)
    xd_wrong[6:9] = airy_velocity(_WAVE, mid, t)  # sampled without the reference point
    assert np.abs(drv(t, np.zeros(12), xd_wrong)[6:9]).max() > 1e-3


def test_fixed_body_feels_the_drag_excitation_scaled_by_ramp_squared() -> None:
    """A fixed body sees F = 0.5*rho*Cd*D*L*|u_n|*u_n from the orbital flow at the absolute
    midpoint (vertical member: u_n = horizontal components), scaled by ramp(t)^2 -- zero at
    t = 0."""
    dk, mid = _single_cylinder_deck()
    drv = _build_drag_state_force(dk, 12, rho=_RHO, wave=_WAVE, ramp=_RAMP)
    assert drv is not None
    zero = np.zeros(12)
    np.testing.assert_array_equal(drv(0.0, zero, zero), np.zeros(12))
    for t in (7.3, 33.1):
        u = airy_velocity(_WAVE, _REFS[1] + mid, t) * _RAMP.value(t)
        un = np.array([u[0], u[1], 0.0])
        f = 0.5 * _RHO * 1.2 * 0.2 * 0.2 * np.linalg.norm(un) * un
        np.testing.assert_allclose(drv(t, zero, zero)[6:9], f, rtol=1e-12, atol=1e-15)


def test_ramp_without_wave_is_rejected() -> None:
    with pytest.raises(ValueError, match="drag_wave_ramp given without drag_wave"):
        _build_drag_state_force(_deck(), 12, rho=_RHO, ramp=_RAMP)
    with pytest.raises(ValueError, match="drag_wave_ramp given without drag_wave"):
        # validated before any kernel work, so this raises immediately
        build_system(
            _deck(),
            bem_databases={"b0": _single_body_hdb(), "b1": _single_body_hdb()},
            dt=_DT,
            t_max_kernel=120.0,
            solve_equilibrium=False,
            drag_wave_ramp=_RAMP,
        )
