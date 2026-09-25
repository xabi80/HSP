"""Flume-mooring Phase C2, integration tier: ``build_system`` measures catenaries against the
TRUE deck anchors for a body whose reference point is not the origin (two synthetic-BEM builds
with the static solve, ~30 s).

The M4 PR6 single-body catenary pair is built twice: once at the origin, once with the body's
reference point AND both anchors moved by the same horizontal shift. The system is the same
one translated, so the static equilibrium (a displacement from the reference point) and the
restoring state force must be identical. Before C2 the shifted deck placed the fairlead at
``xi`` instead of ``reference_point + xi``: its lines were 120 m longer on one side than the
other and the static solve pulled the body ~120 m towards one anchor.
"""

from __future__ import annotations

import numpy as np

from floatsim.driver import build_system
from floatsim.io.deck import (
    Body,
    Catenary,
    CatenaryLine,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    Output,
    RegularWave,
    Simulation,
)
from scripts.m7_pr4_driver_prediction import (
    _ANCHOR_MINUS_GLOBAL,
    _ANCHOR_PLUS_GLOBAL,
    _CATENARY_LINE,
    _single_body_hdb,
)
from tests.validation.test_cummins_free_decay_analytical import _I_OTHER, _M_OTHER

_DT_COARSE = 0.05  # Nyquist-safe for the synthetic BEM (omega_max = 20 rad/s -> dt < 0.157 s)
_SHIFT = np.array([120.0, -45.0, 0.0])


def _deck(shift: np.ndarray) -> Deck:
    line = CatenaryLine(
        length=_CATENARY_LINE.length,
        weight_per_length=_CATENARY_LINE.weight_per_length,
        EA=_CATENARY_LINE.EA,
    )
    return Deck(
        simulation=Simulation(duration=10.0, dt=_DT_COARSE),
        environment=Environment(water_depth=200.0, water_density=1025.0),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[
            Body(
                name="b0",
                reference_point=shift.tolist(),
                mass=_M_OTHER,
                inertia=Inertia(Ixx=_I_OTHER, Iyy=_I_OTHER, Izz=_I_OTHER),
                hydro_database=HydroDatabaseRef(format="wamit", path="synthetic_simple"),
                initial_conditions=InitialConditions(),
            )
        ],
        connections=[
            Catenary(
                type="catenary",
                body_a="b0",
                body_b="earth",
                attach_a_body=[0.0, 0.0, 0.0],
                attach_b_body=(anchor + shift).tolist(),
                line=line,
            )
            for anchor in (_ANCHOR_PLUS_GLOBAL, _ANCHOR_MINUS_GLOBAL)
        ],
        output=Output(file="out.h5", channels=["surge", "heave"], sample_rate=10.0),
    )


def test_build_system_uses_true_anchors_for_a_non_origin_reference() -> None:
    bem = {"b0": _single_body_hdb()}
    kw = {"bem_databases": bem, "dt": _DT_COARSE, "t_max_kernel": 120.0}
    origin = build_system(_deck(np.zeros(3)), solve_equilibrium=True, **kw)
    shifted = build_system(_deck(_SHIFT), solve_equilibrium=True, **kw)
    # Same equilibrium displacement (the catenary pair is symmetric: no surge offset) ...
    np.testing.assert_allclose(shifted.xi0, origin.xi0, rtol=1e-9, atol=1e-9)
    assert abs(origin.xi0[0]) < 1e-6
    # ... and the same restoring force around it.
    rng = np.random.default_rng(3)
    for _ in range(6):
        xi = origin.xi0 + rng.normal(0.0, [2.0, 2.0, 0.5, 0.01, 0.01, 0.02])
        np.testing.assert_allclose(
            shifted.state_force(0.0, xi, np.zeros(6)),
            origin.state_force(0.0, xi, np.zeros(6)),
            rtol=1e-9,
            atol=1e-3,
        )
