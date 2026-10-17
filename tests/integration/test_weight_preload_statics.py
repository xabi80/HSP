"""Weight preload, statics (docs/weight-preload-plan.md §2.1-2.2; Gates A static and B).

Gate A (static): the pinned float's joint force equals its net buoyancy. With ``P = rho g V -
m g`` up, the pin pushes down on the body, so the recovered multiplier on the translational rows
is ``(0, 0, -P)``. That is the M9 sign convention: at rest ``F + G^T Lambda = C x``, and Lambda
is the physical force on body A. The yaw-lock row carries nothing.

Gate B: a deck that does not switch the preload on is byte-identical to before, and switching it
on with every hydro body at its default volume (``m / rho``) adds no static force.
"""

from __future__ import annotations

import numpy as np
import pytest

from floatsim.driver import _static_force
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    Output,
    RegularWave,
    Simulation,
)
from floatsim.solver.newmark import integrate_cummins
from tests.support import pinned_float as pf


@pytest.mark.parametrize("net_up", [30.0, -30.0])
def test_static_pin_force_is_the_net_buoyancy(net_up: float) -> None:
    s = pf.setup(net_up)
    assert s.lam0 is not None
    np.testing.assert_allclose(s.lam0[0:3], [0.0, 0.0, -net_up], rtol=1e-10, atol=1e-9)
    assert abs(float(s.lam0[3])) < 1e-9  # yaw-lock row
    np.testing.assert_allclose(s.xi0, np.zeros(6), atol=1e-9)  # the pin holds the body


@pytest.mark.parametrize("net_up", [30.0, -30.0])
def test_static_force_vector(net_up: float) -> None:
    s = pf.setup(net_up)
    assert s.static_force is not None
    np.testing.assert_allclose(
        s.static_force, [0.0, 0.0, net_up, 0.0, 0.0, 0.0], rtol=1e-12, atol=1e-12
    )


def test_state_force_carries_the_static_force() -> None:
    s = pf.setup(30.0)
    f = s.state_force(0.0, np.zeros(6), np.zeros(6))
    np.testing.assert_allclose(f, [0.0, 0.0, 30.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_preload_off_is_unchanged() -> None:
    """Gate B: the switch off leaves no static force, no multipliers, the old solve."""
    s = pf.setup(30.0, preload=False)
    assert s.static_force is None and s.lam0 is None
    np.testing.assert_array_equal(s.state_force(0.0, np.zeros(6), np.zeros(6)), np.zeros(6))
    np.testing.assert_array_equal(s.xi0, np.zeros(6))


def test_default_volume_adds_no_force() -> None:
    """Gate B: preload on, ``displaced_volume`` left at m / rho -> zero net static force."""
    s = pf.setup(None)
    assert s.static_force is not None
    np.testing.assert_allclose(s.static_force, np.zeros(6), atol=1e-9)
    assert s.lam0 is not None
    np.testing.assert_allclose(s.lam0, np.zeros(4), atol=1e-9)


def test_unheld_structural_body_raises() -> None:
    """A structural body's weight must reach a joint; a free one has nothing to hold it."""
    inertia = Inertia(Ixx=1.0, Iyy=1.0, Izz=1.0)
    dk = Deck(
        simulation=Simulation(duration=1.0, dt=0.01, weight_preload=True),
        environment=Environment(water_depth=200.0, water_density=998.0, gravity=9.806),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[
            Body(
                name="buoy",
                reference_point=[0.0, 0.0, 0.0],
                mass=10.0,
                inertia=inertia,
                hydro_body_label="buoy",
            ),
            Body(
                name="hub",
                reference_point=[0.0, 0.0, 1.0],
                mass=2.0,
                inertia=inertia,
                structural=True,
            ),
        ],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path="x.nc"),
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )
    with pytest.raises(ValueError, match=r"structural body 'hub'.*no joint"):
        _static_force(dk)


def test_preloaded_rest_stays_at_rest() -> None:
    """Started at its preloaded equilibrium, the pinned float does not move: the static solve
    and the dynamics carry the same loads (plan §2.3)."""
    s = pf.setup(30.0)
    r = integrate_cummins(
        lhs=s.lhs,
        kernel=s.kernel,
        xi0=s.xi0,
        xi_dot0=s.xi_dot0,
        duration=5.0,
        dt=pf.DT,
        rho_inf=0.8,
        constraints=s.constraints,
        state_force=s.state_force,
        projection_interval=1,
    )
    assert float(np.max(np.abs(r.xi - s.xi0))) < 1e-10
    assert r.lam is not None
    np.testing.assert_allclose(r.lam[-1][0:3], [0.0, 0.0, -30.0], rtol=1e-9)
