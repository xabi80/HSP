"""Weight preload, Gate C statics: the flume cluster through FloatSim against the payload
study's large-angle statics (docs/weight-preload-plan.md §3).

The cluster is the record's (studies/flume-mooring mooring_verify._cluster_deck):
- four OSU buoys at r = 0.4167 m, 45 / 135 / 225 / 315 deg, on the committed 12-panel coupled
  BEM;
- each buoy yaw-locked to a structural hub at its pin, 1.624 m above its CoG.
A payload on the hub is folded into the hub body (combined mass, at the combined CG).

With the preload on, FloatSim's own static solve now gives what the study computed by hand.
- **Centred 10 kg** (12 kg with the hub). The cluster sinks
  (2 + m) g / (4 C33) = 157.94 mm, as studies/payload-stability floatsim_crosscheck.json and
  cluster_statics on the 12-panel table do. Each pin carries (2 + m) g / 4 down on its buoy.
- **1 kg at 0.1 m.** The hub tilts m g e / (C33 sum x_i^2) = 0.869 deg (the study's 0.869 deg).
  The payload sits in the pin plane, so no preload term changes the hub's seesaw.
Without the preload, FloatSim's static solve sees no load at all: zero sinkage, zero tilt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floatsim.driver import build_system
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    Output,
    RegularWave,
    Simulation,
    YawLockedJoint,
)
from floatsim.solver.newmark import integrate_cummins

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
_NC = REPO / "studies" / "flume-mooring" / "cluster_osu_open_rot45_psd.nc"
_OVR = "weight-preload Gate C: coarse coupled OSU BEM, small-body kernel (flume record)"
RHO, G = 998.0, 9.806
M_BUOY, I_BUOY = 21.52, (10.2, 10.2, 0.063)
Z_COG, Z_PIN = -0.907, 0.717
R_BUOY = 0.5 * 1.25 / 1.5
ANG = np.deg2rad([45.0, 135.0, 225.0, 315.0])
M_HUB = 2.0


def _deck(m_payload: float, e_x: float, *, preload: bool) -> Deck:
    hub_mass = M_HUB + m_payload
    cg = np.array([m_payload * e_x / hub_mass, 0.0, Z_PIN])
    bodies, joints = [], []
    for k, a in enumerate(ANG):
        x, y = R_BUOY * np.cos(a), R_BUOY * np.sin(a)
        bodies.append(
            Body(
                name=f"buoy{k + 1}",
                reference_point=[x, y, Z_COG],
                mass=M_BUOY,
                inertia=Inertia(Ixx=I_BUOY[0], Iyy=I_BUOY[1], Izz=I_BUOY[2]),
                hydro_body_label=f"buoy{k + 1}",
            )
        )
        joints.append(
            YawLockedJoint(
                type="yaw_locked",
                body_a=f"buoy{k + 1}",
                body_b="hub",
                attach_a_body=[0.0, 0.0, Z_PIN - Z_COG],
                attach_b_body=(np.array([x, y, Z_PIN]) - cg).tolist(),
                axis=[0.0, 0.0, 1.0],
            )
        )
    bodies.append(
        Body(
            name="hub",
            reference_point=cg.tolist(),
            mass=hub_mass,
            inertia=Inertia(Ixx=0.3, Iyy=0.3, Izz=0.6),
            structural=True,
        )
    )
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.01, weight_preload=preload),
        environment=Environment(water_depth=200.0, water_density=RHO, gravity=G),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=bodies,
        joints=joints,
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC)),
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def _setup(m_payload: float, e_x: float, *, preload: bool = True):  # type: ignore[no-untyped-def]
    return build_system(
        _deck(m_payload, e_x, preload=preload),
        bem_databases={},
        dt=0.01,
        t_max_kernel=30.0,
        solve_equilibrium=True,
        shared_hydro_database=read_capytaine(_NC),
        asymptote_check_override=_OVR,
        kernel_decay_floor_override=_OVR,
    )


def _c33() -> float:
    return float(np.asarray(read_capytaine(_NC).C)[2, 2])


def test_centred_payload_sinks_as_the_statics() -> None:
    m = 10.0
    s = _setup(m, 0.0)
    sink = (M_HUB + m) * G / (4.0 * _c33())
    assert sink == pytest.approx(0.15794, abs=5e-5)  # floatsim_crosscheck.json
    np.testing.assert_allclose(s.xi0[2::6], -sink, rtol=1e-6)  # buoys and hub together
    assert s.lam0 is not None
    for j in range(4):  # 4 rows per yaw-locked joint
        np.testing.assert_allclose(
            s.lam0[4 * j : 4 * j + 3], [0.0, 0.0, -(M_HUB + m) * G / 4.0], rtol=1e-6, atol=1e-6
        )


def test_off_centre_payload_tilts_as_the_statics() -> None:
    m, e = 1.0, 0.1
    s = _setup(m, e)
    x2 = np.sum((R_BUOY * np.cos(ANG)) ** 2)
    tilt = m * G * e / (_c33() * x2)
    assert np.degrees(tilt) == pytest.approx(0.869, abs=1e-3)  # floatsim_crosscheck.json
    assert s.xi0[6 * 4 + 4] == pytest.approx(tilt, rel=1e-3)  # hub pitch, +x side down


def test_without_preload_the_static_solve_sees_no_load() -> None:
    s = _setup(10.0, 0.1, preload=False)
    np.testing.assert_allclose(s.xi0, 0.0, atol=1e-12)
    assert s.lam0 is None


def test_preloaded_cluster_stays_at_rest() -> None:
    """The cluster with 10 kg on its hub, started at its preloaded equilibrium, stays there."""
    s = _setup(10.0, 0.0)
    r = integrate_cummins(
        lhs=s.lhs,
        kernel=s.kernel,
        xi0=s.xi0,
        xi_dot0=s.xi_dot0,
        duration=5.0,
        dt=0.01,
        rho_inf=0.8,
        constraints=s.constraints,
        state_force=s.state_force,
        projection_interval=1,
    )
    assert float(np.max(np.abs(r.xi - s.xi0))) < 1e-8
