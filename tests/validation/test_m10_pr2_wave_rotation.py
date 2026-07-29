"""M10 PR2 -- regular-wave rotation MEASUREMENT on the coupled articulated
model, and the in-band drag-free rotational-resonance finding.

Q6 fork resolved SUPPORTED: the wave-force builder
``make_regular_wave_force`` (excitation.py) is dimension-agnostic and
composes onto the coupled 18-DOF model through the
``integrate_cummins(external_force=...)`` hook, with the buoy force
scattered into global DOF [0:18] and the structural hub [18:24] carrying
zero wave force. The driver declines turnkey wiring (Q1); the caller
composes.

Per the plan, the rotation gate is a MEASUREMENT, not a pass/fail on
rotation amplitude. These tests gate the *correctness and reproducibility*
of the measurement (so Amendment A4's numbers stay honest), not whether
the rotation is above or below 0.1 rad:

  * convention (fast): the coupled RAO is origin-referenced, so
    ``body_position=(0,0,0)`` is the correct composition. Decisive check
    for heading 0 deg -- b1==b2 (y-mirror) and
    arg(b0)-arg(b1) == -k*(x0-x1).
  * off-resonance sensitivity (slow): the composed wave case at T=10 s
    reproduces ~0.0185 rad/m worst-joint rotation and a physical heave
    (~wave amplitude) -- the composition is correct and steady.
  * in-band rotational mode (slow): the free rotational decay is a
    stable, bounded, LIGHTLY damped oscillation with T_rot in-band
    (~3.26 s, adjacent to the 3.106 s heave resonance) -- the finding,
    and the evidence that the wave-case runaway is resonance not
    instability.

Validity domain (stated, not gated): off resonance (T>=10 s) the linear
model is valid and rotations are small; near the 3.1-3.4 s band the
drag-free measurement is undetermined (INBAND-ROTATIONAL-RESONANCE).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import find_peaks

from floatsim.driver import build_system
from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.excitation import make_regular_wave_force
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    Output,
    Simulation,
    YawLockedJoint,
)
from floatsim.io.deck import RegularWave as DeckWave
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave

REPO = Path(__file__).resolve().parents[2]
_NC = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_REF = REPO / "studies" / "cluster-3buoy-rigid" / "reference_single_bem.nc"
_CONTAM = (4.934, 20.909)
_R = 0.5
_ANG = np.deg2rad([0.0, 120.0, 240.0])
_Z_BUOY = -1.1956674320202696
_Z_ARM = 0.4933695679797303
_G = 9.81


# ---------------------------------------------------------------------------
# Convention gate (fast) -- no integration, just the RAO
# ---------------------------------------------------------------------------


def test_coupled_rao_origin_referenced() -> None:
    """The coupled Capytaine RAO carries each buoy's spatial phase (it is
    origin-referenced), so body_position=(0,0,0) composes correctly. For
    heading 0 deg: buoy1==buoy2 (y-mirror) and arg(b0)-arg(b1) equals the
    origin-referenced prediction -k*(x0-x1). A per-body-referenced RAO
    would give arg(b0)==arg(b1)."""
    h = read_capytaine(_NC)
    w = np.asarray(h.omega)
    rao = np.asarray(h.RAO)
    assert h.reference_point.tolist() == [0.0, 0.0, 0.0]
    x = [_R * np.cos(a) for a in _ANG]  # heading 0 deg -> k.x = k*x
    for wt in (1.6706, 2.0747):
        iw = int(np.argmin(np.abs(w - wt)))
        k = w[iw] ** 2 / _G
        heave = [rao[6 * j + 2, iw, 0] for j in range(3)]
        # y-mirror: buoy1 and buoy2 identical (heading 0 deg)
        assert heave[1] == pytest.approx(heave[2])
        # origin-referenced positional phase (dominant Froude-Krylov term)
        d_meas = float(np.angle(heave[0]) - np.angle(heave[1]))
        d_pred = -k * (x[0] - x[1])
        assert d_meas == pytest.approx(d_pred, abs=0.01)
        # and NOT per-body-referenced (which would be ~0)
        assert abs(d_meas) > 0.05


# ---------------------------------------------------------------------------
# Shared build (slow)
# ---------------------------------------------------------------------------


def _hdb18() -> HydroDatabase:
    h = read_capytaine(_NC)
    w = np.asarray(h.omega)
    drop = {int(np.argmin(np.abs(w - wc))) for wc in _CONTAM}
    keep = np.array([k for k in range(w.size) if k not in drop])
    return HydroDatabase(
        omega=h.omega[keep],
        heading_deg=h.heading_deg,
        A=h.A[:, :, keep],
        B=h.B[:, :, keep],
        A_inf=h.A_inf,
        C=h.C,
        RAO=h.RAO[:, keep, :],
        reference_point=h.reference_point,
        C_source=h.C_source,
        metadata=dict(h.metadata),
        body_labels=h.body_labels,
    )


def _deck() -> Deck:
    buoys = [
        Body(
            name=f"buoy{i + 1}",
            reference_point=[_R * np.cos(a), _R * np.sin(a), _Z_BUOY],
            mass=28.67,
            inertia=Inertia(Ixx=24.0, Iyy=24.0, Izz=0.114),
            hydro_body_label=f"buoy{i + 1}",
            initial_conditions=InitialConditions(),
        )
        for i, a in enumerate(_ANG)
    ]
    hub = Body(
        name="hub",
        reference_point=[0.0, 0.0, _Z_ARM],
        mass=12.0,
        inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0),
        structural=True,
    )
    joints = [
        YawLockedJoint(
            type="yaw_locked",
            body_a=f"buoy{i + 1}",
            body_b="hub",
            attach_a_body=[0.0, 0.0, _Z_ARM - _Z_BUOY],
            attach_b_body=[_R * np.cos(a), _R * np.sin(a), 0.0],
            axis=[0.0, 0.0, 1.0],
        )
        for i, a in enumerate(_ANG)
    ]
    return Deck(
        simulation=Simulation(duration=50.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=1025.0, gravity=_G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[*buoys, hub],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC)),
        hydrostatic_database=HydroDatabaseRef(format="capytaine", path=str(_REF)),
        joints=joints,
        output=Output(file="out.h5", channels=["heave"], sample_rate=10.0),
    )


@pytest.fixture(scope="module")
def hdb() -> HydroDatabase:
    return _hdb18()


@pytest.fixture(scope="module")
def setup(hdb):  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(
            _deck(),
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=hdb,
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override="M10 cluster small-body hulls; ITEM25",
        )


def _rel_rot(r):  # type: ignore[no-untyped-def]
    """max over time of buoy-hub relative rotation |(droll,dpitch)| per joint."""
    hr, hp = r.xi[:, 21], r.xi[:, 22]
    return np.array(
        [np.sqrt((r.xi[:, 6 * j + 3] - hr) ** 2 + (r.xi[:, 6 * j + 4] - hp) ** 2) for j in range(3)]
    )


# ---------------------------------------------------------------------------
# Off-resonance measurement (slow) -- reproducibility of Amendment A4's number
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_offresonance_rotation_sensitivity(setup, hdb) -> None:  # type: ignore[no-untyped-def]
    """A composed regular wave at T=10 s (off resonance, model valid) drives
    a steady, physical response: worst-joint rotation ~0.0185 rad/m and a
    heave near the wave amplitude. This gates that the caller-side wave-force
    composition is CORRECT and steady (A4's off-res sensitivity), not a
    pass/fail on the rotation magnitude."""
    wave = RegularWave(amplitude=1.0, omega=2 * np.pi / 10.0, heading_deg=0.0)
    f18 = make_regular_wave_force(
        hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0), ramp=HalfCosineRamp(duration=20.0)
    )

    def ext(t: float) -> np.ndarray:
        f = np.zeros(24)
        f[:18] = f18(t)
        return f

    r = integrate_cummins(
        lhs=setup.lhs,
        kernel=setup.kernel,
        xi0=np.zeros(24),
        xi_dot0=np.zeros(24),
        duration=160.0,
        dt=0.01,
        rho_inf=0.8,
        constraints=setup.constraints,
        external_force=ext,
        projection_interval=1,
    )
    steady = r.t >= 120.0
    worst = _rel_rot(r)[:, steady].max()
    assert worst == pytest.approx(0.0185, rel=0.20)  # A4 off-res sensitivity (rad/m)
    # physical heave: ~ wave amplitude, and NOT a runaway (off resonance)
    heave = np.abs(r.xi[steady, 2]).max()
    assert 0.7 < heave < 1.4


# ---------------------------------------------------------------------------
# The finding (slow) -- in-band, lightly damped, STABLE rotational mode
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_inband_rotational_mode(setup) -> None:  # type: ignore[no-untyped-def]
    """Free rotational decay (buoy pitch IC, no wave): a stable, BOUNDED,
    lightly-damped oscillation with T_rot in-band (~3.26 s, adjacent to the
    3.106 s heave resonance). Bounded-and-decaying proves the integrator +
    KKT handling are sound, so the wave-case runaway is genuine resonant
    buildup (Q~134), not numerical instability."""
    xi0 = np.zeros(24)
    for j in range(3):
        xi0[6 * j + 4] = 0.02  # pitch IC on the buoys
    r = integrate_cummins(
        lhs=setup.lhs,
        kernel=setup.kernel,
        xi0=xi0,
        xi_dot0=np.zeros(24),
        duration=120.0,
        dt=0.01,
        rho_inf=0.8,
        constraints=setup.constraints,
        projection_interval=1,
    )
    th = r.xi[:, 4] - r.xi[:, 22]  # buoy0 pitch relative to hub
    # bounded + decays -> stable (rules out numerical instability)
    assert th.max() <= 0.0201
    pk, _ = find_peaks(th, height=0.0)
    assert pk.size >= 5
    t_rot = float(np.mean(np.diff(r.t[pk])))
    assert 2.5 < t_rot < 4.0  # in-band, adjacent to the 3.106 s heave mode
    # lightly damped (radiation-only) -> the Q that drives the runaway
    amps = th[pk]
    ld = np.log(amps[:-1] / amps[1:])
    ld = ld[np.isfinite(ld) & (ld > 0)]
    zeta = float(np.mean(ld) / (2 * np.pi))
    assert 0.0 < zeta < 0.01  # < 1% => Q > 50
