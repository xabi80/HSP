"""M10 PR0.85 -- per-body label-resolved hydrostatic C in the coupled path.

A coupled BEM is inter-body radiation/excitation; hydrostatic stiffness is
per-body block-diagonal and cannot live there (the committed 18-DOF fixture
carries C=0). ``build_system(hydrostatic_database=...)`` supplies each
coupled body's 6x6 buoyancy C, resolved BY LABEL (single-body reference =
broadcast, M8's ``kron(I, c_single)``), placed block-diagonally.

GATE 1 (the substantive one, slow) is a DECAY, not an assembly check
(Amendment A3(d): structural assertions do not catch a missing restoring
force): the real M10 topology assembles AND oscillates, with the condensed
heave C33 = 663.2420101. GATE 4 is the label / silent-zero forcing
function.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.signal import find_peaks

from floatsim.driver import build_system
from floatsim.hydro.database import HydroDatabase
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    Output,
    RegularWave,
    Simulation,
    YawLockedJoint,
)
from floatsim.solver.newmark import integrate_cummins

REPO = Path(__file__).resolve().parents[2]
_NC = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_REF = REPO / "studies" / "cluster-3buoy-rigid" / "reference_single_bem.nc"
_C33_COMPOSITE = 663.2420101148214
_R = 0.5
_ANG = np.deg2rad([0.0, 120.0, 240.0])
_Z_BUOY = -1.1956674320202696
_Z_ARM = 0.4933695679797303


def _zeros_c_2body() -> HydroDatabase:
    """Small 2-body labelled coupled database with C=0 (a coupled BEM with
    no hydrostatics -- the real-fixture shape)."""
    nd = 12
    omega = np.linspace(0.1, 3.0, 6)
    a_inf = np.eye(nd) * 1.0e6
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0, 90.0]),
        A=np.stack([a_inf for _ in range(omega.size)], axis=-1),
        B=np.zeros((nd, nd, omega.size)),
        A_inf=a_inf,
        C=np.zeros((nd, nd)),  # <-- coupled BEM carries no hydrostatics
        RAO=np.zeros((nd, omega.size, 2), dtype=np.complex128),
        reference_point=np.zeros(3),
        C_source="full",
        metadata={},
        body_labels=("alpha", "beta"),
    )


def _coupled_2body_deck(*, declare_hydrostatic: bool) -> Deck:
    def _b(name: str, label: str) -> Body:
        return Body(
            name=name,
            reference_point=[0.0, 0.0, 0.0],
            mass=1.0e6,
            inertia=Inertia(Ixx=1e8, Iyy=1e8, Izz=1e8),
            hydro_body_label=label,
        )

    hydro = HydroDatabaseRef(format="capytaine", path="h.nc") if declare_hydrostatic else None
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.1),
        environment=Environment(water_depth=200.0, water_density=1025.0),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[_b("b0", "alpha"), _b("b1", "beta")],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path="s.nc"),
        hydrostatic_database=hydro,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


# --- GATE 4: the silent-zero / label forcing function (fast) ---


def test_zero_coupled_c_no_reference_raises() -> None:
    """A coupled BEM with C=0 and no hydrostatic_database RAISES rather than
    silently assembling a non-oscillating (zero-restoring) system."""
    with pytest.raises(ValueError, match=r"zero hydrostatic C"):
        build_system(
            _coupled_2body_deck(declare_hydrostatic=False),
            bem_databases={},
            dt=0.1,
            t_max_kernel=2.0,
            solve_equilibrium=False,
            shared_hydro_database=_zeros_c_2body(),
        )


def test_unknown_hydrostatic_label_raises() -> None:
    hydro = HydroDatabase(
        omega=np.linspace(0.1, 3.0, 6),
        heading_deg=np.array([0.0, 90.0]),
        A=np.stack([np.eye(12) for _ in range(6)], axis=-1),
        B=np.zeros((12, 12, 6)),
        A_inf=np.eye(12),
        C=np.eye(12) * 1e5,
        RAO=np.zeros((12, 6, 2), dtype=np.complex128),
        reference_point=np.zeros(3),
        C_source="full",
        metadata={},
        body_labels=("gamma", "delta"),  # do not match alpha/beta
    )
    with pytest.raises(ValueError, match=r"not found in hydrostatic_database labels"):
        build_system(
            _coupled_2body_deck(declare_hydrostatic=True),
            bem_databases={},
            dt=0.1,
            t_max_kernel=2.0,
            solve_equilibrium=False,
            shared_hydro_database=_zeros_c_2body(),
            hydrostatic_database=hydro,
        )


def test_deck_declares_hydrostatic_but_none_passed_raises() -> None:
    with pytest.raises(ValueError, match=r"none was passed to"):
        build_system(
            _coupled_2body_deck(declare_hydrostatic=True),
            bem_databases={},
            dt=0.1,
            t_max_kernel=2.0,
            solve_equilibrium=False,
            shared_hydro_database=_zeros_c_2body(),
        )


def test_hydrostatic_without_shared_db_raises() -> None:
    with pytest.raises(ValidationError, match=r"no 'shared_hydro_database'"):
        Deck(
            simulation=Simulation(duration=10.0, dt=0.1),
            environment=Environment(water_depth=200.0, water_density=1025.0),
            waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
            bodies=[
                Body(
                    name="b",
                    reference_point=[0.0, 0.0, 0.0],
                    mass=1.0,
                    inertia=Inertia(Ixx=1, Iyy=1, Izz=1),
                    hydro_database=HydroDatabaseRef(format="wamit", path="x"),
                )
            ],
            hydrostatic_database=HydroDatabaseRef(format="capytaine", path="h.nc"),
            output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
        )


# --- GATE 1: the real M10 topology assembles AND oscillates (slow, a DECAY) ---


def _hdb18() -> HydroDatabase:
    from floatsim.hydro.readers.capytaine import read_capytaine

    h = read_capytaine(_NC)
    w = np.asarray(h.omega)
    keep = np.array(
        [
            k
            for k in range(w.size)
            if k not in {int(np.argmin(np.abs(w - wc))) for wc in (4.934, 20.909)}
        ]
    )
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


def _m10_deck() -> Deck:
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
        environment=Environment(water_depth=200.0, water_density=1025.0, gravity=9.81),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[*buoys, hub],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC)),
        hydrostatic_database=HydroDatabaseRef(format="capytaine", path=str(_REF)),
        joints=joints,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


@pytest.mark.slow
def test_m10_assembles_and_oscillates() -> None:
    """PR0.85's gate is a DECAY: with the hydrostatic reference the coupled
    C33 (condensed heave) is 663.2420101 and the heave decay oscillates
    (peaks >= 3, finite T_n). The 3.106087 s VALUE is PR1's GATE A, not
    asserted here -- PR0.85 proves the system is dynamic, PR1 proves it is
    correct."""
    from floatsim.hydro.readers.capytaine import read_capytaine

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setup = build_system(
            _m10_deck(),
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=_hdb18(),
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override="M10 cluster small-body hulls; ITEM25",
        )

    c = np.asarray(setup.lhs.C)
    # condensed rigid-heave stiffness = sum of the buoys' heave diagonals;
    # the structural hub contributes zero.
    c33_condensed = sum(float(c[6 * k + 2, 6 * k + 2]) for k in range(3))
    assert c33_condensed == pytest.approx(_C33_COMPOSITE, rel=1e-6)
    assert np.max(np.abs(c[18:24, :])) == 0.0  # hub still contributes zero C

    xi0 = np.zeros(24)
    for k in range(4):
        xi0[6 * k + 2] = 0.10  # rigid heave IC
    r = integrate_cummins(
        lhs=setup.lhs,
        kernel=setup.kernel,
        xi0=xi0,
        xi_dot0=np.zeros(24),
        duration=50.0,
        dt=0.01,
        rho_inf=0.8,
        constraints=setup.constraints,
        projection_interval=1,
    )
    pk, _ = find_peaks(r.xi[:, 2], height=0.0)
    assert pk.size >= 3, f"heave decay did not oscillate (peaks={pk.size})"
    t_n = float(np.mean(np.diff(r.t[pk])))
    assert np.isfinite(t_n) and 2.5 < t_n < 3.7  # dynamic + physically plausible; PR1 pins 3.106
