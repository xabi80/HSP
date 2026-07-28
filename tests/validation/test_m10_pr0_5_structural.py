"""M10 PR0.5 -- structural (hydro-free) body support in the coupled assembly.

FORK 1 (driver-local; plan Amendment A1 / Q4-c): a deck may mix hydro
(``hydro_body_label``) bodies backed by a ``shared_hydro_database`` with
``structural: true`` bodies (an articulation hub / arm structure). The
hydro blocks are scattered by label into the global 6*n_deck matrices;
each structural body contributes rigid mass/inertia only -- ZERO to
A_inf, B, the retardation kernel and hydrostatic C. The kernel is
computed on the hydro-only sub-database and embedded to global size.

GATE 1 (the substantive one, slow) uses the REAL M10 topology -- 3 buoys
on the committed 18-DOF coupled fixture + 1 structural hub with the
Q2-locked properties -- so PR0.5's gate is directly PR1's precondition.
GATE 2 (byte-identity of every existing deck path) is held by the
no-structural coupled path being untouched (the structural guard routes
mixed decks to a separate helper) and confirmed by the unchanged pass of
test_deck / test_driver / test_m9_coupled_build in the full suite.
GATE 3 (typo protection) and GATE 4 ("exactly one") are the fast
schema-forcing-function tests.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

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
)

REPO = Path(__file__).resolve().parents[2]
_NC = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_CONTAM = (4.934, 20.909)  # M8 PR3 contaminated-frequency exclusion (grid selection)
_R = 0.5
_ANG = np.deg2rad([0.0, 120.0, 240.0])
_Z_BUOY = -1.1956674320202696
_Z_ARM = 0.4933695679797303


def _inertia() -> Inertia:
    return Inertia(Ixx=1.0e8, Iyy=1.0e8, Izz=1.0e8)


# ---------------------------------------------------------------------------
# GATE 4 -- "exactly one of {hydro_database, hydro_body_label, structural}"
# ---------------------------------------------------------------------------


def test_structural_plus_hydro_source_raises() -> None:
    with pytest.raises(ValidationError, match=r"exactly one of"):
        Body(
            name="hub",
            reference_point=[0.0, 0.0, 0.0],
            mass=12.0,
            inertia=_inertia(),
            structural=True,
            hydro_database=HydroDatabaseRef(format="wamit", path="x"),
        )


def test_structural_plus_label_raises() -> None:
    with pytest.raises(ValidationError, match=r"exactly one of"):
        Body(
            name="hub",
            reference_point=[0.0, 0.0, 0.0],
            mass=12.0,
            inertia=_inertia(),
            structural=True,
            hydro_body_label="alpha",
        )


# ---------------------------------------------------------------------------
# GATE 3 -- typo protection (the amendment's whole point)
# ---------------------------------------------------------------------------


def test_no_declaration_raises_got_zero() -> None:
    """A body that declares NONE of the three raises -- never a silent
    hydro-free body."""
    with pytest.raises(ValidationError, match=r"exactly one of.*got 0"):
        Body(name="b", reference_point=[0.0, 0.0, 0.0], mass=1.0, inertia=_inertia())


def test_misspelled_hydro_key_raises() -> None:
    """A misspelled key (hydro_databse) is rejected by the forbid-extra
    schema before it can become a silent hydro-free body."""
    with pytest.raises(ValidationError, match=r"[Ee]xtra"):
        Body.model_validate(
            {
                "name": "b",
                "reference_point": [0.0, 0.0, 0.0],
                "mass": 1.0,
                "inertia": {"Ixx": 1e8, "Iyy": 1e8, "Izz": 1e8},
                "hydro_databse": {"format": "wamit", "path": "x"},  # typo
            }
        )


def test_structural_body_requires_coupled_deck() -> None:
    """A structural body outside a coupled deck (no shared_hydro_database)
    raises -- structural support lives only in the coupled assembly."""
    with pytest.raises(ValidationError, match=r"require a coupled deck"):
        Deck(
            simulation=Simulation(duration=10.0, dt=0.01),
            environment=Environment(water_depth=200.0, water_density=1025.0),
            waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
            bodies=[
                Body(
                    name="hub",
                    reference_point=[0.0, 0.0, 0.0],
                    mass=12.0,
                    inertia=_inertia(),
                    structural=True,
                )
            ],
            output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
        )


# ---------------------------------------------------------------------------
# GATE 1 -- the real M10 4-body mixed assembly (slow)
# ---------------------------------------------------------------------------


def _hdb18() -> HydroDatabase:
    from floatsim.hydro.readers.capytaine import read_capytaine

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
        inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0),  # Q2 rod-derived hub inertia
        structural=True,
    )
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=1025.0, gravity=9.81),
        waves=RegularWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[*buoys, hub],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC)),
        output=Output(file="out.h5", channels=["heave"], sample_rate=10.0),
    )


@pytest.mark.slow
def test_m10_mixed_assembly_structural_hub() -> None:
    """The 4-body M10 system (3 hydro buoys + 1 structural hub) assembles;
    the hub carries rigid mass/inertia only, contributes exactly zero to
    every hydrodynamic quantity, and rank(M_plus_Ainf) == 24 (PR1's
    precondition -- Phase-1 measured a point-mass hub gives 21/24)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Item-25 bypass warning
        setup = build_system(
            _m10_deck(),
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=_hdb18(),
            asymptote_check_override="M10 cluster small-body hulls; ITEM25-SMALL-BODY",
        )

    mpa = np.asarray(setup.lhs.M_plus_Ainf)
    c = np.asarray(setup.lhs.C)
    k = np.asarray(setup.kernel.K)
    hub = slice(18, 24)  # hub is deck body index 3 -> global DOF [18, 24)

    # shapes: global 6*n_deck = 24; kernel embedded to global, lhs/kernel agree
    assert mpa.shape == (24, 24)
    assert k.shape[:2] == (24, 24)
    assert setup.lhs.n_dof == setup.kernel.n_dof == 24

    # by-label mapping: each buoy's block is at its deck DOF range (identity
    # here -- deck order == label order); the hub is body 3.
    assert setup.body_name_to_index == {"buoy1": 0, "buoy2": 1, "buoy3": 2, "hub": 3}

    # the hub's 6 DOF carry ONLY its rigid mass/inertia (Q2), no A_inf.
    np.testing.assert_allclose(
        np.diag(mpa[hub, hub]), [12.0, 12.0, 12.0, 0.5, 0.5, 1.0], rtol=1e-12
    )
    # ...and EXACTLY zero hydrodynamics: A_inf coupling, C, kernel (asserted).
    assert np.max(np.abs(mpa[hub, :18])) == 0.0  # no hub<->buoy added-mass coupling
    assert np.max(np.abs(mpa[:18, hub])) == 0.0
    assert np.max(np.abs(c[hub, :])) == 0.0  # hub row of hydrostatic C
    assert np.max(np.abs(c[:, hub])) == 0.0
    assert np.max(np.abs(k[hub, :, :])) == 0.0  # hub rows of the kernel
    assert np.max(np.abs(k[:, hub, :])) == 0.0

    # total mass = sum of body masses = 98.01 kg (3x28.67 + 12); the
    # structural hub's mass IS in the assembly (its rigid block above is
    # exactly diag(12,12,12,...), no drop). Recover each body's rigid
    # translational mass as M_plus_Ainf minus the scattered hydro A_inf.
    a_inf18 = np.asarray(_hdb18().A_inf)
    rigid_mass = [float(mpa[6 * b, 6 * b]) for b in range(4)]
    rigid_mass[0] -= float(a_inf18[0, 0])  # buoy1 surge A_inf
    rigid_mass[1] -= float(a_inf18[6, 6])  # buoy2 surge A_inf
    rigid_mass[2] -= float(a_inf18[12, 12])  # buoy3 surge A_inf
    # rigid_mass[3] (hub) has no A_inf to subtract
    assert sum(rigid_mass) == pytest.approx(3 * 28.67 + 12.0, rel=1e-9)  # 98.01 kg
    assert rigid_mass[3] == pytest.approx(12.0, rel=1e-12)  # the structural hub mass

    # PR1 precondition: the projection metric inv(M_plus_Ainf) is well-posed.
    assert np.linalg.matrix_rank(mpa) == 24
