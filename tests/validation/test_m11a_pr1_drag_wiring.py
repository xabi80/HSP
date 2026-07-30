"""M11a PR1 -- Morison drag wiring into build_system (plan Q3-i).

WIRING only: build_system reads each ``Body.drag_elements``, constructs
``MorisonElement``s at the body's slot, composes
``make_morison_state_force`` and sums it into ``SimulationSetup.state_force``
alongside connector/catenary. No new physics (no plate extension Q3-iii, no
spar lateral elements Q3-ii).

GATE 1 (regression, the substantive one -- a DECAY per M10 A3(d)): the deck
path composes the IDENTICAL drag force the committed spar-fin study applied
by hand, reproducing its BEM+Morison decay and effective damping
ζ = 2.5225e-02. Proven by force-identity + trajectory-identity + ζ.

Note (finding, reported in the PR): build_system's PER-BODY path calls
``compute_retardation_kernel`` WITHOUT the Item-25 override, so the
small-body spar-fin BEM cannot be built end-to-end through the per-body
path (the study documents this as why it hand-assembles). GATE 1 therefore
uses the study's own lhs/kernel + the deck-path-composed drag force; GATE 4
exercises the full build_system composition on the COUPLED path (which
accepts the override, and is M11's target).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import find_peaks

from floatsim.driver import _build_drag_state_force, _compose_state_force, build_system
from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    MorisonMember,
    Output,
    Simulation,
    YawLockedJoint,
)
from floatsim.io.deck import RegularWave as DeckWave
from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.newmark import integrate_cummins

REPO = Path(__file__).resolve().parents[2]
_SPAR = REPO / "studies" / "spar-fin-decay"
_NC18 = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_REF = REPO / "studies" / "cluster-3buoy-rigid" / "reference_single_bem.nc"
_STUDY_ZETA = 2.5225e-02  # studies/spar-fin-decay/results/summary.md (effective, first peaks)


def _zeta_first_peaks(t, x):  # type: ignore[no-untyped-def]
    """Study's method (analyze_and_plot.py:_peaks_period_and_zeta)."""
    pk, _ = find_peaks(x, height=1e-4)
    amps = x[pk]
    ratios = amps[:-1] / amps[1:]
    delta = float(np.mean(np.log(ratios[ratios > 0])))
    return float(delta / np.sqrt(4.0 * np.pi**2 + delta**2)), pk.size


# ---------------------------------------------------------------------------
# GATE 1 -- spar-fin BEM+Morison regression through the deck-path drag
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gate1_spar_fin_drag_regression() -> None:
    """The deck path composes the SAME drag force the study hand-assembled,
    reproduces its decay exactly, and its effective ζ matches the committed
    2.5225e-02."""
    sys.path.insert(0, str(_SPAR))
    import study_common as sc  # type: ignore[import-not-found]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdb = sc.load_hdb()
        lhs = sc.build_lhs(hdb)
        kernel = sc.build_kernel(hdb)
    study_force = sc.make_morison_force()

    # deck-path drag: a minimal 1-body deck carrying the same plate element
    deck = Deck(
        simulation=Simulation(duration=sc.DURATION, dt=sc.DT),
        environment=Environment(water_depth=200.0, water_density=sc.RHO, gravity=sc.G),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[
            Body(
                name="spar",
                reference_point=[0.0, 0.0, sc.CoG_Z],
                mass=sc.M_BODY,
                inertia=Inertia(Ixx=sc.I_XX, Iyy=sc.I_YY, Izz=sc.I_ZZ),
                hydro_database=HydroDatabaseRef(format="capytaine", path="x.nc"),
                drag_elements=[
                    MorisonMember(
                        type="morison_member",
                        node_a=[-sc.PLATE_L / 2.0, 0.0, sc.PLATE_Z],
                        node_b=[sc.PLATE_L / 2.0, 0.0, sc.PLATE_Z],
                        diameter=sc.PLATE_D,
                        Cd=sc.PLATE_CD,
                        Ca=0.0,
                    )
                ],
            )
        ],
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )
    deck_force = _build_drag_state_force(deck, n_dof=6, rho=sc.RHO)
    assert deck_force is not None

    # (i) FORCE IDENTITY: deck-path drag == study hand-assembled, exact
    rng = np.random.default_rng(0)
    for _ in range(8):
        xi = rng.standard_normal(6) * 0.1
        xd = rng.standard_normal(6) * 0.2
        np.testing.assert_array_equal(deck_force(1.3, xi, xd), study_force(1.3, xi, xd))

    # study equilibrium + IC (decay_run.py:47,79)
    eq = solve_static_equilibrium(lhs=lhs, state_force=None)
    xi0 = eq.xi_eq.copy()
    xi0[2] += sc.IC_HEAVE

    def _decay(force):  # type: ignore[no-untyped-def]
        return integrate_cummins(
            lhs=lhs,
            kernel=kernel,
            xi0=xi0,
            xi_dot0=np.zeros(6),
            duration=sc.DURATION,
            dt=sc.DT,
            state_force=force,
        )

    r_study = _decay(study_force)
    r_deck = _decay(deck_force)
    # (ii) TRAJECTORY IDENTITY
    np.testing.assert_allclose(r_deck.xi, r_study.xi, rtol=1e-12, atol=1e-12)
    # (iii) effective ζ vs the study's committed value (5 sig figs -> rel 1e-4)
    z_deck, npk = _zeta_first_peaks(r_deck.t, r_deck.xi[:, 2])
    assert npk >= 3
    assert z_deck == pytest.approx(_STUDY_ZETA, rel=1e-4)


# ---------------------------------------------------------------------------
# GATE 2 -- byte-identity: a drag-free deck's state_force is untouched
# ---------------------------------------------------------------------------


def _no_drag_deck() -> Deck:
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.1),
        environment=Environment(water_depth=200.0, water_density=1025.0),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[
            Body(
                name="b",
                reference_point=[0.0, 0.0, 0.0],
                mass=1.0e6,
                inertia=Inertia(Ixx=1e8, Iyy=1e8, Izz=1e8),
                hydro_body_label="alpha",
            )
        ],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path="s.nc"),
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def test_gate2_no_drag_returns_none_and_compose_byte_identical() -> None:
    """A deck with no drag_elements yields drag_force=None, and the composed
    state_force is byte-identical to connector+catenary alone (the pre-M11a
    two-source path)."""
    assert _build_drag_state_force(_no_drag_deck(), n_dof=6, rho=1025.0) is None

    rng = np.random.default_rng(1)

    def cf(_t, xi, xd):  # type: ignore[no-untyped-def]
        return xi * 2.0

    def af(_t, xi, xd):  # type: ignore[no-untyped-def]
        return xd * 3.0

    composed = _compose_state_force(cf, af, None, 6)
    for _ in range(5):
        xi = rng.standard_normal(6)
        xd = rng.standard_normal(6)
        np.testing.assert_array_equal(composed(0.7, xi, xd), cf(0.7, xi, xd) + af(0.7, xi, xd))


# ---------------------------------------------------------------------------
# GATE 3 -- changed-behavior surface: the committed drag-carrying decks
# ---------------------------------------------------------------------------


def test_gate3_committed_drag_decks_now_execute() -> None:
    """The committed decks that carry drag blocks (examples/two_body_semisub_
    barge.yml, studies/cluster-3buoy-rigid/deck_bem_morison.yaml) now compose
    a non-None drag force once built -- documenting the changed-behavior
    surface. Neither is built via build_system in the suite (parse-only /
    study-hand-assembled), so no tested behavior changes."""
    from floatsim.io.deck import load_deck

    barge = load_deck(REPO / "examples" / "two_body_semisub_barge.yml")
    # semisub carries a drag element (Ca=1.0, include_inertia default False -> inert)
    drag = _build_drag_state_force(barge, n_dof=6 * len(barge.bodies), rho=1025.0)
    assert drag is not None  # would now execute if this deck were built


def test_gate3_include_inertia_rejected() -> None:
    """include_inertia=True is rejected structurally (double-count impossible)."""
    deck = Deck(
        simulation=Simulation(duration=10.0, dt=0.1),
        environment=Environment(water_depth=200.0, water_density=1025.0),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[
            Body(
                name="b",
                reference_point=[0.0, 0.0, 0.0],
                mass=1.0,
                inertia=Inertia(Ixx=1, Iyy=1, Izz=1),
                hydro_database=HydroDatabaseRef(format="wamit", path="x"),
                drag_elements=[
                    MorisonMember(
                        type="morison_member",
                        node_a=[-0.5, 0, 0],
                        node_b=[0.5, 0, 0],
                        diameter=1.0,
                        Cd=1.0,
                        include_inertia=True,
                    )
                ],
            )
        ],
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )
    with pytest.raises(ValueError, match=r"include_inertia=True is not supported"):
        _build_drag_state_force(deck, n_dof=6, rho=1025.0)


# ---------------------------------------------------------------------------
# GATE 4 -- no double-counting: drag is force-only, never inertia (coupled path)
# ---------------------------------------------------------------------------


def _hdb18() -> HydroDatabase:
    h = read_capytaine(_NC18)
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


def _coupled_deck(with_drag: bool) -> Deck:
    r, ang, zb, za = 0.5, np.deg2rad([0.0, 120.0, 240.0]), -1.1956674320202696, 0.4933695679797303
    buoys = []
    for i, a in enumerate(ang):
        de = []
        if with_drag and i == 0:
            de = [
                MorisonMember(
                    type="morison_member",
                    node_a=[-0.215, 0.0, -1.278],
                    node_b=[0.215, 0.0, -1.278],
                    diameter=0.3377,
                    Cd=5.0,
                    Ca=0.0,
                )
            ]
        buoys.append(
            Body(
                name=f"buoy{i + 1}",
                reference_point=[r * np.cos(a), r * np.sin(a), zb],
                mass=28.67,
                inertia=Inertia(Ixx=24.0, Iyy=24.0, Izz=0.114),
                hydro_body_label=f"buoy{i + 1}",
                initial_conditions=InitialConditions(),
                drag_elements=de,
            )
        )
    hub = Body(
        name="hub",
        reference_point=[0.0, 0.0, za],
        mass=12.0,
        inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0),
        structural=True,
    )
    joints = [
        YawLockedJoint(
            type="yaw_locked",
            body_a=f"buoy{i + 1}",
            body_b="hub",
            attach_a_body=[0.0, 0.0, za - zb],
            attach_b_body=[r * np.cos(a), r * np.sin(a), 0.0],
            axis=[0.0, 0.0, 1.0],
        )
        for i, a in enumerate(ang)
    ]
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=1025.0, gravity=9.81),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[*buoys, hub],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC18)),
        hydrostatic_database=HydroDatabaseRef(format="capytaine", path=str(_REF)),
        joints=joints,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


@pytest.mark.slow
def test_gate4_drag_is_force_only_no_inertia() -> None:
    """build_system's COUPLED path composes drag end-to-end; drag contributes
    FORCE only -- M_plus_Ainf (and C, kernel) are byte-identical with vs
    without drag, and only state_force differs."""
    hdb = _hdb18()
    ref = read_capytaine(_REF)
    kw = dict(
        bem_databases={},
        dt=0.01,
        t_max_kernel=30.0,
        solve_equilibrium=False,
        shared_hydro_database=hdb,
        hydrostatic_database=ref,
        asymptote_check_override="M11a PR1 GATE4 small-body",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_no = build_system(_coupled_deck(with_drag=False), **kw)  # type: ignore[arg-type]
        s_dr = build_system(_coupled_deck(with_drag=True), **kw)  # type: ignore[arg-type]

    # drag touches NO inertia / stiffness / kernel
    np.testing.assert_array_equal(s_dr.lhs.M_plus_Ainf, s_no.lhs.M_plus_Ainf)
    np.testing.assert_array_equal(s_dr.lhs.C, s_no.lhs.C)
    np.testing.assert_array_equal(s_dr.kernel.K, s_no.kernel.K)

    # ...but state_force DOES now carry drag (nonzero where the no-drag one is
    # zero), at a state with buoy1 heave velocity.
    xi = np.zeros(24)
    xd = np.zeros(24)
    xd[2] = 1.0  # buoy1 heave velocity -> plate drag
    f_no = s_no.state_force(0.0, xi, xd)
    f_dr = s_dr.state_force(0.0, xi, xd)
    assert np.max(np.abs(f_no)) == 0.0
    assert abs(f_dr[2]) > 0.0  # drag opposes buoy1 heave
