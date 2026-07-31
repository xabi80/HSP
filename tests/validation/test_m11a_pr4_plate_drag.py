"""M11a PR4 -- direction-dependent (anisotropic) plate drag (plan Q3-iii).

The ONLY genuinely-new-physics piece in M11a. A heave plate resists broadside
(normal) flow far more than edge-on (tangential) flow; a member-normal
:class:`MorisonElement` cylinder is drag-ISOTROPIC in its normal plane and
cannot represent this (the M11a-PR1 horizontal-cylinder heave-plate stand-in
mis-models a tilting plate). :class:`PlateDragElement` decomposes the flow:

- NORMAL (broadside), Cd_n = 5.0 KNOWN, integrated over the disc face -- captures
  both heave (uniform w) and the tilting-rotational contribution
  (``INT|x|^3 dA = 8a^5/15``);
- TANGENTIAL (edge-on), Cd_t tank-pending (carried as a [1,2] sensitivity),
  lumped at the rim -- a MINOR term.

Gates:
- GATE 1: measured plate rotational drag vs the MODAL-KINEMATICS reference
  (constrained eigenanalysis of the drag-free system, F1 discipline -- NEVER a
  fixed pivot). Normal and tangential reported SEPARATELY; the re-derived
  ``E_normal/E_tangential = 1.8-3.5`` split (STEP 1's 3.9-7.7 used a wrong plate
  depth -0.125; the committed depth is -0.2617 -- the record wins) is a
  PREDICTION the code confirms.
- GATE 2: byte-identity (a drag-free deck builds identically; drag is force-only).
- GATE 3 (structural): in pure heave u_n is uniform, so the plate reduces EXACTLY
  to the single-Cd cylinder -- force-identity at machine precision, and the heave
  decay reproduces the committed zeta = 2.5225e-02.
- GATE 4 (analytical): the disc quadrature converges to ``8a^5/15`` (residual
  asserted at the adopted strip count) and the face area is exact.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh, null_space
from scipy.signal import find_peaks

from floatsim.bodies.rigid_body import quaternion_from_euler_zyx, rotation_matrix
from floatsim.driver import _build_drag_state_force, build_system
from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.morison import (
    MorisonElement,
    PlateDragElement,
    morison_element_force,
    plate_element_force,
)
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    Output,
    PlateMember,
    Simulation,
    YawLockedJoint,
)
from floatsim.io.deck import RegularWave as DeckWave
from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.newmark import integrate_cummins

REPO = Path(__file__).resolve().parents[2]
_SPAR = REPO / "studies" / "spar-fin-decay"
_NC = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_REF = REPO / "studies" / "cluster-3buoy-rigid" / "reference_single_bem.nc"

_R, _ANG = 0.5, np.deg2rad([0.0, 120.0, 240.0])
_ZB, _ZA = -1.1956674320202696, 0.4933695679797303
_RHO = 1025.0
# Plate geometry -- re-derived from committed sources (PR2 test:47, study_common).
_A_PLATE = 0.215  # disc radius (study_common PLATE_RADIUS)
_T_PLATE = 0.0039  # rim thickness ~4mm
_Z_PLATE_B = -1.45737 - _ZB  # plate z, body frame = -0.2617 (PR2 test:47)
_CD_N = 5.0  # KNOWN heave-plate broadside coefficient
_CD_T = 1.5  # edge-on coefficient (mid of the [1,2] tank-pending sensitivity)
_INT_X3 = 8.0 * _A_PLATE**5 / 15.0  # analytical INT INT |x|^3 dA over the disc
_STUDY_ZETA = 2.5225e-02  # committed heave decay (studies/spar-fin-decay/results/summary.md)


# ---------------------------------------------------------------------------
# Coupled cluster fixtures (shared with PR2)
# ---------------------------------------------------------------------------


def _hdb18() -> HydroDatabase:
    h = read_capytaine(_NC)
    w = np.asarray(h.omega)
    keep = np.array(
        [
            k
            for k in range(w.size)
            if k not in {int(np.argmin(np.abs(w - c))) for c in (4.934, 20.909)}
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


def _coupled_deck(with_plate: bool, cd_n: float = _CD_N, cd_t: float = _CD_T) -> Deck:
    plate = (
        [
            PlateMember(
                type="plate",
                center=[0.0, 0.0, _Z_PLATE_B],
                normal=[0.0, 0.0, 1.0],
                radius=_A_PLATE,
                thickness=_T_PLATE,
                Cd_n=cd_n,
                Cd_t=cd_t,
            )
        ]
        if with_plate
        else []
    )
    buoys = [
        Body(
            name=f"buoy{i + 1}",
            reference_point=[_R * np.cos(a), _R * np.sin(a), _ZB],
            mass=28.67,
            inertia=Inertia(Ixx=24.0, Iyy=24.0, Izz=0.114),
            hydro_body_label=f"buoy{i + 1}",
            initial_conditions=InitialConditions(),
            drag_elements=list(plate),
        )
        for i, a in enumerate(_ANG)
    ]
    hub = Body(
        name="hub",
        reference_point=[0.0, 0.0, _ZA],
        mass=12.0,
        inertia=Inertia(Ixx=0.5, Iyy=0.5, Izz=1.0),
        structural=True,
    )
    joints = [
        YawLockedJoint(
            type="yaw_locked",
            body_a=f"buoy{i + 1}",
            body_b="hub",
            attach_a_body=[0.0, 0.0, _ZA - _ZB],
            attach_b_body=[_R * np.cos(a), _R * np.sin(a), 0.0],
            axis=[0.0, 0.0, 1.0],
        )
        for i, a in enumerate(_ANG)
    ]
    return Deck(
        simulation=Simulation(duration=60.0, dt=0.01),
        environment=Environment(water_depth=200.0, water_density=_RHO, gravity=9.81),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[*buoys, hub],
        shared_hydro_database=HydroDatabaseRef(format="capytaine", path=str(_NC)),
        hydrostatic_database=HydroDatabaseRef(format="capytaine", path=str(_REF)),
        joints=joints,
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def _build_coupled(with_plate: bool, cd_n: float = _CD_N, cd_t: float = _CD_T):  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(
            _coupled_deck(with_plate, cd_n, cd_t),
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=_hdb18(),
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override="M11a PR4 plate drag",
        )


# ---------------------------------------------------------------------------
# Modal-kinematics reference (constrained eigenanalysis -- predates drag code)
# ---------------------------------------------------------------------------


def _reference(setup):  # type: ignore[no-untyped-def]
    """Rotational mode shape from the DRAG-FREE constrained eigenanalysis.

    Returns ``(ph, w_mode, i_eff)`` with ``ph`` normalised to buoy0 pitch = 1.
    Independent of the plate drag code (F1 discipline).
    """
    ma = np.asarray(setup.lhs.M_plus_Ainf)
    c = np.asarray(setup.lhs.C)
    g = setup.constraints.jacobian(np.zeros(24))
    z = null_space(g)
    _w2, vq = eigh(z.T @ c @ z, z.T @ ma @ z)
    modes = z @ vq
    w = np.sqrt(np.abs(_w2))
    cand = max((k for k in range(w.size) if 1.5 < w[k] < 2.3), key=lambda k: abs(modes[4, k]))
    ph = modes[:, cand] / modes[4, cand]
    return ph, float(w[cand]), float(ph @ ma @ ph)


def _plate_zeta(ph, i_eff, w_mode, theta, cd_n, cd_t):  # type: ignore[no-untyped-def]
    """Energy-equivalent plate-drag zeta at pitch-angle amplitude ``theta``.

    Returns ``(zeta_normal, zeta_tangential)`` -- reported SEPARATELY. Same
    energy form as PR2: ``E_diss = 0.5*rho*Cd*(8/3w)*vamp^3 * geom``. Normal
    (tilting): ``vamp = |pitch_rate|*thd`` over ``INT|x|^3 dA``. Tangential
    (edge-on): ``vamp = |a_c|*thd`` over rim area ``t*2a`` at the disc centre.
    """
    thd = theta * w_mode
    e_store = 0.5 * i_eff * thd**2
    a_edge = _T_PLATE * 2.0 * _A_PLATE
    e_n = sum(
        0.5 * _RHO * cd_n * (8.0 / 3.0) / w_mode * (thd * abs(ph[6 * b + 4])) ** 3 * _INT_X3
        for b in range(3)
    )
    e_t = sum(
        0.5
        * _RHO
        * cd_t
        * a_edge
        * (8.0 / 3.0)
        / w_mode
        * (thd * abs(ph[6 * b + 0] + ph[6 * b + 4] * _Z_PLATE_B)) ** 3
        for b in range(3)
    )
    return e_n / (4 * np.pi * e_store), e_t / (4 * np.pi * e_store)


def _modal_decay(setup, ph, ma, use_drag, theta0, n_peaks=2):  # type: ignore[no-untyped-def]
    """Decay from the PURE eigenmode IC (``xi0 = ph*theta0``, constraint-
    consistent), measured in the MODAL coordinate ``q = ph^T(M+A)xi /
    ph^T(M+A)ph`` -- the coordinate the energy-equivalent reference is defined
    in. A single/differential DOF over-reads the modal decay rate by ~16-26 %
    (Finding F3 item-3 investigation); the modal coordinate removes that
    artifact, so this end-to-end check is tight (not the loose rel=0.35 a
    differential-DOF measure would force)."""
    r = integrate_cummins(
        lhs=setup.lhs,
        kernel=setup.kernel,
        xi0=ph * theta0,
        xi_dot0=np.zeros(24),
        duration=60.0,
        dt=0.01,
        rho_inf=0.8,
        constraints=setup.constraints,
        projection_interval=1,
        state_force=setup.state_force if use_drag else None,
    )
    q = (r.xi @ ma @ ph) / float(ph @ ma @ ph)
    pk, _ = find_peaks(q, height=0.0)
    amps = q[pk][: n_peaks + 1]
    d = np.log(amps[:-1] / amps[1:])
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.mean(d) / (2 * np.pi)), float(np.mean(amps[:2]))


def _plate_power_at_modal_state(ph, w_mode, cd_n, cd_t):  # type: ignore[no-untyped-def]
    """Instantaneous plate dissipation power ``-F.v`` at the peak-velocity modal
    state (calm sea, identity pose). Ratio of the (Cd_t=0) and (Cd_n=0) builds
    is the normal/tangential energy split the analytical reference predicts."""
    thd = 0.02 * w_mode
    xdot = np.zeros(24)
    for b in range(3):
        xdot[6 * b : 6 * b + 6] = ph[6 * b : 6 * b + 6] * thd
    r_mat = rotation_matrix(quaternion_from_euler_zyx(roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0))
    power = 0.0
    for b in range(3):
        pe = PlateDragElement(
            body_index=b,
            center_body=np.array([0.0, 0.0, _Z_PLATE_B]),
            normal_body=np.array([0.0, 0.0, 1.0]),
            radius=_A_PLATE,
            thickness=_T_PLATE,
            Cd_n=cd_n,
            Cd_t=cd_t,
        )
        f6 = plate_element_force(
            pe,
            rotation_matrix_body=r_mat,
            reference_velocity_inertial=xdot[6 * b : 6 * b + 3],
            angular_velocity_inertial=r_mat @ xdot[6 * b + 3 : 6 * b + 6],
            fluid_velocity=np.zeros(3),
            rho=_RHO,
        )
        power += -float(f6 @ xdot[6 * b : 6 * b + 6])
    return power


# ===========================================================================
# GATE 1 -- plate rotational drag vs the MODAL-KINEMATICS reference
# ===========================================================================


@pytest.mark.slow
def test_gate1_plate_drag_vs_modal_reference() -> None:
    setup = _build_coupled(with_plate=True)
    ph, w_mode, i_eff = _reference(setup)

    # (a) the reference is MODAL, not a fixed pivot (F1): mode reproduces the
    # record period and PR2's rotation-centre beta = -0.330 (NOT the fixed-joint
    # -1.689). a_c (disc-centre edge-on lever) follows from the modal shape.
    assert 2 * np.pi / w_mode == pytest.approx(3.214, rel=0.02)  # vs record 3.257
    beta = ph[0]  # buoy0 surge per unit pitch-rate
    assert beta == pytest.approx(-0.3302, rel=0.02)  # PR2 / Finding F1
    a_c = ph[0] + ph[4] * _Z_PLATE_B
    assert a_c == pytest.approx(-0.592, rel=0.02)  # re-derived (STEP 1's -0.455 used wrong depth)

    # (b) SPLIT (the headline): the code's normal/tangential energy ratio at the
    # modal state matches the analytical E_n/E_t, reported SEPARATELY. Normal
    # DOMINATES using the KNOWN Cd_n = 5.0; the edge-on Cd_t is minor.
    zeta_n, zeta_t = _plate_zeta(ph, i_eff, w_mode, 0.02, _CD_N, _CD_T)
    ratio_analytic = zeta_n / zeta_t
    assert 1.76 < ratio_analytic < 3.52  # re-derived band (Cd_t in [2,1]); STEP 1 had 3.9-7.7
    assert zeta_n > zeta_t  # normal dominant
    p_norm = _plate_power_at_modal_state(ph, w_mode, _CD_N, 0.0)
    p_edge = _plate_power_at_modal_state(ph, w_mode, 0.0, _CD_T)
    assert p_norm / p_edge == pytest.approx(ratio_analytic, rel=0.03)  # code confirms the split

    # (c) end-to-end: the coupled decay's plate zeta_drag, measured in the MODAL
    # coordinate (item-3 investigation: the reference is SOUND -- the earlier
    # ~24 % was a differential-DOF coordinate artifact, not a linearization bias),
    # matches the energy-equivalent modal prediction to <5 %. Positive, and small
    # vs the spar's 0.379 % (F1) -- the plate's rotational drag is a minor addition.
    ma = np.asarray(setup.lhs.M_plus_Ainf)
    nodrag = _build_coupled(with_plate=False)
    ztq, ampq = _modal_decay(setup, ph, ma, use_drag=True, theta0=0.05)
    zfq, _ = _modal_decay(nodrag, ph, ma, use_drag=False, theta0=0.05)
    zeta_drag_modal = ztq - zfq
    assert zeta_drag_modal > 0.0
    k_per_rad = sum(_plate_zeta(ph, i_eff, w_mode, 0.02, _CD_N, _CD_T)) / 0.02
    assert zeta_drag_modal == pytest.approx(k_per_rad * ampq, rel=0.05)
    assert zeta_drag_modal < 0.001  # < 0.1%: far below the spar's 0.379% (PR2)


# ===========================================================================
# GATE 2 -- byte-identity: drag is force-only; a drag-free deck is untouched
# ===========================================================================


@pytest.mark.slow
def test_gate2_plate_drag_is_force_only() -> None:
    """A plate touches NO inertia/stiffness/kernel: build the coupled system with
    and without the plate and confirm M_plus_Ainf, C and the kernel are
    byte-identical; only state_force differs (nonzero under a tilting velocity)."""
    s_no = _build_coupled(with_plate=False)
    s_pl = _build_coupled(with_plate=True)
    np.testing.assert_array_equal(s_pl.lhs.M_plus_Ainf, s_no.lhs.M_plus_Ainf)
    np.testing.assert_array_equal(s_pl.lhs.C, s_no.lhs.C)
    np.testing.assert_array_equal(s_pl.kernel.K, s_no.kernel.K)

    xi = np.zeros(24)
    xd = np.zeros(24)
    xd[4] = 1.0  # buoy1 pitch-rate -> tilting plate normal drag
    assert np.max(np.abs(s_no.state_force(0.0, xi, xd))) == 0.0
    assert np.max(np.abs(s_pl.state_force(0.0, xi, xd))) > 0.0


def test_gate2_no_drag_returns_none() -> None:
    """A deck with no drag_elements yields drag_force=None (the pre-M11a path is
    byte-identical)."""
    assert _build_drag_state_force(_coupled_deck(with_plate=False), n_dof=24, rho=_RHO) is None
    assert _build_drag_state_force(_coupled_deck(with_plate=True), n_dof=24, rho=_RHO) is not None


# ===========================================================================
# Requirement (a) -- supersession is a STRUCTURAL GUARD (raises), not a note
# ===========================================================================


def _one_body_deck(drag_elements) -> Deck:  # type: ignore[no-untyped-def]
    return Deck(
        simulation=Simulation(duration=10.0, dt=0.1),
        environment=Environment(water_depth=200.0, water_density=_RHO, gravity=9.81),
        waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
        bodies=[
            Body(
                name="buoy",
                reference_point=[0.0, 0.0, _ZB],
                mass=28.67,
                inertia=Inertia(Ixx=24.0, Iyy=24.0, Izz=0.114),
                hydro_database=HydroDatabaseRef(format="capytaine", path="x.nc"),
                drag_elements=drag_elements,
            )
        ],
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )


def test_supersession_guard_raises_on_heave_plate_standin() -> None:
    """A body carrying BOTH a PlateMember and a horizontal-cylinder heave-plate
    stand-in (the M11a-PR1 trick) is rejected structurally -- the plate element
    owns the normal drag, so the double-count is made impossible, not documented
    against. A vertical spar cylinder (parallel to the plate normal) is allowed."""
    from floatsim.io.deck import MorisonMember

    plate = PlateMember(
        type="plate",
        center=[0.0, 0.0, _Z_PLATE_B],
        normal=[0.0, 0.0, 1.0],
        radius=_A_PLATE,
        thickness=_T_PLATE,
        Cd_n=_CD_N,
        Cd_t=_CD_T,
    )
    standin = MorisonMember(
        type="morison_member",
        node_a=[-_A_PLATE, 0.0, _Z_PLATE_B],  # HORIZONTAL cylinder in the plate plane
        node_b=[_A_PLATE, 0.0, _Z_PLATE_B],
        diameter=np.pi * _A_PLATE**2 / (2.0 * _A_PLATE),
        Cd=_CD_N,
    )
    with pytest.raises(ValueError, match=r"not parallel to the plate normal"):
        _build_drag_state_force(_one_body_deck([plate, standin]), n_dof=6, rho=_RHO)

    # A vertical spar (axis parallel to the plate normal) coexists fine.
    spar = MorisonMember(
        type="morison_member",
        node_a=[0.0, 0.0, _Z_PLATE_B],  # VERTICAL cylinder (a spar)
        node_b=[0.0, 0.0, 0.5 - _ZB],
        diameter=0.168,
        Cd=1.2,
    )
    assert _build_drag_state_force(_one_body_deck([plate, spar]), n_dof=6, rho=_RHO) is not None


# ===========================================================================
# GATE 3 -- structural: pure heave reduces EXACTLY to the single-Cd cylinder
# ===========================================================================


def test_gate3_heave_reduces_exactly_to_single_cd() -> None:
    """In pure heave u_n is uniform, so the plate's normal branch must reduce
    EXACTLY to the committed single-Cd cylinder (D*L = pi*a^2). Machine-precision
    force identity -- if it does not hold, the normal branch is wrong."""
    pe = PlateDragElement(
        body_index=0,
        center_body=np.array([0.0, 0.0, _Z_PLATE_B]),
        normal_body=np.array([0.0, 0.0, 1.0]),
        radius=_A_PLATE,
        thickness=_T_PLATE,
        Cd_n=_CD_N,
        Cd_t=0.0,  # isolate the normal branch
    )
    v_heave = np.array([0.0, 0.0, -0.7])
    r_mat = np.eye(3)
    f_plate = plate_element_force(
        pe,
        rotation_matrix_body=r_mat,
        reference_velocity_inertial=v_heave,
        angular_velocity_inertial=np.zeros(3),
        fluid_velocity=np.zeros(3),
        rho=_RHO,
    )
    # Equivalent single-Cd horizontal cylinder with D*L = pi*a^2.
    length = 2.0 * _A_PLATE
    diameter = np.pi * _A_PLATE**2 / length
    me = MorisonElement(
        body_index=0,
        node_a_body=np.array([-length / 2.0, 0.0, _Z_PLATE_B]),
        node_b_body=np.array([length / 2.0, 0.0, _Z_PLATE_B]),
        diameter=diameter,
        Cd=_CD_N,
    )
    f_cyl = morison_element_force(
        me,
        midpoint_inertial=np.array([0.0, 0.0, _Z_PLATE_B]),
        axis_hat_inertial=np.array([1.0, 0.0, 0.0]),
        body_velocity_at_midpoint=v_heave,
        body_acceleration_at_midpoint=None,
        fluid_velocity=np.zeros(3),
        fluid_acceleration=None,
        rho=_RHO,
        reference_point_inertial=np.zeros(3),
    )
    np.testing.assert_allclose(f_plate, f_cyl, rtol=1e-13, atol=1e-13)
    # and both equal the analytical 0.5*rho*Cd*A*|v|*v.
    assert f_plate[2] == pytest.approx(0.5 * _RHO * _CD_N * np.pi * _A_PLATE**2 * 0.7 * 0.7)


@pytest.mark.slow
def test_gate3_heave_decay_reproduces_committed_zeta() -> None:
    """The committed spar-fin heave decay run THROUGH the plate element reproduces
    zeta = 2.5225e-02. The plate uses the exact face area pi*a^2 = 0.14522; the
    committed value used the rounded 0.1452 -- the <=1.4e-4 area rounding is the
    only slack (the machine-precision reduction in the sibling test guarantees no
    other difference), well inside rel=1e-3."""
    sys.path.insert(0, str(_SPAR))
    import study_common as sc  # type: ignore[import-not-found]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdb = sc.load_hdb()
        lhs = sc.build_lhs(hdb)
        kernel = sc.build_kernel(hdb)

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
                    PlateMember(
                        type="plate",
                        center=[0.0, 0.0, sc.PLATE_Z],
                        normal=[0.0, 0.0, 1.0],
                        radius=sc.PLATE_RADIUS,
                        thickness=_T_PLATE,
                        Cd_n=sc.PLATE_CD,
                        Cd_t=0.0,  # pure heave: edge-on is inert anyway
                    )
                ],
            )
        ],
        output=Output(file="o.h5", channels=["heave"], sample_rate=10.0),
    )
    force = _build_drag_state_force(deck, n_dof=6, rho=sc.RHO)
    assert force is not None

    eq = solve_static_equilibrium(lhs=lhs, state_force=None)
    xi0 = eq.xi_eq.copy()
    xi0[2] += sc.IC_HEAVE
    r = integrate_cummins(
        lhs=lhs,
        kernel=kernel,
        xi0=xi0,
        xi_dot0=np.zeros(6),
        duration=sc.DURATION,
        dt=sc.DT,
        state_force=force,
    )
    x = r.xi[:, 2]
    pk, _ = find_peaks(x, height=1e-4)
    amps = x[pk]
    ratios = amps[:-1] / amps[1:]
    delta = float(np.mean(np.log(ratios[ratios > 0])))
    zeta = delta / np.sqrt(4.0 * np.pi**2 + delta**2)
    assert pk.size >= 3
    assert zeta == pytest.approx(_STUDY_ZETA, rel=1e-3)


# ===========================================================================
# GATE 4 -- analytical: the disc quadrature converges to 8 a^5 / 15
# ===========================================================================


def _disc_int_x3(n_radial, n_azimuthal):  # type: ignore[no-untyped-def]
    pe = PlateDragElement(
        body_index=0,
        center_body=np.zeros(3),
        normal_body=np.array([0.0, 0.0, 1.0]),
        radius=_A_PLATE,
        thickness=_T_PLATE,
        Cd_n=_CD_N,
        Cd_t=_CD_T,
        n_radial=n_radial,
        n_azimuthal=n_azimuthal,
    )
    x = pe._patch_pos_body[:, 0]  # in-plane coord along e1 (center at origin)
    return float(np.sum(np.abs(x) ** 3 * pe._patch_area)), float(pe._patch_area.sum())


def test_gate4_quadrature_converges_to_analytical_integral() -> None:
    """The radial+azimuthal midpoint quadrature converges to the analytical
    ``INT INT |x|^3 dA = 8 a^5 / 15``; the face area is exact at any count."""
    # Face area exact (the property GATE 3's exact heave reduction relies on).
    val_12, area_12 = _disc_int_x3(12, 24)
    assert area_12 == pytest.approx(np.pi * _A_PLATE**2, rel=0, abs=1e-15)

    # Adopted count (12 x 24): residual is a stated -0.58% (converging from below).
    resid_12 = (val_12 - _INT_X3) / _INT_X3
    assert resid_12 == pytest.approx(-0.0058, abs=0.001)

    # Refining halves the residual (monotone convergence to the analytical value).
    val_16, _ = _disc_int_x3(16, 32)
    resid_16 = (val_16 - _INT_X3) / _INT_X3
    assert abs(resid_16) < abs(resid_12)
    assert resid_16 == pytest.approx(-0.0033, abs=0.001)
    # both underpredict (from below) and approach the analytical 8a^5/15.
    assert val_12 < val_16 < _INT_X3
