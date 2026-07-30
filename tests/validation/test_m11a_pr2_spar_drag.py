"""M11a PR2 -- spar lateral drag elements (plan Q3-ii): the existing
member-normal cylinder model used CORRECTLY for a slender vertical spar
moving laterally. No new physics; the plate mis-model stays PR4.

GATE 1 is judged against a CORRECTED reference (plan Finding F1): the
fixed-joint pendulum idealization overestimates articulated-mode drag ~8x,
so the reference is derived from the DRAG-FREE mode shape (a constrained
eigenanalysis, independent of the drag code), integrating quadratic drag
against the TRUE local velocities. The fixed-joint form is retained only
as a sanity check that must reproduce the original 2.99%.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh, null_space
from scipy.signal import find_peaks

from floatsim.driver import build_system
from floatsim.hydro.database import HydroDatabase
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
    distributed_cylinder_drag,
)
from floatsim.io.deck import RegularWave as DeckWave
from floatsim.solver.newmark import integrate_cummins

REPO = Path(__file__).resolve().parents[2]
_NC = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_REF = REPO / "studies" / "cluster-3buoy-rigid" / "reference_single_bem.nc"
_R, _ANG, _ZB, _ZA = 0.5, np.deg2rad([0.0, 120.0, 240.0]), -1.1956674320202696, 0.4933695679797303
_D, _CD, _NSEG = 0.1682, 1.2, 10
_RHO = 1025.0
_Z_WL_B, _Z_PLATE_B = 0.0 - _ZB, -1.45737 - _ZB  # submerged span, body frame
_ZETA_RAD = 0.00354  # drag-free rotational zeta (this build)


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


def _deck(n_seg: int) -> Deck:
    spar = (
        distributed_cylinder_drag(
            z_bottom=_Z_PLATE_B, z_top=_Z_WL_B, diameter=_D, cd=_CD, n_segments=n_seg
        )
        if n_seg > 0
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
            drag_elements=list(spar),
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


def _build(n_seg: int):  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_system(
            _deck(n_seg),
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=_hdb18(),
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override="M11a PR2 spar drag",
        )


def _zeta_drag_energy(shape, i_store, theta, w_mode):  # type: ignore[no-untyped-def]
    """Energy-equivalent quadratic-drag zeta at amplitude theta, given a
    velocity mode shape (per unit buoy0 pitch-rate) and modal inertia."""
    zz = np.linspace(_Z_PLATE_B, _Z_WL_B, 400)
    thd = theta * w_mode
    j = 0.0
    for b in range(3):
        p = shape[6 * b : 6 * b + 6]
        vb = np.hypot(p[0] + p[4] * zz, p[1] - p[3] * zz)
        j += np.trapezoid(np.abs(vb) ** 3, zz)
    e_diss = 0.5 * _RHO * _D * _CD * (8.0 / 3.0) / w_mode * thd**3 * j
    return e_diss / (4 * np.pi * (0.5 * i_store * thd**2))


def _corrected_prediction(setup):  # type: ignore[no-untyped-def]
    """Constrained eigenanalysis of the drag-free system -> rotational mode
    shape -> corrected zeta_drag per unit amplitude, and the fixed-joint
    sanity value at Theta=0.02."""
    ma = np.asarray(setup.lhs.M_plus_Ainf)
    c = np.asarray(setup.lhs.C)
    g = setup.constraints.jacobian(np.zeros(24))
    z = null_space(g)
    w2, vq = eigh(z.T @ c @ z, z.T @ ma @ z)
    modes = z @ vq
    w = np.sqrt(np.abs(w2))
    # rotational mode: in the 1.5-2.3 band, most buoy-pitch-dominant
    cand = max(
        (k for k in range(w.size) if 1.5 < w[k] < 2.3),
        key=lambda k: abs(modes[4, k]),
    )
    ph = modes[:, cand] / modes[4, cand]  # normalize buoy0 pitch = 1
    i_eff = float(ph @ ma @ ph)
    k_corr = _zeta_drag_energy(ph, i_eff, 0.02, w[cand]) / 0.02  # zeta per rad
    # fixed-joint sanity: single-buoy |a|=s about the joint, I_eff=105.79
    zz = np.linspace(_Z_PLATE_B, _Z_WL_B, 400)
    s = (_ZA - _ZB) - zz
    thd = 0.02 * w[cand]
    e_fj = 0.5 * _RHO * _D * _CD * (8.0 / 3.0) / w[cand] * thd**3 * np.trapezoid(np.abs(s) ** 3, zz)
    zeta_fj = e_fj / (4 * np.pi * (0.5 * 105.79 * thd**2))
    return k_corr, zeta_fj, w[cand]


def _rot_decay(setup, use_drag: bool, n_peaks: int | None = None):  # type: ignore[no-untyped-def]
    xi0 = np.zeros(24)
    for j in range(3):
        xi0[6 * j + 4] = 0.02
    r = integrate_cummins(
        lhs=setup.lhs,
        kernel=setup.kernel,
        xi0=xi0,
        xi_dot0=np.zeros(24),
        duration=60.0,
        dt=0.01,
        rho_inf=0.8,
        constraints=setup.constraints,
        projection_interval=1,
        state_force=setup.state_force if use_drag else None,
    )
    th = r.xi[:, 4] - r.xi[:, 22]
    pk, _ = find_peaks(th, height=0.0)
    amps = th[pk] if n_peaks is None else th[pk][: n_peaks + 1]
    d = np.log(amps[:-1] / amps[1:])
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.mean(d) / (2 * np.pi)), float(np.mean(amps[:2]))


# ---------------------------------------------------------------------------
# GATE 1 -- corrected-reference regression (a DECAY)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gate1_spar_drag_vs_corrected_prediction() -> None:
    setup = _build(_NSEG)
    k_corr, zeta_fj, _w_mode = _corrected_prediction(setup)
    # SANITY: the fixed-joint form reproduces the original 2.99% (arithmetic sound)
    assert zeta_fj == pytest.approx(0.0299, rel=0.05)
    # the correction is a big kinematic reduction (~8x), not a tuning
    assert 0.08 < (k_corr * 0.02) / zeta_fj < 0.20
    # measured (drag decay, first 2 peaks) vs corrected prediction at that amplitude
    zt2, amp = _rot_decay(setup, use_drag=True, n_peaks=2)
    zf2, _ = _rot_decay(setup, use_drag=False, n_peaks=2)
    zeta_drag_meas = zt2 - zf2
    zeta_drag_pred = k_corr * amp
    assert zeta_drag_meas == pytest.approx(zeta_drag_pred, rel=0.35)  # derivation's approximations


# ---------------------------------------------------------------------------
# GATE 3 -- orientation: spar (vertical) contributes ZERO in pure heave
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gate3_no_drag_in_pure_heave() -> None:
    """A vertical member's member-normal drag responds only to lateral
    velocity: in pure heave (motion along the axis) the spar contributes
    nothing. Asserts the element orientation is right."""
    setup = _build(_NSEG)
    xi = np.zeros(24)
    xd = np.zeros(24)
    for k in range(4):
        xd[6 * k + 2] = 1.0  # rigid heave velocity, all bodies
    f = setup.state_force(0.0, xi, xd)
    assert np.max(np.abs(f)) == 0.0  # zero drag under pure heave


# ---------------------------------------------------------------------------
# GATE 4 -- discretization convergence
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gate4_discretization_convergence() -> None:
    """Measured rotational damping converges with element count (the s^3
    moment weighting means 1 element underpredicts). Report the residual at
    the adopted N=10."""
    zf, _ = _rot_decay(_build(0), use_drag=False, n_peaks=2)  # drag-free baseline (no elements)
    zetas = {}
    for n in (1, 4, 10, 16):
        zt, _ = _rot_decay(_build(n), use_drag=True, n_peaks=2)
        zetas[n] = zt - zf
    # monotone increase toward convergence; N=10 within a few % of N=16
    assert zetas[1] < zetas[4] < zetas[10]
    assert zetas[10] == pytest.approx(zetas[16], rel=0.05)
