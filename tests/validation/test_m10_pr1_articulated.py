"""M10 PR1 -- the first articulated-3 run: assemble + equilibrium + the
heave cross-check gate (Measurement E) + the zero-pitch symmetry check.

The articulated-3 model is 3 hydro buoys on the committed 18-DOF coupled
fixture + 1 dry structural hub (Q2), joined by 3 ``yaw_locked`` joints at
the buoy-end attachments (m = 12, 12 free DOF). This is the first place
the coupled hydro path, the reference-aware joint layer (PR0.75), the
structural body (PR0.5) and the per-body hydrostatic C (PR0.85) run
together end-to-end.

Gates (references derived before the model existed -- plan Measurement E):

  GATE A -- heave cross-check (correctness, pass/fail). With every joint
    translation locked the articulated cluster's pure heave is
    rigid-body-identical to the rigid cluster, so the free-decay period
    is a TRUE cross-check against
        T_n = 2*pi*sqrt((M + A33_inf)/C33) = 3.106087 s,
    the committed interaction.json:T_n_with_interaction = 3.1060873561 s.
    Gate rtol 1e-2 (M8 cross-check band).

  GATE B -- zero-pitch symmetry (correctness, pass/fail). 3-fold + y-mirror
    symmetry => a symmetric heave IC excites only symmetric modes =>
    ZERO pitch/roll. Nonzero rotation signals an assembly/constraint
    asymmetry bug, not physics.

Preconditions gated BEFORE the two correctness gates (Amendment A2 -- the
STEP-1 check that stopped the two earlier PR1 runs, and the second
convention-mismatch detector):

  * phi(xi=0) ~ 0: the reference-aware JointSet reads xi as
    displacement-from-reference; a residual constraint at rest means the
    joint / coupled state conventions still disagree (the abs-xi bug gave
    max|phi| = 1.689).
  * equilibrium: from xi=0 with NO IC and NO external force the assembled
    system STAYS at xi=0 (the displacement-xi convention's static solve is
    trivially xi=0; this no-drift run is its substantive verification and
    would expose any residual reference-config force).
  * rank(G) = 12 on the assembled model (risk-register re-check).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
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
    RegularWave,
    Simulation,
    YawLockedJoint,
)
from floatsim.solver.newmark import integrate_cummins

REPO = Path(__file__).resolve().parents[2]
_NC = REPO / "studies" / "cluster-3buoy-rigid" / "capytaine_multibody_18dof.nc"
_REF = REPO / "studies" / "cluster-3buoy-rigid" / "reference_single_bem.nc"
_CONTAM = (4.934, 20.909)  # M8 PR3 contaminated-frequency exclusion (grid selection)
_R = 0.5
_ANG = np.deg2rad([0.0, 120.0, 240.0])
_Z_BUOY = -1.1956674320202696
_Z_ARM = 0.4933695679797303
_T_N_REF = 3.1060873561  # committed interaction.json:T_n_with_interaction (s)


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


def _articulated_deck() -> Deck:
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
    joints = [
        YawLockedJoint(
            type="yaw_locked",
            body_a=f"buoy{i + 1}",
            body_b="hub",
            attach_a_body=[0.0, 0.0, _Z_ARM - _Z_BUOY],  # buoy-end attach -> arm tip
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
        output=Output(file="out.h5", channels=["heave"], sample_rate=10.0),
    )


@pytest.fixture(scope="module")
def setup():  # type: ignore[no-untyped-def]
    """Assemble the articulated-3 model once for all gates (kernel build is
    the slow part)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Item-25 small-body override warning
        return build_system(
            _articulated_deck(),
            bem_databases={},
            dt=0.01,
            t_max_kernel=30.0,
            solve_equilibrium=False,
            shared_hydro_database=_hdb18(),
            hydrostatic_database=read_capytaine(_REF),
            asymptote_check_override="M10 cluster small-body hulls; ITEM25-SMALL-BODY",
        )


@pytest.fixture(scope="module")
def heave_decay(setup):  # type: ignore[no-untyped-def]
    """Free decay from a rigid heave IC (every body +0.10 m in heave),
    shared by GATE A and GATE B."""
    xi0 = np.zeros(24)
    for k in range(4):
        xi0[6 * k + 2] = 0.10
    return integrate_cummins(
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


# --- Preconditions (Amendment A2 STEP-1 + second-convention detector) ------


@pytest.mark.slow
def test_phi_rest_precondition(setup) -> None:  # type: ignore[no-untyped-def]
    """max|phi(xi=0)| ~ 0: the reference-aware JointSet (PR0.75) reads xi as
    displacement-from-reference. The abs-xi convention gave 1.689 here."""
    js = setup.constraints
    assert js is not None
    assert js.n_constraints == 12
    phi0 = js.phi(np.zeros(24))
    assert np.max(np.abs(phi0)) < 1e-9


@pytest.mark.slow
def test_g_rank_full(setup) -> None:  # type: ignore[no-untyped-def]
    """rank(G) = 12 on the assembled model (risk-register re-check): the
    hub topology is not over-constrained; the KKT solve is well-posed."""
    g = setup.constraints.jacobian(np.zeros(24))
    assert g.shape == (12, 24)
    assert np.linalg.matrix_rank(g) == 12


@pytest.mark.slow
def test_equilibrium_no_drift(setup) -> None:  # type: ignore[no-untyped-def]
    """xi=0 is the assembled system's equilibrium: from rest with NO IC and
    NO external force it stays at xi=0. This is the displacement-xi
    convention's static-solve verification and a second-convention-mismatch
    detector (a residual reference-config force would drift xi)."""
    r = integrate_cummins(
        lhs=setup.lhs,
        kernel=setup.kernel,
        xi0=np.zeros(24),
        xi_dot0=np.zeros(24),
        duration=20.0,
        dt=0.01,
        rho_inf=0.8,
        constraints=setup.constraints,
        projection_interval=1,
    )
    assert np.max(np.abs(r.xi)) < 1e-9


# --- GATE A -- heave cross-check (correctness) -----------------------------


@pytest.mark.slow
def test_heave_period_cross_check(heave_decay) -> None:  # type: ignore[no-untyped-def]
    """GATE A: the pure-heave free-decay period reproduces the rigid
    cluster's T_n = 3.106087 s (committed T_n_with_interaction), rtol 1e-2.
    With joint translations locked, cluster heave is rigid-body-identical,
    so this is a true cross-check against the pre-derived reference."""
    z = heave_decay.xi[:, 2]
    pk, _ = find_peaks(z, height=0.0)
    assert pk.size >= 3, f"heave decay did not oscillate (peaks={pk.size})"
    t_n = float(np.mean(np.diff(heave_decay.t[pk])))
    assert t_n == pytest.approx(_T_N_REF, rel=1e-2)


# --- GATE B -- zero-pitch symmetry (correctness) ---------------------------


@pytest.mark.slow
def test_zero_pitch_symmetry(heave_decay) -> None:  # type: ignore[no-untyped-def]
    """GATE B: a symmetric heave IC excites only symmetric modes (3-fold +
    y-mirror), so every body's pitch and roll stay ~0. A nonzero rotation
    would signal an assembly/constraint asymmetry bug. Measured floor is
    ~1e-5 rad (projection numerical noise); the 1e-3 gate is 100x below the
    Item-2 physical threshold (0.1 rad) and 70x above the floor."""
    max_pitch = max(float(np.abs(heave_decay.xi[:, 6 * k + 4]).max()) for k in range(4))
    max_roll = max(float(np.abs(heave_decay.xi[:, 6 * k + 3]).max()) for k in range(4))
    assert max_pitch < 1e-3, f"pitch broke symmetry: {max_pitch:.3e} rad"
    assert max_roll < 1e-3, f"roll broke symmetry: {max_roll:.3e} rad"
