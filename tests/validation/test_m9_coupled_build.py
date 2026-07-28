"""M9 PR3 -- deck schema + ``build_system`` coupled (shared N-body BEM) path.

Covers the three validation layers introduced by PR3, keeping the
kernel-gated full build off the every-PR path:

  * ``Body`` / ``Deck`` schema validators (``floatsim.io.deck``) --
    fast, no kernel: the "exactly one hydro source" rule, the
    shared-database / label cross-checks, and the joint endpoint
    checks. Plus a back-compatibility parse of the committed
    ``two_body_semisub_barge.yml``.
  * ``_build_joint_set`` -- fast: deck ``joints`` -> ``JointSet``
    (body names -> indices, ``'earth'`` -> ``-1``), ``None`` when the
    deck has no joints.
  * ``_build_coupled_lhs_kernel`` label-contract raises -- fast:
    every raise fires *before* ``compute_retardation_kernel``, so the
    fixtures need only be shape-valid (zero ``B`` is never
    transformed). This is the M9 half of the plan-Q5 label contract
    (``tests/support/condensation.py`` is the M8 reference).

  * One ``@pytest.mark.slow`` end-to-end coupled build that actually
    permutes a gate-passing 12x12 database into deck-body order and
    computes the coupled kernel. The kernel gate (M6 PR3) admits no
    fast synthetic -- the narrow-band ``_single_body_hdb`` needs
    ``t_max=120`` -- which is why ``test_driver.py`` is wholly slow;
    this module keeps a single slow case and everything else fast.

Byte-identity of the untouched per-body path is covered by
``tests/unit/test_driver.py::test_step_C_kernel_K_matches_hand_wired``
(build_system vs the independent hand-wired oracle at rtol=atol=1e-12
on ``M_plus_Ainf``, ``C`` and ``kernel.K``); PR3 leaves that path's
statements unchanged (they are merely re-indented under the
``shared_hydro_database is None`` branch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from floatsim.driver import _build_coupled_lhs_kernel, _build_joint_set, build_system
from floatsim.hydro.database import HydroDatabase
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HingeJoint,
    HydroDatabaseRef,
    Inertia,
    Output,
    RegularWave,
    Simulation,
    YawLockedJoint,
    load_deck,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DECK = REPO_ROOT / "examples" / "two_body_semisub_barge.yml"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _sim() -> Simulation:
    return Simulation(duration=10.0, dt=0.01)


def _env() -> Environment:
    return Environment(water_depth=200.0, water_density=1025.0)


def _waves() -> RegularWave:
    return RegularWave(type="regular", height=1.0, period=10.0, heading=0.0)


def _out() -> Output:
    return Output(file="out.h5", channels=["heave"], sample_rate=10.0)


def _inertia() -> Inertia:
    return Inertia(Ixx=1.0e9, Iyy=1.0e9, Izz=1.0e9)


def _labelled_body(name: str, label: str) -> Body:
    return Body(
        name=name,
        reference_point=[0.0, 0.0, 0.0],
        mass=1.0e7,
        inertia=_inertia(),
        hydro_body_label=label,
    )


def _perbody_body(name: str) -> Body:
    return Body(
        name=name,
        reference_point=[0.0, 0.0, 0.0],
        mass=1.0e7,
        inertia=_inertia(),
        hydro_database=HydroDatabaseRef(format="wamit", path="synthetic"),
    )


def _shared_ref() -> HydroDatabaseRef:
    """A deck-level shared-database *file pointer* (content not parsed by
    the schema; the loaded object is passed to build_system separately)."""
    return HydroDatabaseRef(format="capytaine", path="shared_nbody.nc")


def _coupled_deck(labels: tuple[str, str], *, joints: list | None = None) -> Deck:
    return Deck(
        simulation=_sim(),
        environment=_env(),
        waves=_waves(),
        bodies=[_labelled_body("body_0", labels[0]), _labelled_body("body_1", labels[1])],
        shared_hydro_database=_shared_ref(),
        joints=joints or [],
        output=_out(),
    )


def _zeros_labelled_hdb(labels: tuple[str, ...]) -> HydroDatabase:
    """Shape-valid N-body labelled database with zero radiation damping.

    Used only for the label-contract raise tests, which fire before the
    kernel transform -- ``B`` is never touched, so zeros are fine.
    """
    n = len(labels)
    nd = 6 * n
    omega = np.linspace(0.1, 3.0, 5)
    nw = omega.size
    a_inf = np.eye(nd, dtype=np.float64) * 1.0e7
    c = np.eye(nd, dtype=np.float64) * 1.0e6
    a = np.stack([a_inf for _ in range(nw)], axis=-1)
    b = np.zeros((nd, nd, nw), dtype=np.float64)
    rao = np.zeros((nd, nw, 2), dtype=np.complex128)
    return HydroDatabase(
        omega=omega,
        heading_deg=np.array([0.0, 90.0]),
        A=a,
        B=b,
        A_inf=a_inf,
        C=c,
        RAO=rao,
        reference_point=np.zeros(3),
        C_source="full",
        metadata={"src": "m9-pr3-raise-fixture"},
        body_labels=labels,
    )


# ---------------------------------------------------------------------------
# Body / Deck schema validators (fast)
# ---------------------------------------------------------------------------


def test_body_both_hydro_sources_raises() -> None:
    with pytest.raises(ValidationError, match=r"exactly one of 'hydro_database'"):
        Body(
            name="b",
            reference_point=[0.0, 0.0, 0.0],
            mass=1.0e7,
            inertia=_inertia(),
            hydro_database=HydroDatabaseRef(format="wamit", path="x"),
            hydro_body_label="alpha",
        )


def test_body_neither_hydro_source_raises() -> None:
    with pytest.raises(ValidationError, match=r"exactly one of 'hydro_database'"):
        Body(name="b", reference_point=[0.0, 0.0, 0.0], mass=1.0e7, inertia=_inertia())


def test_deck_labelled_body_without_shared_db_raises() -> None:
    with pytest.raises(ValidationError, match=r"no .*shared_hydro_database"):
        Deck(
            simulation=_sim(),
            environment=_env(),
            waves=_waves(),
            bodies=[_labelled_body("body_0", "alpha")],
            output=_out(),
        )


def test_deck_shared_db_without_labelled_body_raises() -> None:
    with pytest.raises(ValidationError, match=r"no body selects a block"):
        Deck(
            simulation=_sim(),
            environment=_env(),
            waves=_waves(),
            bodies=[_perbody_body("body_0")],
            shared_hydro_database=_shared_ref(),
            output=_out(),
        )


def test_deck_duplicate_hydro_body_label_raises() -> None:
    with pytest.raises(ValidationError, match=r"duplicate hydro_body_label"):
        _coupled_deck(("alpha", "alpha"))


def test_deck_joint_unknown_body_raises() -> None:
    bad = HingeJoint(
        type="hinge",
        body_a="body_0",
        body_b="ghost",
        attach_a_body=[0.0, 0.0, 0.0],
        attach_b_body=[0.0, 0.0, 0.0],
        axis=[0.0, 0.0, 1.0],
    )
    with pytest.raises(ValidationError, match=r"unknown body 'ghost'"):
        _coupled_deck(("alpha", "beta"), joints=[bad])


def test_deck_joint_self_connection_raises() -> None:
    bad = HingeJoint(
        type="hinge",
        body_a="body_0",
        body_b="body_0",
        attach_a_body=[0.0, 0.0, 0.0],
        attach_b_body=[0.0, 0.0, 0.0],
        axis=[0.0, 0.0, 1.0],
    )
    with pytest.raises(ValidationError, match=r"connects body 'body_0' to itself"):
        _coupled_deck(("alpha", "beta"), joints=[bad])


def test_deck_joint_to_earth_is_allowed() -> None:
    j = HingeJoint(
        type="hinge",
        body_a="body_0",
        body_b="earth",
        attach_a_body=[0.0, 0.0, 0.0],
        attach_b_body=[1.0, 0.0, 0.0],
        axis=[0.0, 1.0, 0.0],
    )
    deck = _coupled_deck(("alpha", "beta"), joints=[j])
    assert deck.joints[0].body_b == "earth"


def test_two_body_semisub_barge_still_parses() -> None:
    """Back-compat: the committed per-body deck parses unchanged under the
    new 'exactly one hydro source' rule (both bodies set hydro_database)."""
    deck = load_deck(SAMPLE_DECK)
    assert len(deck.bodies) == 2
    for b in deck.bodies:
        assert b.hydro_database is not None
        assert b.hydro_body_label is None
    assert deck.shared_hydro_database is None
    assert deck.joints == []


# ---------------------------------------------------------------------------
# _build_joint_set (fast)
# ---------------------------------------------------------------------------


def test_build_joint_set_none_when_no_joints() -> None:
    deck = _coupled_deck(("alpha", "beta"))
    assert _build_joint_set(deck, {"body_0": 0, "body_1": 1}) is None


def test_build_joint_set_maps_hinge_and_yaw_locked() -> None:
    hinge = HingeJoint(
        type="hinge",
        body_a="body_0",
        body_b="body_1",
        attach_a_body=[1.0, 0.0, 0.0],
        attach_b_body=[-1.0, 0.0, 0.0],
        axis=[0.0, 1.0, 0.0],
    )
    yaw = YawLockedJoint(
        type="yaw_locked",
        body_a="body_1",
        body_b="earth",
        attach_a_body=[0.0, 0.0, 0.0],
        attach_b_body=[2.0, 0.0, 0.0],
    )
    deck = _coupled_deck(("alpha", "beta"), joints=[hinge, yaw])
    js = _build_joint_set(deck, {"body_0": 0, "body_1": 1})
    assert js is not None
    assert js.n_bodies == 2
    assert len(js.joints) == 2
    j0, j1 = js.joints
    assert j0.kind == "hinge"
    assert (j0.body_a, j0.body_b) == (0, 1)
    np.testing.assert_allclose(j0.axis, [0.0, 1.0, 0.0])
    np.testing.assert_allclose(j0.attach_a, [1.0, 0.0, 0.0])
    assert j1.kind == "yaw_locked"
    assert (j1.body_a, j1.body_b) == (1, -1)  # earth sentinel
    np.testing.assert_allclose(j1.axis, [0.0, 0.0, 1.0])  # yaw_locked default axis


# ---------------------------------------------------------------------------
# Coupled label-contract raises (fast -- fire before the kernel)
# ---------------------------------------------------------------------------


def test_coupled_single_body_db_raises() -> None:
    """A shared database with no body_labels (single-body) cannot back a
    coupled deck."""
    single = _zeros_labelled_hdb(("alpha",))
    # Rebuild as a genuine single-body (body_labels=None) 6x6 database.
    single_nolabel = HydroDatabase(
        omega=single.omega,
        heading_deg=single.heading_deg,
        A=single.A,
        B=single.B,
        A_inf=single.A_inf,
        C=single.C,
        RAO=single.RAO,
        reference_point=single.reference_point,
        C_source="full",
        metadata={},
    )
    deck = _coupled_deck(("alpha", "beta"))
    with pytest.raises(ValueError, match=r"single-body \(no body_labels\)"):
        _build_coupled_lhs_kernel(
            deck, single_nolabel, dt=0.01, t_max_kernel=120.0, gravity=9.80665
        )


def test_coupled_mixing_per_body_and_shared_raises() -> None:
    """Every body must select a block when a shared database is present."""
    db = _zeros_labelled_hdb(("alpha", "beta"))
    deck = Deck(
        simulation=_sim(),
        environment=_env(),
        waves=_waves(),
        bodies=[_labelled_body("body_0", "alpha"), _perbody_body("body_1")],
        shared_hydro_database=_shared_ref(),
        output=_out(),
    )
    with pytest.raises(ValueError, match=r"EVERY body must select its"):
        _build_coupled_lhs_kernel(deck, db, dt=0.01, t_max_kernel=120.0, gravity=9.80665)


def test_coupled_missing_label_raises() -> None:
    db = _zeros_labelled_hdb(("alpha", "beta"))
    deck = _coupled_deck(("alpha", "gamma"))  # 'gamma' absent from db
    with pytest.raises(ValueError, match=r"\['gamma'\] not found"):
        _build_coupled_lhs_kernel(deck, db, dt=0.01, t_max_kernel=120.0, gravity=9.80665)


def test_coupled_unused_label_raises() -> None:
    db = _zeros_labelled_hdb(("alpha", "beta", "delta"))  # 'delta' unused by deck
    deck = _coupled_deck(("alpha", "beta"))
    with pytest.raises(ValueError, match=r"label\(s\) \['delta'\] unused"):
        _build_coupled_lhs_kernel(deck, db, dt=0.01, t_max_kernel=120.0, gravity=9.80665)


def test_build_system_declares_shared_but_none_passed_raises() -> None:
    """The deck declares a shared database but the caller forgot to pass
    the loaded object to build_system."""
    deck = _coupled_deck(("alpha", "beta"))
    with pytest.raises(ValueError, match=r"none was passed to"):
        build_system(deck, bem_databases={}, dt=0.01, t_max_kernel=120.0)


# ---------------------------------------------------------------------------
# Coupled end-to-end (slow -- real permutation + kernel gate)
# ---------------------------------------------------------------------------


def _two_body_gate_passing_hdb(
    labels: tuple[str, str], scale_a: float, scale_b: float
) -> HydroDatabase:
    """Block-diagonal 12x12 labelled database from two scaled copies of the
    gate-passing single-body ``_single_body_hdb`` (narrow-band, passes the
    M6 PR3 kernel gate at t_max=120/dt=0.002).

    Body ``labels[0]`` carries ``scale_a`` x the template, ``labels[1]``
    carries ``scale_b`` x -- distinct scalings so the permutation is
    observable in the assembled diagonal blocks. Off-diagonal (coupling)
    blocks are zero; the permutation ``np.ix_(perm, perm)`` indexes the
    whole matrix uniformly, so distinguishable diagonal blocks suffice to
    verify the reorder.
    """
    from scripts.m7_pr4_driver_prediction import _single_body_hdb

    single = _single_body_hdb()
    nw = np.asarray(single.omega).size
    nh = np.asarray(single.heading_deg).size

    def _bd(m6: np.ndarray) -> np.ndarray:
        out = np.zeros((12, 12), dtype=np.float64)
        out[:6, :6] = scale_a * m6
        out[6:, 6:] = scale_b * m6
        return out

    a_inf = _bd(np.asarray(single.A_inf))
    c = _bd(np.asarray(single.C))
    a = np.stack([_bd(np.asarray(single.A)[:, :, k]) for k in range(nw)], axis=-1)
    b = np.stack([_bd(np.asarray(single.B)[:, :, k]) for k in range(nw)], axis=-1)
    rao = np.zeros((12, nw, nh), dtype=np.complex128)
    rao[:6] = scale_a * np.asarray(single.RAO)
    rao[6:] = scale_b * np.asarray(single.RAO)
    return HydroDatabase(
        omega=np.asarray(single.omega),
        heading_deg=np.asarray(single.heading_deg),
        A=a,
        B=b,
        A_inf=a_inf,
        C=c,
        RAO=rao,
        reference_point=np.asarray(single.reference_point),
        C_source="full",
        metadata={"src": "m9-pr3-coupled-fixture"},
        body_labels=labels,
    )


@pytest.mark.slow
def test_coupled_build_permutes_blocks_by_label() -> None:
    """End-to-end coupled build: the label -> block map is *by label*, so
    a deck that maps body_0 -> 'beta' and body_1 -> 'alpha' must place the
    'beta' (scale 2x) block first and the 'alpha' (scale 1x) block second,
    independent of the database's own label order."""
    from scripts.m7_pr4_driver_prediction import _single_body_hdb

    single = _single_body_hdb()
    a_inf_single_33 = float(np.asarray(single.A_inf)[2, 2])

    db = _two_body_gate_passing_hdb(("alpha", "beta"), scale_a=1.0, scale_b=2.0)
    deck = _coupled_deck(("beta", "alpha"))  # body_0 -> beta(2x), body_1 -> alpha(1x)

    setup = build_system(
        deck,
        bem_databases={},
        dt=2.0e-3,
        t_max_kernel=120.0,
        solve_equilibrium=False,
        shared_hydro_database=db,
    )

    mpa = np.asarray(setup.lhs.M_plus_Ainf)
    c = np.asarray(setup.lhs.C)
    assert mpa.shape == (12, 12)
    assert np.asarray(setup.kernel.K).shape[:2] == (12, 12)

    # Rigid heave mass is identical for both deck bodies (same deck mass).
    rigid_33 = 1.0e7  # body.mass, per _labelled_body
    # Block 0 == 'beta' == 2x template added mass; block 1 == 'alpha' == 1x.
    assert mpa[2, 2] == pytest.approx(rigid_33 + 2.0 * a_inf_single_33, rel=1e-12)
    assert mpa[8, 8] == pytest.approx(rigid_33 + 1.0 * a_inf_single_33, rel=1e-12)
    # C carries only the (scaled) hydrostatic heave restoring, no rigid part.
    c33 = float(np.asarray(single.C)[2, 2])
    assert c[2, 2] == pytest.approx(2.0 * c33, rel=1e-12)
    assert c[8, 8] == pytest.approx(1.0 * c33, rel=1e-12)
    # No joints declared -> no constraints wired.
    assert setup.constraints is None
