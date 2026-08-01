"""Deck-driven composition driver -- M7-Foundation PR4 / F1.

``build_system(deck, *, bem_databases, dt, t_max_kernel, solve_equilibrium=True)``
takes a validated :class:`floatsim.io.deck.Deck` plus a dict of
pre-loaded BEM databases (keyed by body name) and produces a
:class:`SimulationSetup` containing everything an integrator needs::

    setup = build_system(deck, bem_databases={"body_0": hdb_0, ...},
                          dt=0.01, t_max_kernel=120.0)
    result = integrate_cummins(
        lhs=setup.lhs, kernel=setup.kernel,
        xi0=setup.xi0, xi_dot0=setup.xi_dot0,
        duration=10.0, state_force=setup.state_force,
    )

The driver itself does NOT load BEM files -- that choice is the
caller's, and is the natural place for the OrcaFlex / WAMIT /
Capytaine dispatch. The driver also does NOT build the wave-force
callable -- wave forcing is an ``external_force`` to the
integrator, conceptually distinct from connector / catenary
``state_force`` couplings. Both are intentional scope declines
(plan Q1).

Locked scope (M7-Foundation PR4):

  - Block-diagonal hydrodynamics: each body backed by an
    independent ``HydroDatabase``. Multi-body BEM cross-coupling
    requires shape promotion (tracker entry B4) and is out of
    scope.
  - Deck ``RigidLink`` connection -> existing
    :func:`floatsim.bodies.connector.heave_rigid_link` (M4 PR3
    heave-only). General N-DOF rigid link is queued as tracker
    entry A1.
  - Deck ``Catenary`` connection ->
    :func:`floatsim.mooring.catenary_analytic.make_catenary_state_force`
    via :class:`CatenaryAttachment`. Body-to-earth only (PR3
    locked scope); body-to-body catenaries raise
    ``NotImplementedError`` here.
  - Deck ``LinearSpring`` connection ->
    :func:`floatsim.bodies.connector.assemble_attachment_transformed_connector`
    via F2. **K_attach = stiffness * I_3** in the translational
    block, zero rotational. ``rest_length`` other than zero is not
    supported at PR4 (it requires equilibrium-dependent
    linearisation; see TODO in the error message). Body-body with
    any non-zero attach offset raises NotImplementedError citing
    ``phase2-followups.md#bb-offset-connector`` per the Xabier-
    pinned PR4 disposition (plan Q9).
  - Wave forcing not assembled -- caller composes
    ``make_regular_wave_force`` separately.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from floatsim.bodies.connector import (
    LinearConnector,
    assemble_attachment_transformed_connector,
    heave_rigid_link,
    make_connector_state_force,
)
from floatsim.bodies.joints import JointSet, hinge_joint, yaw_locked_joint
from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.hydrostatics import gravity_restoring_contribution
from floatsim.hydro.morison import MorisonElement, PlateDragElement, make_morison_state_force
from floatsim.hydro.radiation import CumminsLHS, assemble_cummins_lhs
from floatsim.hydro.retardation import RetardationKernel, compute_retardation_kernel
from floatsim.io.deck import (
    Body,
    Catenary,
    Deck,
    HingeJoint,
    LinearSpring,
    PlateMember,
    RigidLink,
    YawLockedJoint,
)
from floatsim.mooring.catenary_analytic import (
    CatenaryAttachment,
    CatenaryLine,
    make_catenary_state_force,
)
from floatsim.solver.equilibrium import solve_static_equilibrium
from floatsim.solver.state import (
    assemble_global_kernel,
    assemble_global_lhs,
    pack_state,
)

_EARTH_NAME: Final[str] = "earth"
_EARTH_INDEX: Final[int] = -1

# Equilibrium tolerance for the default solve. Tighter than the M4 PR6
# fixture's 1 N because the driver is the cross-deck path; if a future
# deck genuinely needs a looser tol, expose it as a kwarg later.
_EQUILIBRIUM_TOL_N: Final[float] = 1.0


_StateForce = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class SimulationSetup:
    """Output of :func:`build_system` -- everything integrate_cummins needs.

    Per plan Q1: ``xi0`` and ``xi_dot0`` are returned separately
    matching the ``integrate_cummins(xi0=..., xi_dot0=...)`` contract;
    they are NOT packed into a 12N-vector state.

    Attributes
    ----------
    lhs
        Block-diagonal global :class:`CumminsLHS` of size 6N x 6N.
    kernel
        Block-diagonal :class:`RetardationKernel` of shape
        (6N, 6N, N_t).
    state_force
        Composed (t, xi, xi_dot) -> F[6N] closure summing connector
        and catenary contributions for this deck. Returns a zero
        vector when the deck has no connections.
    xi0
        Length-6N position IC. Post-equilibrium-solve if
        ``solve_equilibrium=True`` was passed to ``build_system``;
        otherwise the deck-stated InitialConditions packed into the
        global vector.
    xi_dot0
        Length-6N velocity IC, packed from deck-stated
        InitialConditions.
    body_name_to_index
        Deck-order mapping from body name to integer slot index.
        Useful for extracting per-body channels by name from
        integrator output.
    """

    lhs: CumminsLHS
    kernel: RetardationKernel
    state_force: _StateForce
    xi0: NDArray[np.float64]
    xi_dot0: NDArray[np.float64]
    body_name_to_index: dict[str, int]
    constraints: JointSet | None = None
    """Velocity-level joint constraints from the deck's ``joints`` section
    (M9), or ``None`` if the deck declares no joints. Passed to
    ``integrate_cummins(constraints=...)`` to activate the KKT path."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_body_names(deck: Deck) -> dict[str, int]:
    """Build the body-name-to-index map; validate uniqueness; reject
    name collision with the ``earth`` sentinel."""
    name_to_index: dict[str, int] = {}
    duplicates: list[str] = []
    for k, body in enumerate(deck.bodies):
        if body.name == _EARTH_NAME:
            raise ValueError(
                f"body name {_EARTH_NAME!r} collides with the earth "
                "sentinel; rename the body in the deck."
            )
        if body.name in name_to_index:
            duplicates.append(body.name)
        else:
            name_to_index[body.name] = k
    if duplicates:
        raise ValueError(
            f"duplicate body names in deck: {duplicates}. Body names " "must be unique."
        )
    return name_to_index


def _resolve_endpoint(name: str, name_to_index: dict[str, int], context: str) -> int:
    """Map a deck-string body endpoint to an integer index or the earth sentinel."""
    if name == _EARTH_NAME:
        return _EARTH_INDEX
    if name not in name_to_index:
        raise ValueError(
            f"{context}: unknown body name {name!r}; known bodies = "
            f"{sorted(name_to_index)} plus the {_EARTH_NAME!r} sentinel."
        )
    return name_to_index[name]


def _body_mass_matrix(body: Body) -> NDArray[np.float64]:
    """Build the 6x6 rigid-body mass matrix from deck-level body data.

    M7-Foundation PR4 assumes the deck's reference_point IS the body
    frame origin and the CoG (no explicit CoG-offset field in the
    deck). Phase 2 may add an off-CoG reference-point field.
    """
    inertia_3x3 = np.array(
        [
            [body.inertia.Ixx, body.inertia.Ixy, body.inertia.Ixz],
            [body.inertia.Ixy, body.inertia.Iyy, body.inertia.Iyz],
            [body.inertia.Ixz, body.inertia.Iyz, body.inertia.Izz],
        ],
        dtype=np.float64,
    )
    return rigid_body_mass_matrix(
        mass=body.mass,
        inertia_at_reference=inertia_3x3,
        cog_offset_body=None,
    )


def _per_body_lhs(body: Body, hdb: HydroDatabase, gravity: float) -> CumminsLHS:
    """Assemble single-body CumminsLHS, passing gravity inputs only if
    hdb.C_source == 'buoyancy_only'."""
    M = _body_mass_matrix(body)
    if hdb.C_source == "buoyancy_only":
        return assemble_cummins_lhs(
            rigid_body_mass=M,
            hdb=hdb,
            mass=body.mass,
            cog_offset_from_bem_origin=np.zeros(3, dtype=np.float64),
            gravity=gravity,
        )
    return assemble_cummins_lhs(rigid_body_mass=M, hdb=hdb)


def _materialise_rigid_link(
    conn: RigidLink, name_to_index: dict[str, int], penalty_k: float
) -> LinearConnector:
    """RigidLink -> heave_rigid_link (M4 PR3 heave-only scope)."""
    a = _resolve_endpoint(conn.body_a, name_to_index, "RigidLink.body_a")
    b = _resolve_endpoint(conn.body_b, name_to_index, "RigidLink.body_b")
    return heave_rigid_link(
        body_a=a,
        body_b=b,
        penalty_stiffness=conn.penalty_stiffness_factor * penalty_k,
        penalty_damping=conn.penalty_damping_factor * penalty_k,
    )


def _materialise_linear_spring(
    conn: LinearSpring, name_to_index: dict[str, int]
) -> LinearConnector:
    """LinearSpring -> assemble_attachment_transformed_connector via F2.

    K_attach = stiffness * I_3 in the translational block (isotropic
    3-axis translational spring at the attachment point), zero
    rotational. rest_length = 0 is required at M7-Foundation PR4.
    """
    if conn.rest_length != 0.0:
        raise NotImplementedError(
            f"LinearSpring rest_length = {conn.rest_length} != 0 is not "
            "supported at M7-Foundation PR4. Non-zero rest_length requires "
            "equilibrium-dependent linearisation; deferred to a future PR. "
            "For now, use rest_length = 0 (the linear spring is anchored at "
            "the attachment-to-attachment line)."
        )
    a = _resolve_endpoint(conn.body_a, name_to_index, "LinearSpring.body_a")
    b = _resolve_endpoint(conn.body_b, name_to_index, "LinearSpring.body_b")

    # Body-body with non-zero offset is the BB-OFFSET-CONNECTOR framework limit.
    # Q9 pinned disposition: raise NotImplementedError citing the tracker entry.
    attach_a = np.asarray(conn.anchor_a_body, dtype=np.float64)
    attach_b = (
        np.zeros(3, dtype=np.float64)
        if conn.anchor_b_body is None
        else np.asarray(conn.anchor_b_body, dtype=np.float64)
    )
    a_offset = not bool(np.allclose(attach_a, 0.0))
    b_offset = not bool(np.allclose(attach_b, 0.0))

    if a != _EARTH_INDEX and b != _EARTH_INDEX and (a_offset or b_offset):
        raise NotImplementedError(
            f"LinearSpring(body_a={conn.body_a!r}, body_b={conn.body_b!r}) is a "
            "body-body connection with a non-zero attachment offset. The "
            "LinearConnector framework cannot represent this without per-"
            "endpoint K factors (Newton-III asymmetry at reference points). "
            "Tracked as BB-OFFSET-CONNECTOR in "
            "docs/phase2-followups.md#bb-offset-connector; PR4 disposition "
            "pinned at docs/m7-foundation-plan.md Q9. To proceed, either set "
            "both anchor_a_body and anchor_b_body to zero, OR change one "
            "endpoint to 'earth'."
        )

    # Isotropic translational K from scalar stiffness.
    K_attach = np.zeros((6, 6), dtype=np.float64)
    K_attach[:3, :3] = conn.stiffness * np.eye(3)

    # F2 needs to know which side carries the offset. For body-earth cases
    # F2 handles routing internally; we pass attach_a_body / attach_b_body
    # straight through.
    return assemble_attachment_transformed_connector(
        body_a=a,
        body_b=b,
        K_attach=K_attach,
        B_attach=None,
        attach_a_body=attach_a if a != _EARTH_INDEX else None,
        attach_b_body=attach_b if b != _EARTH_INDEX else None,
        rest_offset_attach=None,
    )


def _materialise_catenary(conn: Catenary, name_to_index: dict[str, int]) -> CatenaryAttachment:
    """Catenary -> CatenaryAttachment for the F3 composer.

    Body-to-earth only at PR3-locked scope. The deck-side anchor
    (whichever is 'earth') has its attach_*_body field interpreted as
    the inertial-frame anchor position.
    """
    a = _resolve_endpoint(conn.body_a, name_to_index, "Catenary.body_a")
    b = _resolve_endpoint(conn.body_b, name_to_index, "Catenary.body_b")

    if a != _EARTH_INDEX and b != _EARTH_INDEX:
        raise NotImplementedError(
            f"Catenary(body_a={conn.body_a!r}, body_b={conn.body_b!r}) is "
            "body-to-body. The F3 composer (M7-Foundation PR3 locked scope) "
            "supports body-to-earth catenaries only -- see plan Q4. "
            "Change one endpoint to 'earth' or queue a tracker entry for "
            "body-body catenary support."
        )

    if a == _EARTH_INDEX and b == _EARTH_INDEX:
        raise ValueError(
            f"Catenary cannot have both endpoints as earth; got "
            f"body_a={conn.body_a!r}, body_b={conn.body_b!r}."
        )

    if a == _EARTH_INDEX:
        # body_b is the real body; body_a is earth.
        return CatenaryAttachment(
            body_index=b,
            fairlead_body=np.asarray(conn.attach_b_body, dtype=np.float64),
            anchor_global=np.asarray(conn.attach_a_body, dtype=np.float64),
            line=CatenaryLine(
                length=conn.line.length,
                weight_per_length=conn.line.weight_per_length,
                EA=conn.line.EA,
            ),
            seabed_depth=200.0,  # default fallback; deck schema doesn't carry seabed_depth
        )
    # body_a is the real body; body_b is earth.
    return CatenaryAttachment(
        body_index=a,
        fairlead_body=np.asarray(conn.attach_a_body, dtype=np.float64),
        anchor_global=np.asarray(conn.attach_b_body, dtype=np.float64),
        line=CatenaryLine(
            length=conn.line.length,
            weight_per_length=conn.line.weight_per_length,
            EA=conn.line.EA,
        ),
        seabed_depth=200.0,
    )


def _compose_state_force(
    connector_force: _StateForce | None,
    catenary_force: _StateForce | None,
    drag_force: _StateForce | None,
    n_dof: int,
) -> _StateForce:
    """Sum the (optional) connector, catenary and drag force closures.

    Additive-generic over the present sources (order:
    connector, catenary, drag). Returns a zero-vector closure if all are
    None. With only connector + catenary present the result is
    ``connector(...) + catenary(...)`` -- byte-identical to the pre-M11a
    two-source path (drag_force=None adds nothing).
    """
    sources = [f for f in (connector_force, catenary_force, drag_force) if f is not None]

    if not sources:
        zeros = np.zeros(n_dof, dtype=np.float64)

        def _zero_force(
            _t: float, _xi: NDArray[np.float64], _xd: NDArray[np.float64]
        ) -> NDArray[np.float64]:
            return zeros

        return _zero_force

    if len(sources) == 1:
        return sources[0]

    def _composed(
        t: float, xi: NDArray[np.float64], xi_dot: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        total = sources[0](t, xi, xi_dot)
        for f in sources[1:]:
            total = total + f(t, xi, xi_dot)
        return total

    return _composed


def _build_drag_state_force(deck: Deck, n_dof: int, *, rho: float) -> _StateForce | None:
    """Compose the deck's Morison ``drag_elements`` into a state-force
    closure (M11a PR1 / plan Q3-i). WIRING only -- no new physics.

    Drag ONLY: the BEM database already carries added mass and
    Froude-Krylov, so a Morison inertia term would double-count. This is
    made **impossible**, not merely avoided -- each element is constructed
    with ``include_inertia=False`` and ``include_inertia=True`` in the deck
    is rejected with a clear error. ``Ca`` feeds only the inertia term and
    is therefore inert on this path (permitted but unused; matches the
    committed ``two_body_semisub_barge.yml`` element's ``Ca=1.0``).

    Fluid is still water (calm) -- the free-decay convention the M10/M9
    studies used. Wave-orbital-velocity-relative drag is a follow-on that
    couples to the wave field (composed separately, M10 A4).

    Returns ``None`` when no body declares ``drag_elements`` (the
    common case), so a drag-free deck's ``state_force`` is untouched.
    """
    elements: list[MorisonElement | PlateDragElement] = []
    for k, body in enumerate(deck.bodies):
        for e in body.drag_elements:
            if isinstance(e, PlateMember):
                elements.append(
                    PlateDragElement(
                        body_index=k,
                        center_body=np.asarray(e.center, dtype=np.float64),
                        normal_body=np.asarray(e.normal, dtype=np.float64),
                        radius=e.radius,
                        thickness=e.thickness,
                        Cd_n=e.Cd_n,
                        Cd_t=e.Cd_t,
                        n_radial=e.n_radial,
                        n_azimuthal=e.n_azimuthal,
                    )
                )
                continue
            # MorisonMember cylinder (spar / brace).
            if e.include_inertia:
                raise ValueError(
                    f"body {body.name!r} drag element: include_inertia=True is not "
                    "supported by build_system's drag wiring (M11a PR1 is drag-only). "
                    "The BEM database already carries added mass + Froude-Krylov, so the "
                    "Morison inertia term would double-count. Set include_inertia=False."
                )
            elements.append(
                MorisonElement(
                    body_index=k,
                    node_a_body=np.asarray(e.node_a, dtype=np.float64),
                    node_b_body=np.asarray(e.node_b, dtype=np.float64),
                    diameter=e.diameter,
                    Cd=e.Cd,
                    Ca=e.Ca,  # inert: include_inertia is forced False below
                    include_inertia=False,
                )
            )
    if not elements:
        return None

    calm = np.zeros(3, dtype=np.float64)

    def _calm_fluid(_point: NDArray[np.float64], _t: float) -> NDArray[np.float64]:
        return calm

    return make_morison_state_force(elements, n_dof=n_dof, fluid_velocity_fn=_calm_fluid, rho=rho)


# ---------------------------------------------------------------------------
# Public API: build_system
# ---------------------------------------------------------------------------


def _build_joint_set(deck: Deck, name_to_index: dict[str, int]) -> JointSet | None:
    """Materialise the deck's ``joints`` section into a :class:`JointSet`
    (M9). Body names -> indices; ``'earth'`` -> ``-1``."""
    if not deck.joints:
        return None

    def idx(name: str) -> int:
        return -1 if name == "earth" else name_to_index[name]

    built = []
    for j in deck.joints:
        body_a = idx(j.body_a)
        body_b = idx(j.body_b)
        attach_a = np.asarray(j.attach_a_body, dtype=np.float64)
        attach_b = np.asarray(j.attach_b_body, dtype=np.float64)
        axis = np.asarray(j.axis, dtype=np.float64)
        if isinstance(j, HingeJoint):
            built.append(
                hinge_joint(body_a, body_b, attach_a=attach_a, attach_b=attach_b, axis=axis)
            )
        elif isinstance(j, YawLockedJoint):  # pragma: no branch
            built.append(
                yaw_locked_joint(body_a, body_b, attach_a=attach_a, attach_b=attach_b, axis=axis)
            )
    # M10 PR0.75: supply each body's world reference position so the joint
    # layer reads xi as displacement-from-reference (the coupled Cummins
    # convention), not absolute position (Amendment A2).
    refs = tuple(np.asarray(b.reference_point, dtype=np.float64) for b in deck.bodies)
    return JointSet(joints=tuple(built), n_bodies=len(deck.bodies), body_references=refs)


def _inject_hydrostatic_c(
    c_mat: NDArray[np.float64],
    hydro: list[tuple[int, str]],
    shared_db: HydroDatabase,
    hydrostatic_database: HydroDatabase | None,
) -> None:
    """Add per-body block-diagonal buoyancy stiffness into ``c_mat`` in
    place, resolved BY LABEL from a hydrostatic reference (M10 PR0.85).

    ``hydro`` is a list of ``(global_body_index, hydro_body_label)``. A
    single-body reference (``body_labels is None``) is broadcast to every
    body (M8's ``kron(I, c_single)`` degenerate case); a labelled
    reference resolves each body's ``6x6`` block by label. Hydrostatic
    stiffness is per-body block-diagonal and cannot live in a coupled
    (cross-coupled) BEM, so a coupled ``C`` that is all-zero means "no
    restoring" -- never "provided as zero": with no reference supplied
    this raises rather than assembling a silently non-oscillating system.
    """
    if hydrostatic_database is None:
        if not np.any(np.asarray(shared_db.C)):
            raise ValueError(
                "coupled shared_hydro_database carries zero hydrostatic C (a coupled "
                "BEM is inter-body radiation/excitation only; hydrostatic stiffness "
                "is per-body block-diagonal and cannot live there) and no "
                "'hydrostatic_database' was provided. Supply a per-body hydrostatic "
                "reference (M10 PR0.85)."
            )
        return
    labels = hydrostatic_database.body_labels
    c_ref = np.asarray(hydrostatic_database.C, dtype=np.float64)
    for k, label in hydro:
        if labels is None:
            blk = c_ref  # single-body 6x6 reference, broadcast to every body
        else:
            if label not in labels:
                raise ValueError(
                    f"hydro_body_label {label!r} not found in hydrostatic_database "
                    f"labels {list(labels)} (M10 PR0.85 label contract)"
                )
            j = labels.index(label)
            blk = c_ref[6 * j : 6 * j + 6, 6 * j : 6 * j + 6]
        c_mat[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] += blk


def _build_coupled_mixed(
    deck: Deck,
    shared_db: HydroDatabase,
    dt: float,
    t_max_kernel: float,
    gravity: float,
    asymptote_check_override: str | None = None,
    hydrostatic_database: HydroDatabase | None = None,
    kernel_decay_floor_override: str | None = None,
) -> tuple[CumminsLHS, RetardationKernel]:
    """Coupled ``6*n_deck`` LHS + kernel for a deck mixing hydro (labelled)
    bodies with structural (hydro-free) bodies (M10 PR0.5, plan Q4-c).

    FORK 1 (driver-local; no ``HydroDatabase`` contract change): the hydro
    bodies' blocks are permuted by label out of ``shared_db`` and
    **scattered** into the global matrices at their deck-body DOF slots;
    each structural body contributes its rigid mass/inertia only, and
    exactly ZERO to ``A_inf`` / ``B`` / the kernel / hydrostatic ``C``. The
    kernel is computed on the hydro-only sub-database (contract-valid) and
    **embedded** to global size (option i -- the integrator requires
    ``lhs.n_dof == kernel.n_dof``, ``newmark.py``).
    """
    labels = shared_db.body_labels
    assert labels is not None  # caller checked
    n = len(deck.bodies)

    for b in deck.bodies:
        if b.hydro_body_label is None and not b.structural:
            raise ValueError(
                f"coupled deck body {b.name!r} has neither 'hydro_body_label' nor "
                "'structural: true'; a shared_hydro_database deck cannot also carry "
                "a per-body 'hydro_database'"
            )

    hydro = [
        (k, str(b.hydro_body_label))
        for k, b in enumerate(deck.bodies)
        if b.hydro_body_label is not None
    ]
    deck_hydro_labels = [lab for _, lab in hydro]
    missing = sorted(set(deck_hydro_labels) - set(labels))
    if missing:
        raise ValueError(
            f"hydro_body_label(s) {missing} not found in shared database labels "
            f"{list(labels)} (M8 label contract)"
        )
    unused = sorted(set(labels) - set(deck_hydro_labels))
    if unused:
        raise ValueError(
            f"shared database has label(s) {unused} unused by any deck body "
            f"(deck hydro labels: {deck_hydro_labels}); coupling requires a full "
            "mapping of the hydro bodies"
        )

    # Hydro-only permutation (db block order -> deck hydro-body order).
    perm = np.concatenate([6 * labels.index(lab) + np.arange(6) for lab in deck_hydro_labels])
    a_inf_h = np.asarray(shared_db.A_inf)[np.ix_(perm, perm)]
    c_h = np.asarray(shared_db.C)[np.ix_(perm, perm)].copy()
    a_h = np.asarray(shared_db.A)[np.ix_(perm, perm)]
    b_h = np.asarray(shared_db.B)[np.ix_(perm, perm)]
    rao_h = np.asarray(shared_db.RAO)[perm]

    # Scatter map: the global 6-DOF slots of the hydro bodies, in
    # hydro-body order (same order as ``perm`` / the hydro blocks above).
    hydro_dof = np.concatenate([6 * k + np.arange(6) for k, _ in hydro])

    # Global LHS (6*n_deck): rigid mass/inertia for EVERY body; hydro A_inf
    # and C scattered in by label; structural blocks stay hydro-zero.
    m_plus_ainf = np.zeros((6 * n, 6 * n), dtype=np.float64)
    c_mat = np.zeros((6 * n, 6 * n), dtype=np.float64)
    for k, body in enumerate(deck.bodies):
        m_plus_ainf[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] = _body_mass_matrix(body)
    m_plus_ainf[np.ix_(hydro_dof, hydro_dof)] += a_inf_h
    c_mat[np.ix_(hydro_dof, hydro_dof)] += c_h
    if shared_db.C_source == "buoyancy_only":
        for k, _lab in hydro:  # gravity restoring only for the hydro (floating) bodies
            c_mat[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] += gravity_restoring_contribution(
                mass=deck.bodies[k].mass,
                cog_offset_from_bem_origin=np.zeros(3, dtype=np.float64),
                gravity=gravity,
            )
    # M10 PR0.85: per-body block-diagonal buoyancy stiffness from a reference.
    _inject_hydrostatic_c(c_mat, hydro, shared_db, hydrostatic_database)
    lhs = CumminsLHS(M_plus_Ainf=m_plus_ainf, C=c_mat)

    # Kernel: compute on the hydro-only sub-database (N-body-contract-valid),
    # then embed into the global 6*n_deck kernel (structural rows/cols zero).
    reordered = HydroDatabase(
        omega=shared_db.omega,
        heading_deg=shared_db.heading_deg,
        A=a_h,
        B=b_h,
        A_inf=a_inf_h,
        C=c_h,
        RAO=rao_h,
        reference_point=shared_db.reference_point,
        C_source=shared_db.C_source,
        metadata=dict(shared_db.metadata),
        body_labels=tuple(deck_hydro_labels),
    )
    kernel_h = compute_retardation_kernel(
        reordered,
        t_max=t_max_kernel,
        dt=dt,
        asymptote_check_override=asymptote_check_override,
        kernel_decay_floor_override=kernel_decay_floor_override,
    )
    k_h = np.asarray(kernel_h.K)
    k_global = np.zeros((6 * n, 6 * n, k_h.shape[-1]), dtype=np.float64)
    k_global[np.ix_(hydro_dof, hydro_dof)] = k_h
    kernel = RetardationKernel(K=k_global, t=kernel_h.t, dt=kernel_h.dt)
    return lhs, kernel


def _build_coupled_lhs_kernel(
    deck: Deck,
    shared_db: HydroDatabase,
    dt: float,
    t_max_kernel: float,
    gravity: float,
    asymptote_check_override: str | None = None,
    hydrostatic_database: HydroDatabase | None = None,
    kernel_decay_floor_override: str | None = None,
) -> tuple[CumminsLHS, RetardationKernel]:
    """Coupled ``6N`` LHS + kernel from a shared N-body database (M9 Q5).

    Deck body -> database block mapping is **by label** (the M8 contract,
    ``tests/support/condensation.py`` the reference implementation): the
    ``hydro_body_label`` of each body selects a ``6x6`` block of the
    shared database, and the coupled matrices are permuted into deck-body
    order. Hard-raises on missing / duplicate / unused labels.
    """
    labels = shared_db.body_labels
    if labels is None:
        raise ValueError(
            "shared_hydro_database is single-body (no body_labels); a coupled "
            "deck requires an N-body database (M8 reader with per-body labels)"
        )
    # M10 PR0.5: a deck mixing hydro (labelled) bodies with structural
    # (hydro-free) bodies takes the scatter/embed path. Decks with NO
    # structural body take the original M9 PR3 path below, UNCHANGED
    # (byte-identity, the M8/M9 N=1 pattern).
    if any(b.structural for b in deck.bodies):
        return _build_coupled_mixed(
            deck,
            shared_db,
            dt,
            t_max_kernel,
            gravity,
            asymptote_check_override,
            hydrostatic_database,
            kernel_decay_floor_override,
        )
    if any(b.hydro_body_label is None for b in deck.bodies):
        raise ValueError(
            "shared_hydro_database is declared, so EVERY body must select its "
            "block via hydro_body_label (no mixing per-body and shared)"
        )
    deck_labels = [str(b.hydro_body_label) for b in deck.bodies]
    missing = sorted(set(deck_labels) - set(labels))
    if missing:
        raise ValueError(
            f"hydro_body_label(s) {missing} not found in shared database labels "
            f"{list(labels)} (M8 label contract)"
        )
    unused = sorted(set(labels) - set(deck_labels))
    if unused:
        raise ValueError(
            f"shared database has label(s) {unused} unused by any deck body "
            f"(deck labels: {deck_labels}); coupling requires a full mapping"
        )

    # Permutation: deck body k <- database block at labels.index(deck_labels[k]).
    perm = np.concatenate([6 * labels.index(dl) + np.arange(6) for dl in deck_labels])
    a_inf = np.asarray(shared_db.A_inf)[np.ix_(perm, perm)]
    c_mat = np.asarray(shared_db.C)[np.ix_(perm, perm)].copy()
    a_omega = np.asarray(shared_db.A)[np.ix_(perm, perm)]
    b_omega = np.asarray(shared_db.B)[np.ix_(perm, perm)]
    rao = np.asarray(shared_db.RAO)[perm]

    n = len(deck.bodies)
    rigid = np.zeros((6 * n, 6 * n), dtype=np.float64)
    for k, body in enumerate(deck.bodies):
        rigid[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] = _body_mass_matrix(body)
    m_plus_ainf = rigid + a_inf

    # Per-block gravity restoring (M5 hydrostatic-gravity lesson) when the
    # database carries buoyancy-only C.
    if shared_db.C_source == "buoyancy_only":
        for k, body in enumerate(deck.bodies):
            c_mat[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] += gravity_restoring_contribution(
                mass=body.mass,
                cog_offset_from_bem_origin=np.zeros(3, dtype=np.float64),
                gravity=gravity,
            )

    # M10 PR0.85: per-body block-diagonal buoyancy stiffness from a reference
    # (or raise if the coupled C is zero and no reference was supplied).
    _inject_hydrostatic_c(
        c_mat, [(k, deck_labels[k]) for k in range(n)], shared_db, hydrostatic_database
    )

    lhs = CumminsLHS(M_plus_Ainf=m_plus_ainf, C=c_mat)
    reordered = HydroDatabase(
        omega=shared_db.omega,
        heading_deg=shared_db.heading_deg,
        A=a_omega,
        B=b_omega,
        A_inf=a_inf,
        C=c_mat,
        RAO=rao,
        reference_point=shared_db.reference_point,
        C_source=shared_db.C_source,
        metadata=dict(shared_db.metadata),
        body_labels=tuple(deck_labels),
    )
    kernel = compute_retardation_kernel(
        reordered,
        t_max=t_max_kernel,
        dt=dt,
        asymptote_check_override=asymptote_check_override,
        kernel_decay_floor_override=kernel_decay_floor_override,
    )
    return lhs, kernel


def build_system(
    deck: Deck,
    *,
    bem_databases: dict[str, HydroDatabase],
    dt: float,
    t_max_kernel: float,
    solve_equilibrium: bool = True,
    shared_hydro_database: HydroDatabase | None = None,
    asymptote_check_override: str | None = None,
    hydrostatic_database: HydroDatabase | None = None,
    kernel_decay_floor_override: str | None = None,
) -> SimulationSetup:
    """Materialise a deck-driven simulation setup.

    ``asymptote_check_override`` (M10 PR0): a non-empty rationale string
    that bypasses the retardation-kernel high-frequency asymptote gate
    (M7.5 PR1, ``ITEM25-SMALL-BODY-APPLICABILITY``) on the **coupled**
    (``shared_hydro_database``) path only. Small-body BEM (e.g. the
    cluster hulls, L~1.85 m) does not reach the ``1/omega^4`` regime by
    ``omega_max``; the override is a forcing function — the kernel still
    raises on an empty / whitespace rationale. ``None`` (default) leaves
    the gate active and the assembly byte-identical to pre-PR0.

    ``kernel_decay_floor_override`` (M11b PR8): a non-empty rationale
    string that exempts **noise-floor DOFs** from the retardation-kernel
    post-extension decay gate (Check 3) on the **coupled** path only. A
    DOF qualifies for exemption ONLY by a measured criterion — its kernel
    peak ``|K|`` relative to the matrix's dominant diagonal entry falls
    below ``retardation._KERNEL_DECAY_NOISE_FLOOR`` (see the constant's
    derivation). The 12-buoy platform's yaw radiation is
    ``|K|/dominant ~ 4e-15`` (numerical noise, not physics); its
    non-decay on the coarse 13-omega grid is a gate false-positive. The
    exemption is per-DOF and reported (a ``warnings.warn`` lists the
    exempted DOFs and their measured ratios); physical DOFs that fail to
    decay still raise. Empty / whitespace rationale raises, exactly as
    ``asymptote_check_override``. ``None`` (default) leaves Check 3 fully
    active. See tracker ``KERNEL-DECAY-COARSE-GRID``.

    Parameters
    ----------
    deck
        Validated :class:`floatsim.io.deck.Deck` instance.
    bem_databases
        Dict mapping body name (as in ``deck.bodies[k].name``) to a
        pre-loaded :class:`HydroDatabase`. The driver does NOT call
        any reader; the caller dispatches OrcaFlex / WAMIT / Capytaine
        per body. Every body in the deck must have an entry.
    dt
        Integration step in seconds. Passed to
        :func:`compute_retardation_kernel` per body.
    t_max_kernel
        Retardation kernel duration in seconds (the convolution
        memory). Same value used for every body.
    solve_equilibrium
        When True (default), the returned ``xi0`` is the result of
        :func:`solve_static_equilibrium` starting from the deck-
        stated initial position. When False, ``xi0`` is the
        deck-stated initial position packed into the global vector.

    Returns
    -------
    SimulationSetup
        See class docstring.

    Raises
    ------
    ValueError
        On deck-content issues: duplicate body names, body name
        colliding with the ``earth`` sentinel, missing BEM database
        for a body, unresolved connection endpoint, Catenary with
        both endpoints earth.
    NotImplementedError
        For features out of M7-Foundation PR4 scope: LinearSpring
        rest_length != 0; body-body LinearSpring with non-zero
        attach offset (BB-OFFSET-CONNECTOR); body-body Catenary.
    """
    # --- Body bookkeeping ---------------------------------------------------
    name_to_index = _validate_body_names(deck)

    # --- LHS + kernel: coupled (shared database) or per-body block-diagonal --
    if deck.shared_hydro_database is not None:
        if shared_hydro_database is None:
            raise ValueError(
                "deck declares 'shared_hydro_database' but none was passed to "
                "build_system(shared_hydro_database=...). The driver does not "
                "load BEM files itself."
            )
        if deck.hydrostatic_database is not None and hydrostatic_database is None:
            raise ValueError(
                "deck declares 'hydrostatic_database' but none was passed to "
                "build_system(hydrostatic_database=...). The driver does not load "
                "BEM files itself (M10 PR0.85)."
            )
        lhs_global, kernel_global = _build_coupled_lhs_kernel(
            deck,
            shared_hydro_database,
            dt,
            t_max_kernel,
            deck.environment.gravity,
            hydrostatic_database=hydrostatic_database,
            asymptote_check_override=asymptote_check_override,
            kernel_decay_floor_override=kernel_decay_floor_override,
        )
    else:
        for body in deck.bodies:
            if body.name not in bem_databases:
                raise ValueError(
                    f"bem_databases missing entry for body {body.name!r}. The "
                    "driver does not load BEM files itself; caller must pre-load "
                    "each body's database and pass it via the bem_databases dict."
                )
        per_body_lhs = [
            _per_body_lhs(body, bem_databases[body.name], gravity=deck.environment.gravity)
            for body in deck.bodies
        ]
        per_body_kernel = [
            compute_retardation_kernel(bem_databases[body.name], t_max=t_max_kernel, dt=dt)
            for body in deck.bodies
        ]
        lhs_global = assemble_global_lhs(per_body_lhs)
        kernel_global = assemble_global_kernel(per_body_kernel)
    n_dof = lhs_global.n_dof

    # --- Connection materialisation ----------------------------------------
    # Rigid-link penalty stiffness scale: max(diag(C_global)).
    penalty_k_scale = float(np.max(np.diag(lhs_global.C)))

    connectors: list[LinearConnector] = []
    catenary_attachments: list[CatenaryAttachment] = []

    for conn in deck.connections:
        if isinstance(conn, RigidLink):
            connectors.append(_materialise_rigid_link(conn, name_to_index, penalty_k_scale))
        elif isinstance(conn, LinearSpring):
            connectors.append(_materialise_linear_spring(conn, name_to_index))
        elif isinstance(conn, Catenary):
            catenary_attachments.append(_materialise_catenary(conn, name_to_index))
        else:  # pragma: no cover -- pydantic discriminator forbids unknown types
            raise TypeError(f"unknown Connection type: {type(conn).__name__}")

    connector_force = make_connector_state_force(connectors, n_dof=n_dof) if connectors else None
    catenary_force = (
        make_catenary_state_force(catenary_attachments, n_dof=n_dof)
        if catenary_attachments
        else None
    )
    # M11a PR1 (Q3-i): compose the deck's Morison drag_elements into the
    # state-force alongside connector/catenary. Common to both the coupled
    # and per-body assembly paths. Drag-only (no inertia double-count);
    # None when no body declares drag_elements (drag-free decks untouched).
    drag_force = _build_drag_state_force(deck, n_dof, rho=deck.environment.water_density)
    state_force = _compose_state_force(connector_force, catenary_force, drag_force, n_dof)

    # --- Initial conditions -------------------------------------------------
    xi0_packed = pack_state(
        [np.asarray(b.initial_conditions.position, dtype=np.float64) for b in deck.bodies]
    )
    xi_dot0_packed = pack_state(
        [np.asarray(b.initial_conditions.velocity, dtype=np.float64) for b in deck.bodies]
    )

    if solve_equilibrium:
        eq = solve_static_equilibrium(
            lhs=lhs_global,
            state_force=state_force,
            xi0=xi0_packed,
            tol=_EQUILIBRIUM_TOL_N,
        )
        xi0 = eq.xi_eq
    else:
        xi0 = xi0_packed

    return SimulationSetup(
        lhs=lhs_global,
        kernel=kernel_global,
        state_force=state_force,
        xi0=xi0,
        xi_dot0=xi_dot0_packed,
        body_name_to_index=name_to_index,
        constraints=_build_joint_set(deck, name_to_index),
    )
