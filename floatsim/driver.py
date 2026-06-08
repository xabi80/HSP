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
  - Deck ``Catenary`` connection -> :func:`floatsim.mooring.catenary_analytic.make_catenary_state_force`
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
from floatsim.bodies.mass_properties import rigid_body_mass_matrix
from floatsim.hydro.database import HydroDatabase
from floatsim.hydro.radiation import CumminsLHS, assemble_cummins_lhs
from floatsim.hydro.retardation import RetardationKernel, compute_retardation_kernel
from floatsim.io.deck import (
    Body,
    Catenary,
    Deck,
    LinearSpring,
    RigidLink,
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


_StateForce = Callable[
    [float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]
]


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
            f"duplicate body names in deck: {duplicates}. Body names "
            "must be unique."
        )
    return name_to_index


def _resolve_endpoint(
    name: str, name_to_index: dict[str, int], context: str
) -> int:
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


def _per_body_lhs(
    body: Body, hdb: HydroDatabase, gravity: float
) -> CumminsLHS:
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


def _materialise_catenary(
    conn: Catenary, name_to_index: dict[str, int]
) -> CatenaryAttachment:
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
    n_dof: int,
) -> _StateForce:
    """Sum the (optional) connector and catenary force closures.

    Returns a zero-vector closure if both are None.
    """
    zeros = np.zeros(n_dof, dtype=np.float64)

    if connector_force is None and catenary_force is None:

        def _zero_force(_t: float, _xi: NDArray[np.float64], _xd: NDArray[np.float64]) -> NDArray[np.float64]:
            return zeros

        return _zero_force

    if catenary_force is None:
        return connector_force  # type: ignore[return-value]

    if connector_force is None:
        return catenary_force

    cf = connector_force
    af = catenary_force

    def _composed(
        t: float, xi: NDArray[np.float64], xi_dot: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return cf(t, xi, xi_dot) + af(t, xi, xi_dot)

    return _composed


# ---------------------------------------------------------------------------
# Public API: build_system
# ---------------------------------------------------------------------------


def build_system(
    deck: Deck,
    *,
    bem_databases: dict[str, HydroDatabase],
    dt: float,
    t_max_kernel: float,
    solve_equilibrium: bool = True,
) -> SimulationSetup:
    """Materialise a deck-driven simulation setup.

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
    for body in deck.bodies:
        if body.name not in bem_databases:
            raise ValueError(
                f"bem_databases missing entry for body {body.name!r}. The "
                "driver does not load BEM files itself; caller must pre-load "
                "each body's database and pass it via the bem_databases dict."
            )

    # --- LHS + kernel per body ---------------------------------------------
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
            connectors.append(
                _materialise_rigid_link(conn, name_to_index, penalty_k_scale)
            )
        elif isinstance(conn, LinearSpring):
            connectors.append(_materialise_linear_spring(conn, name_to_index))
        elif isinstance(conn, Catenary):
            catenary_attachments.append(_materialise_catenary(conn, name_to_index))
        else:  # pragma: no cover -- pydantic discriminator forbids unknown types
            raise TypeError(f"unknown Connection type: {type(conn).__name__}")

    connector_force = (
        make_connector_state_force(connectors, n_dof=n_dof) if connectors else None
    )
    catenary_force = (
        make_catenary_state_force(catenary_attachments, n_dof=n_dof)
        if catenary_attachments
        else None
    )
    state_force = _compose_state_force(connector_force, catenary_force, n_dof)

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
    )
