"""YAML deck schema for FloatSim — ARCHITECTURE.md §5.

Milestone 0 scope: structural validation only. We check shape, types, ranges
and enum membership. We do NOT check that referenced paths exist, parse BEM
files, or cross-reference body names against connectors — those land in
later milestones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------
# Numeric aliases. ``allow_inf_nan=False`` on every float shuts the door on
# silent NaN/inf contamination — a class of bug that is miserable to debug
# once it reaches the integrator.
# --------------------------------------------------------------------------

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]

Vec3 = Annotated[list[FiniteFloat], Field(min_length=3, max_length=3)]
Vec6 = Annotated[list[FiniteFloat], Field(min_length=6, max_length=6)]


class _Base(BaseModel):
    """Project-wide pydantic defaults: forbid unknown fields, assign defaults on validate."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# Simulation block (§5, §9.1, §9.3, §9.4)
# --------------------------------------------------------------------------


class Simulation(_Base):
    """Time-integration settings and startup policy (ARCHITECTURE.md §9)."""

    duration: PositiveFloat
    dt: PositiveFloat
    integrator: Literal["generalized_alpha", "rk4"] = "generalized_alpha"
    spectral_radius_inf: UnitInterval = 0.8
    retardation_memory: PositiveFloat = 60.0  # §9.1 default
    ramp_duration: NonNegativeFloat = 20.0  # §9.3 default
    skip_static_equilibrium: bool = False  # §9.4 — debug only


# --------------------------------------------------------------------------
# Environment (§5)
# --------------------------------------------------------------------------


class Environment(_Base):
    """Ambient water and gravity. Units: m, kg/m^3, m/s^2."""

    water_depth: PositiveFloat
    water_density: PositiveFloat
    gravity: PositiveFloat = 9.80665


# --------------------------------------------------------------------------
# Waves — Phase 1 supports regular Airy waves only (§1.1).
# --------------------------------------------------------------------------


class RegularWave(_Base):
    """First-order Airy wave. ``heading`` is in degrees at the deck boundary (§3.2)."""

    type: Literal["regular"]
    height: PositiveFloat  # wave height H = 2 * amplitude
    period: PositiveFloat
    heading: FiniteFloat  # degrees


# --------------------------------------------------------------------------
# Rigid-body block (§5, §3.3)
# --------------------------------------------------------------------------


class Inertia(_Base):
    """Mass-moment tensor about the body reference point, in the body frame."""

    Ixx: PositiveFloat
    Iyy: PositiveFloat
    Izz: PositiveFloat
    Ixy: FiniteFloat = 0.0
    Ixz: FiniteFloat = 0.0
    Iyz: FiniteFloat = 0.0


class HydroDatabaseRef(_Base):
    """Pointer to a BEM database file; content not parsed until Milestone 1+."""

    format: Literal["orcawave", "wamit", "capytaine"]
    path: str
    body_index: Annotated[int, Field(ge=0)] = 0


class MorisonMember(_Base):
    """Slender-cylinder drag (and optional inertia) element between two body-frame nodes.

    By default (``include_inertia=False``) the element contributes only
    the quadratic-drag term ``½·ρ·D·Cd·|u_n|·u_n`` per unit length;
    ``Ca`` is then unused. Setting ``include_inertia=True`` adds the
    Froude-Krylov + added-mass term ``ρ·A_x·(1+Ca)·a_fluid_n − ρ·A_x·Ca·a_body_n``.

    For bodies whose ``hydro_database`` is non-empty (a BEM run),
    ``include_inertia=True`` double-counts inertia and the deck loader
    emits a startup warning naming the offending member. See M5 PR4
    plan Q1 for the rationale.
    """

    type: Literal["morison_member"]
    node_a: Vec3
    node_b: Vec3
    diameter: PositiveFloat
    Cd: NonNegativeFloat
    Ca: NonNegativeFloat = 0.0
    include_inertia: bool = False


class PlateMember(_Base):
    """Direction-dependent circular-plate (heave-plate) drag element.

    A thin disc resists broadside (normal) flow far more than edge-on
    (tangential) flow. Unlike :class:`MorisonMember` (an isotropic
    member-normal cylinder), this element decomposes the local flow into a
    NORMAL term -- ``½·ρ·Cd_n·|w|·w`` integrated over the disc face, capturing
    both heave and the tilting-rotational contribution -- and a minor
    TANGENTIAL (edge-on) term -- ``½·ρ·Cd_t·(t·2a)·|u_t|·u_t`` at the rim.
    Maps to :class:`floatsim.hydro.morison.PlateDragElement`; drag-only (the
    BEM carries added mass). See M11a PR4 (plan Q3-iii, Finding F3).

    A body carrying a ``PlateMember`` may not also carry a ``MorisonMember``
    cylinder lying in the plate plane (the M11a-PR1 horizontal-cylinder
    heave-plate stand-in) -- the plate element supersedes it; ``build_system``
    raises on the double-count (only spars parallel to the plate normal are
    permitted alongside a plate).
    """

    type: Literal["plate"]
    center: Vec3
    normal: Vec3
    radius: PositiveFloat
    thickness: NonNegativeFloat
    Cd_n: NonNegativeFloat
    Cd_t: NonNegativeFloat = 0.0
    n_radial: Annotated[int, Field(ge=1)] = 12
    n_azimuthal: Annotated[int, Field(ge=1)] = 24


DragElement = Annotated[MorisonMember | PlateMember, Field(discriminator="type")]


def distributed_cylinder_drag(
    *,
    z_bottom: float,
    z_top: float,
    diameter: float,
    cd: float,
    n_segments: int,
) -> list[MorisonMember]:
    """Build ``n_segments`` stacked vertical Morison members spanning the
    body-frame z-range ``[z_bottom, z_top]`` (M11a PR2, plan Q3-ii).

    Each member is a short vertical cylinder segment on the body axis
    (``node_a = [0,0,z_lo]``, ``node_b = [0,0,z_hi]``). Because the axis is
    vertical, the member-normal drag responds ONLY to the body's LATERAL
    velocity -- in pure heave (motion along the axis) it contributes
    nothing. The same elements damp BOTH articulated rotational families
    (buoy-vs-hub and cluster-vs-platform), since each moves the spar
    laterally.

    Distributing the drag along the span (rather than one lumped element)
    matters because the drag moment weights as ``s^3`` along the member:
    a single element underpredicts by ~26 % on the spar geometry; the
    midpoint-rule error falls as ``1/N^2`` (M11a PR2 STEP 2c). This is the
    CORRECT use of the existing member-normal cylinder model for a slender
    vertical spar (no new physics); the heave-plate mis-model stays PR4.

    ``Ca``/inertia are omitted (drag-only): the BEM carries added mass, and
    ``build_system`` rejects ``include_inertia=True`` (M11a PR1).
    """
    if n_segments < 1:
        raise ValueError(f"n_segments must be >= 1; got {n_segments}")
    if z_top <= z_bottom:
        raise ValueError(f"z_top ({z_top}) must exceed z_bottom ({z_bottom})")
    edges = [z_bottom + (z_top - z_bottom) * k / n_segments for k in range(n_segments + 1)]
    return [
        MorisonMember(
            type="morison_member",
            node_a=[0.0, 0.0, edges[k]],
            node_b=[0.0, 0.0, edges[k + 1]],
            diameter=diameter,
            Cd=cd,
        )
        for k in range(n_segments)
    ]


class InitialConditions(_Base):
    """Initial 6-DOF position (from equilibrium) and velocity.

    Component order: surge, sway, heave, roll, pitch, yaw.
    """

    position: Vec6 = Field(default_factory=lambda: [0.0] * 6)
    velocity: Vec6 = Field(default_factory=lambda: [0.0] * 6)


class Body(_Base):
    """One rigid floating body.

    Hydro source (exactly one, M9 PR3 / plan Q5; ``structural`` M10 PR0.5):
    * ``hydro_database`` — a per-body single-body BEM database (the
      pre-M9 path; block-diagonal assembly), or
    * ``hydro_body_label`` — a label selecting this body's ``6x6`` block
      of the deck-level ``shared_hydro_database`` (the coupled path), or
    * ``structural: true`` — a hydro-free rigid body (an articulation
      hub / arm structure): rigid mass + inertia only, ZERO contribution
      to A_inf, B, the retardation kernel and hydrostatic C. Permitted
      only in a coupled deck (with a ``shared_hydro_database``).

    The "exactly one" rule is a forcing function: a body that declares
    NONE (e.g. a misspelled ``hydro_databse:`` key) raises rather than
    silently becoming a hydro-free body (M10 plan Amendment A1).
    """

    name: Annotated[str, Field(min_length=1)]
    reference_point: Vec3
    mass: PositiveFloat
    inertia: Inertia
    hydro_database: HydroDatabaseRef | None = None
    hydro_body_label: str | None = None
    structural: bool = False
    drag_elements: list[DragElement] = Field(default_factory=list)
    initial_conditions: InitialConditions = Field(default_factory=InitialConditions)

    @model_validator(mode="after")
    def _exactly_one_hydro_source(self) -> Body:
        n_set = (
            (self.hydro_database is not None)
            + (self.hydro_body_label is not None)
            + bool(self.structural)
        )
        if n_set != 1:
            raise ValueError(
                f"body {self.name!r}: exactly one of 'hydro_database' (per-body), "
                "'hydro_body_label' (shared coupled database), or 'structural: true' "
                f"(a hydro-free rigid body) must be set; got {n_set}"
            )
        return self


# --------------------------------------------------------------------------
# Connections — springs and catenary lines (§5, §1.1).
# --------------------------------------------------------------------------


class LinearSpring(_Base):
    """6-DOF linear spring between two bodies, or between a body and the earth sentinel."""

    type: Literal["linear_spring"]
    body_a: Annotated[str, Field(min_length=1)]
    body_b: Annotated[str, Field(min_length=1)]
    anchor_a_body: Vec3
    anchor_b_body: Vec3 | None = None
    anchor_b_global: Vec3 | None = None
    stiffness: PositiveFloat
    rest_length: NonNegativeFloat = 0.0


class CatenaryLine(_Base):
    """Irvine analytic catenary line parameters."""

    length: PositiveFloat
    weight_per_length: FiniteFloat  # N/m in water (may be negative for buoyant sections)
    EA: PositiveFloat


class Catenary(_Base):
    """Mooring catenary between two bodies (or a body and earth via body_b='earth')."""

    type: Literal["catenary"]
    body_a: Annotated[str, Field(min_length=1)]
    body_b: Annotated[str, Field(min_length=1)]
    attach_a_body: Vec3
    attach_b_body: Vec3
    line: CatenaryLine


class RigidLink(_Base):
    """Heave-only penalty rigid link between two bodies.

    Per ``docs/milestone-4-plan.md`` Q1, the stiffness is specified as a
    dimensionless factor ``10^3 ... 10^4`` (default ``10^4``) multiplied
    at solve-setup time by ``max(diag(C_global))`` to obtain N/m. The
    ceiling at ``10^5`` guards against a prohibitive explicit-integrator
    stability floor (``dt < 2 / sqrt(2 k / mu_eff)`` — see
    ``floatsim.bodies.connector.check_connector_stability``).

    M4 PR3 only implements the heave-only constraint
    (``floatsim.bodies.connector.heave_rigid_link``); a general
    N-DOF rigid link is deferred to Phase 2.
    """

    type: Literal["rigid_link"]
    body_a: Annotated[str, Field(min_length=1)]
    body_b: Annotated[str, Field(min_length=1)]
    penalty_stiffness_factor: Annotated[float, Field(ge=1.0e3, le=1.0e5, allow_inf_nan=False)] = (
        1.0e4
    )
    penalty_damping_factor: NonNegativeFloat = 0.0


Connection = Annotated[LinearSpring | Catenary | RigidLink, Field(discriminator="type")]


# --------------------------------------------------------------------------
# Joints — velocity-level KKT constraints (M9 B1/B2).
# --------------------------------------------------------------------------


class HingeJoint(_Base):
    """Revolute joint: free rotation about ``axis``, all else locked."""

    type: Literal["hinge"]
    body_a: Annotated[str, Field(min_length=1)]
    body_b: Annotated[str, Field(min_length=1)]  # a body name or 'earth'
    attach_a_body: Vec3
    attach_b_body: Vec3
    axis: Vec3


class YawLockedJoint(_Base):
    """The 12-buoy joint: 3 translations + yaw locked, roll/pitch free."""

    type: Literal["yaw_locked"]
    body_a: Annotated[str, Field(min_length=1)]
    body_b: Annotated[str, Field(min_length=1)]
    attach_a_body: Vec3
    attach_b_body: Vec3
    axis: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 1.0])


class RigidJoint(_Base):
    """Weld: all 6 relative DOF locked (3 translations + all 3 rotations) --
    the two bodies move as one rigid body. ``axis`` is unused (kept for
    schema uniformity with the other joints)."""

    type: Literal["rigid"]
    body_a: Annotated[str, Field(min_length=1)]
    body_b: Annotated[str, Field(min_length=1)]  # a body name or 'earth'
    attach_a_body: Vec3
    attach_b_body: Vec3
    axis: Vec3 = Field(default_factory=lambda: [0.0, 0.0, 1.0])


Joint = Annotated[HingeJoint | YawLockedJoint | RigidJoint, Field(discriminator="type")]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


class Output(_Base):
    """HDF5 output configuration."""

    file: Annotated[str, Field(min_length=1)]
    channels: Annotated[list[str], Field(min_length=1)]
    sample_rate: PositiveFloat


# --------------------------------------------------------------------------
# Top-level deck
# --------------------------------------------------------------------------


class Deck(_Base):
    """Full simulation input deck (ARCHITECTURE.md §5)."""

    simulation: Simulation
    environment: Environment
    waves: RegularWave
    bodies: Annotated[list[Body], Field(min_length=1)]
    connections: list[Connection] = Field(default_factory=list)
    shared_hydro_database: HydroDatabaseRef | None = None
    hydrostatic_database: HydroDatabaseRef | None = None
    joints: list[Joint] = Field(default_factory=list)
    output: Output

    @model_validator(mode="after")
    def _validate_shared_hydro_and_joints(self) -> Deck:
        labelled = [b for b in self.bodies if b.hydro_body_label is not None]
        if labelled and self.shared_hydro_database is None:
            raise ValueError(
                "bodies use 'hydro_body_label' but the deck declares no " "'shared_hydro_database'"
            )
        if self.shared_hydro_database is not None and not labelled:
            raise ValueError(
                "'shared_hydro_database' is declared but no body selects a block "
                "via 'hydro_body_label'"
            )
        if labelled:
            labels = [b.hydro_body_label for b in labelled]
            if len(set(labels)) != len(labels):
                raise ValueError(f"duplicate hydro_body_label among bodies: {labels}")
        # Structural (hydro-free) bodies are supported only in the coupled
        # assembly (M10 PR0.5): they need a shared_hydro_database context.
        structural = [b.name for b in self.bodies if b.structural]
        if structural and self.shared_hydro_database is None:
            raise ValueError(
                f"structural (hydro-free) bodies {structural} require a coupled deck "
                "declaring a 'shared_hydro_database'; they are supported only in the "
                "coupled assembly (M10 PR0.5)"
            )
        # A 'hydrostatic_database' (per-body block-diagonal buoyancy C source,
        # M10 PR0.85) is meaningful only for the coupled path.
        if self.hydrostatic_database is not None and self.shared_hydro_database is None:
            raise ValueError(
                "'hydrostatic_database' is declared but there is no "
                "'shared_hydro_database'; the per-body hydrostatic C source applies "
                "only to the coupled assembly (M10 PR0.85)"
            )
        names = {b.name for b in self.bodies} | {"earth"}
        for j in self.joints:
            for endpoint in (j.body_a, j.body_b):
                if endpoint not in names:
                    raise ValueError(f"joint references unknown body {endpoint!r}")
            if j.body_a == j.body_b:
                raise ValueError(f"joint connects body {j.body_a!r} to itself")
        return self


def load_deck(path: str | Path) -> Deck:
    """Read a YAML deck file from disk and return a validated ``Deck``.

    Parameters
    ----------
    path
        Filesystem path to a YAML deck.

    Returns
    -------
    Deck
        A pydantic-validated deck object. Raises ``pydantic.ValidationError``
        if the file content does not conform to the schema.
    """
    raw: Any = yaml.safe_load(Path(path).read_text())
    return Deck.model_validate(raw)
