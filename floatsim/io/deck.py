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
from pydantic import BaseModel, ConfigDict, Field

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
    """Slender-cylinder drag element between two body-frame nodes."""

    type: Literal["morison_member"]
    node_a: Vec3
    node_b: Vec3
    diameter: PositiveFloat
    Cd: NonNegativeFloat
    Ca: NonNegativeFloat = 0.0


DragElement = Annotated[MorisonMember, Field(discriminator="type")]


class InitialConditions(_Base):
    """Initial 6-DOF position (from equilibrium) and velocity.

    Component order: surge, sway, heave, roll, pitch, yaw.
    """

    position: Vec6 = Field(default_factory=lambda: [0.0] * 6)
    velocity: Vec6 = Field(default_factory=lambda: [0.0] * 6)


class Body(_Base):
    """One rigid floating body."""

    name: Annotated[str, Field(min_length=1)]
    reference_point: Vec3
    mass: PositiveFloat
    inertia: Inertia
    hydro_database: HydroDatabaseRef
    drag_elements: list[DragElement] = Field(default_factory=list)
    initial_conditions: InitialConditions = Field(default_factory=InitialConditions)


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


Connection = Annotated[LinearSpring | Catenary, Field(discriminator="type")]


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
    output: Output


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
