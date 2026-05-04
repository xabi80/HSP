"""Minimal OpenFAST deck parser for M6 cross-check static-residual computation.

Used by :mod:`tests.validation.test_m6_openfast_static_eq` to compute
the net static load FloatSim must apply to its Cummins linearisation
to reproduce the OpenFAST equilibrium offset.

Why a parser instead of comparing equilibria directly
-----------------------------------------------------
The Cummins formulation linearises about the BEM reference position;
FloatSim's static-equilibrium solver returns ``xi=0`` for any deck
because equilibrium IS the linearisation point (see
``docs/openfast-cross-check-conventions.md`` Item 15). OpenFAST's
nonlinear solver settles into a non-zero offset whenever the deck's
total mass and displaced volume don't exactly balance at the BEM
reference; that offset is a deck-bookkeeping artifact, not physics
disagreement.

To turn this into a real cross-check (rather than a "we can't
reproduce the offset" report), PR2 computes the net residual force
from the OpenFAST input files, applies it to FloatSim's solver as
``F_external``, and asserts the resulting displacement matches
OpenFAST's last-30-s mean. Match validates: parsing of OpenFAST
inputs + deck-mass aggregation + buoyancy calculation + Cummins
linearisation + gravity decomposition (Item 5).

Scope
-----
This is **not** a general-purpose OpenFAST input parser. It reads the
specific named scalars the residual computation requires, and uses
OC4-specific literature values for quantities that require deeper
geometry (water ballast inside columns). For non-OC4 decks the
caller must override the literature constants.

Parsed scalars (deck-specific, no literature):

- ``.fst`` driver: ``Gravity``, ``WtrDens``
- ``HydroDyn.dat``: ``PtfmVol0`` (displaced volume at undisplaced
  position), ``PtfmCOBxt``, ``PtfmCOByt``
- ``ElastoDyn.dat``: ``PtfmMass``, ``PtfmCMzt`` (platform CoG depth),
  ``HubMass``, ``NacMass``, ``YawBrMass``, ``NacCMxn``, ``NacCMyn``,
  ``NacCMzn``, ``TowerHt``, ``TowerBsHt``
- Tower distributed mass: integrated from the ``TwrFile`` station
  table (``HtFract``, ``TMassDen`` columns).
- Blade distributed mass: integrated from one ``BldFile`` station
  table (assumes all 3 blades use the same file, which is true for
  the OC4-DeepCwind reference).

Literature constants (cited):

- ``OC4_PLATFORM_TOTAL_MASS_KG`` = 1.3473e7 (Robertson 2014 Table 3-1,
  NREL/TP-5000-60601). Total platform mass *including* fixed ballast
  water; supersedes the parsed ``PtfmMass`` (which is steel/structure
  only). Deeper-than-5-scalars parsing of ``HydroDyn.dat`` ``FillGroups``
  + member geometry would reproduce this from inputs but adds
  ~300 lines of OC4-specific geometry handling; the literature
  constant is the cleaner trade-off.

Returned ``F_residual`` 6-vector (inertial frame, at the body's
reference point z=0 = SWL):

- ``F_residual[2]``  = ``rho * V0 * g - m_total * g`` (net vertical;
  positive = net upward = body pushed up from BEM reference).
- ``F_residual[3]``  = roll moment from horizontal CoG/CoB offsets.
- ``F_residual[4]``  = pitch moment, dominated for OC4 by the
  off-axis NacCMxn (downwind nacelle CoG).
- ``F_residual[0,1,5]`` = surge / sway / yaw, all zero for any deck
  with axisymmetric mass distribution + on-axis CoB. OC4 satisfies
  both. Non-zero values would indicate a bookkeeping error.

References
----------
- Robertson, A. et al., 2014. *Definition of the Semisubmersible
  Floating System for Phase II of OC4*, NREL/TP-5000-60601.
- Jonkman, J. et al., 2009. *Definition of a 5-MW Reference Wind
  Turbine for Offshore System Development*, NREL/TP-500-38060.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

# Robertson 2014 Table 3-1: total platform mass including fixed
# ballast (water-filled compartments in the offset columns + center
# column). The OpenFAST `PtfmMass` parameter (3.852e6 kg) covers
# steel/structure only; the difference is the ballast water which
# OpenFAST applies via HydroDyn `FillGroups` + member geometry.
OC4_PLATFORM_TOTAL_MASS_KG: Final[float] = 1.3473e7

# NREL 5-MW reference values (Jonkman 2009 Table 6-1) for the values
# we don't read from the deck. Used as fallbacks if the per-deck
# integration fails (e.g. TwrFile or BldFile unparseable).
_NREL5MW_TOWER_MASS_KG_FALLBACK: Final[float] = 2.4990e5  # 249.9 t
_NREL5MW_BLADE_MASS_KG_FALLBACK: Final[float] = 1.7740e4  # 17.74 t per blade


@dataclass(frozen=True)
class DeckResidual:
    """Result of :func:`compute_openfast_deck_residual`.

    Attributes
    ----------
    F_residual
        Length-6 net static load at the body reference point. ``F[2]``
        is the dominant heave residual; ``F[0:2]`` and ``F[5]`` are
        zero by symmetry for axisymmetric decks (OC4 satisfies);
        ``F[3:5]`` carry small moments from off-axis CoG (e.g. the
        nacelle's downwind offset).
    m_total_kg
        Sum of platform (incl. ballast), tower, hub, nacelle, yaw
        bearing, and blade masses.
    cog_total_z_m
        Combined CoG vertical coordinate (relative to the BEM
        reference at SWL).
    buoyancy_n
        ``rho * V0 * g`` (positive scalar, the magnitude of the
        upward buoyancy at the BEM reference).
    weight_n
        ``m_total * g`` (positive scalar, magnitude of the downward
        weight).
    components
        Per-element mass breakdown for diagnostic logging.
    """

    F_residual: NDArray[np.float64]
    m_total_kg: float
    cog_total_z_m: float
    buoyancy_n: float
    weight_n: float
    components: dict[str, float]


# ---------------------------------------------------------------------------
# Lightweight key=value scanner for OpenFAST .dat / .fst input files
# ---------------------------------------------------------------------------


_VALUE_LINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r'^\s*("[^"]*"|\S+)\s+(\S+)\s*'  # value, then KEY in some position
)


def _scan_named_scalar(path: Path, key: str) -> str:
    """Return the value associated with ``KEY`` in an OpenFAST input file.

    OpenFAST .dat / .fst files use a value-then-name layout, e.g.::

        9.80665                Gravity     - Gravitational acceleration ...
        1025.0                 WtrDens     - Water density (kg/m^3)
        "tower.dat"            TwrFile     - ...

    This scans line by line, looking for the specified ``key`` as the
    second whitespace-separated token. Returns the raw string value
    (the first token, with surrounding quotes preserved). Caller is
    responsible for type conversion.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith(("---", "==", "//", "#")):
                continue
            match = _VALUE_LINE_PATTERN.match(line)
            if match is None:
                continue
            value, name = match.group(1), match.group(2)
            if name == key:
                return value.strip('"')
    raise ValueError(f"key {key!r} not found in {path}")


def _scan_named_float(path: Path, key: str) -> float:
    """Like :func:`_scan_named_scalar`, with float conversion."""
    raw = _scan_named_scalar(path, key)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"key {key!r} value {raw!r} in {path} is not a float") from exc


def _scan_named_path(path: Path, key: str) -> Path:
    """Like :func:`_scan_named_scalar`, returning a path resolved next to ``path``."""
    raw = _scan_named_scalar(path, key)
    return path.parent / raw


# ---------------------------------------------------------------------------
# Distributed-mass integration helpers (tower + blade)
# ---------------------------------------------------------------------------


def _integrate_distributed_mass(
    file_path: Path,
    *,
    span_m: float,
    htfract_col: int = 0,
    tmassden_col: int = 1,
) -> tuple[float, float]:
    """Integrate ``TMassDen``-style station table -> ``(mass_kg, cog_relative)``.

    Parameters
    ----------
    file_path
        Path to the ``TwrFile`` or ``BldFile``.
    span_m
        Physical length corresponding to ``HtFract = 1.0``. For tower:
        ``TowerHt - TowerBsHt``; for blade: ``TipRad - HubRad``.
    htfract_col, tmassden_col
        Column indices of the fractional-height and mass-density
        columns within the data block. Default 0/1 covers the tower
        table layout; blade tables may have a different layout (the
        caller passes the right indices).

    Returns
    -------
    (mass_kg, cog_relative)
        ``mass_kg`` is the trapezoidally-integrated total mass over
        the span; ``cog_relative`` is the mass-weighted mean of
        ``HtFract`` (i.e. the centroid as a fraction of ``span_m``,
        in [0, 1]).
    """
    htfract: list[float] = []
    tmassden: list[float] = []
    in_data = False
    with open(file_path, encoding="utf-8") as fh:
        for line in fh:
            tokens = line.split()
            if not tokens:
                continue
            # First all-numeric row begins the data block; subsequent
            # all-numeric rows continue it; anything else (mode-shape
            # block, etc.) ends it.
            try:
                values = [float(t) for t in tokens]
            except ValueError:
                if in_data:
                    break
                continue
            if len(values) <= max(htfract_col, tmassden_col):
                if in_data:
                    break
                continue
            in_data = True
            htfract.append(values[htfract_col])
            tmassden.append(values[tmassden_col])
    if len(htfract) < 2:
        raise ValueError(
            f"{file_path.name}: distributed-mass table has {len(htfract)} stations; "
            "need at least 2 for trapezoidal integration."
        )
    h = np.asarray(htfract, dtype=np.float64) * span_m
    rho = np.asarray(tmassden, dtype=np.float64)
    mass = float(np.trapezoid(rho, h))
    if mass <= 0.0:
        raise ValueError(f"{file_path.name}: integrated mass {mass} not positive")
    centroid = float(np.trapezoid(rho * h, h) / mass)
    return mass, centroid / span_m


# ---------------------------------------------------------------------------
# Top-level residual computation
# ---------------------------------------------------------------------------


def compute_openfast_deck_residual(
    deck_dir: Path,
    *,
    platform_total_mass_kg: float = OC4_PLATFORM_TOTAL_MASS_KG,
) -> DeckResidual:
    """Net static-residual 6-vector for an OpenFAST deck at FloatSim's reference.

    Parameters
    ----------
    deck_dir
        Path to the per-scenario directory holding the ``.fst`` driver
        plus its referenced submodule files.
    platform_total_mass_kg
        Total platform mass *including any fixed ballast water*. Default
        is the Robertson 2014 OC4 reference value (1.3473e7 kg). Override
        for non-OC4 decks. The OpenFAST ``PtfmMass`` parameter is
        steel/structure only and does NOT include the column-ballast
        water; using ``PtfmMass`` directly here would underestimate the
        weight by ~10e6 kg for OC4.

    Returns
    -------
    DeckResidual
        See dataclass docstring. ``F_residual`` is in inertial-frame
        Newtons / Newton-metres at the BEM reference origin.
    """
    fst_files = sorted(deck_dir.glob("*.fst"))
    if len(fst_files) != 1:
        raise ValueError(f"{deck_dir}: expected exactly one .fst driver; found {len(fst_files)}")
    fst = fst_files[0]
    g = _scan_named_float(fst, "Gravity")
    rho = _scan_named_float(fst, "WtrDens")

    # ElastoDyn provides platform & RNA mass details.
    elastodyn = _scan_named_path(fst, "EDFile")
    ed_ptfm_cmzt = _scan_named_float(elastodyn, "PtfmCMzt")
    ed_hub_mass = _scan_named_float(elastodyn, "HubMass")
    ed_nac_mass = _scan_named_float(elastodyn, "NacMass")
    ed_yawbr_mass = _scan_named_float(elastodyn, "YawBrMass")
    ed_nac_cmxn = _scan_named_float(elastodyn, "NacCMxn")
    ed_nac_cmyn = _scan_named_float(elastodyn, "NacCMyn")
    ed_nac_cmzn = _scan_named_float(elastodyn, "NacCMzn")
    ed_tower_ht = _scan_named_float(elastodyn, "TowerHt")
    ed_tower_bs_ht = _scan_named_float(elastodyn, "TowerBsHt")

    # Tower distributed mass via the TwrFile station table.
    twr_span = ed_tower_ht - ed_tower_bs_ht
    try:
        twr_file = _scan_named_path(elastodyn, "TwrFile")
        tower_mass, tower_centroid_frac = _integrate_distributed_mass(twr_file, span_m=twr_span)
    except (ValueError, FileNotFoundError):
        tower_mass = _NREL5MW_TOWER_MASS_KG_FALLBACK
        tower_centroid_frac = 0.5  # rough; only matters for the moment term
    tower_cog_z = ed_tower_bs_ht + tower_centroid_frac * twr_span

    # Blade mass: assume all 3 blades use BldFile(1). Blade z is
    # secondary for the residual (blades sit at the rotor, fairly far
    # off-axis only when the rotor isn't at the tower-top).
    try:
        bld_file = _scan_named_path(elastodyn, "BldFile(1)")
        # Blade table layout (6 cols): BlFract, PitchAxis, StrcTwst,
        # BMassDen, FlpStff, EdgStff -- BMassDen is column 3, not 1.
        # AdjBlMs is a calibration factor applied to the mass density
        # by ElastoDyn at runtime.
        try:
            adj_bl_ms = _scan_named_float(bld_file, "AdjBlMs")
        except ValueError:
            adj_bl_ms = 1.0
        blade_mass_each_raw, _ = _integrate_distributed_mass(
            bld_file,
            span_m=63.0,  # NREL 5-MW blade length (Jonkman 2009 Table 6-1)
            htfract_col=0,
            tmassden_col=3,
        )
        blade_mass_each = adj_bl_ms * blade_mass_each_raw
    except (ValueError, FileNotFoundError):
        blade_mass_each = _NREL5MW_BLADE_MASS_KG_FALLBACK
    blade_total = 3.0 * blade_mass_each

    # HydroDyn provides the buoyancy reference.
    hd_file = _scan_named_path(fst, "HydroFile")
    hd_vol0 = _scan_named_float(hd_file, "PtfmVol0")
    hd_cobxt = _scan_named_float(hd_file, "PtfmCOBxt")
    hd_cobyt = _scan_named_float(hd_file, "PtfmCOByt")

    # Tower-top in inertial coords (RNA components are positioned
    # relative to it).
    tower_top_z = ed_tower_ht
    # Nacelle CoG in inertial coords (NacCMxn is downwind, +x; NacCMzn
    # is above tower-top).
    nac_cog_x = ed_nac_cmxn
    nac_cog_y = ed_nac_cmyn
    nac_cog_z = tower_top_z + ed_nac_cmzn
    # Hub: at the rotor apex which sits at the tower-top + overhang;
    # we approximate by the nacelle CoG horizontal position with hub
    # at tower-top vertical (good to within a few metres given the
    # small magnitudes involved).
    hub_cog_x = nac_cog_x
    hub_cog_y = nac_cog_y
    hub_cog_z = tower_top_z
    # Blade CoG: assume the 3-blade rotor's combined CoG sits near
    # the rotor centre (small offsets cancel by 3-fold symmetry).
    blade_cog_x = nac_cog_x  # ~rotor apex x; close enough.
    blade_cog_y = 0.0
    blade_cog_z = tower_top_z

    # Combined CoG (mass-weighted).
    masses = {
        "platform_with_ballast": platform_total_mass_kg,
        "tower": tower_mass,
        "hub": ed_hub_mass,
        "nacelle": ed_nac_mass,
        "yaw_bearing": ed_yawbr_mass,
        "blades_total": blade_total,
    }
    m_total = sum(masses.values())
    cogs_z = {
        "platform_with_ballast": ed_ptfm_cmzt,
        "tower": tower_cog_z,
        "hub": hub_cog_z,
        "nacelle": nac_cog_z,
        "yaw_bearing": tower_top_z,
        "blades_total": blade_cog_z,
    }
    cogs_x = {
        "platform_with_ballast": 0.0,
        "tower": 0.0,
        "hub": hub_cog_x,
        "nacelle": nac_cog_x,
        "yaw_bearing": 0.0,
        "blades_total": blade_cog_x,
    }
    cogs_y = {
        "platform_with_ballast": 0.0,
        "tower": 0.0,
        "hub": hub_cog_y,
        "nacelle": nac_cog_y,
        "yaw_bearing": 0.0,
        "blades_total": blade_cog_y,
    }
    cog_total_x = sum(masses[k] * cogs_x[k] for k in masses) / m_total
    cog_total_y = sum(masses[k] * cogs_y[k] for k in masses) / m_total
    cog_total_z = sum(masses[k] * cogs_z[k] for k in masses) / m_total

    # Net forces and moments at the BEM reference point (origin).
    weight_n = m_total * g
    buoyancy_n = rho * hd_vol0 * g

    # Vertical residual: + upward = +z direction = +F[2].
    F = np.zeros(6, dtype=np.float64)
    F[2] = buoyancy_n - weight_n

    # Moments from horizontal CoG/CoB offsets:
    # Weight is at (cog_total_x, cog_total_y, cog_total_z) with force
    # (0, 0, -weight_n). Moment = r x F:
    #   M_x =  r_y · F_z - r_z · F_y =  cog_total_y · (-weight_n) - 0
    #   M_y =  r_z · F_x - r_x · F_z = 0 - cog_total_x · (-weight_n)
    #   M_z = 0
    # Buoyancy at (cobxt, cobyt, z_B) with force (0, 0, +buoyancy_n);
    # we only have the horizontal CoB, the z component cancels with
    # the weight's moment about z=0 (both lever arms on the x/y axis):
    #   M_x_B =  cobyt · buoyancy_n
    #   M_y_B = -cobxt · buoyancy_n
    F[3] = -cog_total_y * weight_n + hd_cobyt * buoyancy_n
    F[4] = +cog_total_x * weight_n - hd_cobxt * buoyancy_n
    # F[0], F[1], F[5] are zero for vertically-aligned conservative loads.

    return DeckResidual(
        F_residual=F,
        m_total_kg=m_total,
        cog_total_z_m=cog_total_z,
        buoyancy_n=buoyancy_n,
        weight_n=weight_n,
        components=masses,
    )
