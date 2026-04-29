"""OrcaFlex VesselType YAML reader — ARCHITECTURE.md §8 M1.5.

An OrcaFlex VesselType YAML is the human-readable text export produced when
an OrcaWave ``.owr`` diffraction result is imported into OrcaFlex and saved
as a model YAML. The reader consumes that export and returns a validated
:class:`floatsim.hydro.database.HydroDatabase`.

Supported fixture conventions (rejected otherwise with a descriptive error):

    UnitsSystem                   SI      (mass in tonnes, force in kN)
    WavesReferredToBy             frequency (rad/s)
    RAOPhaseConvention            leads
    RAOPhaseUnitsConvention       degrees or radians

Unit handling
-------------
OrcaFlex "SI" uses tonnes for mass and kN for force. Every mass/force-related
block (A(omega), B(omega), A_inf, C, LoadRAO amplitudes) therefore scales
uniformly by 1000 into pure SI (kg, kg*m, kg*m^2, N, N*m, ...). There is no
cross-block unit change to reason about.

RAO complex convention
----------------------
For ``RAOPhaseConvention: leads``, the file stores ``amp`` and ``phase_leads``
per DOF. FloatSim encodes the complex excitation transfer function as::

    X(omega) = amp_SI * exp(1j * phase_leads_rad)

so that the physical force per unit wave amplitude is
``F(t) = Re[X * A_wave * exp(1j * omega * t)]``. This matches the
time-harmonic convention used by the Cummins assembly in
:mod:`floatsim.hydro.radiation`.

Scope
-----
M1.5 handles a single vessel type with a single draught, and treats the
``LoadRAOs`` block as the first-order wave excitation RAO stored in
``HydroDatabase.RAO``. ``DisplacementRAOs``, ``WaveDrift``, ``SumFrequencyQTFs``
and ``SeaStateRAOs`` are not consumed — they are out-of-scope for Phase 1
(QTFs / irregular seas land in Phase 2). Multi-body and multi-draught
extensions wait for Milestone 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from numpy.typing import NDArray

from floatsim.hydro.database import HydroDatabase

_OFX_SI_SCALE: Final[float] = 1000.0

_REQUIRED_UNITS_SYSTEM: Final[str] = "SI"
_REQUIRED_WAVES_REFERRED_TO_BY: Final[str] = "frequency (rad/s)"
_SUPPORTED_PHASE_CONVENTION: Final[str] = "leads"
_SUPPORTED_PHASE_UNITS: Final[tuple[str, ...]] = ("degrees", "radians")

_DOF_ORDER: Final[tuple[str, ...]] = (
    "Surge",
    "Sway",
    "Heave",
    "Roll",
    "Pitch",
    "Yaw",
)

# Column layout of each row in a LoadRAOs / DisplacementRAOs block:
#   [freq, Surge_amp, Surge_phase, Sway_amp, Sway_phase, Heave_amp,
#    Heave_phase, Roll_amp, Roll_phase, Pitch_amp, Pitch_phase, Yaw_amp,
#    Yaw_phase]
_RAO_ROW_LENGTH: Final[int] = 13

# Hydrostatic stiffness is stored as a 3x3 block spanning (heave, roll, pitch).
_HYDROSTAT_DOF_INDICES: Final[list[int]] = [2, 3, 4]

_FREQ_MATCH_TOL: Final[float] = 1.0e-9


def read_orcaflex_vessel_yaml(
    path: str | Path,
    *,
    vessel_type_index: int = 0,
    draught_name: str | None = None,
) -> HydroDatabase:
    """Parse an OrcaFlex VesselType YAML export into a HydroDatabase.

    Parameters
    ----------
    path
        Path to the OrcaFlex model YAML (``.yml``) produced via
        ``Model → Save As → YAML`` after importing an ``.owr`` via
        ``Import → OrcaWave Results``.
    vessel_type_index
        Index into the ``VesselTypes`` list. Defaults to the first vessel type.
    draught_name
        If given, selects the named draught under the chosen vessel type.
        Defaults to the first draught.

    Returns
    -------
    HydroDatabase
        Validated BEM database in pure SI units. Shape/symmetry invariants
        are enforced by :class:`HydroDatabase.__post_init__`.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"OrcaFlex VesselType YAML not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    _validate_units_system(data)

    vessel_types = data.get("VesselTypes") or []
    if not vessel_types:
        raise ValueError("YAML does not contain a non-empty VesselTypes block")
    if vessel_type_index >= len(vessel_types):
        raise ValueError(
            f"vessel_type_index={vessel_type_index} out of range "
            f"(file contains {len(vessel_types)} vessel types)"
        )
    vt: dict[str, Any] = vessel_types[vessel_type_index]

    _validate_vessel_type_conventions(vt)

    draughts = vt.get("Draughts") or []
    if not draughts:
        raise ValueError(f"VesselType '{vt.get('Name', '?')}' has no Draughts block")
    draught = _pick_draught(draughts, draught_name)

    omega = _extract_omega_from_load_raos(draught)
    A_inf, A_freq, B_freq = _extract_added_mass_and_damping(draught, omega)
    C = _extract_hydrostatic_stiffness(draught)
    heading_deg, RAO = _extract_load_raos(draught, vt, omega)

    ref = draught.get("ReferenceOrigin") or [0.0, 0.0, 0.0]
    reference_point = np.asarray(ref, dtype=np.float64)

    metadata: dict[str, str] = {
        "source": "orcaflex-vessel-yaml",
        "file": p.name,
        "vessel_type": str(vt.get("Name", "?")),
        "draught": str(draught.get("Name", "?")),
    }

    return HydroDatabase(
        omega=omega,
        heading_deg=heading_deg,
        A=A_freq,
        B=B_freq,
        A_inf=A_inf,
        C=C,
        RAO=RAO,
        reference_point=reference_point,
        # OrcaFlex's VesselType bundles the body's mass distribution
        # into the same record as the BEM output, so its
        # HydrostaticStiffness block is the FULL linearised restoring
        # (buoyancy + waterplane + gravity m*g*z_G already combined).
        # Empirically verified against `platform_small.yml`: the OC4
        # fixture reports pitch C_55 ~ 9.97e8 N*m/rad, close to the
        # Robertson 2014 Table 3-3 full value 1.078e9; the
        # buoyancy-only contribution would be -7e8 (negative) for OC4.
        # See test_orcaflex_roundtrip_oc4 + the gravity-bug post-mortem.
        # The raw-BEM readers (WAMIT, Capytaine) declare "buoyancy_only".
        C_source="full",
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_units_system(data: dict[str, Any]) -> None:
    """UnitsSystem can live under General (model YAML) or at the top level."""
    units = None
    general = data.get("General")
    if isinstance(general, dict):
        units = general.get("UnitsSystem")
    if units is None:
        units = data.get("UnitsSystem")
    if units != _REQUIRED_UNITS_SYSTEM:
        raise ValueError(
            f"UnitsSystem {units!r} not supported by orcaflex_vessel_yaml reader; "
            f"only {_REQUIRED_UNITS_SYSTEM!r} is recognized."
        )


def _validate_vessel_type_conventions(vt: dict[str, Any]) -> None:
    wrb = vt.get("WavesReferredToBy")
    if wrb != _REQUIRED_WAVES_REFERRED_TO_BY:
        raise ValueError(
            f"WavesReferredToBy {wrb!r} not supported by orcaflex_vessel_yaml reader; "
            f"only {_REQUIRED_WAVES_REFERRED_TO_BY!r} is recognized."
        )
    phase_conv = vt.get("RAOPhaseConvention")
    if phase_conv != _SUPPORTED_PHASE_CONVENTION:
        raise ValueError(
            f"RAOPhaseConvention {phase_conv!r} not supported; "
            f"only {_SUPPORTED_PHASE_CONVENTION!r} is recognized."
        )
    phase_units = vt.get("RAOPhaseUnitsConvention")
    if phase_units not in _SUPPORTED_PHASE_UNITS:
        raise ValueError(
            f"RAOPhaseUnitsConvention {phase_units!r} not supported; must be "
            f"one of {_SUPPORTED_PHASE_UNITS}."
        )


def _pick_draught(draughts: list[dict[str, Any]], name: str | None) -> dict[str, Any]:
    if name is None:
        return draughts[0]
    for d in draughts:
        if d.get("Name") == name:
            return d
    available = [d.get("Name") for d in draughts]
    raise ValueError(f"Draught {name!r} not found; available: {available}")


def _find_value_by_key_prefix(block: dict[str, Any], prefix: str) -> Any:
    """Return the value whose key begins with ``prefix``.

    OrcaFlex's YAML uses comma-joined composite keys (e.g. ``"AddedMassMatrixX,
    AddedMassMatrixY, ..."``) that yaml parses as a single string. Matching on
    prefix is the cleanest way to find the matrix value without hard-coding
    the exact comma-joined key.
    """
    for k, v in block.items():
        if isinstance(k, str) and k.startswith(prefix):
            return v
    raise ValueError(f"Key with prefix {prefix!r} not found in block")


def _as_6x6(value: Any, label: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (6, 6):
        raise ValueError(f"{label} must be 6x6; got shape {arr.shape}")
    return arr


def _extract_omega_from_load_raos(draught: dict[str, Any]) -> NDArray[np.float64]:
    load_raos = draught.get("LoadRAOs") or {}
    raos = load_raos.get("RAOs") or []
    if not raos:
        raise ValueError("LoadRAOs.RAOs block missing or empty")
    rows = _find_value_by_key_prefix(raos[0], "RAOPeriodOrFrequency")
    if not isinstance(rows, list) or not rows:
        raise ValueError("RAOPeriodOrFrequency data rows missing")
    freqs = sorted({float(row[0]) for row in rows})
    return np.asarray(freqs, dtype=np.float64)


def _extract_added_mass_and_damping(
    draught: dict[str, Any],
    omega: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    amd_blocks = draught.get("FrequencyDependentAddedMassAndDamping") or []
    if not amd_blocks:
        raise ValueError("FrequencyDependentAddedMassAndDamping block missing")

    A_inf: NDArray[np.float64] | None = None
    A_map: dict[float, NDArray[np.float64]] = {}
    B_map: dict[float, NDArray[np.float64]] = {}

    for block in amd_blocks:
        period_or_freq = block.get("AMDPeriodOrFrequency")
        added_mass = (
            _as_6x6(_find_value_by_key_prefix(block, "AddedMassMatrix"), "AddedMass")
            * _OFX_SI_SCALE
        )
        is_infinity = period_or_freq == "Infinity" or (
            isinstance(period_or_freq, float) and not np.isfinite(period_or_freq)
        )
        if is_infinity:
            A_inf = added_mass
            # Damping at infinity is not meaningful and not needed.
            continue
        damping = _as_6x6(_find_value_by_key_prefix(block, "Damping"), "Damping") * _OFX_SI_SCALE
        key = float(period_or_freq)
        A_map[key] = added_mass
        B_map[key] = damping

    if A_inf is None:
        raise ValueError("No AMDPeriodOrFrequency: Infinity entry found — cannot populate A_inf")

    n_w = int(omega.size)
    A_freq = np.zeros((6, 6, n_w), dtype=np.float64)
    B_freq = np.zeros((6, 6, n_w), dtype=np.float64)
    for k, w in enumerate(omega):
        w_f = float(w)
        matched = min(A_map.keys(), key=lambda x: abs(x - w_f))
        if abs(matched - w_f) > _FREQ_MATCH_TOL:
            raise ValueError(
                f"omega={w_f} rad/s not present in FrequencyDependentAddedMassAndDamping "
                f"(nearest available = {matched})"
            )
        A_freq[:, :, k] = A_map[matched]
        B_freq[:, :, k] = B_map[matched]

    return A_inf, A_freq, B_freq


def _extract_hydrostatic_stiffness(
    draught: dict[str, Any],
) -> NDArray[np.float64]:
    try:
        raw = _find_value_by_key_prefix(draught, "HydrostaticStiffness")
    except ValueError as exc:
        raise ValueError("HydrostaticStiffness{z,Rx,Ry} block not found") from exc
    mat_3x3 = np.asarray(raw, dtype=np.float64)
    if mat_3x3.shape != (3, 3):
        raise ValueError(f"HydrostaticStiffness block must be 3x3; got {mat_3x3.shape}")
    C = np.zeros((6, 6), dtype=np.float64)
    C[np.ix_(_HYDROSTAT_DOF_INDICES, _HYDROSTAT_DOF_INDICES)] = mat_3x3 * _OFX_SI_SCALE
    return C


def _extract_load_raos(
    draught: dict[str, Any],
    vt: dict[str, Any],
    omega: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    load_raos = draught.get("LoadRAOs") or {}
    raos = load_raos.get("RAOs") or []
    if not raos:
        raise ValueError("LoadRAOs.RAOs block missing or empty")

    phase_units = vt["RAOPhaseUnitsConvention"]
    deg_to_rad = np.pi / 180.0 if phase_units == "degrees" else 1.0

    n_w = int(omega.size)
    n_h = len(raos)
    heading_deg = np.zeros(n_h, dtype=np.float64)
    RAO = np.zeros((6, n_w, n_h), dtype=np.complex128)
    filled = np.zeros((6, n_w, n_h), dtype=bool)

    for h_idx, direction_block in enumerate(raos):
        heading_deg[h_idx] = float(direction_block["RAODirection"])
        rows = _find_value_by_key_prefix(direction_block, "RAOPeriodOrFrequency")
        for row in rows:
            if len(row) != _RAO_ROW_LENGTH:
                raise ValueError(
                    f"LoadRAO row must have {_RAO_ROW_LENGTH} entries (freq + 6 amp/phase pairs); "
                    f"got {len(row)}"
                )
            freq = float(row[0])
            w_idx = int(np.argmin(np.abs(omega - freq)))
            if abs(float(omega[w_idx]) - freq) > _FREQ_MATCH_TOL:
                raise ValueError(f"LoadRAO frequency {freq} not present in omega grid")
            for dof_idx in range(6):
                amp = float(row[1 + 2 * dof_idx]) * _OFX_SI_SCALE
                phase_rad = float(row[2 + 2 * dof_idx]) * deg_to_rad
                RAO[dof_idx, w_idx, h_idx] = amp * np.exp(1j * phase_rad)
                filled[dof_idx, w_idx, h_idx] = True

    if not np.all(filled):
        missing = np.argwhere(~filled)
        raise ValueError(
            f"LoadRAO grid incompletely populated; missing entries at "
            f"(dof, omega_idx, heading_idx): {missing.tolist()}"
        )

    return heading_deg, RAO
