"""Loader for committed OpenFAST cross-check CSV fixtures (M6 PR1).

The M6 OpenFAST cross-check (see ``docs/milestone-6-plan.md`` v2 Q3)
relies on per-scenario CSV time histories committed under
``tests/fixtures/openfast/oc4_deepcwind/outputs/``. The CSVs are
extracted from OpenFAST `.out` files by
``scripts/extract_openfast_fixtures.py`` and are *not* regenerated in
CI -- the regeneration step requires OpenFAST installed locally on
the contributor's machine.

This module provides the canonical loader that scenario-specific
validation tests (M6 PR2-PR6) call to ingest those fixtures.

Canonical CSV format
--------------------
A normalised, SI-canonical CSV with a single header row. Required
columns:

    time_s   -- seconds, monotonically increasing, dt > 0
    surge_m  -- platform surge (inertial-frame x), metres
    sway_m   -- platform sway (inertial-frame y), metres
    heave_m  -- platform heave (inertial-frame z), metres, +z up
    roll_rad   -- platform roll (ZYX-intrinsic Euler), radians
    pitch_rad  -- platform pitch
    yaw_rad    -- platform yaw

Any number of additional columns (line tensions in N, body-frame
accelerations in m/s^2, etc.) may follow; these are surfaced in
:attr:`OpenFASTHistory.extra_columns` keyed by their CSV header name.

The choice of SI-canonical (radians, metres, Newtons, kilograms) is
deliberate: OpenFAST writes degrees by default, so the unit
conversion happens once in the extraction script and is an auditable
single-source transformation. The loader does not silently re-convert
units; if the JSON sidecar reports a non-canonical unit system the
loader raises ``ValueError`` rather than guessing.

Companion JSON sidecar
----------------------
Every CSV must ship with a ``{stem}.json`` sidecar in the same
directory containing the metadata required for downstream
verification. The schema is the one written by
``scripts/extract_openfast_fixtures.py`` after consuming
``inputs/manifest.json``. Required keys:

    scenario_name      -- one of {"s1_static_eq", "s2_pitch_decay",
                           "s3_rao_sweep", "s4_moored_eq",
                           "s5_drag_decay"}. The CSV stem may carry
                           additional context (e.g. the S3 sweep
                           stems include the wave-period suffix
                           ``s3_rao_sweep_WaveTp_004p0``); the
                           loader does NOT enforce stem-equals-
                           scenario_name.
    openfast_version   -- runtime-detected version string.
    openfast_version_required -- pin from manifest.json (e.g.
                           "v4.1.2"); the loader warns if the
                           runtime version disagrees.
    deck_dir           -- relative path (under tests/fixtures/openfast/
                           oc4_deepcwind/) of the deck the .outb
                           was produced from. Lets a future audit
                           re-run the same case from the same inputs.
    dt_s               -- output sample rate in seconds (positive).
    duration_s         -- total duration in seconds (positive).
    n_samples          -- row count in the CSV (positive integer).
    unit_system        -- must be "SI_canonical" (radians, metres,
                           Newtons). The extraction script is the
                           single point of unit conversion.
    extracted_by       -- the script invocation that produced the file.

Optional keys (``purpose``, ``moordyn_active``, ``sweep_value``,
``r_test_tag_required``, ``extracted_at``, ``channels_canonical``)
flow through unchanged into :attr:`OpenFASTHistory.metadata` for
use by tests.

Why a JSON sidecar rather than a CSV-preamble comment
-----------------------------------------------------
The metadata is structured (lists, nested dicts) and machine-read by
the loader for shape and version validation. CSV-preamble comments
would either need a custom parser or restrict the metadata to flat
key=value pairs. JSON keeps the contract explicit without adding any
new dependency (Python's stdlib ``json`` suffices).

Why SI-canonical and not OpenFAST-native units in the CSV
---------------------------------------------------------
OpenFAST mixes degrees (rotations), metres, kilonewtons (in some
configurations), and tonnes (in others). A CSV that preserves the
native mix would force every test to redo the conversions, with the
risk that one test forgets and reports e.g. pitch period in degrees.
SI-canonical at extraction time means every consumer is talking the
same units, with the conversion code in one place
(``scripts/extract_openfast_fixtures.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

_REQUIRED_DOF_COLUMNS: Final[tuple[str, ...]] = (
    "time_s",
    "surge_m",
    "sway_m",
    "heave_m",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
)

_REQUIRED_METADATA_KEYS: Final[tuple[str, ...]] = (
    "scenario_name",
    "deck_dir",
    "openfast_version",
    "dt_s",
    "duration_s",
    "n_samples",
    "unit_system",
    "extracted_by",
)

_CANONICAL_UNIT_SYSTEM: Final[str] = "SI_canonical"


@dataclass(frozen=True)
class OpenFASTHistory:
    """One scenario's loaded time history plus its sidecar metadata.

    Attributes
    ----------
    t
        Time in seconds, shape ``(N,)``, monotonically increasing.
    xi
        Platform 6-DOF state, shape ``(N, 6)``, SI canonical:
        ``xi[:, 0:3]`` are surge/sway/heave in metres,
        ``xi[:, 3:6]`` are roll/pitch/yaw in radians (ZYX-intrinsic).
    extra_columns
        Mapping from CSV column name to ``(N,)`` array for every
        column beyond the seven required ones (e.g. line tensions).
    metadata
        Parsed JSON sidecar. Includes the required keys plus any
        scenario-specific extras.
    """

    t: NDArray[np.float64]
    xi: NDArray[np.float64]
    extra_columns: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Number of time samples in the history."""
        return int(self.t.size)

    @property
    def dt_s(self) -> float:
        """Mean sample spacing in seconds (averaged over all intervals).

        Equals the JSON sidecar's ``dt_s`` to numerical noise; we
        compute from the time array so the property reflects the data
        rather than the metadata claim.
        """
        if self.t.size < 2:
            raise ValueError("dt_s requires at least 2 samples")
        return float(np.mean(np.diff(self.t)))

    @property
    def duration_s(self) -> float:
        """Total time span ``t[-1] - t[0]`` in seconds."""
        if self.t.size < 2:
            return 0.0
        return float(self.t[-1] - self.t[0])


def load_openfast_history(csv_path: str | Path) -> OpenFASTHistory:
    """Load an OpenFAST scenario CSV (and its JSON sidecar) into a history.

    Parameters
    ----------
    csv_path
        Path to the ``.csv`` file. The companion sidecar
        ``{stem}.json`` must exist alongside.

    Returns
    -------
    OpenFASTHistory
        Validated time history with metadata.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` or the JSON sidecar are missing.
    ValueError
        If the CSV is missing one of the seven required DOF columns,
        the time column is non-monotonic, the data shape is
        inconsistent, the JSON sidecar is missing a required key,
        the ``unit_system`` is not ``"SI_canonical"``, or the
        ``scenario_id`` does not match the CSV stem.
    """
    csv = Path(csv_path)
    if not csv.is_file():
        raise FileNotFoundError(f"OpenFAST CSV not found: {csv}")
    json_path = csv.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(
            f"JSON sidecar not found alongside {csv.name}: expected {json_path.name} "
            "in the same directory. Every committed CSV must ship its metadata."
        )

    metadata = _load_metadata(json_path)
    t, xi, extras = _load_csv_columns(csv)
    _validate_time_column(t, dt_s_metadata=float(metadata["dt_s"]))

    return OpenFASTHistory(t=t, xi=xi, extra_columns=extras, metadata=metadata)


def _load_metadata(json_path: Path) -> dict[str, Any]:
    """Read and validate the JSON sidecar."""
    with open(json_path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{json_path.name} must be a JSON object; got {type(raw).__name__}")
    missing = [k for k in _REQUIRED_METADATA_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"{json_path.name} is missing required keys: {missing}. "
            f"See tests/support/openfast_csv.py module docstring for the schema."
        )
    if raw["unit_system"] != _CANONICAL_UNIT_SYSTEM:
        raise ValueError(
            f"{json_path.name} unit_system={raw['unit_system']!r}; the loader "
            f"requires {_CANONICAL_UNIT_SYSTEM!r}. The extraction script "
            f"(scripts/extract_openfast_fixtures.py) is the single point of "
            f"unit conversion -- if your CSV is in OpenFAST-native units, "
            f"re-run the extraction."
        )
    if not isinstance(raw["dt_s"], int | float) or float(raw["dt_s"]) <= 0.0:
        raise ValueError(f"{json_path.name} dt_s must be a positive number; got {raw['dt_s']!r}")
    if not isinstance(raw["duration_s"], int | float) or float(raw["duration_s"]) <= 0.0:
        raise ValueError(
            f"{json_path.name} duration_s must be a positive number; " f"got {raw['duration_s']!r}"
        )
    if not isinstance(raw["n_samples"], int) or int(raw["n_samples"]) <= 0:
        raise ValueError(
            f"{json_path.name} n_samples must be a positive integer; " f"got {raw['n_samples']!r}"
        )
    if not isinstance(raw["deck_dir"], str) or not raw["deck_dir"]:
        raise ValueError(
            f"{json_path.name} deck_dir must be a non-empty string; " f"got {raw['deck_dir']!r}"
        )
    return raw


def _load_csv_columns(
    csv_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, NDArray[np.float64]]]:
    """Read the CSV, split into (time, xi, extras)."""
    with open(csv_path, encoding="utf-8") as fh:
        header_line = fh.readline().strip()
    if not header_line:
        raise ValueError(f"{csv_path.name} is empty or has no header row")
    headers = [h.strip() for h in header_line.split(",")]
    missing = [c for c in _REQUIRED_DOF_COLUMNS if c not in headers]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {missing}. "
            f"Required: {list(_REQUIRED_DOF_COLUMNS)}; got: {headers}"
        )
    if len(set(headers)) != len(headers):
        dupes = [h for h in headers if headers.count(h) > 1]
        raise ValueError(f"{csv_path.name} has duplicate column names: {sorted(set(dupes))}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)
    if data.shape[1] != len(headers):
        raise ValueError(
            f"{csv_path.name} has {data.shape[1]} data columns but "
            f"{len(headers)} headers; the two must agree."
        )
    if data.shape[0] < 2:
        raise ValueError(f"{csv_path.name} has only {data.shape[0]} data rows; need at least 2.")

    col_index = {h: i for i, h in enumerate(headers)}
    t = data[:, col_index["time_s"]].astype(np.float64, copy=True)
    xi = np.column_stack(
        [
            data[:, col_index["surge_m"]],
            data[:, col_index["sway_m"]],
            data[:, col_index["heave_m"]],
            data[:, col_index["roll_rad"]],
            data[:, col_index["pitch_rad"]],
            data[:, col_index["yaw_rad"]],
        ]
    ).astype(np.float64, copy=False)

    extras: dict[str, NDArray[np.float64]] = {}
    for h in headers:
        if h in _REQUIRED_DOF_COLUMNS:
            continue
        extras[h] = data[:, col_index[h]].astype(np.float64, copy=True)
    return t, xi, extras


def _validate_time_column(t: NDArray[np.float64], *, dt_s_metadata: float) -> None:
    """Reject non-monotonic times or a sample rate inconsistent with the sidecar."""
    if not np.all(np.isfinite(t)):
        raise ValueError("time_s column contains non-finite values")
    diffs = np.diff(t)
    if np.any(diffs <= 0.0):
        raise ValueError(
            "time_s column must be strictly monotonically increasing; "
            f"min(diff) = {float(np.min(diffs)):.3e} s"
        )
    dt_observed = float(np.mean(diffs))
    rel_err = abs(dt_observed - dt_s_metadata) / max(dt_s_metadata, 1.0e-12)
    if rel_err > 1.0e-3:
        raise ValueError(
            f"observed mean dt_s = {dt_observed:.6e} disagrees with sidecar "
            f"dt_s = {dt_s_metadata:.6e} by rel-err {rel_err:.3e}; the JSON "
            f"metadata may be stale relative to the CSV."
        )
