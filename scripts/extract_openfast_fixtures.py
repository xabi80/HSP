"""Extract canonical CSV time histories from OpenFAST output for M6.

This script is the **single point of unit conversion** between
OpenFAST's native output format (`.outb` binary, with degrees for
rotations and a mix of metres / kilonewtons / tonnes for forces) and
FloatSim's canonical SI fixture format (metres / radians / Newtons /
kilograms; see ``tests/support/openfast_csv.py`` module docstring for
the full schema). It is **not** run in CI -- it requires OpenFAST and
``openfast-toolbox`` installed locally on the contributor's machine.
The committed CSV+JSON pairs are the artifacts; this script regenerates
them on demand.

Locked spec per ``docs/milestone-6-plan.md`` v2 Q3:

- Reference deck: OpenFAST/r-test ``5MW_OC4Semi_Linear`` vendored under
  ``tests/fixtures/openfast/oc4_deepcwind/`` per the layout described
  in that directory's ``README.md``.
- Five scenarios in increasing-complexity order plus a 14-period S3
  RAO sweep, totalling 18 manifest entries.
- Wind turbine disabled (``CompElast=1`` for static-equilibrium
  scenarios so ElastoDyn applies gravity; ``CompAero=CompInflow=
  CompServo=0`` everywhere).

Manifest-driven configuration
-----------------------------
The scenario list, output channels, deck overrides, and OpenFAST/r-test
version pins are all in
``tests/fixtures/openfast/oc4_deepcwind/inputs/manifest.json``. This
script iterates over ``scenarios[]``, locates each scenario's deck
directory, and either runs OpenFAST or reads an existing ``.outb``,
then converts to the canonical CSV+JSON format.

Output channel names follow the conventions documented in
``docs/openfast-cross-check-conventions.md`` Item 11:

- Platform DOFs: ``PtfmSurge``, ``PtfmHeave``, ``PtfmRoll``, etc.,
  without module prefix.
- MoorDyn tensions: ``FairTen{1,2,3}`` and ``AnchTen{1,2,3}``,
  capitalised, no underscore.

Channel access uses ``output.info["attribute_names"]`` (per Xabier's
M6-PR1.1 lock); the alternative ``output.channels`` attribute is not
guaranteed to exist on the ``openfast_toolbox.io.FASTOutputFile``
return value across all versions.

Modes
-----
- ``--mode run``: invoke the OpenFAST executable on each scenario's
  ``.fst``, then read the produced ``.outb``, convert, and write the
  CSV+JSON. Requires OpenFAST on PATH.
- ``--mode read-only``: skip the OpenFAST invocation and just read the
  ``.outb`` files that already sit next to each ``.fst``. Used when
  someone else has already run the simulations (e.g. on a machine where
  the live OpenFAST run was owned by another contributor).

Both modes require ``openfast-toolbox`` to read the binary outputs.

Usage
-----
::

    pip install openfast-toolbox  # adds openfast_toolbox.io.FASTOutputFile
    python scripts/extract_openfast_fixtures.py --mode read-only --scenario all
    python scripts/extract_openfast_fixtures.py --mode read-only --scenario s1_static_eq
    python scripts/extract_openfast_fixtures.py --mode run --scenario s5_drag_decay

Each scenario writes
``tests/fixtures/openfast/oc4_deepcwind/inputs/{scenario}/{scenario}.csv``
plus ``{scenario}.json`` with the full metadata required by
:func:`tests.support.openfast_csv.load_openfast_history`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

# ---------------------------------------------------------------------------
# Repo-relative paths.
# ---------------------------------------------------------------------------

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_FIXTURE_ROOT: Final[Path] = _REPO_ROOT / "tests" / "fixtures" / "openfast" / "oc4_deepcwind"
_INPUTS_DIR: Final[Path] = _FIXTURE_ROOT / "inputs"
_MANIFEST_PATH: Final[Path] = _INPUTS_DIR / "manifest.json"

_OPENFAST_DEFAULT_BINARY: Final[str] = "openfast"


# ---------------------------------------------------------------------------
# Channel-rename table (OpenFAST native -> FloatSim canonical SI).
# Add to this table as new scenarios surface new channels.
# ---------------------------------------------------------------------------

_DEG_TO_RAD: Final[float] = float(np.pi / 180.0)
_KN_TO_N: Final[float] = 1.0e3

# (canonical_name, conversion_factor)
_RENAME_TABLE: Final[dict[str, tuple[str, float]]] = {
    # Platform DOFs (per conventions doc Item 11: no module prefix).
    "PtfmSurge": ("surge_m", 1.0),
    "PtfmSway": ("sway_m", 1.0),
    "PtfmHeave": ("heave_m", 1.0),
    "PtfmRoll": ("roll_rad", _DEG_TO_RAD),
    "PtfmPitch": ("pitch_rad", _DEG_TO_RAD),
    "PtfmYaw": ("yaw_rad", _DEG_TO_RAD),
    # Platform velocities -- OpenFAST's "T" prefix is translational, "R"
    # rotational; "xt"/"yt"/"zt" are body-frame components.
    "PtfmTVxt": ("surge_dot_m_per_s", 1.0),
    "PtfmTVyt": ("sway_dot_m_per_s", 1.0),
    "PtfmTVzt": ("heave_dot_m_per_s", 1.0),
    "PtfmRVxt": ("roll_dot_rad_per_s", _DEG_TO_RAD),
    "PtfmRVyt": ("pitch_dot_rad_per_s", _DEG_TO_RAD),
    "PtfmRVzt": ("yaw_dot_rad_per_s", _DEG_TO_RAD),
    # MoorDyn outputs (per conventions doc Item 11; capitalised, no
    # underscore). OpenFAST writes line tensions in kN -> N.
    "FairTen1": ("fair_ten_line1_n", _KN_TO_N),
    "FairTen2": ("fair_ten_line2_n", _KN_TO_N),
    "FairTen3": ("fair_ten_line3_n", _KN_TO_N),
    "AnchTen1": ("anch_ten_line1_n", _KN_TO_N),
    "AnchTen2": ("anch_ten_line2_n", _KN_TO_N),
    "AnchTen3": ("anch_ten_line3_n", _KN_TO_N),
    # Wave elevation and B1 wave loads (S3 RAO sweep) -- forces in N,
    # moments in N*m, no conversion.
    "Wave1Elev": ("wave_elev_m", 1.0),
    "B1WvsF1xi": ("wave_force_x_n", 1.0),
    "B1WvsF1yi": ("wave_force_y_n", 1.0),
    "B1WvsF1zi": ("wave_force_z_n", 1.0),
    "B1WvsM1xi": ("wave_moment_x_nm", 1.0),
    "B1WvsM1yi": ("wave_moment_y_nm", 1.0),
    "B1WvsM1zi": ("wave_moment_z_nm", 1.0),
}


@dataclass(frozen=True)
class ScenarioEntry:
    """One row from the canonical manifest."""

    scenario_name: str
    deck_dir: Path  # absolute, resolved from manifest's relative path
    purpose: str
    moordyn_active: bool
    output_channels: tuple[str, ...]
    sweep_value: float | None

    @property
    def fst_path(self) -> Path:
        """Top-level ``.fst`` driver in the deck directory."""
        # Each deck dir holds exactly one .fst file per the manifest layout.
        candidates = sorted(self.deck_dir.glob("*.fst"))
        if len(candidates) != 1:
            raise SystemExit(
                f"{self.deck_dir.name}: expected exactly one .fst file; "
                f"found {len(candidates)}: {[c.name for c in candidates]}"
            )
        return candidates[0]

    @property
    def outb_path(self) -> Path:
        """The ``.outb`` next to the ``.fst`` (OpenFAST default output)."""
        return self.fst_path.with_suffix(".outb")

    @property
    def csv_path(self) -> Path:
        """Where the canonical CSV lands (next to the deck inputs)."""
        return self.deck_dir / f"{self.fst_path.stem}.csv"

    @property
    def json_path(self) -> Path:
        """JSON metadata sidecar."""
        return self.deck_dir / f"{self.fst_path.stem}.json"


def _load_manifest() -> tuple[dict[str, Any], list[ScenarioEntry]]:
    """Load the manifest and convert ``scenarios[]`` to typed entries."""
    if not _MANIFEST_PATH.is_file():
        raise SystemExit(
            f"manifest not found at {_MANIFEST_PATH.relative_to(_REPO_ROOT)}; "
            f"see {_FIXTURE_ROOT.relative_to(_REPO_ROOT) / 'README.md'} for "
            "the vendoring procedure."
        )
    raw = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    entries: list[ScenarioEntry] = []
    for scenario in raw["scenarios"]:
        deck_dir = _FIXTURE_ROOT / Path(scenario["deck_dir"].replace("\\", "/"))
        entries.append(
            ScenarioEntry(
                scenario_name=scenario["scenario_name"],
                deck_dir=deck_dir,
                purpose=scenario["purpose"],
                moordyn_active=scenario["moordyn_active"],
                output_channels=tuple(scenario["output_channels"]),
                sweep_value=scenario["sweep_value"],
            )
        )
    return raw, entries


# ---------------------------------------------------------------------------
# OpenFAST execution and binary-output reading.
# ---------------------------------------------------------------------------


def _detect_openfast_version(binary: str) -> str:
    """Probe ``{binary} -v`` and return a version string."""
    if shutil.which(binary) is None:
        return "unknown (binary not on PATH)"
    try:
        proc = subprocess.run(
            [binary, "-v"], check=True, capture_output=True, text=True, timeout=10
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        return f"unknown (probe failed: {exc!s})"
    for line in proc.stdout.splitlines():
        if "OpenFAST" in line and "v" in line.lower():
            return line.strip()
    return proc.stdout.splitlines()[0].strip() if proc.stdout else "unknown"


def _run_openfast(binary: str, fst_path: Path) -> Path:  # pragma: no cover -- live-OpenFAST path
    """Invoke OpenFAST on ``fst_path``; return the produced ``.outb`` path."""
    if shutil.which(binary) is None:
        raise SystemExit(
            f"OpenFAST binary {binary!r} not found on PATH. Install from "
            "https://github.com/OpenFAST/openfast/releases or pass --binary."
        )
    proc = subprocess.run([binary, str(fst_path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"OpenFAST exited {proc.returncode} on {fst_path.name}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    outb_path = fst_path.with_suffix(".outb")
    if not outb_path.is_file():
        raise SystemExit(f"OpenFAST did not produce expected output {outb_path}")
    return outb_path


def _read_outb(outb_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Read an OpenFAST ``.outb`` binary into (channel_names, units, data).

    Uses ``openfast_toolbox.io.FASTOutputFile``. Channel names are
    fetched via ``output.info["attribute_names"]`` (NOT via
    ``output.channels`` -- that attribute is unreliable across
    library versions). Units are similarly via
    ``output.info["attribute_units"]``.
    """
    try:
        from openfast_toolbox.io import FASTOutputFile  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "openfast-toolbox is required to read .outb binaries. "
            "Install via `pip install openfast-toolbox`."
        ) from exc

    output = FASTOutputFile(str(outb_path))
    info = output.info
    if "attribute_names" not in info:
        raise SystemExit(
            f"{outb_path.name}: openfast-toolbox FASTOutputFile.info "
            "missing 'attribute_names' key. Library version mismatch?"
        )
    channels = list(info["attribute_names"])
    units = list(info.get("attribute_units", ["?"] * len(channels)))
    # FASTOutputFile exposes the underlying array via .data (an
    # ndarray of shape (n_samples, n_channels)).
    data = np.asarray(output.data, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(channels):
        raise SystemExit(
            f"{outb_path.name}: data shape {data.shape} mismatches channel "
            f"count {len(channels)}"
        )
    return channels, units, data


def _convert_to_canonical_si(
    channels: list[str], units: list[str], data: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert the OpenFAST native units to canonical SI."""
    if "Time" not in channels:  # pragma: no cover
        raise SystemExit("OpenFAST output missing Time channel")
    t_idx = channels.index("Time")
    t_unit = units[t_idx].strip("()") if t_idx < len(units) else "s"
    if t_unit not in ("s", "sec"):  # pragma: no cover
        raise SystemExit(f"unexpected Time unit {t_unit!r} (expected 's')")
    time_s = data[:, t_idx].astype(np.float64)

    canonical: dict[str, np.ndarray] = {}
    for ch in channels:
        if ch in _RENAME_TABLE:
            new_name, factor = _RENAME_TABLE[ch]
            canonical[new_name] = factor * data[:, channels.index(ch)].astype(np.float64)
    return time_s, canonical


def _write_canonical_pair(
    entry: ScenarioEntry,
    time_s: np.ndarray,
    canonical: dict[str, np.ndarray],
    *,
    openfast_version: str,
    extraction_command: str,
    manifest_meta: dict[str, Any],
) -> tuple[Path, Path]:
    """Write the canonical CSV + JSON sidecar for one scenario."""
    csv_path = entry.csv_path
    json_path = entry.json_path

    # The canonical schema requires the six platform DOFs at minimum;
    # additional columns (velocities, tensions, wave channels) are
    # written if present.
    required = ["surge_m", "sway_m", "heave_m", "roll_rad", "pitch_rad", "yaw_rad"]
    missing = [c for c in required if c not in canonical]
    if missing:
        raise SystemExit(
            f"{entry.scenario_name}: missing canonical columns {missing}. "
            "Update _RENAME_TABLE in extract_openfast_fixtures.py."
        )
    optional_order = [
        "surge_dot_m_per_s",
        "sway_dot_m_per_s",
        "heave_dot_m_per_s",
        "roll_dot_rad_per_s",
        "pitch_dot_rad_per_s",
        "yaw_dot_rad_per_s",
        "wave_elev_m",
        "wave_force_x_n",
        "wave_force_y_n",
        "wave_force_z_n",
        "wave_moment_x_nm",
        "wave_moment_y_nm",
        "wave_moment_z_nm",
        "fair_ten_line1_n",
        "fair_ten_line2_n",
        "fair_ten_line3_n",
        "anch_ten_line1_n",
        "anch_ten_line2_n",
        "anch_ten_line3_n",
    ]
    extras = [c for c in optional_order if c in canonical]
    headers = ["time_s", *required, *extras]
    columns = [time_s] + [canonical[c] for c in required + extras]
    arr = np.column_stack(columns)
    csv_path.write_text(
        ",".join(headers)
        + "\n"
        + "\n".join(",".join(f"{v:.10e}" for v in row) for row in arr)
        + "\n",
        encoding="utf-8",
    )

    # Compute dt and duration from the time column (more reliable than
    # parsing fst_edits).
    dt_s = float(np.median(np.diff(time_s))) if time_s.size > 1 else float("nan")
    duration_s = float(time_s[-1] - time_s[0]) if time_s.size > 1 else 0.0

    metadata: dict[str, Any] = {
        "scenario_name": entry.scenario_name,
        "deck_dir": str(entry.deck_dir.relative_to(_FIXTURE_ROOT).as_posix()),
        "purpose": entry.purpose,
        "moordyn_active": entry.moordyn_active,
        "sweep_value": entry.sweep_value,
        "openfast_version": openfast_version,
        "openfast_version_required": manifest_meta.get("openfast_version_required"),
        "r_test_tag_required": manifest_meta.get("r_test_tag_required"),
        "dt_s": dt_s,
        "duration_s": duration_s,
        "n_samples": int(time_s.size),
        "unit_system": "SI_canonical",
        "extracted_by": extraction_command,
        "extracted_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "channels_canonical": headers,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path, json_path


def _extract_one(
    entry: ScenarioEntry,
    *,
    mode: str,
    binary: str,
    manifest_meta: dict[str, Any],
) -> None:
    """Run OpenFAST (if mode=run) and extract canonical CSV+JSON for one scenario."""
    if not entry.deck_dir.is_dir():
        raise SystemExit(
            f"deck directory {entry.deck_dir.relative_to(_REPO_ROOT)} not "
            "found; manifest may be stale."
        )

    if mode == "run":  # pragma: no cover -- live-OpenFAST path
        print(f"  scenario {entry.scenario_name}: running OpenFAST...", flush=True)
        outb_path = _run_openfast(binary, entry.fst_path)
    elif mode == "read-only":
        outb_path = entry.outb_path
        if not outb_path.is_file():
            raise SystemExit(
                f"--mode read-only but {outb_path.relative_to(_REPO_ROOT)} not "
                "found. Either run OpenFAST first (--mode run) or check that "
                "the .outb file was committed/extracted to the deck directory."
            )
        print(f"  scenario {entry.scenario_name}: reading {outb_path.name}", flush=True)
    else:  # pragma: no cover -- argparse choices guard
        raise SystemExit(f"unknown mode {mode!r}")

    channels, units, data = _read_outb(outb_path)
    time_s, canonical = _convert_to_canonical_si(channels, units, data)
    csv_path, json_path = _write_canonical_pair(
        entry,
        time_s,
        canonical,
        openfast_version=_detect_openfast_version(binary),
        extraction_command=(
            f"scripts/extract_openfast_fixtures.py --mode {mode} "
            f"--scenario {entry.scenario_name}"
            + (f" --sweep {entry.sweep_value}" if entry.sweep_value is not None else "")
        ),
        manifest_meta=manifest_meta,
    )
    print(
        f"    wrote {csv_path.relative_to(_REPO_ROOT)} + {json_path.name}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _select_entries(entries: list[ScenarioEntry], scenario_arg: str) -> list[ScenarioEntry]:
    """Filter the manifest entries by the ``--scenario`` CLI argument."""
    if scenario_arg == "all":
        return entries
    selected = [e for e in entries if e.scenario_name == scenario_arg]
    if not selected:
        valid = sorted({e.scenario_name for e in entries})
        raise SystemExit(f"unknown --scenario {scenario_arg!r}; valid: {', '.join(valid)} or 'all'")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode",
        choices=("run", "read-only"),
        default="read-only",
        help=(
            "'run': invoke OpenFAST against each .fst (requires OpenFAST on "
            "PATH); 'read-only': just read existing .outb files (default)."
        ),
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help=(
            "Scenario name from manifest.json (e.g. s1_static_eq), or 'all' "
            "for the full set. The S3 RAO sweep registers as 14 entries with "
            "scenario_name='s3_rao_sweep' and distinct sweep_value -- 'all' "
            "processes all of them."
        ),
    )
    parser.add_argument(
        "--binary",
        default=_OPENFAST_DEFAULT_BINARY,
        help="OpenFAST executable name or absolute path (default: 'openfast').",
    )
    args = parser.parse_args(argv)

    manifest_meta, entries = _load_manifest()
    selected = _select_entries(entries, args.scenario)
    print(
        f"extract_openfast_fixtures: mode={args.mode}, "
        f"selected {len(selected)} of {len(entries)} manifest entries",
        flush=True,
    )
    for entry in selected:
        _extract_one(entry, mode=args.mode, binary=args.binary, manifest_meta=manifest_meta)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
