"""Run OpenFAST locally and extract canonical CSV time histories for M6.

This script is the **single point of unit conversion** between
OpenFAST's native output format (`.out` text or `.outb` binary,
typically with degrees for rotations and a mix of metres / kilonewtons /
tonnes for forces) and FloatSim's canonical SI fixture format
(metres / radians / Newtons / kilograms; see
``tests/support/openfast_csv.py`` module docstring for the full
schema). It is *not* run in CI -- it requires OpenFAST installed
locally on the contributor's machine. The committed CSV+JSON pairs
are the artifacts; this script regenerates them on demand.

Locked spec per ``docs/milestone-6-plan.md`` v2 Q3:

- Reference deck: OpenFAST/r-test ``5MW_OC4Semi_Linear/`` vendored
  under ``tests/fixtures/openfast/oc4_deepcwind/inputs/``.
- Five scenarios in increasing-complexity order:
  S1 static_eq, S2 free_decay, S3 rao_*, S4 moored_eq, S5 drag_decay.
- Wind turbine disabled (``CompElast=CompAero=CompInflow=CompServo=0``)
  for hydrodynamics-only isolation. **Footgun:** with
  ``CompElast=0`` ElastoDyn does not apply gravity -- HydroDyn alone
  provides only buoyancy-referenced restoring. For S1 (static
  equilibrium) we use the workaround documented in
  ``docs/openfast-cross-check-conventions.md`` (CompElast=1 with
  unused platform DOFs locked, OR HydroDyn standalone driver with
  explicit gravity input).

Status (M6 PR1)
---------------
This is a **scaffolding** commit. The end-to-end orchestration --
copying inputs into a working directory, invoking the OpenFAST
executable, parsing its `.out` output, performing the unit
conversions, writing the canonical CSV + JSON sidecar -- is sketched
in skeleton form below but not exercised against a real OpenFAST
binary. The goal of this PR is to land the contract (canonical
fixture format, metadata schema, scenario list) so M6 PR2 can write
its first failing assertion against a hand-authored fixture matching
this contract; PR2 then either uses Xabier's locally-extracted CSVs
or this script's output once OpenFAST is installed.

Usage (when OpenFAST is installed)
----------------------------------
::

    pip install --editable .[dev]
    # Ensure 'openfast' is on PATH (download from
    # https://github.com/OpenFAST/openfast/releases).
    python scripts/extract_openfast_fixtures.py --scenario s2_free_decay
    python scripts/extract_openfast_fixtures.py --scenario all  # all five

Each scenario writes ``tests/fixtures/openfast/oc4_deepcwind/outputs/{scenario}.csv``
plus ``{scenario}.json`` with the full metadata required by
:func:`tests.support.openfast_csv.load_openfast_history`.

Re-running is idempotent in the sense that the same OpenFAST inputs
+ same OpenFAST version produce a byte-similar CSV (some BEM-integrator
noise in the last few significant digits is expected). The JSON
sidecar's ``openfast_version`` and ``extracted_at`` capture the
non-determinism so an audit can detect a silent OpenFAST version bump.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Repo-relative paths and scenario registry.
# ---------------------------------------------------------------------------

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_FIXTURE_ROOT: Final[Path] = _REPO_ROOT / "tests" / "fixtures" / "openfast" / "oc4_deepcwind"
_INPUTS_DIR: Final[Path] = _FIXTURE_ROOT / "inputs"
_OUTPUTS_DIR: Final[Path] = _FIXTURE_ROOT / "outputs"

_OPENFAST_DEFAULT_BINARY: Final[str] = "openfast"


@dataclass(frozen=True)
class Scenario:
    """One M6 cross-check scenario."""

    scenario_id: str
    description: str
    fst_filename: str  # the top-level .fst driver file, relative to _INPUTS_DIR
    duration_s: float
    dt_s: float
    notes: str


# Locked per docs/milestone-6-plan.md v2 Q2 (in increasing-complexity
# order: statics -> decay -> RAO -> moored statics -> drag decay).
SCENARIOS: Final[tuple[Scenario, ...]] = (
    Scenario(
        scenario_id="s1_static_eq",
        description="Static equilibrium (no waves, no wind, no mooring)",
        fst_filename="OC4Semi_S1_static_eq.fst",
        duration_s=200.0,
        dt_s=0.05,
        notes=(
            "CompElast=1 with platform DOFs locked except heave/roll/pitch (the "
            "CompElast=0 alternative would skip gravity in ElastoDyn -- see Q2 "
            "footgun in docs/milestone-6-plan.md v2)."
        ),
    ),
    Scenario(
        scenario_id="s2_free_decay",
        description="Heave + pitch free decay (Cummins free response)",
        fst_filename="OC4Semi_S2_free_decay.fst",
        duration_s=600.0,
        dt_s=0.05,
        notes="Heave IC = 0.5 m, pitch IC = 5 deg (no waves, no wind, no mooring).",
    ),
    Scenario(
        scenario_id="s3_rao_T08",
        description="Regular wave at T = 8 s (RAO sweep point)",
        fst_filename="OC4Semi_S3_rao_T08.fst",
        duration_s=400.0,
        dt_s=0.05,
        notes="Heading 0 deg, amplitude 0.5 m. Steady-state extracted from last 3 cycles.",
    ),
    Scenario(
        scenario_id="s3_rao_T10",
        description="Regular wave at T = 10 s",
        fst_filename="OC4Semi_S3_rao_T10.fst",
        duration_s=400.0,
        dt_s=0.05,
        notes="Heading 0 deg, amplitude 0.5 m.",
    ),
    Scenario(
        scenario_id="s3_rao_T12",
        description="Regular wave at T = 12 s",
        fst_filename="OC4Semi_S3_rao_T12.fst",
        duration_s=400.0,
        dt_s=0.05,
        notes="Heading 0 deg, amplitude 0.5 m.",
    ),
    Scenario(
        scenario_id="s3_rao_T14",
        description="Regular wave at T = 14 s",
        fst_filename="OC4Semi_S3_rao_T14.fst",
        duration_s=400.0,
        dt_s=0.05,
        notes="Heading 0 deg, amplitude 0.5 m.",
    ),
    Scenario(
        scenario_id="s3_rao_T16",
        description="Regular wave at T = 16 s (near OC4 heave natural)",
        fst_filename="OC4Semi_S3_rao_T16.fst",
        duration_s=400.0,
        dt_s=0.05,
        notes="Heading 0 deg, amplitude 0.5 m. Spans heave resonance ~17 s.",
    ),
    Scenario(
        scenario_id="s3_rao_T18",
        description="Regular wave at T = 18 s",
        fst_filename="OC4Semi_S3_rao_T18.fst",
        duration_s=400.0,
        dt_s=0.05,
        notes="Heading 0 deg, amplitude 0.5 m.",
    ),
    Scenario(
        scenario_id="s4_moored_eq",
        description="Moored static equilibrium (3-line catenary via MAP++)",
        fst_filename="OC4Semi_S4_moored_eq.fst",
        duration_s=300.0,
        dt_s=0.05,
        notes="MAP++ analytical catenary (NOT MoorDyn -- transient mooring is Phase 2).",
    ),
    Scenario(
        scenario_id="s5_drag_decay",
        description="Drag-on heave free decay (Morison Members populated)",
        fst_filename="OC4Semi_S5_drag_decay.fst",
        duration_s=600.0,
        dt_s=0.05,
        notes=(
            "Same IC as S2 with HydroDyn Members block populated. Cross-checks "
            "M5 PR4 Morison wiring against HydroDyn's drag-element implementation."
        ),
    ),
)


def _scenario_by_id(sid: str) -> Scenario:
    for s in SCENARIOS:
        if s.scenario_id == sid:
            return s
    raise SystemExit(
        f"unknown scenario_id {sid!r}; valid ids: " + ", ".join(s.scenario_id for s in SCENARIOS)
    )


# ---------------------------------------------------------------------------
# Skeleton extraction pipeline.
# ---------------------------------------------------------------------------


def _detect_openfast_version(binary: str) -> str:
    """Probe ``{binary} -v`` and return a version string (e.g. ``v3.5.3``).

    Skeleton: shells out to OpenFAST and parses the leading version
    line from stdout. Falls back to ``"unknown"`` if the call fails;
    the caller decides whether to abort.
    """
    if shutil.which(binary) is None:
        return "unknown (binary not on PATH)"
    try:
        proc = subprocess.run(
            [binary, "-v"], check=True, capture_output=True, text=True, timeout=10
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        return f"unknown (probe failed: {exc!s})"
    # OpenFAST prints e.g. "OpenFAST-v3.5.3" on the first stdout line.
    for line in proc.stdout.splitlines():
        if "OpenFAST" in line and "v" in line:
            return line.strip()
    return proc.stdout.splitlines()[0].strip() if proc.stdout else "unknown"


def _run_openfast(binary: str, fst_path: Path) -> Path:
    """Invoke OpenFAST on ``fst_path``; return the produced ``.out`` path.

    Skeleton: assumes OpenFAST writes ``{stem}.out`` next to the input
    by default. Real implementation will copy inputs into a fresh
    working directory first (OpenFAST is sensitive to relative
    sub-input paths).
    """
    if shutil.which(binary) is None:  # pragma: no cover
        raise SystemExit(
            f"OpenFAST binary {binary!r} not found on PATH. Install from "
            "https://github.com/OpenFAST/openfast/releases or pass --binary."
        )
    proc = subprocess.run(  # pragma: no cover -- depends on OpenFAST install
        [binary, str(fst_path)], capture_output=True, text=True
    )
    if proc.returncode != 0:  # pragma: no cover
        raise SystemExit(
            f"OpenFAST exited {proc.returncode} on {fst_path.name}.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    out_path = fst_path.with_suffix(".out")
    if not out_path.is_file():  # pragma: no cover
        raise SystemExit(f"OpenFAST did not produce expected output {out_path}")
    return out_path


def _parse_openfast_out_text(out_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Parse an OpenFAST text ``.out`` file into (channel_names, units, data).

    OpenFAST's text output has two header lines (channel names; units),
    then whitespace-separated float rows. ``data`` is shape
    ``(n_samples, n_channels)``.
    """
    with open(out_path, encoding="utf-8") as fh:
        # OpenFAST sometimes writes a banner of dash lines before the
        # channel-name row. Skip until we hit a row that contains "Time".
        for line in fh:
            if line.strip().startswith("Time") or "Time " in line:
                channel_line = line
                break
        else:  # pragma: no cover
            raise SystemExit(f"could not find channel-names row in {out_path}")
        units_line = next(fh)
        # Remaining lines are the data block; some OpenFAST builds emit
        # a trailing blank line which np.loadtxt happily ignores.
    channels = channel_line.split()
    units = units_line.split()
    if len(channels) != len(units):  # pragma: no cover
        raise SystemExit(
            f"{out_path.name}: channel/unit header mismatch " f"({len(channels)} vs {len(units)})"
        )
    data = np.loadtxt(out_path, skiprows=8)  # OpenFAST default: 8 header lines
    return channels, units, data


def _convert_to_canonical_si(
    channels: list[str], units: list[str], data: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Convert OpenFAST native units to SI canonical (rad, m, N, kg).

    Returns ``(time_s, named_columns)`` with ``named_columns`` keyed by
    a normalised SI-suffixed name (e.g. ``"surge_m"``, ``"pitch_rad"``,
    ``"tension_line1_N"``). Caller is responsible for assembling these
    into the canonical CSV header order.
    """
    if "Time" not in channels:  # pragma: no cover
        raise SystemExit("OpenFAST output missing Time channel")
    t_idx = channels.index("Time")
    t_unit = units[t_idx].strip("()")
    if t_unit not in ("s", "sec"):  # pragma: no cover
        raise SystemExit(f"unexpected Time unit {t_unit!r} (expected 's')")
    time_s = data[:, t_idx].astype(np.float64)

    canonical: dict[str, np.ndarray] = {}
    # Map OpenFAST channel names to canonical (SI) names + conversion factor.
    # Add to this table as new scenarios surface new channels.
    rename_table: dict[str, tuple[str, float]] = {
        "PtfmSurge": ("surge_m", 1.0),
        "PtfmSway": ("sway_m", 1.0),
        "PtfmHeave": ("heave_m", 1.0),
        "PtfmRoll": ("roll_rad", np.pi / 180.0),  # OpenFAST writes degrees by default
        "PtfmPitch": ("pitch_rad", np.pi / 180.0),
        "PtfmYaw": ("yaw_rad", np.pi / 180.0),
        # M6 PR5 (S4 moored) extension: per-line top tensions in kN -> N.
        # Add conversions here when MoorDyn / MAP++ output channel names
        # are confirmed against a live OpenFAST run.
    }
    for ch in channels:
        if ch in rename_table:
            new_name, factor = rename_table[ch]
            canonical[new_name] = factor * data[:, channels.index(ch)].astype(np.float64)
    return time_s, canonical


def _write_canonical_pair(
    scenario: Scenario,
    time_s: np.ndarray,
    canonical: dict[str, np.ndarray],
    extra: dict[str, np.ndarray],
    *,
    openfast_version: str,
    binary: str,
) -> tuple[Path, Path]:
    """Write the SI-canonical CSV + JSON sidecar for one scenario."""
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _OUTPUTS_DIR / f"{scenario.scenario_id}.csv"
    json_path = _OUTPUTS_DIR / f"{scenario.scenario_id}.json"

    required = ["surge_m", "sway_m", "heave_m", "roll_rad", "pitch_rad", "yaw_rad"]
    missing = [c for c in required if c not in canonical]
    if missing:  # pragma: no cover -- rename_table coverage gap
        raise SystemExit(
            f"{scenario.scenario_id}: extracted history missing canonical columns "
            f"{missing}. Update rename_table in _convert_to_canonical_si."
        )

    headers = ["time_s", *required, *extra.keys()]
    columns = [time_s] + [canonical[c] for c in required] + list(extra.values())
    arr = np.column_stack(columns)
    csv_path.write_text(
        ",".join(headers)
        + "\n"
        + "\n".join(",".join(f"{v:.10e}" for v in row) for row in arr)
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "scenario_id": scenario.scenario_id,
        "openfast_version": openfast_version,
        "dt_s": scenario.dt_s,
        "duration_s": scenario.duration_s,
        "unit_system": "SI_canonical",
        "extracted_by": (
            f"scripts/extract_openfast_fixtures.py --scenario {scenario.scenario_id} "
            f"--binary {binary}"
        ),
        "extracted_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_inputs": [str(_INPUTS_DIR.relative_to(_REPO_ROOT) / scenario.fst_filename)],
        "description": scenario.description,
        "notes": scenario.notes,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path, json_path


def _extract_one(scenario: Scenario, binary: str) -> None:  # pragma: no cover
    """Run OpenFAST + extract one scenario. Real-execution path."""
    fst_path = _INPUTS_DIR / scenario.fst_filename
    if not fst_path.is_file():
        raise SystemExit(
            f"OpenFAST input {fst_path} not found. The committed fixture set "
            f"may be incomplete; see {_FIXTURE_ROOT / 'README.md'} for the "
            "vendoring procedure."
        )
    print(f"  scenario {scenario.scenario_id}: running OpenFAST...", flush=True)
    out_path = _run_openfast(binary, fst_path)
    channels, units, data = _parse_openfast_out_text(out_path)
    time_s, canonical = _convert_to_canonical_si(channels, units, data)
    csv, json_path = _write_canonical_pair(
        scenario,
        time_s,
        canonical,
        extra={},  # M6 PR5/PR6 extend with per-line tensions
        openfast_version=_detect_openfast_version(binary),
        binary=binary,
    )
    print(f"    wrote {csv.relative_to(_REPO_ROOT)} + {json_path.name}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        default="all",
        help=(
            "Scenario id to extract, or 'all' for the full M6 set. Valid ids: "
            + ", ".join(s.scenario_id for s in SCENARIOS)
        ),
    )
    parser.add_argument(
        "--binary",
        default=_OPENFAST_DEFAULT_BINARY,
        help="OpenFAST executable name or absolute path (default: 'openfast').",
    )
    args = parser.parse_args(argv)

    if not _INPUTS_DIR.is_dir():
        print(
            f"OpenFAST inputs directory {_INPUTS_DIR.relative_to(_REPO_ROOT)} not found. "
            f"See {_FIXTURE_ROOT.relative_to(_REPO_ROOT) / 'README.md'} for the vendoring "
            "recipe.",
            file=sys.stderr,
        )
        return 1

    scenarios = SCENARIOS if args.scenario == "all" else (_scenario_by_id(args.scenario),)
    for sc in scenarios:
        _extract_one(sc, args.binary)  # pragma: no cover -- live OpenFAST path
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
