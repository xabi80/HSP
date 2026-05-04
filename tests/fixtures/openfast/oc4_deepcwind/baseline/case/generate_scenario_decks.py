"""Generate per-scenario OpenFAST decks for the M6 cross-check.

Reads the OpenFAST OC4 DeepCwind baseline from the local r-test clone,
vendors a clean copy into the HSP repo, and writes scenario-specific
decks per the definitions in scenario_config.py.

Usage:
    python generate_scenario_decks.py \\
        --r-test-root C:/path/to/r-test \\
        --hsp-root C:/path/to/HSP \\
        [--clean]

Architecture:
    openfast_io reads an entire OpenFAST deck (.fst plus all referenced
    module .dat files) into a single nested dict, fst_vt, with sub-dicts
    keyed by module name ('Fst', 'ElastoDyn', 'HydroDyn', 'MoorDyn',
    'ServoDyn', etc.). To produce a scenario deck:

        1. Read baseline once -> fst_vt with all module data.
        2. For each scenario:
           a. Deep-copy fst_vt so scenarios don't pollute each other.
           b. Apply scenario edits (sparse overrides) to the right
              sub-dicts.
           c. Configure InputWriter_OpenFAST with the modified fst_vt
              and an output directory.
           d. writer.execute() emits the whole deck.

    Sweep scenarios (S3 RAO) re-use the same baseline read but produce
    one output directory per swept value.

Dependencies:
    pip install openfast_io

This script does NOT run OpenFAST. Use run_scenarios.py for that.

Author: HSP project, M6 setup, generated 2026-05.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from openfast_io.FAST_reader import InputReader_OpenFAST
    from openfast_io.FAST_writer import InputWriter_OpenFAST
except ImportError:
    print(
        "ERROR: openfast_io not installed.\n"
        "Run: pip install openfast_io",
        file=sys.stderr,
    )
    sys.exit(1)

from scenario_config import SCENARIOS, Scenario


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# The WSt_WavesWN variant is the time-domain configuration of OC4
# DeepCwind. The Linear variant (5MW_OC4Semi_Linear) is for
# linearization analysis only and unsuitable for time-domain cross-
# check.
BASELINE_CASE_DIR = "glue-codes/openfast/5MW_OC4Semi_WSt_WavesWN"
SUPPORT_DIR = "glue-codes/openfast/5MW_Baseline"

# Files in 5MW_Baseline that the OC4 deck references. The full
# 5MW_Baseline directory contains material for many test cases; we
# vendor only what OC4 needs to keep repo size reasonable.
SUPPORT_FILES_NEEDED = (
    "HydroData/marin_semi.1",
    "HydroData/marin_semi.3",
    "HydroData/marin_semi.hst",
    "HydroData/marin_semi.12d",
    "HydroData/marin_semi.12s",
    "NRELOffshrBsline5MW_Blade.dat",
    "NRELOffshrBsline5MW_Tower_OC4DeepCwindSemi.dat",
)

# Vendored layout under HSP repo.
VENDOR_REL = "tests/fixtures/openfast/oc4_deepcwind/baseline"
INPUTS_REL = "tests/fixtures/openfast/oc4_deepcwind/inputs"


# ---------------------------------------------------------------------
# Edit dispatch: which fst_vt sub-dict each scenario edit-set targets
# ---------------------------------------------------------------------
#
# scenario_config.py groups edits by name (fst_edits, elastodyn_edits,
# hydrodyn_edits). openfast_io stores them in fst_vt['Fst'],
# fst_vt['ElastoDyn'], fst_vt['HydroDyn']. This mapping connects them.
EDIT_TARGETS: dict[str, str] = {
    "fst_edits": "Fst",
    "elastodyn_edits": "ElastoDyn",
    "hydrodyn_edits": "HydroDyn",
}


# ---------------------------------------------------------------------
# Vendoring
# ---------------------------------------------------------------------

def vendor_baseline(r_test_root: Path, hsp_root: Path, *, clean: bool) -> Path:
    """Copy the OC4 baseline from r-test into the HSP repo.

    Parameters
    ----------
    r_test_root : Path
        Root of the r-test clone (already at v4.1.2 tag).
    hsp_root : Path
        Root of the HSP repo.
    clean : bool
        If True, wipe the existing vendored copy first.

    Returns
    -------
    Path
        The vendored baseline directory inside HSP.
    """
    src_case = r_test_root / BASELINE_CASE_DIR
    src_support = r_test_root / SUPPORT_DIR
    if not src_case.is_dir():
        raise FileNotFoundError(
            f"Baseline case not found: {src_case}\n"
            "Confirm r-test is checked out at v4.1.2 tag:\n"
            "  cd r-test && git fetch --tags && git checkout v4.1.2"
        )

    vendor_dir = hsp_root / VENDOR_REL
    if clean and vendor_dir.exists():
        shutil.rmtree(vendor_dir)
    vendor_dir.mkdir(parents=True, exist_ok=True)

    # Copy the case directory verbatim. Small (~50 KB).
    case_dst = vendor_dir / "case"
    if case_dst.exists():
        shutil.rmtree(case_dst)
    shutil.copytree(src_case, case_dst)
    print(f"  Vendored case: {src_case.name} -> {case_dst.relative_to(hsp_root)}")

    # Copy only the needed support files.
    support_dst = vendor_dir / "5MW_Baseline"
    support_dst.mkdir(exist_ok=True)
    for rel in SUPPORT_FILES_NEEDED:
        src = src_support / rel
        if not src.exists():
            print(f"  WARNING: support file missing, skipping: {rel}")
            continue
        dst = support_dst / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Vendored support: {rel}")

    return vendor_dir


# ---------------------------------------------------------------------
# Reading the baseline
# ---------------------------------------------------------------------

def read_baseline(vendor_dir: Path) -> InputReader_OpenFAST:
    """Read the vendored baseline once into a populated reader.

    The reader's fst_vt nested dict is the single source of truth for
    all scenario derivatives. Each scenario deep-copies it before
    mutating.
    """
    case_dir = vendor_dir / "case"
    fst_files = list(case_dir.glob("*.fst"))
    if len(fst_files) != 1:
        raise RuntimeError(
            f"Expected exactly one .fst in {case_dir}, found {len(fst_files)}."
        )
    fst_path = fst_files[0]

    reader = InputReader_OpenFAST()
    reader.FAST_InputFile = fst_path.name
    reader.FAST_directory = str(case_dir)
    reader.execute()
    return reader


# ---------------------------------------------------------------------
# fst_vt mutation
# ---------------------------------------------------------------------

def apply_scenario_edits(fst_vt: dict[str, Any], scenario: Scenario) -> None:
    """Apply scenario edits to a fst_vt nested dict in place.

    Raises KeyError if any edit targets a key absent from the baseline.
    Surfacing this immediately prevents silent insertion of unknown
    fields, which OpenFAST would later refuse to parse anyway.
    """
    edit_groups: dict[str, dict[str, Any]] = {
        "fst_edits": dict(scenario.fst_edits),
        "elastodyn_edits": dict(scenario.elastodyn_edits),
        "hydrodyn_edits": dict(scenario.hydrodyn_edits),
    }
    for group_name, edits in edit_groups.items():
        if not edits:
            continue
        target_key = EDIT_TARGETS[group_name]
        target_dict = fst_vt[target_key]
        for key, value in edits.items():
            if key not in target_dict:
                # Sample of available keys helps the user spot typos
                # or version mismatches without dumping 200+ entries.
                sample = sorted(target_dict.keys())[:30]
                raise KeyError(
                    f"Key '{key}' not found in fst_vt['{target_key}']. "
                    f"Available keys (first 30): {sample}"
                )
            target_dict[key] = value


def apply_sweep_value(
    fst_vt: dict[str, Any],
    scenario: Scenario,
    sweep_value: float,
) -> None:
    """Set the swept parameter value in fst_vt.

    The S3 RAO sweep targets HydroDyn.WaveTp; future sweeps over other
    parameters or modules would need this function extended.
    """
    if scenario.sweep_param is None:
        return
    sweep_name, _ = scenario.sweep_param

    # Currently all sweep parameters live in HydroDyn. If a future
    # scenario sweeps something elsewhere, extend this lookup.
    if sweep_name in fst_vt["HydroDyn"]:
        fst_vt["HydroDyn"][sweep_name] = sweep_value
    elif sweep_name in fst_vt["Fst"]:
        fst_vt["Fst"][sweep_name] = sweep_value
    else:
        raise KeyError(
            f"Sweep parameter '{sweep_name}' not found in HydroDyn or Fst."
        )


# ---------------------------------------------------------------------
# Per-scenario deck generation
# ---------------------------------------------------------------------

def _deck_directory(
    scenario: Scenario,
    inputs_root: Path,
    sweep_value: float | None,
) -> Path:
    """Compute output directory for a scenario (or one of its sweep variants)."""
    if sweep_value is None:
        return inputs_root / scenario.name
    assert scenario.sweep_param is not None
    sweep_name, _ = scenario.sweep_param
    base_dir = inputs_root / scenario.name
    # Encode value with 'p' for decimal so the path is filesystem-safe.
    return base_dir / f"{sweep_name}_{sweep_value:05.1f}".replace(".", "p")


def _naming_prefix(scenario: Scenario, sweep_value: float | None) -> str:
    """Filename prefix for the generated module files."""
    if sweep_value is None:
        return scenario.name
    assert scenario.sweep_param is not None
    sweep_name, _ = scenario.sweep_param
    suffix = f"_{sweep_name}_{sweep_value:05.1f}".replace(".", "p")
    return f"{scenario.name}{suffix}"


def generate_deck_for_scenario(
    scenario: Scenario,
    baseline_reader: InputReader_OpenFAST,
    inputs_root: Path,
    sweep_value: float | None = None,
) -> Path:
    """Create one scenario deck via InputWriter_OpenFAST.

    Returns the directory the deck was written to.
    """
    deck_dir = _deck_directory(scenario, inputs_root, sweep_value)
    if deck_dir.exists():
        shutil.rmtree(deck_dir)
    deck_dir.mkdir(parents=True)

    # Configure writer. Deep-copy fst_vt so mutations stay scenario-
    # local; the baseline reader keeps clean state for the next call.
    writer = InputWriter_OpenFAST()
    writer.fst_vt = copy.deepcopy(baseline_reader.fst_vt)
    writer.FAST_runDirectory = str(deck_dir)
    writer.FAST_namingOut = _naming_prefix(scenario, sweep_value)

    # Apply edits (will raise on unknown keys — fail-loud preferred).
    apply_scenario_edits(writer.fst_vt, scenario)
    if sweep_value is not None:
        apply_sweep_value(writer.fst_vt, scenario, sweep_value)

    # Write the deck. The writer emits .fst + every module file
    # referenced by fst_vt's flags (CompElast, CompHydro, CompMooring,
    # etc.) into FAST_runDirectory.
    writer.execute()
    return deck_dir


# ---------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------

def write_manifest(
    inputs_root: Path,
    generated: list[tuple[Scenario, Path, float | None]],
) -> Path:
    """Write a JSON manifest describing every generated deck.

    Consumed by run_scenarios.py and (later) by FloatSim's fixture
    loader to know which decks correspond to which scenarios.
    """
    scenarios_payload: list[dict[str, Any]] = []
    for scenario, deck_dir, sweep_value in generated:
        scenarios_payload.append({
            "scenario_name": scenario.name,
            "deck_dir": str(deck_dir.relative_to(inputs_root.parent)),
            "purpose": scenario.purpose,
            "moordyn_active": scenario.moordyn_active,
            "output_channels": list(scenario.output_channels),
            "sweep_value": sweep_value,
            "fst_edits": dict(scenario.fst_edits),
            "elastodyn_edits": dict(scenario.elastodyn_edits),
            "hydrodyn_edits": dict(scenario.hydrodyn_edits),
        })
    manifest: dict[str, Any] = {
        "openfast_version_required": "v4.1.2",
        "r_test_tag_required": "v4.1.2",
        "baseline_case": "5MW_OC4Semi_WSt_WavesWN",
        "scenarios": scenarios_payload,
    }
    manifest_path = inputs_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return manifest_path


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r-test-root",
        type=Path,
        required=True,
        help="Path to local r-test clone (must be at v4.1.2 tag).",
    )
    parser.add_argument(
        "--hsp-root",
        type=Path,
        required=True,
        help="Path to HSP repo root.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe existing vendored baseline and inputs before regenerating.",
    )
    args = parser.parse_args()

    r_test_root = args.r_test_root.resolve()
    hsp_root = args.hsp_root.resolve()

    print("=" * 70)
    print("M6 OpenFAST scenario deck generator")
    print("=" * 70)
    print(f"r-test root: {r_test_root}")
    print(f"HSP root:    {hsp_root}")
    print()

    # 1. Vendor baseline.
    print("[1/4] Vendoring OC4 DeepCwind baseline...")
    vendor_dir = vendor_baseline(r_test_root, hsp_root, clean=args.clean)
    print()

    # 2. Read baseline.
    print("[2/4] Reading baseline OpenFAST deck...")
    baseline_reader = read_baseline(vendor_dir)
    n_top = len(baseline_reader.fst_vt["Fst"])
    n_ed = len(baseline_reader.fst_vt["ElastoDyn"])
    n_hd = len(baseline_reader.fst_vt["HydroDyn"])
    print(
        f"      Loaded fst_vt: {n_top} Fst keys, "
        f"{n_ed} ElastoDyn keys, {n_hd} HydroDyn keys"
    )
    print()

    # 3. Generate per-scenario decks.
    print("[3/4] Generating scenario decks...")
    inputs_root = hsp_root / INPUTS_REL
    if args.clean and inputs_root.exists():
        shutil.rmtree(inputs_root)
    inputs_root.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[Scenario, Path, float | None]] = []
    for scenario in SCENARIOS:
        if scenario.sweep_param is None:
            deck_dir = generate_deck_for_scenario(
                scenario, baseline_reader, inputs_root,
            )
            generated.append((scenario, deck_dir, None))
            print(f"      [{scenario.name}] -> {deck_dir.relative_to(hsp_root)}")
        else:
            sweep_name, values = scenario.sweep_param
            print(f"      [{scenario.name}] sweep over {sweep_name}:")
            for value in values:
                deck_dir = generate_deck_for_scenario(
                    scenario, baseline_reader, inputs_root, sweep_value=value,
                )
                generated.append((scenario, deck_dir, value))
                print(
                    f"        {sweep_name}={value} -> "
                    f"{deck_dir.relative_to(hsp_root)}"
                )
    print()

    # 4. Manifest.
    print("[4/4] Writing manifest...")
    manifest_path = write_manifest(inputs_root, generated)
    print(f"      -> {manifest_path.relative_to(hsp_root)}")
    print()

    print("=" * 70)
    print(
        f"Done. Generated {len(generated)} deck(s) "
        f"across {len(SCENARIOS)} scenario(s)."
    )
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Inspect a generated deck to confirm edits look correct:")
    if generated:
        sample = generated[0][1]
        print(f"       cd {sample.relative_to(hsp_root)}")
        print("       Open the .fst and verify CompAero, CompServo,")
        print("       CompMooring values match your scenario.")
    print("  2. Smoke-test with: run_scenarios.py --scenarios s1_static_eq")
    print("  3. Run all with:    run_scenarios.py --continue-on-error")
    return 0


if __name__ == "__main__":
    sys.exit(main())
