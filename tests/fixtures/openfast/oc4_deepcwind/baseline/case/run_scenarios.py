"""Run the generated M6 OpenFAST scenarios.

Reads the manifest produced by generate_scenario_decks.py and invokes
fast_x64.exe (or openfast on Linux) on each deck. Captures stdout/
stderr per scenario, records exit codes, summarizes at the end.

Usage:
    python run_scenarios.py \\
        --hsp-root C:/path/to/HSP \\
        --openfast-bin C:/OpenFAST/fast_x64.exe \\
        [--scenarios s1_static_eq s2_pitch_decay] \\
        [--continue-on-error]

What it does:
    - Reads tests/fixtures/openfast/oc4_deepcwind/inputs/manifest.json
    - For each deck listed (or filtered by --scenarios), runs OpenFAST
      with cwd set to the deck directory.
    - Writes stdout/stderr/exit-code into <deck_dir>/run_log/.
    - Summarizes pass/fail at the end.

Does NOT convert outputs to fixtures — that's extract_openfast_fixtures.py.

Author: HSP project, M6 setup, generated 2026-05.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


INPUTS_REL = "tests/fixtures/openfast/oc4_deepcwind/inputs"


def run_one_deck(
    deck_dir: Path,
    openfast_bin: Path,
) -> tuple[bool, float, str]:
    """Run OpenFAST in a single deck directory.

    Returns
    -------
    (success, elapsed_seconds, summary_line)
    """
    fst_files = list(deck_dir.glob("*.fst"))
    if len(fst_files) != 1:
        return False, 0.0, f"FAIL (no unique .fst in {deck_dir.name})"

    fst_path = fst_files[0]
    log_dir = deck_dir / "run_log"
    log_dir.mkdir(exist_ok=True)

    start = time.monotonic()
    with (log_dir / "stdout.log").open("w", encoding="utf-8") as out_fh, \
         (log_dir / "stderr.log").open("w", encoding="utf-8") as err_fh:
        proc = subprocess.run(
            [str(openfast_bin), fst_path.name],
            cwd=deck_dir,
            stdout=out_fh,
            stderr=err_fh,
            text=True,
            check=False,
        )
    elapsed = time.monotonic() - start

    (log_dir / "exit_code.txt").write_text(str(proc.returncode), encoding="utf-8")

    if proc.returncode == 0:
        # OpenFAST sometimes returns 0 on FATAL error too — check stdout.
        stdout = (log_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
        if "FATAL ERROR" in stdout or "Aborting OpenFAST" in stdout:
            return False, elapsed, f"FAIL  (FATAL in log, {elapsed:.1f}s)"
        return True, elapsed, f"OK    ({elapsed:.1f}s)"
    return False, elapsed, f"FAIL  (exit {proc.returncode}, {elapsed:.1f}s)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsp-root", type=Path, required=True)
    parser.add_argument(
        "--openfast-bin",
        type=Path,
        required=True,
        help="Path to fast_x64.exe (Windows) or openfast (Linux).",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Restrict to named scenarios (e.g. s1_static_eq). Default: all.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining scenarios after a failure.",
    )
    args = parser.parse_args()

    hsp_root = args.hsp_root.resolve()
    inputs_root = hsp_root / INPUTS_REL
    manifest_path = inputs_root / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", file=sys.stderr)
        print("Run generate_scenario_decks.py first.", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decks = manifest["scenarios"]

    if args.scenarios:
        decks = [d for d in decks if d["scenario_name"] in args.scenarios]
        if not decks:
            print(f"ERROR: no decks match --scenarios {args.scenarios}", file=sys.stderr)
            return 1

    print("=" * 70)
    print(f"Running {len(decks)} OpenFAST deck(s)")
    print("=" * 70)

    results: list[tuple[str, str, str, bool]] = []
    for entry in decks:
        deck_dir = (hsp_root / entry["deck_dir"]).resolve()
        # entry['deck_dir'] is relative to hsp_root/<INPUTS_REL>'s parent,
        # which is `tests/fixtures/openfast/oc4_deepcwind/`. Adjust:
        if not deck_dir.exists():
            # Fall back to manifest path semantics.
            deck_dir = (inputs_root.parent / entry["deck_dir"]).resolve()
        sweep_str = (
            f" [{entry['sweep_value']}]" if entry.get("sweep_value") is not None else ""
        )
        label = f"{entry['scenario_name']}{sweep_str}"
        print(f"\n[{label}]")
        print(f"  deck: {deck_dir.relative_to(hsp_root)}")

        success, elapsed, summary = run_one_deck(deck_dir, args.openfast_bin)
        print(f"  {summary}")
        results.append((label, summary, str(deck_dir), success))

        if not success and not args.continue_on_error:
            print("\nAborting due to failure. Use --continue-on-error to skip.")
            break

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    n_ok = sum(1 for _, _, _, ok in results if ok)
    n_fail = len(results) - n_ok
    for label, summary, deck, _ in results:
        print(f"  {label:30s}  {summary}")
    print(f"\n  {n_ok} OK, {n_fail} FAIL, {len(decks) - len(results)} not run")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
