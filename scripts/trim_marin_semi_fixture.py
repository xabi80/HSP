"""Trim the OpenFAST r-test marin_semi WAMIT files to a small test fixture.

The marin_semi WAMIT case from
``OpenFAST/r-test/glue-codes/openfast/5MW_Baseline/HydroData`` contains:

    marin_semi.1    503 KB    9000 lines    ~250 frequencies + Inf + Zero
    marin_semi.3   10  MB   110 KK lines    37 headings, same freq grid
    marin_semi.hst   1 KB     36 lines      6x6 hydrostatic stiffness

Only the ``.hst`` file is small enough to commit verbatim. We trim the
``.1`` and ``.3`` to:

  * Periods: ``-1`` (Inf row), ``0`` (Zero row, discarded by reader),
    plus three finite periods near the typical wave range (~5 s, ~8 s,
    ~12.5 s — chosen as the values nearest to those targets that
    actually appear in the file).
  * Headings (``.3`` only): ``0`` deg only.

Run from the repo root::

    python scripts/trim_marin_semi_fixture.py \\
        --src C:/path/to/openfast-r-test/glue-codes/openfast/5MW_Baseline/HydroData \\
        --dst tests/fixtures/bem/wamit

The script is idempotent and safe to re-run. It does not modify the
sources; only writes ``marin_semi_trimmed.{1,3,hst}`` under ``--dst``.

License of the *output* fixture follows the source — Apache 2.0
(OpenFAST/r-test) — and is documented in
``docs/wamit-fixture-attribution.md``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Periods we keep (in seconds). Chosen to span low / mid / high wave bands.
# The actual values written are the closest matches found in marin_semi.1 —
# WAMIT's frequency grid does not land on round numbers.
_TARGET_PERIODS = (5.0, 8.0, 12.5)
_KEEP_HEADING_DEG = 0.0


def _load_periods(path_dot_1: Path) -> list[float]:
    """Return sorted unique finite periods (excludes -1 and 0 sentinels)."""
    periods = set()
    with path_dot_1.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            try:
                p = float(tokens[0])
            except ValueError:
                continue
            if p > 0.0:
                periods.add(p)
    return sorted(periods)


def _select_periods(all_periods: list[float]) -> list[float]:
    chosen: list[float] = []
    for target in _TARGET_PERIODS:
        closest = min(all_periods, key=lambda p: abs(p - target))
        if closest not in chosen:
            chosen.append(closest)
    return chosen


def _trim_dot_1(src: Path, dst: Path, kept_periods: set[float]) -> None:
    """Write rows with PER in {-1, 0} ∪ kept_periods to dst."""
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for line in fi:
            tokens = line.split()
            if not tokens:
                fo.write(line)
                continue
            try:
                p = float(tokens[0])
            except ValueError:
                fo.write(line)
                continue
            if p == -1.0 or p == 0.0 or p in kept_periods:
                fo.write(line)


def _trim_dot_3(src: Path, dst: Path, kept_periods: set[float]) -> None:
    """Write rows with PER in kept_periods AND BETA == 0 to dst."""
    with src.open("r", encoding="utf-8") as fi, dst.open("w", encoding="utf-8") as fo:
        for line in fi:
            tokens = line.split()
            if not tokens:
                fo.write(line)
                continue
            try:
                p = float(tokens[0])
                beta = float(tokens[1])
            except (ValueError, IndexError):
                fo.write(line)
                continue
            if p in kept_periods and abs(beta - _KEEP_HEADING_DEG) < 1.0e-9:
                fo.write(line)


def _file_should_be_writable(p: Path) -> None:
    if p.is_dir():
        raise IsADirectoryError(f"Destination {p} is a directory; expected a file path")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source dir containing marin_semi.{1,3,hst} (OpenFAST r-test HydroData).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("tests/fixtures/bem/wamit"),
        help="Destination dir for trimmed marin_semi_trimmed.{1,3,hst}.",
    )
    args = parser.parse_args(argv)

    src: Path = args.src
    dst: Path = args.dst
    dst.mkdir(parents=True, exist_ok=True)

    src_1 = src / "marin_semi.1"
    src_3 = src / "marin_semi.3"
    src_hst = src / "marin_semi.hst"
    for p in (src_1, src_3, src_hst):
        if not p.is_file():
            raise FileNotFoundError(p)

    all_periods = _load_periods(src_1)
    chosen = _select_periods(all_periods)
    kept = set(chosen)
    print(f"  keeping finite periods: {chosen}")

    out_1 = dst / "marin_semi_trimmed.1"
    out_3 = dst / "marin_semi_trimmed.3"
    out_hst = dst / "marin_semi_trimmed.hst"
    for p in (out_1, out_3, out_hst):
        _file_should_be_writable(p)

    _trim_dot_1(src_1, out_1, kept)
    _trim_dot_3(src_3, out_3, kept)
    shutil.copyfile(src_hst, out_hst)

    print(f"  wrote {out_1} ({out_1.stat().st_size} bytes)")
    print(f"  wrote {out_3} ({out_3.stat().st_size} bytes)")
    print(f"  wrote {out_hst} ({out_hst.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
