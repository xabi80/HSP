"""REV A (WITHDRAWN 2026-09-25; rev B's runs are attachment_sweep.py extremes). Kept as the record.

Targeted worst-case runs for MOORING-SPEC.md (2026-09-25): cluster and platform with their
pretension raised over the Phase C design (line stiffness k kept), H = 0.5 m at T = 2.35 and
2.65 s, the CONSERVATIVE drift sum (the full recorded bound applied in-run + FloatSim's own mean
force), via line_hardware.extreme_run.

  python run_spec_extremes.py          T0 x1.2 (the instructed +20 %)  -> spec_extremes.json
  python run_spec_extremes.py scan     T0 x1.4 and x1.6 (+20 % failed) -> spec_extremes_t0scan.json
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import line_hardware as lh


def cases(scales: tuple[float, ...]) -> list[tuple]:
    return [(a, 0.5, T, True, sc) for sc in scales for a in ("cluster", "platform")
            for T in (2.35, 2.65)]


if __name__ == "__main__":
    scan = len(sys.argv) > 1 and sys.argv[1] == "scan"
    todo = cases((1.4, 1.6) if scan else (1.2,))
    with ProcessPoolExecutor(max_workers=len(todo)) as ex:
        rows = list(ex.map(lh.extreme_run, todo))
    out = HERE / ("spec_extremes_t0scan.json" if scan else "spec_extremes.json")
    out.write_text(json.dumps(rows, indent=1, default=float))
