"""Orchestrate the platform fin fan as a sequence of BOUNDED fresh subprocesses.

Why chunked fresh processes: the constrained (KKT) integrator's per-step
allocations -- joint-Jacobian evaluation, the saddle-point solve, position
projection, Morison drag -- churn the native allocator ~40 GB/case and retain
~2 GB/case through heap fragmentation that gc cannot reclaim (object counts stay
flat while RSS climbs). The retardation convolution buffer was initially
suspected but proven flat in isolation (both np.roll and an allocation-free
variant leave RSS unchanged over 40 k pushes) -- see tracker
CONSTRAINED-INTEGRATOR-SWEEP-MEMORY. At ~2 GB/case even one 55-case config OOMs a
single process (~64 GB box); the original all-in-one fan died at case ~96.

Each ``platform_fin_fan.py`` invocation runs up to FIN_MAX_NEW (default 12) NEW
cases for the first incomplete config, persists each as a per-case row JSON (the
resume unit), assembles a config's summary CSV once all 55 rows exist, and writes
the outstanding count to ``fin_study/_fin_remaining.txt``. This loop re-invokes a
fresh process (memory reclaimed by the OS on each exit) until that count is 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FAN = _HERE / "platform_fin_fan.py"
_REMAIN = _HERE / "fin_study" / "_fin_remaining.txt"
_MAX_ITERS = 300  # safety cap (>> 220 cases / 12 per chunk)


def _remaining() -> int:
    try:
        return int(_REMAIN.read_text().strip())
    except (OSError, ValueError):
        return -1


def main() -> None:
    for it in range(_MAX_ITERS):
        print(f"\n########## chunk {it} -> fresh process ##########", flush=True)
        r = subprocess.run([sys.executable, str(_FAN)])  # stdout/stderr stream through
        if r.returncode != 0:
            print(f"chunk {it} FAILED (rc={r.returncode}); stopping.", flush=True)
            sys.exit(r.returncode)
        rem = _remaining()
        print(f"[orchestrator] after chunk {it}: REMAINING {rem}", flush=True)
        if rem == 0:
            print("[orchestrator] all platform fin cases complete.", flush=True)
            return
        if rem < 0:
            print("[orchestrator] could not read remaining count; stopping.", flush=True)
            sys.exit(1)
    print(f"[orchestrator] hit iteration cap {_MAX_ITERS} with work remaining.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
