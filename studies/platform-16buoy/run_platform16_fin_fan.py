"""Orchestrate the 16-buoy platform fin fan as bounded fresh subprocesses (same
memory discipline as the 12-buoy runner: the constrained KKT integrator retains
~GB/case via native-heap fragmentation, so each chunk runs FIN_MAX_NEW new cases
then exits to reclaim RAM). Loops ``platform16_fin_fan.py`` until
``fin_study/_fin16_remaining.txt`` reads 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FAN = _HERE / "platform16_fin_fan.py"
_REMAIN = _HERE / "fin_study" / "_fin16_remaining.txt"
_MAX_ITERS = 400  # >> 275 cases / 8 per chunk


def _remaining() -> int:
    try:
        return int(_REMAIN.read_text().strip())
    except (OSError, ValueError):
        return -1


def main() -> None:
    for it in range(_MAX_ITERS):
        print(f"\n########## chunk {it} -> fresh process ##########", flush=True)
        r = subprocess.run([sys.executable, str(_FAN)])
        if r.returncode != 0:
            print(f"chunk {it} FAILED (rc={r.returncode}); stopping.", flush=True)
            sys.exit(r.returncode)
        rem = _remaining()
        print(f"[orchestrator] after chunk {it}: REMAINING {rem}", flush=True)
        if rem == 0:
            print("[orchestrator] all platform16 fin cases complete.", flush=True)
            return
        if rem < 0:
            print("[orchestrator] could not read remaining count; stopping.", flush=True)
            sys.exit(1)
    print(f"[orchestrator] hit iteration cap {_MAX_ITERS} with work remaining.", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
