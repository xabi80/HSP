"""Ordered 16-buoy fin-fan runner: process fins in the order their BEM databases
finish (none, then 0.15, then 0.215), so the fan can start on the ready fins
while the slowest BEM is still solving. For each fin it waits (polling) for the
BEM .nc, then loops ``platform16_fin_fan.py <tag>`` (fresh subprocess per chunk,
memory reclaimed on exit) until that fin's cases are all done. Single process ->
no race on the shared remaining-count file.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FAN = _HERE / "platform16_fin_fan.py"
_STUDY = _HERE / "fin_study"
_ORDER = ["none", "015", "0215"]  # fastest BEM first
_MAX_ITERS = 400


def main() -> None:
    for tag in _ORDER:
        nc = _STUDY / f"capytaine_platform16_fin{tag}.nc"
        waited = 0
        while not nc.exists():
            if waited % 600 == 0:
                print(f"[{tag}] waiting for {nc.name} (BEM still solving)...", flush=True)
            time.sleep(60)
            waited += 60
        print(f"\n===== FAN fin {tag} (BEM ready: {nc.stat().st_size / 1e6:.1f} MB) =====", flush=True)
        for it in range(_MAX_ITERS):
            r = subprocess.run([sys.executable, str(_FAN), tag])
            if r.returncode != 0:
                print(f"[{tag}] chunk {it} FAILED rc={r.returncode}; stopping.", flush=True)
                sys.exit(r.returncode)
            rem = int((_STUDY / "_fin16_remaining.txt").read_text().strip())
            print(f"[{tag}] after chunk {it}: REMAINING {rem}", flush=True)
            if rem == 0:
                break
        else:
            print(f"[{tag}] hit iteration cap with work remaining.", flush=True)
            sys.exit(1)
    print("\nALL platform16 fin cases complete.", flush=True)


if __name__ == "__main__":
    main()
