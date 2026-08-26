"""Final headline figure for the pin-vs-rigid study: deck tilt (the verdict
metric) with the drag-limited time-domain overlaid on the FD map, plus the
connection-moment cost and the (identical) heave response.

Reads pvr_fd_summary.csv and (if present) pvr_td_summary.csv. Writes
pvr_verdict.png next to this script.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
CA, CR = "#0c8b96", "#c0392b"  # articulated (pin) teal, rigid (weld) red


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _col(rows: list[dict], cfg: str, x: str, y: str):
    r = [(float(d[x]), float(d[y])) for d in rows if d["config"] == cfg]
    r.sort()
    return np.array([a for a, _ in r]), np.array([b for _, b in r])


def main() -> None:
    fd = _read(_HERE / "pvr_fd_summary.csv")
    td = _read(_HERE / "pvr_td_summary.csv")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

    # (1) deck tilt — FD map (radiation-only) + TD drag-limited points
    for cfg, col, lab in [("artic", CA, "articulated (pin)"), ("rigid", CR, "rigid (weld)")]:
        Tf, pf = _col(fd, cfg, "T_s", "pitch_RAO_radpm")
        ax[0].plot(Tf, pf * 1e3, "-", color=col, lw=1.6, alpha=0.55,
                   label=f"{lab} — FD (rad-only, upper bound)")
        if td:
            Tt, pt = _col(td, cfg, "T", "pitch_RAO")
            ax[0].plot(Tt, pt * 1e3, "o", color=col, ms=8, mec="k", mew=0.6,
                       label=f"{lab} — drag-limited (real)")
    ax[0].set_yscale("log")
    ax[0].set_title("Deck TILT — platform pitch RAO\n(the rocket / datacenter metric)", fontsize=11)
    ax[0].set_ylabel("|pitch| (mrad per m wave-amp)")

    # (2) connection moment — the weld's structural cost (FD)
    for cfg, col, lab in [("artic", CA, "articulated (pin)"), ("rigid", CR, "rigid (weld)")]:
        Tf, mf = _col(fd, cfg, "T_s", "Mjoint_buoyhub_Nm")
        ax[1].plot(Tf, mf, "-o", ms=3, color=col, label=lab)
    ax[1].set_title("Connection MOMENT at buoy→hub\n(pin carries ~0 by construction)", fontsize=11)
    ax[1].set_ylabel("max |joint moment| (N·m per m wave-amp)")

    # (3) heave — identical (connection type doesn't change vertical motion)
    for cfg, col, lab in [("artic", CA, "articulated (pin)"), ("rigid", CR, "rigid (weld)")]:
        Tf, hf = _col(fd, cfg, "T_s", "heave_RAO")
        ax[2].plot(Tf, hf, "-", color=col, lw=2.2 if cfg == "artic" else 1.3,
                   alpha=0.9 if cfg == "artic" else 1.0, label=lab)
    ax[2].set_title("Platform HEAVE RAO\n(identical — connection irrelevant to heave)", fontsize=11)
    ax[2].set_ylabel("|heave| (m per m wave-amp)")

    for a in ax:
        a.set_xlabel("wave period T (s)")
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8)
    fig.suptitle("Articulated (pin) vs whole-chain-rigid 16-buoy platform — for a still, level "
                 "deck the PIN wins: same heave, far less tilt, ~no connection moment",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_HERE / "pvr_verdict.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {_HERE / 'pvr_verdict.png'} (fd={len(fd)} rows, td={len(td)} rows)")


if __name__ == "__main__":
    main()
