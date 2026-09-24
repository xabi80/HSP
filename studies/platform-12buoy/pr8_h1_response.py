"""Why the M11b platform case H = 1.0 m, T = 3.141 s does not settle under relative-velocity drag
(window-to-window amplitude spread 11.7 % at the 508 s cap). Is the response non-stationary or
multi-periodic, which would break the adaptive-settle criterion's single-frequency assumption?

Input: the full run saved by ``pr8_reldrag_check.py 1.0 3.141 driver --history`` (FloatSim, relative
drag wired through build_system). Diagnostics on platform heave (and buoy pitch/roll):
  1. envelope: the single-frequency (omega) fit amplitude in consecutive windows -- exactly the
     settle criterion's measure -- over the whole run;
  2. spectrum of the settled part (Hann window): lines at omega, its harmonics/subharmonics, and
     any other line (the buoys' 2.9 s tilt mode, the 3.14 s heave mode, slow drift);
  3. stroboscopic section: the state sampled once per forcing period. A periodic response
     collapses to a point, a quasi-periodic one traces a closed curve, a chaotic one scatters;
  4. symmetry: head seas excite no roll/sway by symmetry; roll growing from round-off would be
     a symmetry-breaking (parametric) instability.

Writes pr8_reldrag_out/h1_t3141_response.{png,json}.  Run: python pr8_h1_response.py [stem]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "pr8_reldrag_out"


def _fit(t: np.ndarray, x: np.ndarray, w: float) -> tuple[float, float]:
    d = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(d, x, rcond=None)
    return float(np.hypot(c[0], c[1])), float(c[2])


def main() -> None:
    stem = sys.argv[1] if len(sys.argv) > 1 else "H1_T3p141_driver"
    h = np.load(OUT / f"history_{stem}.npz")
    t, w, H = h["t"], float(h["omega"]), float(h["H"])
    A, T = 0.5 * H, 2 * np.pi / float(h["omega"])
    z = h["platform"][:, 2]
    dt = float(t[1] - t[0])
    win = 6.0 * T
    # 1. envelope (settle-criterion measure) in consecutive windows after the 20 s ramp
    edges = np.arange(20.0, t[-1] - win + 1e-9, win)
    env = [(e + win / 2, _fit(t[(t >= e) & (t < e + win)], z[(t >= e) & (t < e + win)], w)[0] / A)
           for e in edges]
    # 2. spectrum of the settled part (last 60 %)
    m = t >= t[-1] - 0.6 * (t[-1] - 20.0)
    zz = z[m] - z[m].mean()
    win_h = np.hanning(zz.size)
    spec = np.abs(np.fft.rfft(zz * win_h)) * 2 / win_h.sum()
    freq = 2 * np.pi * np.fft.rfftfreq(zz.size, dt)
    order = np.argsort(spec)[::-1]
    lines = []
    for i in order:
        if spec[i] < 0.01 * spec.max() or len(lines) >= 8:
            break
        if all(abs(freq[i] - f) > 0.02 for f, _ in lines):
            lines.append((float(freq[i]), float(spec[i])))
    # 3. stroboscopic section (sampled at the forcing phase, after the ramp)
    ts = np.arange(20.0 + T, t[-1], T)
    idx = np.clip(np.searchsorted(t, ts), 0, t.size - 1)
    vz = np.gradient(z, dt)
    strob = np.column_stack([z[idx], vz[idx]])
    late = strob[len(strob) // 2:]
    spread_strobe = float(np.hypot(*(late.max(0) - late.min(0))) / (A * max(1.0, w)))
    # 4. symmetry-breaking: roll / sway growth
    roll = h["buoy_roll"]
    sway = h["platform"][:, 1]
    early = (t > 20) & (t < 60)
    rec = {"H": H, "T": T, "duration_s": float(t[-1]),
           "envelope_rao": [[round(a, 1), round(b, 4)] for a, b in env],
           "envelope_last_spread_pct": float(100 * (max(e[1] for e in env[-6:])
                                                    - min(e[1] for e in env[-6:]))
                                             / np.mean([e[1] for e in env[-6:]])),
           "spectral_lines_rad_s": [[round(f, 4), round(a / A, 4), round(f / w, 4)]
                                    for f, a in lines],
           "strobe_late_extent_norm": spread_strobe,
           "max_buoy_roll_deg_early": float(np.degrees(np.abs(roll[early]).max())),
           "max_buoy_roll_deg_late": float(np.degrees(np.abs(roll[m]).max())),
           "max_platform_sway_m": float(np.abs(sway).max())}
    (OUT / "h1_t3141_response.json").write_text(json.dumps(rec, indent=1))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    ax[0, 0].plot(t, z / A, lw=0.5, color="#0c6d76")
    ax[0, 0].plot([e[0] for e in env], [e[1] for e in env], "o-", color="#c0562b", ms=3,
                  label="single-frequency fit per 6T window (settle measure)")
    ax[0, 0].set(xlabel="t (s)", ylabel="platform heave / A", title=f"H {H} m, T {T:.3f} s")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].semilogy(freq / w, spec / A, lw=0.8)
    ax[0, 1].set(xlim=(0, 4), xlabel="frequency / forcing frequency", ylabel="|heave| / A",
                 title="spectrum, settled part")
    for f in (0.5, 1, 2, 3):
        ax[0, 1].axvline(f, color="0.7", lw=0.6, ls=":")
    ax[1, 0].plot(strob[:, 0] / A, strob[:, 1] / (A * w), ".", ms=3, color="0.5", label="all")
    ax[1, 0].plot(late[:, 0] / A, late[:, 1] / (A * w), ".", ms=5, color="#c0562b",
                  label="second half")
    ax[1, 0].set(xlabel="heave / A", ylabel="heave rate / (A w)",
                 title="stroboscopic section (once per forcing period)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, np.degrees(roll), lw=0.4)
    ax[1, 1].set(xlabel="t (s)", ylabel="buoy roll (deg)", title="symmetry: buoy roll, head seas")
    fig.tight_layout()
    fig.savefig(OUT / "h1_t3141_response.png", dpi=120)
    print(json.dumps({k: v for k, v in rec.items() if k != "envelope_rao"}, indent=1))


if __name__ == "__main__":
    main()
