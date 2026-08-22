"""Pre-test heave free-decay PREDICTION for the OSU Test Buoy.

Splits the decay into the part we can predict and the part the test measures:

* PERIOD is predictable — T = 2π·√((M + A33) / C33). M = 21.52 kg and
  C33 = 194.5 N/m are pinned by the measured mass/waterline and the 6" pipe
  diameter; only the heave-plate added mass A33 is uncertain, and it is bounded
  (spar-only ~1.5 kg .. near-solid disc ~8.5 kg). Best estimate T ≈ 2.3–2.4 s.
* DAMPING is the unknown the test exists to measure — quadratic (Morison) drag on
  the perforated/webbed heave plate, so it is amplitude-dependent (curved decay
  envelope, large log-decrement on the first swing). Cd of an open frame is not
  something potential-flow BEM can give, hence the light/central/heavy band.

Writes OSU_heave_decay_prediction.png next to this script. See
OSU-TEST-BUOY-GEOMETRY.md ("Heave decay prediction (pre-test)").
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

# Knowns (measured geometry + spreadsheet mass + waterline):
M = 21.52       # kg total floating mass
C33 = 194.5     # N/m waterplane heave stiffness (= rho*g*A_wp)
X0 = 0.10       # m release amplitude (test)


def period(a33: float) -> float:
    """Heave natural period (s) with plate added mass ``a33`` (kg)."""
    return 2 * np.pi * np.sqrt((M + a33) / C33)


def simulate(a33: float, bq: float, c_lin: float, t_end: float = 25.0, dt: float = 0.002):
    """SDOF free-decay: quadratic plate drag ``bq`` + small linear radiation ``c_lin``."""
    m = M + a33
    n = int(t_end / dt)
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = X0
    for i in range(n - 1):
        a = (-C33 * x[i] - bq * abs(v[i]) * v[i] - c_lin * v[i]) / m
        v[i + 1] = v[i] + a * dt
        x[i + 1] = x[i] + v[i + 1] * dt
    return np.arange(n) * dt, x


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    # Added-mass scenarios (the only period unknown):
    scen = {
        "spar-only (plate transparent)": 1.5,
        "open/webbed ~40% solid": 3.5,
        "MOST LIKELY (~60% solid)": 5.5,
        "near-solid disc (upper bound)": 8.5,
    }
    print("HEAVE PERIOD vs plate added mass:")
    for k, a in scen.items():
        print(f"  A33={a:4.1f} kg -> T = {period(a):.2f} s   ({k})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # left: period band
    aa = np.linspace(1, 9, 50)
    ax1.plot(aa, [period(a) for a in aa], "-", color="#0c8b96", lw=2.5)
    for k, a in scen.items():
        ax1.plot(a, period(a), "o", ms=9)
        ax1.annotate(f"{period(a):.2f}s", (a, period(a)),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax1.axhspan(period(3.5), period(8.5), alpha=0.12, color="#0c8b96")
    ax1.axhline(period(5.5), ls="--", color="#d1543a", lw=1.5)
    ax1.set_xlabel("heave plate added mass A33 (kg)")
    ax1.set_ylabel("heave natural period T (s)")
    ax1.set_title("PERIOD is predictable: T = 2π√((M+A33)/C33)\n"
                  "M=21.5 kg, C33=194.5 N/m known; only A33 uncertain", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.text(1.2, period(8.5) + 0.01, "most-likely band 2.3–2.5 s", color="#0c8b96", fontsize=9)

    # right: predicted decay (central A33=5.5) for a damping band
    for bq, c_lin, lab, col in [
        (70, 3, "light damping (ζ₁~6%)", "#8fbfc6"),
        (130, 5, "central (ζ₁~10%)", "#0c8b96"),
        (220, 8, "heavy (perforated jets, ζ₁~16%)", "#0a5560"),
    ]:
        t, x = simulate(5.5, bq, c_lin)
        ax2.plot(t, x * 1000, color=col, lw=1.8, label=lab)
    ax2.set_xlim(0, 18)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("heave (mm)")
    ax2.set_title("DECAY is the unknown the test measures\n"
                  "central A33=5.5 (T=2.34 s); damping band from the perforated plate", fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.axhline(0, color="k", lw=0.5)
    fig.suptitle("OSU buoy — heave free-decay PREDICTION (release 100 mm)  |  "
                 "most likely: T ≈ 2.3–2.4 s, ζ₁ ≈ 8–15% (amplitude-dependent)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "OSU_heave_decay_prediction.png"
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"\nwrote {out}")
    _ = find_peaks  # (available for interactive zeta checks)


if __name__ == "__main__":
    main()
