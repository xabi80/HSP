"""Single-DOF platform-heave RESPONSE wall effect at the true 2.7 m flume depth.

The coupled/articulated BEM runs at deep water (finite depth is impractically slow at the
coupled panel count). Free-decay is depth-robust, but the wave-EXCITATION wall effect is
depth-sensitive (a narrow shallow channel blocks long waves far more than deep water). This
script carries that piece: it reads the native-2.7 m single-array heave coefficients from
flume_wall_effect.py (sweep_results.npy: A33, B33, |F| for walls-out vs walls-in, all at
2.7 m -- the "He" DOF is every buoy heaving in unison = the platform heave mode) and forms a
linear single-DOF impedance RAO,

    RAO(w) = |F(w)| / | C - (M + A33(w)) w^2 + i w (B33(w) + Bv) | ,

for walls-out and walls-in, so the wall effect is the ratio at each period. C is the platform
heave stiffness (16 x single-buoy), M is calibrated to the ~2.6 s heave period on the walls-out
A33, and Bv is a linearised viscous term set to a representative zeta (the wall-effect RATIO is
insensitive to zeta -- near resonance it tracks the excitation ratio, off-resonance the F/C
ratio). This is a linear impedance model, NOT the full FloatSim solver; it isolates the
depth-sensitive response wall effect that the deep-water articulated run cannot see.

Writes floatsim_wall_rao.png + prints the per-period response wall effect. Run after
flume_wall_effect.py: python floatsim_wall_1dof.py
"""
# ruff: noqa: E702  -- compact plotting/print setup lines in a one-off analysis script.
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
C33_BUOY, N_BUOY, T_HEAVE, ZETA = 194.5, 16, 2.6, 0.10
TEAL, RED = "#0c8b96", "#b2432c"


def _series(rows, which, kind):
    """(T, value) arrays for coefficient `kind` ('A'|'B'|'F') of heave, from o|wl (walls-out|in)."""
    idx = 1 if which == "o" else 2          # row = (T, o, wl, od)
    T = np.array([r[0] for r in rows])
    v = np.array([r[idx][(kind, "He")] for r in rows])
    return T, v


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    rows = np.load(HERE / "sweep_results.npy", allow_pickle=True)
    T, A_o = _series(rows, "o", "A")
    _, B_o = _series(rows, "o", "B")
    _, F_o = _series(rows, "o", "F")
    _, A_w = _series(rows, "wl", "A")
    _, B_w = _series(rows, "wl", "B")
    _, F_w = _series(rows, "wl", "F")
    w = 2 * np.pi / T
    C = N_BUOY * C33_BUOY
    wn = 2 * np.pi / T_HEAVE
    A_n = np.interp(wn, w[::-1], A_o[::-1])          # walls-out A33 at the heave frequency
    M = C / wn**2 - A_n                              # calibrate effective mass to T_HEAVE
    Bv = 2 * ZETA * np.sqrt(C * (M + A_n))           # linearised viscous term for target zeta

    def rao(A, B, F):
        return np.abs(F) / np.abs(C - (M + A) * w**2 + 1j * w * (B + Bv))

    R_o, R_w = rao(A_o, B_o, F_o), rao(A_w, B_w, F_w)
    pct = 100 * (R_w / R_o - 1)

    print(f"Single-DOF platform-heave RESPONSE wall effect @ 2.7 m (M={M:.0f} kg, C={C:.0f} N/m, "
          f"zeta={ZETA:.0%}):")
    print(" T(s)  RAO_out  RAO_in   wall%")
    for i in range(len(T)):
        print(f"{T[i]:5.2f} {R_o[i]:8.4f} {R_w[i]:8.4f} {pct[i]:+7.1f}")
    band = (T >= 1.5) & (T <= 2.9)
    tail = T > 2.9
    print(f"\noperating band (1.5-2.9 s): |wall effect| <= {np.max(np.abs(pct[band])):.1f}%")
    print(f"long-period tail (T > 2.9 s): |wall effect| up to {np.max(np.abs(pct[tail])):.1f}%")

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(T, R_o, "o-", color=TEAL, lw=2, label="walls out (2.7 m)")
    ax.plot(T, R_w, "s--", color=RED, lw=2, label="walls in (2.7 m)")
    ax.axvspan(2.09, 2.29, color="0.85", alpha=0.7, zorder=0)     # first transverse cut-on band
    ax.text(2.19, ax.get_ylim()[1] * 0.96, "cut-on\n2.19 s", ha="center", va="top", fontsize=8,
            color="#8a6d1a")
    ax.set_xlabel("wave period T (s)"); ax.set_ylabel("platform-heave RAO (m/m)")
    ax.set_title("Single-DOF platform-heave RAO at 2.7 m — walls in vs out\n"
                 "(response wall effect; the depth-sensitive piece the deep coupled run misses)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "floatsim_wall_rao.png", dpi=130, bbox_inches="tight")
    print("\nwrote floatsim_wall_rao.png")


if __name__ == "__main__":
    main()
