"""M6 PR3 Mod 2 -- damping stability diagnostic.

Locked workflow: before pinning the OpenFAST pitch zeta as the cross-
check assertion target, verify the value is stable across the response
envelope. Compute log-decrement zeta on peaks 1-5, 5-10, 10-20 of the
regenerated S2 reference (post-Mod-1, with PtfmSurge=0 IC). If the
three windows agree to within rtol=5e-2, the damping is well-defined
and the value can be locked. If they disagree by more than 5%, there
is secondary-mode contamination or nonlinearity; pause PR3 and
investigate.

Run from the repo root:
    python scripts/m6_pr3_mod2_damping_stability.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
S2_CSV = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "openfast"
    / "oc4_deepcwind"
    / "inputs"
    / "s2_pitch_decay"
    / "s2_pitch_decay.csv"
)


def _zeta_log_decrement(peaks: np.ndarray, window: tuple[int, int]) -> float:
    """Log-decrement zeta from peak indices [start, end] inclusive.

    delta = (1 / N) * ln(p_start / p_end), N = end - start
    zeta  = delta / sqrt(delta^2 + 4 pi^2)
    """
    start, end = window
    n = end - start
    if n <= 0:
        raise ValueError(f"empty window {window}")
    if end >= peaks.size:
        raise ValueError(f"need at least {end + 1} peaks; got {peaks.size}")
    delta = float(np.log(peaks[start] / peaks[end]) / n)
    return float(delta / np.sqrt(delta * delta + 4.0 * np.pi * np.pi))


def main() -> None:
    print("M6 PR3 Mod 2 -- pitch damping stability across the envelope")
    print("=" * 64)
    print(f"Reference: {S2_CSV.relative_to(REPO_ROOT)}")
    print()

    data = np.genfromtxt(S2_CSV, delimiter=",", skip_header=1)
    t = data[:, 0]
    pitch = data[:, 5]  # pitch_rad column

    # Subtract last-60-s mean to remove the deck residual offset.
    pitch_eq = float(np.mean(pitch[t >= t[-1] - 60.0]))
    pitch_zm = pitch - pitch_eq
    print(
        f"Pitch settles to: {np.degrees(pitch_eq):.4f} deg "
        f"(deck-residual offset, subtracted before fitting)"
    )

    # Positive peaks of the AC pitch envelope.
    is_peak = (
        (pitch_zm[1:-1] > pitch_zm[:-2]) & (pitch_zm[1:-1] > pitch_zm[2:]) & (pitch_zm[1:-1] > 0)
    )
    peak_idx = np.where(is_peak)[0] + 1
    peaks = pitch_zm[peak_idx]
    print(f"Total positive peaks: {peaks.size}")
    print()
    print(f"{'i':>3}  {'t [s]':>10}  {'peak [deg]':>11}")
    for i in range(min(peaks.size, 22)):
        print(f"{i:>3}  {t[peak_idx[i]]:>10.3f}  {np.degrees(peaks[i]):>11.5f}")

    print()
    windows = [
        ("peaks 1-5", (0, 5)),
        ("peaks 5-10", (5, 10)),
        ("peaks 10-20", (10, 20)),
    ]
    print(f"{'window':>14}  {'N cycles':>9}  {'zeta':>10}")
    print("-" * 64)
    zetas: list[float] = []
    for label, (start, end) in windows:
        try:
            z = _zeta_log_decrement(peaks, (start, end))
            print(f"{label:>14}  {end - start:>9d}  {z:>10.5f}")
            zetas.append(z)
        except ValueError as e:
            print(f"{label:>14}  -- {e}")
            zetas.append(float("nan"))

    if any(np.isnan(z) for z in zetas):
        print()
        print("CANNOT EVALUATE: not enough peaks for the requested windows.")
        return

    print()
    z_min, z_max = float(min(zetas)), float(max(zetas))
    z_mean = float(np.mean(zetas))
    spread = (z_max - z_min) / abs(z_mean)
    print(f"min zeta: {z_min:.5f}, max zeta: {z_max:.5f}, mean: {z_mean:.5f}")
    print(f"relative spread (max-min)/mean: {spread:.4f}")

    rtol = 5.0e-2
    print()
    if spread < rtol:
        print(f"PASS: zeta is stable across the envelope (spread {spread:.3f} < rtol {rtol:.3f}).")
        print(f"      LOCK assertion target zeta = {z_mean:.5f} (mean of three windows).")
        print(f"      OR  use peaks 1-5 (matches PR3 plan default): zeta = {zetas[0]:.5f}.")
    else:
        print(f"FAIL: zeta varies more than {rtol:.0%} across windows ({spread:.4f}).")
        print("      Possible causes: secondary mode contamination, nonlinearity, or")
        print("      heave-pitch coupling pumping energy in/out of the AC envelope.")
        print("      PAUSE PR3 -- investigate before locking the assertion target.")


if __name__ == "__main__":
    main()
