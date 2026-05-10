"""M6 PR4 — F-WAVE-FORCE-CONV pre-investigation: |Im(F_exc)|/|F_exc|
across the 14-period sweep, with predicted TD amp error under the
sign-flip-on-Im hypothesis.

Per Group D addition of the post-PR4 disposition: predict which TD
amp assertions should fail under "make_regular_wave_force conjugates
Im of F_exc". Compare to empirical TD amp errors at WaveTp = 10 s
and 25 s (the only TD-evaluated periods in PR4).

Mechanism prediction (informal):

  WAMIT data F = Re_F + i*Im_F (in +i convention per WAMIT reader).
  make_regular_wave_force computes F(t) = Re[F * exp(-i omega t)]
                                = Re_F cos(wt) + Im_F sin(wt)
  Physically correct (under +i convention): F(t) = Re[F * exp(+i omega t)]
                                = Re_F cos(wt) - Im_F sin(wt)
  So F_used has Im flipped sign vs F_correct. The integrator response
  to F_used is the same physical magnitude as the response to
  conj(F_correct) = Re_F - i*Im_F. Hence under impedance Z, the FS
  TD response amplitude is:

      |xi_FS_TD| = |Z^-1 * (Re_F - i*Im_F)| = |Z^-1 * conj(F_correct)|

  The "correct" impedance amplitude:
      |xi_correct| = |Z^-1 * F_correct|

  The ratio:
      |xi_FS_TD| / |xi_correct| ≈ 1 when Im_F is small relative to Re_F
                                ≠ 1 when Im_F is large

So a small |Im(F)|/|F| ratio predicts the TD amp will agree with
impedance amp despite the convention bug. A large ratio predicts the
TD amp will diverge.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from floatsim.hydro.readers.wamit import (  # noqa: E402
    read_added_mass_and_damping,
    read_excitation_force,
)
from tests.support.rao_extraction import (  # noqa: E402
    quantised_wave_period_s,
    read_wave_tmax_from_seastate,
)
from tests.validation.test_m6_openfast_free_decay import _MARIN_SEMI as MARIN_PATH  # noqa: E402
from tests.validation.test_m6_openfast_regular_wave import (  # noqa: E402
    DOF_INDICES,
    WAVE_PERIODS_S,
    _wt_dir,
)

S3_INPUTS = REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/inputs/s3_rao_sweep"


def main() -> None:
    print("M6 PR4 -- |Im(F_exc)|/|F_exc| diagnostic for F-WAVE-FORCE-CONV")
    print("=" * 78)
    print()
    omega_grid, _A, _B, _A_inf = read_added_mass_and_damping(MARIN_PATH)
    _headings, F_exc = read_excitation_force(MARIN_PATH.parent / "marin_semi.3", omega=omega_grid)

    # Per-period at the IFFT-quantised omega. Heading 0 (h_idx=0).
    h_idx = 0
    print(
        f"{'WaveTp_lbl':>10}  {'omega_q':>8}  "
        + "".join(f"{'|Im/|F|| ' + dof:>16}" for dof in DOF_INDICES)
    )
    print("-" * 78)
    rows: list[dict[str, float]] = []
    for wave_tp_label in WAVE_PERIODS_S:
        deck_dir = _wt_dir(wave_tp_label)
        seastate = next(deck_dir.glob("*_SeaState.dat"))
        wave_tmax = read_wave_tmax_from_seastate(seastate)
        wave_tp_q = quantised_wave_period_s(wave_tp_label, wave_tmax)
        omega_q = 2.0 * np.pi / wave_tp_q
        ratios: dict[str, float] = {}
        line = f"{wave_tp_label:>9.1f}s  {omega_q:>8.4f}  "
        for dof_name, dof_idx in DOF_INDICES.items():
            re = float(np.interp(omega_q, omega_grid, F_exc[dof_idx, :, h_idx].real))
            im = float(np.interp(omega_q, omega_grid, F_exc[dof_idx, :, h_idx].imag))
            mag = (re**2 + im**2) ** 0.5
            ratio = abs(im) / max(mag, 1.0e-30)
            ratios[dof_name] = ratio
            line += f"{ratio:>16.4e}"
        print(line)
        rows.append({"WaveTp": wave_tp_label, **ratios})
    print()
    print("Empirical TD amp errors (post-F2 sweep, from")
    print("docs/diagnostics/m6-pr4-rao-sweep-results.md TD section):")
    print("  WaveTp=10s heave: TD amp +0.42% vs impedance (within 1% gate)")
    print("  WaveTp=10s pitch: TD amp -6.43% vs impedance (FAIL)")
    print("  WaveTp=25s heave: TD amp -0.77% vs impedance (within 1% gate)")
    print("  WaveTp=25s pitch: TD amp -5.93% vs impedance (FAIL)")
    print()
    print("Prediction check at WaveTp = 10 s and 25 s:")
    print(f"  10s heave |Im|/|F| = {next(r for r in rows if r['WaveTp'] == 10.0)['heave']:.3e}")
    print(f"  10s pitch |Im|/|F| = {next(r for r in rows if r['WaveTp'] == 10.0)['pitch']:.3e}")
    print(f"  25s heave |Im|/|F| = {next(r for r in rows if r['WaveTp'] == 25.0)['heave']:.3e}")
    print(f"  25s pitch |Im|/|F| = {next(r for r in rows if r['WaveTp'] == 25.0)['pitch']:.3e}")
    print()
    print("Hypothesis: |Im|/|F| << 1 -> TD amp ≈ impedance amp (passes gate);")
    print("            |Im|/|F| comparable to 1 -> TD amp differs (fails gate).")


if __name__ == "__main__":
    main()
