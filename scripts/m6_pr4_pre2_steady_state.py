"""M6 PR4 Pre-2 -- S3 wave-generation steady-state verification.

Locked workflow (post-WaveMod=1 fix, 2026-05-06):

  - The Pre-2 GATE is on wave generation only: the
    wave-elevation fit must show clean monochromatic content at
    the configured WaveTp on all 14 sweep frequencies, with peak
    amplitude within 2 % of WaveHs/2.
  - The pitch-response transient at the natural frequency
    (T_n ≈ 26.8 s for OC4) is expected to persist at small
    amplitude even after 1200 s; this is handled at RAO
    extraction by the sinusoidal lstsq fit, NOT here.

Procedure (per scenario):
  - Load the OpenFAST CSV.
  - Take the last N_WAVE_PERIODS wave periods of the simulation.
  - Fit y(t) = A * cos(omega * t) + B * sin(omega * t) + C at the
    wave frequency via numpy.linalg.lstsq.
  - Wave amplitude = sqrt(A^2 + B^2); compare to WaveHs/2.
  - Fit-residual norm = ||y - fit||_2 / ||y||_2.

Verdict gate (per scenario):
  - PASS if |amp(wave_elev) - 0.5 m| / 0.5 < 0.02
    AND fit_residual_norm(wave_elev) < 0.05.
  - FAIL otherwise -- the regular-Airy generator is producing the
    wrong amplitude or contaminated wave train at that frequency.

Run from the repo root:
    python scripts/m6_pr4_pre2_steady_state.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

S3_INPUTS = REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/inputs/s3_rao_sweep"
DIAG_DIR = REPO_ROOT / "docs/diagnostics"
DIAG_CSV = DIAG_DIR / "m6-pr4-pre2-summary-table.csv"

# All 14 S3 sweep frequencies.
ALL_PERIODS_S = (
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    12.0,
    14.0,
    16.0,
    18.0,
    20.0,
    22.0,
    25.0,
    30.0,
)

# Configured wave height (set in scenario_config.py for s3_rao_sweep).
WAVE_HS_M = 1.0
EXPECTED_WAVE_AMP_M = WAVE_HS_M / 2.0

# Wave-generation faithfulness gates.
WAVE_AMP_RTOL = 2.0e-2
WAVE_FIT_RESIDUAL_GATE = 5.0e-2

# Steady-state extraction window: last N wave periods.
N_WAVE_PERIODS = 5


@dataclass(frozen=True)
class WaveCheckResult:
    """Per-scenario wave-generation diagnostic result."""

    wave_tp_s: float
    fitted_wave_amp_m: float
    amp_rel_err: float
    fit_residual_normalized: float
    is_clean: bool


def _wave_tmax_from_seastate(wave_tp_s: float) -> float:
    """Read WaveTMax from the per-scenario *_SeaState.dat (default 600 s)."""
    import re

    wt_str = f"{wave_tp_s:05.1f}".replace(".", "p")
    sea = S3_INPUTS / f"WaveTp_{wt_str}" / f"s3_rao_sweep_WaveTp_{wt_str}_SeaState.dat"
    if not sea.exists():
        return 600.0
    pattern = re.compile(r"^\s*([\d.eE+-]+)\s+WaveTMax\b")
    with sea.open() as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                return float(m.group(1))
    return 600.0


def _quantised_wave_period_s(wave_tp_s: float, wave_tmax_s: float) -> float:
    """Return the IFFT-quantised wave period that OpenFAST actually generates.

    OpenFAST's wave generator constructs a wave train via IFFT on a
    frequency grid with spacing ``WaveDOmega = 2*pi / WaveTMax``. The
    requested ``WaveTp`` is silently snapped to the nearest grid point
    ``omega_k = k * WaveDOmega``, with ``k = round(WaveTMax / WaveTp)``.
    For ``WaveTp`` values that don't evenly divide ``WaveTMax`` the
    actual wave period differs from the labelled one by up to half a
    bin-width ratio (~ 1.3 % for WaveTp=16 with WaveTMax=600). Using
    the labelled period for an lstsq fit produces a basis-frequency
    mismatch and inflates the residual.

    See conventions doc Item 21 (added 2026-05-06).
    """
    k = round(wave_tmax_s / wave_tp_s)
    return wave_tmax_s / k


def _load_csv(wave_tp_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Load (t, wave_elev) for the WaveTp scenario."""
    wt_str = f"{wave_tp_s:05.1f}".replace(".", "p")
    csv = S3_INPUTS / f"WaveTp_{wt_str}" / f"s3_rao_sweep_WaveTp_{wt_str}.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing S3 CSV: {csv}")
    data = np.genfromtxt(csv, delimiter=",", skip_header=1)
    return data[:, 0], data[:, 7]  # time_s, wave_elev_m (column 7 per .json schema)


def _lstsq_fit_at_omega(
    t: np.ndarray, x: np.ndarray, omega: float
) -> tuple[float, float, float, float]:
    """Fit x(t) = A*cos(omega t) + B*sin(omega t) + C; return amp, phase, residual_norm, mean.

    Residual norm is ||x - fit||_2 / ||x||_2 (pure-signal-relative; small
    values mean the harmonic at omega captures essentially all variance).
    """
    basis = np.column_stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)])
    coeffs, *_ = np.linalg.lstsq(basis, x, rcond=None)
    A, B, C = coeffs
    fit = basis @ coeffs
    amp = float(np.hypot(A, B))
    phase = float(np.arctan2(B, A))
    x_zm = x - float(np.mean(x))  # subtract DC for residual normalisation
    residual = float(np.linalg.norm(x - fit) / max(np.linalg.norm(x_zm), 1.0e-12))
    return amp, phase, residual, float(C)


def _check_wave_generation(wave_tp_s: float) -> WaveCheckResult:
    """Pre-2 wave-generation gate at one frequency.

    Fits the wave_elev channel at the OpenFAST IFFT-quantised wave
    period (NOT the labelled WaveTp) -- see conventions doc Item 21.
    For WaveTp values that don't evenly divide WaveTMax (=600s in
    the S3 sweep), the actual generated wave is at the nearest IFFT
    bin period; fitting at the labelled period would otherwise
    inflate the residual via basis-frequency mismatch.
    """
    t, we = _load_csv(wave_tp_s)
    wave_tmax_s = _wave_tmax_from_seastate(wave_tp_s)
    t_actual = _quantised_wave_period_s(wave_tp_s, wave_tmax_s)
    omega = 2.0 * np.pi / t_actual
    window_s = N_WAVE_PERIODS * t_actual
    mask = t >= t[-1] - window_s
    t_win = t[mask]
    we_win = we[mask]

    amp, _phase, residual_norm, _C = _lstsq_fit_at_omega(t_win, we_win, omega)
    rel_err = (amp - EXPECTED_WAVE_AMP_M) / EXPECTED_WAVE_AMP_M
    is_clean = abs(rel_err) < WAVE_AMP_RTOL and residual_norm < WAVE_FIT_RESIDUAL_GATE

    return WaveCheckResult(
        wave_tp_s=wave_tp_s,
        fitted_wave_amp_m=amp,
        amp_rel_err=rel_err,
        fit_residual_normalized=residual_norm,
        is_clean=is_clean,
    )


def main() -> None:
    print("M6 PR4 Pre-2 -- S3 wave-generation steady-state check (lstsq fit)")
    print("=" * 78)
    print(f"Reference: each scenario's last {N_WAVE_PERIODS} wave periods " f"of wave_elev channel")
    print(
        f"Gates: |amp - {EXPECTED_WAVE_AMP_M:.3f} m|/{EXPECTED_WAVE_AMP_M:.3f} < "
        f"{WAVE_AMP_RTOL:.2%}; fit residual / signal < {WAVE_FIT_RESIDUAL_GATE:.0%}"
    )
    print()
    print(
        f"{'WaveTp [s]':>10}  {'fitted amp [m]':>15}  {'amp rel-err':>12}  "
        f"{'residual':>10}  {'verdict':>8}"
    )
    print("-" * 78)

    all_results: list[WaveCheckResult] = []
    for wt in ALL_PERIODS_S:
        try:
            r = _check_wave_generation(wt)
        except FileNotFoundError as e:
            print(f"  {wt:>8.1f}  -- MISSING: {e}")
            continue
        verdict = "PASS" if r.is_clean else "FAIL"
        print(
            f"  {wt:>8.1f}  {r.fitted_wave_amp_m:>15.5f}  "
            f"{r.amp_rel_err:>+12.5f}  {r.fit_residual_normalized:>10.5f}  "
            f"{verdict:>8}"
        )
        all_results.append(r)

    # Persist a CSV summary for the Pre-2 doc.
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    with DIAG_CSV.open("w", encoding="utf-8") as fh:
        fh.write("wave_tp_s,fitted_wave_amp_m,amp_rel_err,fit_residual_normalized,is_clean\n")
        for r in all_results:
            fh.write(
                f"{r.wave_tp_s},{r.fitted_wave_amp_m},{r.amp_rel_err},"
                f"{r.fit_residual_normalized},{int(r.is_clean)}\n"
            )
    print(f"\nWrote summary: {DIAG_CSV.relative_to(REPO_ROOT)}")

    n_failed = sum(1 for r in all_results if not r.is_clean)
    if n_failed == 0:
        print(
            f"\nAll {len(all_results)} S3 wave-generation gates PASSED. "
            "Pre-2 closes; PR4 may use these fixtures."
        )
    else:
        print(f"\nFAIL: {n_failed} of {len(all_results)} scenarios failed wave-gen gate.")
        for r in all_results:
            if not r.is_clean:
                print(
                    f"  WaveTp={r.wave_tp_s:.0f}s: amp={r.fitted_wave_amp_m:.5f} m "
                    f"(rel-err {r.amp_rel_err:+.5f}), "
                    f"residual={r.fit_residual_normalized:.5f}"
                )


if __name__ == "__main__":
    main()
