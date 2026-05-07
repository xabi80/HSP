"""M6 PR4 Pre-2 step 4 -- single-frequency smoke test on the WaveMod=1 fix.

After flipping `WaveMod` from 2 (JONSWAP) to 1 (regular Airy) in
``openfast_setup/scenario_config.py`` and regenerating the
``WaveTp_010p0`` deck (only), this script confirms the fix works
by FFT-verifying the OpenFAST output:

  - Wave-elevation FFT over the last 200 s must show a single
    peak at T = 10 s, with no other peaks above 5 % of that
    main peak's amplitude.
  - Pitch FFT over the last 200 s must show the same clean peak
    at T = 10 s.

If both pass, the WaveMod=1 path is verified and we can proceed
to regenerate the other 13 sweep variants. If either fails, the
regular-Airy code path has a configuration issue that must be
diagnosed before going further.

Run from the repo root:
    python scripts/m6_pr4_pre2_smoke_wavetp_10.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
S3_DECK = (
    REPO_ROOT
    / "tests/fixtures/openfast/oc4_deepcwind/inputs/s3_rao_sweep/WaveTp_010p0"
    / "s3_rao_sweep_WaveTp_010p0.outb"
)
DIAG_PNG = REPO_ROOT / "docs/diagnostics/m6-pr4-pre2-smoke-WaveTp_10p0.png"

WAVE_TP_S = 10.0
WINDOW_S = 200.0
SECONDARY_GATE = 5.0e-2  # secondary peaks must be < 5% of main peak


def _read_outb_via_openfast_io() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (t, wave_elev, pitch_rad) from the .outb file."""
    from openfast_io.FAST_output_reader import FASTOutputFile

    out = FASTOutputFile(str(S3_DECK))
    names = list(out.info["attribute_names"])
    units = list(out.info.get("attribute_units", ["?"] * len(names)))
    data = np.asarray(out.data, dtype=np.float64)
    print(f"  channels (n={len(names)}): first 12 = {names[:12]}")
    t_idx = names.index("Time")
    we_idx = names.index("Wave1Elev")
    pitch_idx = names.index("PtfmPitch")
    pitch_rad = (
        np.deg2rad(data[:, pitch_idx]) if "deg" in units[pitch_idx].lower() else data[:, pitch_idx]
    )
    return data[:, t_idx], data[:, we_idx], pitch_rad


def _spectral_peaks(signal_zm: np.ndarray, dt: float, top_n: int = 5) -> list[tuple[float, float]]:
    """Return [(period_s, amplitude)] of the top_n spectral peaks."""
    n = signal_zm.size
    freqs = np.fft.rfftfreq(n, dt)
    spec = np.abs(np.fft.rfft(signal_zm)) / n
    # Skip DC and Nyquist; find local maxima only.
    is_peak = np.zeros_like(spec, dtype=bool)
    is_peak[1:-1] = (spec[1:-1] > spec[:-2]) & (spec[1:-1] > spec[2:])
    is_peak[0] = False
    peak_idx = np.where(is_peak)[0]
    sorted_peaks = peak_idx[np.argsort(spec[peak_idx])[-top_n:][::-1]]
    return [(1.0 / freqs[k], float(spec[k])) for k in sorted_peaks if freqs[k] > 0]


def _check_clean_peak(
    label: str, t: np.ndarray, signal: np.ndarray, expected_T_s: float
) -> tuple[bool, list[tuple[float, float]]]:
    mask = t >= t[-1] - WINDOW_S
    t_w = t[mask]
    s_w = signal[mask] - float(np.mean(signal[mask]))
    dt = float(t_w[1] - t_w[0])
    peaks = _spectral_peaks(s_w, dt, top_n=5)
    if not peaks:
        print(f"  {label}: NO peaks found in the last 200 s")
        return False, []
    main_T, main_A = peaks[0]
    rel_T_err = abs(main_T - expected_T_s) / expected_T_s
    main_close = rel_T_err < 0.05
    secondaries_clean = all(amp / main_A < SECONDARY_GATE for _, amp in peaks[1:])
    print(
        f"  {label}: main peak at T={main_T:.4f} s "
        f"(expected {expected_T_s:.1f} s, rel err {rel_T_err:.4f}), "
        f"amplitude {main_A:.4e}"
    )
    for sub_T, sub_A in peaks[1:]:
        ratio = sub_A / main_A
        flag = "OK" if ratio < SECONDARY_GATE else "FAIL"
        print(
            f"    secondary at T={sub_T:.4f} s: A={sub_A:.4e} "
            f"({100 * ratio:.2f}% of main, {flag})"
        )
    return main_close and secondaries_clean, peaks


def main() -> None:
    print("M6 PR4 Pre-2 step 4 -- WaveMod=1 smoke test on WaveTp=10s")
    print("=" * 70)
    print(f"OpenFAST output: {S3_DECK.relative_to(REPO_ROOT)}")
    print()

    t, we, pitch = _read_outb_via_openfast_io()
    print(f"  TMax = {t[-1]:.1f} s, n_samples = {t.size}")
    print()

    print("Wave elevation FFT (last 200 s, top 5 peaks):")
    we_pass, _we_peaks = _check_clean_peak("wave_elev", t, we, WAVE_TP_S)
    print()
    print("Pitch FFT (last 200 s, top 5 peaks):")
    pitch_pass, _pitch_peaks = _check_clean_peak("pitch", t, pitch, WAVE_TP_S)
    print()

    # Diagnostic plot.
    mask = t >= t[-1] - WINDOW_S
    t_w = t[mask]
    we_w = we[mask] - float(np.mean(we[mask]))
    pitch_w = np.rad2deg(pitch[mask] - float(np.mean(pitch[mask])))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    ax_we_t, ax_we_f = axes[0]
    ax_p_t, ax_p_f = axes[1]

    # Time-domain.
    ax_we_t.plot(t_w, we_w, "k", lw=1.0)
    ax_we_t.set_title(f"Wave elevation, last {int(WINDOW_S)} s")
    ax_we_t.set_xlabel("t [s]")
    ax_we_t.set_ylabel("eta [m]")
    ax_we_t.grid(True, alpha=0.3)

    ax_p_t.plot(t_w, pitch_w, "k", lw=1.0)
    ax_p_t.set_title(f"Pitch, last {int(WINDOW_S)} s")
    ax_p_t.set_xlabel("t [s]")
    ax_p_t.set_ylabel("pitch [deg]")
    ax_p_t.grid(True, alpha=0.3)

    # Frequency-domain (period axis).
    dt = float(t_w[1] - t_w[0])
    freqs = np.fft.rfftfreq(t_w.size, dt)
    we_spec = np.abs(np.fft.rfft(we_w)) / t_w.size
    pitch_spec_deg = np.abs(np.fft.rfft(np.deg2rad(pitch_w))) / t_w.size  # rad
    nz = freqs > 0

    for ax, spec, title, ylabel in (
        (ax_we_f, we_spec, "Wave elev FFT vs period", "|X(T)|/N [m]"),
        (ax_p_f, pitch_spec_deg, "Pitch FFT vs period", "|X(T)|/N [rad]"),
    ):
        periods = 1.0 / freqs[nz]
        ax.semilogy(periods, spec[nz], "b", lw=1.0)
        ax.axvline(WAVE_TP_S, color="r", ls="--", lw=0.8, label=f"WaveTp={WAVE_TP_S} s")
        ax.set_xlim(2.0, 60.0)
        ax.set_title(title)
        ax.set_xlabel("T [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"S3 WaveMod=1 smoke test: WaveTp={WAVE_TP_S}s  --  "
        f"{'PASS' if (we_pass and pitch_pass) else 'FAIL'}"
    )
    fig.tight_layout()
    fig.savefig(DIAG_PNG, dpi=110)
    plt.close(fig)
    print(f"-> {DIAG_PNG.relative_to(REPO_ROOT)}")
    print()

    print(f"Verdict (gate: secondary peaks < {SECONDARY_GATE:.0%} of main peak):")
    print(f"  wave_elev: {'PASS' if we_pass else 'FAIL'}")
    print(f"  pitch:     {'PASS' if pitch_pass else 'FAIL'}")
    if we_pass and pitch_pass:
        print()
        print("SMOKE TEST PASSED. Proceed to step 5: regenerate the other 13 sweeps.")
    else:
        print()
        print("SMOKE TEST FAILED. Stop. Diagnose. Do NOT regenerate the other 13.")


if __name__ == "__main__":
    main()
