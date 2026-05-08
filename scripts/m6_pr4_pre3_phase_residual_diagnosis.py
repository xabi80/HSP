"""M6 PR4 Pre-3 -- WaveTp=10s phase-residual diagnosis.

Pre-3 dual-path verification (post-WAMIT-dimensionalisation-fix)
agreed on amplitude (0.1% rel-err) but disagreed on phase by 12.7°
at WaveTp=10s. At 25s the phase agreement is 0.42° (well within
the 1° gate). This script splits the steady-state window into
EARLY and LATE sub-windows and fits the heave response separately
on each, to discriminate two theories:

  Theory (a) -- transient bleed: heave free-decay mode at T_n=17s
    has not fully decayed in the last 50s window of a 600s run;
    coherent contamination shifts lstsq phase. Diagnostic
    signature: phase moves between EARLY and LATE sub-windows.
    Fix: widen the steady-state window further in time
    (longer simulation, or skip more transient).

  Theory (b) -- static convention issue: WAMIT phase reference
    point or sign convention disagrees with OpenFAST in a way
    that's amplitude-invariant. Diagnostic signature: phase is
    constant across EARLY and LATE sub-windows but offset from
    Path A. Fix: investigate the WAMIT reader.

Output: docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.support.openfast_csv import load_openfast_history  # noqa: E402
from tests.support.rao_extraction import (  # noqa: E402
    lstsq_fit_at_omega,
    quantised_wave_period_s,
    read_wave_tmax_from_seastate,
)

S3_INPUTS = REPO_ROOT / "tests/fixtures/openfast/oc4_deepcwind/inputs/s3_rao_sweep"
DECK_DIR = S3_INPUTS / "WaveTp_010p0"
OUT_PNG = REPO_ROOT / "docs/diagnostics/m6-pr4-pre3-phase-residual-diagnosis.png"

# Path A's reported phase at this freq (from previous Pre-3 run, post-fix).
PATH_A_PHASE_DEG = -6.1735


def main() -> None:
    csv = next(DECK_DIR.glob("*.csv"))
    history = load_openfast_history(csv)
    seastate = next(DECK_DIR.glob("*_SeaState.dat"))
    wave_tmax = read_wave_tmax_from_seastate(seastate)
    wave_tp_label = 10.0
    t_q = quantised_wave_period_s(wave_tp_label, wave_tmax)
    omega_q = 2.0 * np.pi / t_q

    t = history.t
    heave = history.xi[:, 2]
    wave = history.extra_columns["wave_elev_m"]

    print("M6 PR4 Pre-3 -- WaveTp=10s phase-residual diagnosis")
    print("=" * 72)
    print()
    print(f"OpenFAST sim duration: {t[0]:.1f} -- {t[-1]:.1f} s ({t[-1]-t[0]:.1f} s)")
    print(f"WaveTp_label = {wave_tp_label} s, WaveTMax = {wave_tmax:.1f} s")
    print(f"Quantised wave period = {t_q:.6f} s, omega = {omega_q:.6f} rad/s")
    print(f"Path A reported phase (from impedance): {PATH_A_PHASE_DEG:+.4f} deg")
    print()

    # Slide a 50-second (5-period) window through the simulation. Skip
    # the initial 60 s (ramp + first ~3.5 transient cycles).
    n_periods_per_window = 5
    window_s = n_periods_per_window * t_q
    skip_initial_s = 60.0
    sim_end = float(t[-1])
    starts = np.arange(skip_initial_s, sim_end - window_s, window_s)

    rows = []
    for t_start in starts:
        mask = (t >= t_start) & (t < t_start + window_s)
        if mask.sum() < 50:
            continue
        t_w = t[mask]
        heave_w = heave[mask]
        wave_w = wave[mask]
        fit_h = lstsq_fit_at_omega(t_w, heave_w, omega_q)
        fit_w = lstsq_fit_at_omega(t_w, wave_w, omega_q)
        # RAO phase is response - wave_elev (circular).
        diff = fit_h.phase_rad - fit_w.phase_rad
        diff = ((diff + np.pi) % (2.0 * np.pi)) - np.pi
        rao_phase_deg = float(np.rad2deg(diff))
        rao_amp = fit_h.amplitude / fit_w.amplitude
        rows.append(
            {
                "t_center_s": float(t_start + window_s / 2.0),
                "rao_amp": rao_amp,
                "rao_phase_deg": rao_phase_deg,
                "resp_resid": fit_h.fit_residual_normalized,
                "wave_resid": fit_w.fit_residual_normalized,
            }
        )

    print(f"{'t_center':>9}  {'amp':>8}  {'phase_deg':>10}  {'resp_resid':>10}  {'wave_resid':>10}")
    print("-" * 60)
    for r in rows:
        print(
            f"  {r['t_center_s']:>7.1f}  {r['rao_amp']:>8.5f}  "
            f"{r['rao_phase_deg']:>+10.4f}  {r['resp_resid']:>10.5f}  {r['wave_resid']:>10.5f}"
        )

    # Compare EARLY (first window after skip) vs LATE (last window).
    early = rows[0]
    late = rows[-1]
    phase_drift_deg = late["rao_phase_deg"] - early["rao_phase_deg"]
    print()
    print(f"EARLY (t_center={early['t_center_s']:.1f}s): phase = {early['rao_phase_deg']:+.4f} deg")
    print(f"LATE  (t_center={late['t_center_s']:.1f}s): phase = {late['rao_phase_deg']:+.4f} deg")
    print(f"Phase drift EARLY -> LATE: {phase_drift_deg:+.4f} deg")
    print()
    print(f"Path A (impedance) phase: {PATH_A_PHASE_DEG:+.4f} deg")
    print(f"Path B EARLY - Path A:    {early['rao_phase_deg'] - PATH_A_PHASE_DEG:+.4f} deg")
    print(f"Path B LATE  - Path A:    {late['rao_phase_deg'] - PATH_A_PHASE_DEG:+.4f} deg")
    print()
    if abs(phase_drift_deg) > 2.0:
        print("VERDICT: phase drifts by >2 deg between EARLY and LATE sub-windows.")
        print("  Theory (a) supported: transient bleed from heave free-decay (T_n=17s)")
        print("  contaminates the lstsq phase. Fix: widen extractor window further.")
    elif (
        abs(early["rao_phase_deg"] - PATH_A_PHASE_DEG) > 2.0
        and abs(late["rao_phase_deg"] - PATH_A_PHASE_DEG) > 2.0
    ):
        print("VERDICT: phase is approximately constant across EARLY and LATE,")
        print("  but persistently offset from Path A by >2 deg. Theory (b)")
        print("  supported: static convention issue. Investigate WAMIT phase")
        print("  reference / sign convention.")
    else:
        print("VERDICT: ambiguous -- phase drift small AND offset from Path A")
        print("  small. The 12.7-deg gap on the canonical 50-s window may be")
        print("  an artefact of where in the simulation the canonical window")
        print("  lands.")

    # Plot.
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Top: full heave time series.
    axes[0].plot(t, heave, lw=0.8, label="heave_m (OpenFAST)")
    axes[0].axvspan(skip_initial_s, sim_end, alpha=0.05, color="grey")
    early_start = early["t_center_s"] - window_s / 2.0
    late_start = late["t_center_s"] - window_s / 2.0
    axes[0].axvspan(
        early_start, early_start + window_s, alpha=0.3, color="C1", label="EARLY window"
    )
    axes[0].axvspan(late_start, late_start + window_s, alpha=0.3, color="C2", label="LATE window")
    axes[0].set_ylabel("heave (m)")
    axes[0].set_title(f"WaveTp = {wave_tp_label} s (T_q = {t_q:.4f} s) -- heave time series")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Middle: phase vs window center.
    centers = [r["t_center_s"] for r in rows]
    phases = [r["rao_phase_deg"] for r in rows]
    axes[1].plot(centers, phases, "o-", color="C0", label="Path B (lstsq)")
    axes[1].axhline(
        PATH_A_PHASE_DEG, color="C3", ls="--", label=f"Path A = {PATH_A_PHASE_DEG:+.2f} deg"
    )
    axes[1].axhline(0.0, color="k", lw=0.5, alpha=0.3)
    axes[1].set_ylabel("RAO phase (deg)")
    axes[1].set_title("Heave RAO phase vs sliding-window center")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)

    # Bottom: response residual vs window center.
    resids = [r["resp_resid"] for r in rows]
    axes[2].plot(centers, resids, "s-", color="C4", label="resp_resid (||x-fit||/||x-mean||)")
    axes[2].set_ylabel("response fit residual")
    axes[2].set_xlabel("window center time (s)")
    axes[2].set_title("Heave-response sinusoidal-fit residual")
    axes[2].set_yscale("log")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].grid(alpha=0.3, which="both")

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150)
    print()
    print(f"Plot saved to {OUT_PNG}")


if __name__ == "__main__":
    main()
