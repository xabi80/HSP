"""Pre-flight 2 diagnostic: characterise B_heave high-omega behavior.

Three cases to discriminate:
  (a) B_heave monotonically decreasing with clean 1/omega^4 asymptote
      -- surge and heave are physically different at high omega;
      truncation works.
  (b) B_heave noisy at relative magnitudes similar to B_surge but at
      smaller absolute values (numerical-floor noise) -- gate
      coverage gap; truncation also works.
  (c) Something else -- STOP.

Outputs:
  - results/figures/B_heave_high_omega.png (log-log plot)
  - stdout: B_heave * omega^4 std/mean over the last 10 grid points
  - stdout: relative noise at the highest 5 omegas via 5-point
    moving average
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
_NC = _HERE / "capytaine_bem.nc"
_FIG = _HERE / "results" / "figures" / "B_heave_high_omega.png"
_FIG.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ds = xr.open_dataset(_NC)
    omegas = ds["omega"].values
    finite_mask = np.isfinite(omegas)
    omegas_finite = omegas[finite_mask]

    B_heave = ds["radiation_damping"].sel(
        radiating_dof="Heave", influenced_dof="Heave"
    ).values
    B_heave_finite = (
        B_heave[finite_mask] if B_heave.size == omegas.size else B_heave[: omegas_finite.size]
    )

    B_surge = ds["radiation_damping"].sel(
        radiating_dof="Surge", influenced_dof="Surge"
    ).values
    B_surge_finite = (
        B_surge[finite_mask] if B_surge.size == omegas.size else B_surge[: omegas_finite.size]
    )

    print("=" * 70)
    print("Pre-flight 2 diagnostic: B_heave high-omega behavior")
    print("=" * 70)
    print(
        f"omega grid: {omegas_finite.size} finite pts, "
        f"range [{omegas_finite[0]:.3f}, {omegas_finite[-1]:.3f}]"
    )
    print()

    # (1) Log-log plot of B_heave for omega >= 5 (focus on the regime in question).
    high_mask = omegas_finite >= 5.0
    fig, ax = plt.subplots(figsize=(8, 5))
    abs_B_heave = np.abs(B_heave_finite[high_mask])
    abs_B_surge = np.abs(B_surge_finite[high_mask])
    # Mark sign on the markers.
    sign_heave = np.sign(B_heave_finite[high_mask])
    ax.loglog(omegas_finite[high_mask], abs_B_heave, "o-",
              label="|B_heave|", color="steelblue", markersize=4)
    ax.loglog(omegas_finite[high_mask], abs_B_surge, "s-",
              label="|B_surge|", color="firebrick", markersize=4, alpha=0.6)
    # 1/omega^4 reference line through B_heave at omega = 5 (or first non-zero entry)
    # Anchor through smallest omega in the plot range that has finite non-zero data
    finite_nonzero = np.where((omegas_finite[high_mask] > 0) & (abs_B_heave > 0))[0]
    if finite_nonzero.size:
        anchor_idx = finite_nonzero[0]
        anchor_w = omegas_finite[high_mask][anchor_idx]
        anchor_B = abs_B_heave[anchor_idx]
        # Reference: B_ref(w) = anchor_B * (anchor_w/w)^4
        ws_ref = np.logspace(np.log10(anchor_w), np.log10(omegas_finite[-1]), 50)
        Bs_ref = anchor_B * (anchor_w / ws_ref) ** 4
        ax.loglog(ws_ref, Bs_ref, "k--", lw=0.7, alpha=0.6,
                  label="1/omega^4 ref (anchored at first |B_heave|)")
    # Mark sign changes on B_heave
    sign_changes = np.where(sign_heave[:-1] != sign_heave[1:])[0]
    if sign_changes.size:
        for i in sign_changes:
            ax.axvline(omegas_finite[high_mask][i + 1], color="orange",
                       lw=0.4, alpha=0.4)
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("|B(omega)|  (kg/s)")
    ax.set_title("B(omega) high-omega behavior: heave vs surge (log-log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(_FIG, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {_FIG}")
    print()

    # (2) B_heave * omega^4 over last 10 grid points (Item 25's metric, on heave).
    last10_idx = slice(-10, None)
    last10_w = omegas_finite[last10_idx]
    last10_Bh = B_heave_finite[last10_idx]
    last10_Bs = B_surge_finite[last10_idx]
    Bh_w4 = last10_Bh * last10_w ** 4
    Bs_w4 = last10_Bs * last10_w ** 4
    Bh_mean = np.mean(Bh_w4)
    Bs_mean = np.mean(Bs_w4)
    Bh_std = np.std(Bh_w4)
    Bs_std = np.std(Bs_w4)
    print(
        f"Item 25 metric on last 10 grid points "
        f"(omega in [{last10_w[0]:.2f}, {last10_w[-1]:.2f}]):"
    )
    print(f"  B_heave * omega^4: mean = {Bh_mean:+.4e}, std = {Bh_std:.4e}, "
          f"std/|mean| = {Bh_std / abs(Bh_mean) if abs(Bh_mean) > 1e-30 else float('inf'):.4f}")
    print(f"  B_surge * omega^4: mean = {Bs_mean:+.4e}, std = {Bs_std:.4e}, "
          f"std/|mean| = {Bs_std / abs(Bs_mean) if abs(Bs_mean) > 1e-30 else float('inf'):.4f}")
    print("  Item 25 gate threshold: std/mean < 0.10")
    print()

    # (3) Relative noise at the highest 5 omegas via 5-point moving average.
    def _smooth5(x):
        # 5-point centered moving average (boundary uses available values)
        n = len(x)
        out = np.empty_like(x)
        for i in range(n):
            lo = max(0, i - 2)
            hi = min(n, i + 3)
            out[i] = np.mean(x[lo:hi])
        return out

    smooth_Bh = _smooth5(B_heave_finite)
    smooth_Bs = _smooth5(B_surge_finite)
    print("Relative noise (|B - smooth(B)| / |smooth(B)|) at the highest 5 omegas:")
    print(
        f"  {'omega':>8} | {'B_heave':>14} | {'noise_heave':>12} "
        f"| {'B_surge':>12} | {'noise_surge':>12}"
    )
    for i in range(-5, 0):
        w = omegas_finite[i]
        Bh = B_heave_finite[i]
        Bs = B_surge_finite[i]
        sh_Bh = smooth_Bh[i]
        sh_Bs = smooth_Bs[i]
        nh = abs(Bh - sh_Bh) / abs(sh_Bh) if abs(sh_Bh) > 1e-30 else float("inf")
        ns = abs(Bs - sh_Bs) / abs(sh_Bs) if abs(sh_Bs) > 1e-30 else float("inf")
        print(f"  {w:8.3f} | {Bh:+14.5e} | {nh:12.3f} | {Bs:+12.5e} | {ns:12.3f}")
    print()

    # (4) Find the largest omega where the local std/mean of B_heave*omega^4
    # over the last 10 ending-at-that-omega samples is < 0.10 (the Item 25 gate
    # applied locally). That's a proxy for omega_clean for heave.
    print("Local asymptote check for B_heave (rolling last-10 window):")
    print(f"  {'omega_end':>10} | {'std/|mean|':>12}")
    for i in range(15, omegas_finite.size + 1):
        win_w = omegas_finite[i - 10:i]
        win_Bh = B_heave_finite[i - 10:i]
        win_w4 = win_Bh * win_w ** 4
        m = np.mean(win_w4)
        s = np.std(win_w4)
        ratio = s / abs(m) if abs(m) > 1e-30 else float("inf")
        ratio_str = "  inf" if not np.isfinite(ratio) else f"{ratio:12.4f}"
        # Print every fifth one to keep output tractable
        if i % 5 == 0 or i in (15, omegas_finite.size):
            print(f"  {omegas_finite[i-1]:10.3f} | {ratio_str}")
    print()

    # Same for surge for comparison.
    print("Local asymptote check for B_surge (rolling last-10 window):")
    print(f"  {'omega_end':>10} | {'std/|mean|':>12}")
    for i in range(15, omegas_finite.size + 1):
        win_w = omegas_finite[i - 10:i]
        win_Bs = B_surge_finite[i - 10:i]
        win_w4 = win_Bs * win_w ** 4
        m = np.mean(win_w4)
        s = np.std(win_w4)
        ratio = s / abs(m) if abs(m) > 1e-30 else float("inf")
        ratio_str = "  inf" if not np.isfinite(ratio) else f"{ratio:12.4f}"
        if i % 5 == 0 or i in (15, omegas_finite.size):
            print(f"  {omegas_finite[i-1]:10.3f} | {ratio_str}")
    print()

    # Determine omega_clean for surge (when the rolling std/mean first DROPS below 0.1).
    omega_clean_surge = None
    for i in range(omegas_finite.size, 10, -1):
        win_w = omegas_finite[i - 10:i]
        win_Bs = B_surge_finite[i - 10:i]
        win_w4 = win_Bs * win_w ** 4
        m = np.mean(win_w4)
        s = np.std(win_w4)
        ratio = s / abs(m) if abs(m) > 1e-30 else float("inf")
        if ratio < 0.10:
            omega_clean_surge = float(omegas_finite[i - 1])
            break
    print(f"Largest omega_end where B_surge rolling asymptote check passes: "
          f"{omega_clean_surge}")

    # Conclusion summary.
    print()
    print("=" * 70)
    print("Hypothesis classification:")
    print("=" * 70)
    print(f"|B_heave|  at highest omega:   {abs(B_heave_finite[-1]):.3e} kg/s")
    print(f"|B_surge|  at highest omega:   {abs(B_surge_finite[-1]):.3e} kg/s")
    hs_ratio = (
        abs(B_heave_finite[-1]) / abs(B_surge_finite[-1])
        if abs(B_surge_finite[-1]) > 1e-30
        else float("inf")
    )
    print(f"ratio (heave/surge):           {hs_ratio:.3e}")


if __name__ == "__main__":
    main()
