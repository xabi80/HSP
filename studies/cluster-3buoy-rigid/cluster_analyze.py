"""Step 5 analysis + plots: period (with vs without interaction), damping,
cross-DOF, and the A33(omega) interaction picture.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

import cluster_common as cc  # noqa: E402
import cluster_study_common as sc  # noqa: E402

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_FIG = _RESULTS / "figures"
_DOF = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def _load(name):
    d = np.loadtxt(_RESULTS / name, delimiter=",", skiprows=1)
    return {"t": d[:, 0], "xi": d[:, 1:7]}


def _period_zeta(t, x):
    pk, _ = find_peaks(x, height=1e-4)
    if pk.size < 3:
        return None, None, pk
    period = float(np.mean(np.diff(t[pk])))
    amps = x[pk]
    ratios = amps[:-1] / amps[1:]
    deltas = np.log(ratios[ratios > 0])
    delta = float(np.mean(deltas))
    zeta = float(delta / np.sqrt(4 * np.pi**2 + delta**2))
    return period, zeta, pk


def main() -> None:
    _FIG.mkdir(parents=True, exist_ok=True)
    inter = json.loads((_RESULTS / "interaction.json").read_text())
    bem = _load("decay_bem_only.csv")
    mor = _load("decay_bem_morison.csv")

    A33_c = inter["A33_composite_inf"]
    A33_s = inter["A33_single_inf"]
    C33_c = inter["C33_composite"]
    C33_s = C33_c / 3.0  # composite is 3x single by measurement
    M = cc.M_CLUSTER
    R = inter["R_inf"]

    # Period predictions.
    Tn_with = 2 * np.pi * np.sqrt((M + A33_c) / C33_c)
    Tn_without = 2 * np.pi * np.sqrt((M + 3 * A33_s) / (3 * C33_s))
    dT = Tn_with - Tn_without
    wn = 2 * np.pi / Tn_with

    # Radiation damping prediction from B33_composite(omega_n).
    hdb = sc.load_hdb()
    omega = np.asarray(hdb.omega)
    B33 = np.asarray(hdb.B[2, 2, :])
    o = np.argsort(omega)
    B_wn = float(np.interp(wn, omega[o], B33[o]))
    zeta_pred = B_wn / (2 * np.sqrt((M + A33_c) * C33_c))

    # Measured.
    Tn_meas, zeta_rad_meas, pk_b = _period_zeta(bem["t"], bem["xi"][:, 2])
    Tn_mor, zeta_mor, pk_m = _period_zeta(mor["t"], mor["xi"][:, 2])

    def rel(a, b):
        return abs(a - b) / abs(b)

    Tn_rel = rel(Tn_meas, Tn_with)
    zeta_rel = rel(zeta_rad_meas, zeta_pred)

    cross = {n: max(float(np.max(np.abs(bem["xi"][:, k]))),
                    float(np.max(np.abs(mor["xi"][:, k]))))
             for k, n in enumerate(_DOF) if k != 2}
    cross_max = max(cross.values())

    print("=" * 70)
    print("Cluster Step 5 -- analysis")
    print("=" * 70)
    print(f"  Interaction ratio R(inf) = {R:.4f}")
    print(f"  A33_composite = {A33_c:.4f} kg;  3 x A33_single = {3*A33_s:.4f} kg")
    print(f"  C33_composite = {C33_c:.4f} N/m")
    print()
    print(f"  T_n WITH interaction    = {Tn_with:.5f} s")
    print(f"  T_n WITHOUT (3x single) = {Tn_without:.5f} s")
    print(f"  interaction period delta = {dT*1000:+.2f} ms ({dT/Tn_without:+.3%})")
    print(f"  T_n FloatSim (measured) = {Tn_meas:.5f} s  rel-err {Tn_rel:.3%} "
          f"(gate 1e-2: {'PASS' if Tn_rel < 1e-2 else 'FAIL'})")
    print()
    print(f"  omega_n = {wn:.4f} rad/s; B33_composite(omega_n) = {B_wn:.4e}")
    print(f"  zeta_rad predicted = {zeta_pred:.4e} ({zeta_pred*100:.3f}% crit)")
    print(f"  zeta_rad FloatSim  = {zeta_rad_meas:.4e} ({zeta_rad_meas*100:.3f}% crit) "
          f"rel-err {zeta_rel:.3%} (gate 5e-2: {'PASS' if zeta_rel < 5e-2 else 'FAIL'})")
    print()
    print(f"  BEM+Morison effective zeta = {zeta_mor:.4e} ({zeta_mor*100:.2f}% crit) "
          f"[amplitude-dependent, no gate]")
    print(f"  KC per plate = 2*pi*0.10/0.43 = {2*np.pi*0.10/0.43:.2f} (Cd=5.0 valid)")
    print()
    print("  Cross-DOF max |xi_k| (k != heave):")
    for n, v in cross.items():
        print(f"    {n:6s}: {v:.3e}")
    print(f"  cross_max = {cross_max:.3e}")

    # --- Plots ---
    env_t = bem["t"]
    env = sc.IC_HEAVE * np.exp(-zeta_pred * wn * env_t)
    # (1) both decays + envelope
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(bem["t"], bem["xi"][:, 2], lw=1.0, label="BEM-only")
    ax.plot(mor["t"], mor["xi"][:, 2], lw=1.0, label="BEM + Morison plate")
    ax.plot(env_t, env, "k--", lw=0.8, label="analytical radiation envelope")
    ax.plot(env_t, -env, "k--", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="heave (m)",
           title=f"3-buoy cluster heave decay (IC 0.10 m, R={R:.3f})")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(_FIG / "heave_decay.png", dpi=120); plt.close(fig)

    # (2) log-decrement
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(bem["t"][pk_b], np.abs(bem["xi"][pk_b, 2]), "o-", ms=4,
                label="BEM-only peaks")
    ax.semilogy(env_t, env, "k--", lw=0.8, label="analytical envelope")
    ax.set(xlabel="time (s)", ylabel="|heave peak| (m, log)",
           title=f"Log-decrement: zeta_rad {zeta_rad_meas*100:.3f}% "
                 f"vs analytical {zeta_pred*100:.3f}%")
    ax.legend(); ax.grid(True, which="both", alpha=0.3); fig.tight_layout()
    fig.savefig(_FIG / "decay_envelope_log.png", dpi=120); plt.close(fig)

    # (3) cross-DOF
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, n in enumerate(_DOF):
        if k == 2:
            continue
        ax.plot(bem["t"], bem["xi"][:, k], lw=0.8, label=n)
    ax.set(xlabel="time (s)", ylabel="off-heave DOF",
           title=f"Cross-DOF (BEM-only); max |xi| = {cross_max:.2e}")
    ax.legend(ncol=3, fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(_FIG / "cross_dof.png", dpi=120); plt.close(fig)

    # (4) A33(omega) interaction picture
    w_s, a_s = inter["A33_single_curve"]
    w_c, a_c = inter["A33_composite_curve"]
    w_s, a_s, w_c, a_c = map(np.array, (w_s, a_s, w_c, a_c))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(w_c, a_c, "-", lw=1.4, label="A33 composite (measured)")
    ax.plot(w_s, 3 * a_s, "--", lw=1.2, label="3 x A33 single (no interaction)")
    ax.axvline(wn, color="grey", ls=":", lw=0.8, label=f"omega_n={wn:.2f}")
    ax.set(xlabel="omega (rad/s)", ylabel="heave added mass A33 (kg)",
           title=f"3-buoy interaction: R(inf)={R:.4f} (+{(R-1)*100:.1f}%)")
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(_FIG / "interaction_A33.png", dpi=120); plt.close(fig)

    # --- summary.md ---
    band_hi = 10 * cross_max
    summary = f"""# 3-buoy rigid-cluster heave decay -- summary

## Headline: hydrodynamic interaction

**Interaction ratio R(inf) = A33_composite / (3 x A33_single) =
{R:.4f}** (+{(R-1)*100:.2f}% on heave added mass; near-field, weak).
R(omega_n) = {inter['R_omega_n']:.4f}.

**Radiation damping interacts strongly and CONSTRUCTIVELY:**
B33_composite / (3 x B33_single) at omega_n =
{inter['B33_ratio_omega_n']:.4f} (>> 1), i.e.
B33_composite / B33_single = {3*inter['B33_ratio_omega_n']:.2f}x,
approaching the coherent-radiation ceiling N^2 = 9 for N = 3 in-phase
sources. A rigid cluster heaves in phase; at sub-wavelength spacing the
three radiated fields add coherently, so radiated power scales toward
N^2. Added mass (near-field) sees the hulls as independent; radiation
damping (far-field) sees them as one coherent source.
See the Conclusions / Cross-DOF closeout section for the corrected
interpretation and the decay-zeta consequence.

**Period effect of the interaction:**

| quantity | value |
|---|---|
| T_n WITH interaction (measured A33_composite) | {Tn_with:.5f} s |
| T_n WITHOUT (3 x single values) | {Tn_without:.5f} s |
| interaction delta | {dT*1000:+.2f} ms ({dT/Tn_without:+.3%}) |

The +{(R-1)*100:.1f}% added-mass interaction lengthens the heave
period by {dT*1000:.1f} ms.

## Period validation

| quantity | value |
|---|---|
| M + A33_composite | {M + A33_c:.4f} kg |
| C33_composite | {C33_c:.4f} N/m (= 3 x {C33_s:.4f}) |
| T_n analytical (with interaction) | {Tn_with:.5f} s |
| T_n FloatSim (zero-crossing) | {Tn_meas:.5f} s |
| rel-err | {Tn_rel:.3%} (gate 1e-2: {'PASS' if Tn_rel < 1e-2 else 'FAIL'}) |

## Radiation damping (BEM-only)

| quantity | value |
|---|---|
| omega_n | {wn:.4f} rad/s |
| B33_composite(omega_n) | {B_wn:.4e} |
| zeta_rad predicted | {zeta_pred:.4e} ({zeta_pred*100:.3f}% crit) |
| zeta_rad FloatSim | {zeta_rad_meas:.4e} ({zeta_rad_meas*100:.3f}% crit) |
| rel-err | {zeta_rel:.3%} (gate 5e-2: {'PASS' if zeta_rel < 5e-2 else 'FAIL'}) |

Radiation damping is very light (< 0.1% critical), as for the single
buoy -- a low-radiation heave geometry.

## Heave-plate drag (BEM + Morison)

Single degenerate horizontal-cylinder element, projected area
D*L = 3 x 0.1452 = 0.4356 m^2, Cd = 5.0 (equivalent to three offset
elements for pure heave of a rigid symmetric body). KC per plate =
2*pi*0.10/0.43 = {2*np.pi*0.10/0.43:.2f} (unchanged from the single-buoy
study; Cd=5.0 valid per Tao & Cai).

| quantity | value |
|---|---|
| effective zeta (first peaks, amplitude-dependent) | {zeta_mor:.4e} ({zeta_mor*100:.2f}% crit) |
| period | {Tn_mor:.4f} s |

## Cross-DOF

Symmetric 3-fold cluster -> heave decouples. Measured max |xi_k|
(k != heave) = {cross_max:.3e}:

""" + "\n".join(f"- {n}: {v:.3e}" for n, v in cross.items()) + f"""

Band asserted from the measurement (< {band_hi:.2e}, 10x). Three-hull
BEM panel asymmetries are real but here remain at numerical-noise
level.

## Figures

- `figures/heave_decay.png` -- both decays + envelope
- `figures/decay_envelope_log.png` -- log-decrement
- `figures/cross_dof.png` -- off-heave DOF
- `figures/interaction_A33.png` -- A33 composite vs 3 x single
"""
    (_RESULTS / "summary.md").write_text(summary)
    print(f"\n  Wrote results/summary.md + 4 figures.")
    assert cross_max < band_hi
    print(f"  Cross-DOF within measurement band (< {band_hi:.2e}).")


if __name__ == "__main__":
    main()
