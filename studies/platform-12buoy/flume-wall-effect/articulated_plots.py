"""Regenerate the articulated-run figures for the flume wall-effect study from the saved
run outputs (no BEM/solve here — pure plotting, so it is fast and reproducible):

  articulated_decay.json   (articulated_wall.py decay)  -> heave period/damping
  articulated_rao.npz      (articulated_wall.py rao)     -> platform-heave RAO + vert. accel
  accel_multidof.npz       (accel_multidof.py)           -> all-DOF accel at the sensor points
  sweep_results.npy        (flume_wall_effect.py)        -> frequency-domain per-DOF A/F shifts

Writes: articulated_summary.png, articulated_accel.png, accel_multidof.png.
Run after the articulated + frequency-domain sweeps: python articulated_plots.py
"""
# ruff: noqa: E702, B905  -- compact plotting setup lines (E702); zips here are same-length by
# construction (B905), so an explicit strict= adds nothing in this one-off figure generator.
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROT = float(__import__("os").environ.get("PLAT_ROT_DEG", "0"))
SUF = "" if ROT == 0 else f"_rot{int(ROT)}"
TEAL, TEALD, RED, GREY = "#0c8b96", "#0a5560", "#b2432c", "#51606a"
PTS = ["platform_centre", "cluster_1", "cluster_2", "cluster_3", "cluster_4"]
LBL = ["platform", "cl.1", "cl.2", "cl.3", "cl.4"]
COLS = ["#0c8b96", "#12a5b0", "#3fbfc9", "#7ad4db", "#b3e6ea"]

# heave-mode constants (single-buoy, for the freq-domain dA -> period-shift conversion)
C33_BUOY, T_HEAVE = 194.5, 2.52


def freq_domain_heave_period_pct() -> float | None:
    """Implied heave natural-period shift from the frequency-domain added-mass shift:
    dT/T = 0.5 * dA/(M+A) = 0.5 * (dA/A) * (A/(M+A)), with A/(M+A) = A33*wn^2/C33."""
    f = HERE / f"sweep_results{SUF}.npy"
    if not f.exists():
        return None
    rows = np.load(f, allow_pickle=True)
    wn = 2 * np.pi / T_HEAVE
    # pick the sweep row nearest the heave period; row = (T, o=walls-out@2.7m, wl=walls-in@2.7m, od)
    ts = np.array([r[0] for r in rows])
    row = rows[int(np.argmin(np.abs(ts - T_HEAVE)))]
    o, wl = row[1], row[2]
    A_o = o[("A", "He")]
    dA_over_A = wl[("A", "He")] / A_o - 1.0
    A_over_MA = A_o * wn**2 / C33_BUOY
    return 100.0 * 0.5 * dA_over_A * A_over_MA


def fig_summary() -> None:
    dec = json.loads((HERE / f"articulated_decay{SUF}.json").read_text())
    o, w = dec["open"], dec["walled"]
    art_T_pct = 100 * (w["heave_T"] / o["heave_T"] - 1)
    fd_T_pct = freq_domain_heave_period_pct()
    npz = np.load(HERE / f"articulated_rao{SUF}.npz")
    P, Ro, Rw = npz["P"], npz["Ro"], npz["Rw"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3))
    # (a) heave period shift: frequency-domain vs articulated
    labels, vals = [], []
    if fd_T_pct is not None:
        labels.append("frequency\ndomain"); vals.append(fd_T_pct)
    labels.append("articulated\n21-body"); vals.append(art_T_pct)
    ax1.axhspan(-5, 5, color="0.9", zorder=0)
    ax1.axhline(0, color="0.5", lw=0.8)
    bars = ax1.bar(labels, vals, width=0.55, color=[TEAL, TEALD][: len(vals)])
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + (0.05 if v >= 0 else -0.12),
                 f"{v:+.2f}%", ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5, 5)
    ax1.set_ylabel("heave natural-period shift (%)")
    ax1.set_title(f"Free-decay period shift\n(damping unchanged: ζ "
                  f"{o['heave_zeta'] * 100:.1f}% → {w['heave_zeta'] * 100:.1f}%)",
                  fontsize=11, fontweight="bold")
    ax1.text(0.5, -4.4, "grey band = typical ±5% model-test scatter", transform=ax1.transData,
             ha="center", fontsize=8.5, color=GREY)

    # (b) platform-heave RAO, walls in vs out
    ax2.plot(P, Ro, "o-", color=TEAL, lw=2, label="walls out")
    ax2.plot(P, Rw, "s--", color=RED, lw=2, label="walls in")
    ax2.set_xlabel("wave period T (s)")
    ax2.set_ylabel("platform-heave RAO (m/m)")
    mx = max(100 * abs(rw / ro - 1) for ro, rw in zip(Ro, Rw))
    ax2.set_title(f"Platform-heave RAO (wall effect ≤ {mx:.1f}%)", fontsize=11,
                  fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.suptitle("Flume wall effect — articulated 21-body FloatSim (16-buoy platform)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / f"articulated_summary{SUF}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_accel_vertical() -> None:
    """Vertical (heave) acceleration wall effect at the 5 sensor points."""
    npz = np.load(HERE / f"articulated_rao{SUF}.npz")
    P = npz["P"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    wdt = 0.16
    for j, (pt, pl) in enumerate(zip(PTS, LBL)):
        pct = 100 * (npz[f"Aw_{pt}"] / npz[f"Ao_{pt}"] - 1)
        ax.bar(np.arange(len(P)) + (j - 2) * wdt, pct, wdt, label=pl, color=COLS[j])
    ax.axhspan(-5, 5, color="0.9", zorder=0)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(range(len(P))); ax.set_xticklabels([f"{t:.2f}s" for t in P])
    ax.set_ylim(-6, 6); ax.set_ylabel("vertical-acceleration wall effect (%)")
    ax.set_title("Wall effect on vertical acceleration at the sensor points\n"
                 "(grey band = typical ±5% model-test scatter)", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=5, loc="lower center"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / f"articulated_accel{SUF}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_accel_multidof() -> None:
    d = np.load(HERE / f"accel_multidof{SUF}.npz")
    P = d["P"]
    dofs = [("surge", "fore-aft accel"), ("heave", "vertical accel"), ("pitch", "pitch ang. accel")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    wdt = 0.16
    for ax, (dof, title) in zip(axs, dofs):
        for j, (pt, pl) in enumerate(zip(PTS, LBL)):
            pct = 100 * (d[f"w_{pt}_{dof}"] / d[f"o_{pt}_{dof}"] - 1)
            ax.bar(np.arange(len(P)) + (j - 2) * wdt, pct, wdt, label=pl, color=COLS[j])
        ax.axhline(0, color="0.4", lw=0.8)
        ax.axhspan(-5, 5, color="0.85", alpha=0.5, zorder=0)
        ax.set_title(f"{title}\nwall effect (%)", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(P))); ax.set_xticklabels([f"{t:.2f}s" for t in P])
        ax.set_ylim(-6, 6); ax.grid(axis="y", alpha=0.3)
    axs[0].set_ylabel("walls-in vs walls-out (%)", fontsize=10)
    axs[0].legend(fontsize=8, ncol=2, loc="lower left")
    fig.suptitle("Flume wall effect on accelerations, all excited DOFs — articulated 21-body "
                 "FloatSim\n(grey band = typical ±5% model-test scatter)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(HERE / f"accel_multidof{SUF}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_wall_vs_depth() -> None:
    """Isolated sidewall effect vs the finite-depth effect on heave excitation (both at head
    seas), from the 2.7 m frequency-domain sweep. Makes the point that DEPTH, not the walls,
    is the dominant flume artifact -- and it is largest at long periods."""
    rows = np.load(HERE / f"sweep_results{SUF}.npy", allow_pickle=True)
    Ts = np.array([r[0] for r in rows])
    wall = np.array([100 * (r[2][("F", "He")] / r[1][("F", "He")] - 1) for r in rows])  # wl/o
    depth = np.array([100 * (r[1][("F", "He")] / r[3][("F", "He")] - 1) for r in rows])  # o/od
    x = np.arange(len(Ts))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    ax.bar(x - 0.2, wall, 0.4, label="sidewall effect (walls in vs out, 2.7 m)", color=TEAL)
    ax.bar(x + 0.2, depth, 0.4, label="finite-depth effect (2.7 m vs deep)", color=RED)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{t:.2f}" for t in Ts])
    ax.set_xlabel("wave period T (s)")
    ax.set_ylabel("effect on heave wave excitation (%)")
    ax.set_title("Sidewall effect vs finite-depth effect on heave excitation\n"
                 "(the reviewer's concern is the walls; the dominant artifact is the depth)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / f"wall_vs_depth{SUF}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    made = []
    for name, fn in [("articulated_summary.png", fig_summary),
                     ("articulated_accel.png", fig_accel_vertical),
                     ("accel_multidof.png", fig_accel_multidof),
                     ("wall_vs_depth.png", fig_wall_vs_depth)]:
        try:
            fn(); made.append(name.replace(".png", f"{SUF}.png"))
        except FileNotFoundError as e:
            print(f"skip {name}: missing {e.filename}")
    print("wrote:", ", ".join(made))


if __name__ == "__main__":
    main()
