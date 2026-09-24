"""Flume station-keeping mooring sizing for the OSU LWF test articles: 1 buoy (Phase 1),
1 cluster of 4 buoys (Phase 2), and the 4x4 platform (Phase 3, 45 deg test orientation).

Scope (per the TEAMER response): the flume mooring is STATION-KEEPING ONLY -- it holds the
article in the test section against the mean wave-drift force without disturbing the measured
wave-frequency response (free decay, heave/pitch RAOs, accelerations). The field site is deep
water, so a 2.7 m flume cannot host a scale model of the field mooring; this is an equivalent
soft horizontal mooring.

Per article it computes:
  1. Horizontal inertia  M_h = structural mass + low-frequency surge added mass (Capytaine BEM).
  2. Mean wave-drift force (head seas), conservative upper bound: splash-zone Morison drag drift
     on every surface-piercing spar with the body held fixed,
         F_d = (2 / 3 pi) rho Cd D A U^2 ,   U = A w coth(k h),
     (drag integrated to the moving free surface; below the trough the cycle-mean drag is zero)
     plus a Havelock small-ka bound on the potential-flow drift,
         F_p = (5 pi^2/16) rho g a A^2 (ka)^3.
     No shielding credit between spars. Relative-velocity drift (body moving with the wave) is
     lower; the FloatSim moored runs refine it.
  3. Stiffness window: a SOFT bound so the surge natural period sits well above the wave band
     (T_surge >= 3-5 x T_wave_max), and the resulting mean offset delta = F / K_x.
  4. Line design for an X-spread of 4 horizontal lines at the waterline to wall anchors a
     distance L_a up/downstream: per-line axial stiffness, pretension that keeps the slack-side
     lines taut, the sway stiffness it gives, and two side-effect checks -- the heave geometric
     stiffness 4 T0 / l of the pretensioned lines vs the hydrostatic C33, and the mean trim if
     the lines attach ABOVE the waterline (the splash-zone drift acts at the waterline, so a line
     attached at height z makes a pitch moment F z; attaching at the waterline makes it zero).

Writes mooring_sizing.png + mooring_sizing.csv next to this script.  Run: python mooring_sizing.py
"""
# ruff: noqa: E702, RUF001  -- compact numeric lines; display typography in labels
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RHO, G = 998.0, 9.806
H_FLUME, W_FLUME = 2.7, 3.66                 # OSU LWF water depth (storm-wave max) and width
SPAR_D, CD_SPAR, PLATE_R = 0.1593, 1.2, 0.1437
M_BUOY = 21.52                               # OSU buoy, unloaded floating mass (osu_buoy_common)
C33_BUOY = 194.5                             # N/m, single-buoy heave waterplane stiffness
A11_BUOY = 18.87                             # kg, single-buoy low-freq surge added mass (BEM)
A11_ARRAY_PER_BUOY = 310.6 / 16              # kg, per buoy inside the 16-buoy array (coupled BEM)
A11_CLUSTER = 78.81                          # kg, 4-buoy cluster rigid surge (bem_cluster.py)
C55_BUOY = (10.2 + 5.03) * (2 * np.pi / 2.11) ** 2   # N m/rad, from the validated 2.11 s pitch
T_WAVE = np.linspace(1.4, 4.0, 53)           # regular-wave sweep band (s)
T_WAVE_MAX = float(T_WAVE.max())
H_LIST = [0.2, 0.3, 0.4, 0.5]                # wave heights (m); moderate matrix, max 0.5 m
STEEP = 1 / 15                               # practical non-breaking cap on regular-wave H/L
S = 1.25 / 1.5                               # Phase-3 layout scale (2.5 m to buoy centres)
LAT_DIST = 0.10                              # nominal lateral disturbance = 10 % of surge drift


def k_fin(T: float, h: float = H_FLUME) -> float:
    w = 2 * np.pi / T; k = w * w / G
    for _ in range(100):
        k = w * w / (G * np.tanh(k * h))
    return k


def drift_per_spar(H: float, T: float) -> tuple[float, float, float]:
    """(F_drag, F_potential, H_used): mean drift force on one fixed surface-piercing spar."""
    k = k_fin(T); L = 2 * np.pi / k
    Hu = min(H, STEEP * L)
    A = Hu / 2; w = 2 * np.pi / T
    U = A * w / np.tanh(k * H_FLUME)
    Fd = (2 / (3 * np.pi)) * RHO * CD_SPAR * SPAR_D * A * U * U
    a = SPAR_D / 2
    Fp = (5 * np.pi**2 / 16) * RHO * G * a * A * A * (k * a) ** 3
    return Fd, Fp, Hu


def centres(n_cluster: int, rot_deg: float) -> np.ndarray:
    ang_c = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0])[:n_cluster] + rot_deg)
    ang_b = np.deg2rad(np.array([0.0, 90.0, 180.0, 270.0]) + rot_deg)
    arm = S * 1.0 if n_cluster > 1 else 0.0
    return np.array([(arm * np.cos(c) + 0.5 * S * np.cos(b),
                      arm * np.sin(c) + 0.5 * S * np.sin(b)) for c in ang_c for b in ang_b])


CL = centres(1, 45.0)             # Phase-2 cluster, 45 deg test orientation (2 buoys upwave)
PF = centres(4, 45.0)             # Phase-3 platform, 45 deg test orientation
ARTICLES = {
    "1 buoy": dict(n=1, M=M_BUOY, A=A11_BUOY, half_w=PLATE_R, C55=C55_BUOY,
                   z_top=0.718, top="spar top"),
    "1 cluster (4 buoys)": dict(n=4, M=4 * M_BUOY + 2.0, A=A11_CLUSTER,
                                half_w=float(np.abs(CL[:, 1]).max()) + PLATE_R,
                                C55=C33_BUOY * float((CL[:, 0] ** 2).sum()),
                                z_top=0.717, top="hub"),
    "4x4 platform (45°)": dict(n=16, M=16 * M_BUOY + 4 * 2.0 + 6.0, A=310.6,
                               half_w=float(np.abs(PF[:, 1]).max()) + PLATE_R,
                               C55=C33_BUOY * float((PF[:, 0] ** 2).sum()),
                               z_top=0.90, top="deck"),
}


def xspread(Kx: float, delta: float, A_w: float, L_a: float) -> dict:
    """X-spread of 4 waterline lines to wall anchors L_a up/downstream (lateral W/2)."""
    alpha = np.arctan2(W_FLUME / 2, L_a); ell = float(np.hypot(W_FLUME / 2, L_a))
    kl = Kx / (4 * np.cos(alpha) ** 2)                        # per-line axial stiffness
    T0 = 1.2 * kl * np.cos(alpha) * (delta + A_w)             # slack side stays taut (+20 %)
    Ky = 4 * kl * np.sin(alpha) ** 2 + 4 * T0 * np.cos(alpha) ** 2 / ell
    Kz = 4 * T0 / ell                                          # heave geometric stiffness
    return dict(alpha_deg=float(np.degrees(alpha)), ell=ell, kl=kl, T0=T0, Ky=Ky, Kz=Kz)


def main() -> None:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    env = {}
    for H in H_LIST:
        vals = [drift_per_spar(H, T) for T in T_WAVE]
        tot = np.array([v[0] + v[1] for v in vals])
        i = int(np.argmax(tot))
        env[H] = dict(F=float(tot[i]), T=float(T_WAVE[i]), Fp=float(vals[i][1]))
    print("Mean drift per spar (fixed-body upper bound, head seas, h = 2.7 m, H/L <= 1/15):")
    for H, e in env.items():
        print(f"  H = {H:.1f} m: max {e['F']:5.2f} N at T = {e['T']:.2f} s "
              f"(potential-flow part {e['Fp']:.3f} N)")

    rows = []
    for name, a in ARTICLES.items():
        Mh = a["M"] + a["A"]
        F = a["n"] * env[max(H_LIST)]["F"]
        clear = W_FLUME / 2 - a["half_w"]
        trim = np.degrees(F * a["z_top"] / a["C55"])
        print(f"\n{name}:  M_h = {Mh:.1f} kg  drift(H=0.5) = {F:.1f} N  "
              f"clearance {clear:.2f} m/side"
              f"  C55 = {a['C55']:.0f} N·m/rad  trim if attached on {a['top']}: {trim:.1f}°"
              f" (waterline: 0°)")
        for Ts in (12.0, 15.0, 20.0):
            Kx = Mh * (2 * np.pi / Ts) ** 2
            off = {H: a["n"] * env[H]["F"] / Kx for H in H_LIST}
            for L_a in (5.0, 10.0):
                x = xspread(Kx, off[max(H_LIST)], max(H_LIST) / 2, L_a)
                Tsway = 2 * np.pi * np.sqrt(Mh / x["Ky"])
                lat = LAT_DIST * F / x["Ky"]
                print(f"  T_surge {Ts:4.0f} s, anchors {L_a:4.1f} m: Kx {Kx:6.1f} N/m, k_line "
                      f"{x['kl']:5.1f}, T0 {x['T0']:5.1f} N | offset H0.3 {off[0.3]:.2f} m, "
                      f"H0.5 {off[0.5]:.2f} m | "
                      f"heave geo {100 * x['Kz'] / (a['n'] * C33_BUOY):.2f}%"
                      f" | T_sway {Tsway:4.0f} s, lateral {lat:.2f} m")
                rows.append(dict(article=name, T_surge_s=Ts, anchor_dist_m=L_a,
                                 Mh_kg=round(Mh, 1), Kx_Npm=round(Kx, 2),
                                 k_line_Npm=round(x["kl"], 2), line_len_m=round(x["ell"], 2),
                                 line_angle_deg=round(x["alpha_deg"], 1),
                                 pretension_N=round(x["T0"], 2), drift_H05_N=round(F, 2),
                                 **{f"offset_H{H}_m": round(off[H], 3) for H in H_LIST},
                                 heave_geo_pct=round(100 * x["Kz"] / (a["n"] * C33_BUOY), 3),
                                 T_sway_s=round(Tsway, 1), lateral_offset_m=round(lat, 3),
                                 clearance_m=round(clear, 3), trim_top_deg=round(trim, 2)))
    with (HERE / "mooring_sizing.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for H, c in zip(H_LIST, ["#9ecae1", "#6baed6", "#3182bd", "#08519c"], strict=True):
        ax[0].plot(T_WAVE, [sum(drift_per_spar(H, T)[:2]) for T in T_WAVE], color=c, lw=2,
                   label=f"H = {H:.1f} m")
    ax[0].set_xlabel("wave period T (s)"); ax[0].set_ylabel("mean drift per spar (N)")
    ax[0].set_title("Mean wave-drift force per spar\n(fixed-body upper bound, H/L ≤ 1/15)",
                    fontsize=11, fontweight="bold")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    Ts = np.linspace(8, 30, 100)
    for (name, a), c in zip(ARTICLES.items(), ["#0c8b96", "#e08214", "#b2432c"], strict=True):
        Mh = a["M"] + a["A"]
        for H, ls in [(0.5, "-"), (0.3, "--")]:
            ax[1].plot(Ts, a["n"] * env[H]["F"] / (Mh * (2 * np.pi / Ts) ** 2), ls, color=c,
                       lw=2, label=f"{name}, H = {H} m")
    ax[1].axvspan(8, 3 * T_WAVE_MAX, color="#b2432c", alpha=0.08)
    ax[1].text(8.3, 1.9, "too stiff:\nT_surge < 3×T_wave", fontsize=8.5, color="#b2432c")
    ax[1].axvline(5 * T_WAVE_MAX, color="0.4", ls=":", lw=1)
    ax[1].text(5 * T_WAVE_MAX + 0.4, 1.9, "5× T_wave", fontsize=8.5)
    ax[1].set_ylim(0, 2.2)
    ax[1].set_xlabel("surge (mooring) natural period T_surge (s)")
    ax[1].set_ylabel("mean offset along the flume (m)")
    ax[1].set_title("Stiffness–offset trade-off (the three articles nearly\noverlap: drift and "
                    "inertia both scale with buoy count)", fontsize=11, fontweight="bold")
    ax[1].legend(fontsize=7.5); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(HERE / "mooring_sizing.png", dpi=130, bbox_inches="tight")
    print("\nwrote mooring_sizing.csv, mooring_sizing.png")


if __name__ == "__main__":
    main()
