"""In-model PITCH-damping verification for the isolated single spar-buoy (the
platform buoy: 0.215 m heave plate Cd_n=5, spar Cd=1.2). Option (a) of the
pitch-damping test plan.

Method (mirrors the M11a spar-drag PR2 + plate-drag PR4 validation, extended to
the free single-buoy pitch mode with BOTH drag elements):

1. Drag-free constrained eigenanalysis -> the pitch-restoring mode phi (surge-pitch
   coupled), its frequency w_p and modal inertia i_eff. Drag is force-only, so the
   mode is independent of the drag code (F1 discipline).
2. Pitch free-decay released along phi*theta0 at several amplitudes; damping ratio
   measured in the modal coordinate by log-decrement. zeta_drag = zeta_total - zeta_rad.
3. First-principles prediction: energy-equivalent quadratic-drag zeta from phi --
   spar transverse INT|v|^3 dz (Cd=1.2, N segments) + plate normal INT|x|^3 dA
   (Cd_n, 8a^5/15) + plate rim edge-on (Cd_t) -- the SAME energy form as M11a.
4. Discretization convergence (spar n_segments) and spar-vs-plate split.

Quadratic drag => zeta grows with amplitude; reported as zeta(theta) (and KC).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.signal import find_peaks

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))  # repo root for floatsim
warnings.simplefilter("ignore")

import study_common as sc  # noqa: E402
from floatsim.hydro.morison import (  # noqa: E402
    MorisonElement,
    PlateDragElement,
    make_morison_state_force,
)
from floatsim.solver.newmark import integrate_cummins  # noqa: E402

_SPAR_D, _SPAR_CD = 0.1682, 1.2          # buoy spar Ø, transverse Cd (platform value)
_WL_Z, _PLATE_Z = 0.0, sc.PLATE_Z        # waterline, plate z (eqdraft frame)
_A, _T = sc.PLATE_RADIUS, 0.0039         # plate radius 0.215 m, rim thickness
_CD_N, _CD_T = sc.PLATE_CD, 1.5          # plate normal 5.0, tangential 1.5
_RHO = sc.RHO
_INT_X3 = 8.0 * _A**5 / 15.0             # INT INT |x|^3 dA over the disc


def _calm(_p, _t):  # type: ignore[no-untyped-def]
    return np.zeros(3, dtype=np.float64)


def _drag_force(n_seg=10, spar=True, plate=True, n_rad=12, n_az=24):  # type: ignore[no-untyped-def]
    elems: list = []
    if spar and n_seg > 0:
        edges = np.linspace(_PLATE_Z, _WL_Z, n_seg + 1)
        for i in range(n_seg):
            elems.append(MorisonElement(
                body_index=0, node_a_body=np.array([0.0, 0.0, edges[i]]),
                node_b_body=np.array([0.0, 0.0, edges[i + 1]]), diameter=_SPAR_D, Cd=_SPAR_CD))
    if plate:
        elems.append(PlateDragElement(
            body_index=0, center_body=np.array([0.0, 0.0, _PLATE_Z]),
            normal_body=np.array([0.0, 0.0, 1.0]), radius=_A, thickness=_T,
            Cd_n=_CD_N, Cd_t=_CD_T, n_radial=n_rad, n_azimuthal=n_az))
    return make_morison_state_force(elems, n_dof=6, fluid_velocity_fn=_calm, rho=_RHO)


def _pitch_mode(lhs):  # type: ignore[no-untyped-def]
    ma = np.asarray(lhs.M_plus_Ainf)
    w2, v = eigh(np.asarray(lhs.C), ma)
    k = max((j for j in range(6) if w2[j] > 1e-6), key=lambda j: abs(v[4, j]))
    phi = v[:, k] / v[4, k]                 # normalise pitch (dof 4) = 1
    return phi, float(np.sqrt(w2[k])), float(phi @ ma @ phi), ma


def _zeta_pred(phi, w, i_eff, theta, spar=True, plate=True):  # type: ignore[no-untyped-def]
    """Energy-equivalent quadratic-drag zeta at pitch amplitude theta."""
    thd = theta * w
    e = 0.0
    if spar:
        zz = np.linspace(_PLATE_Z, _WL_Z, 400)
        vb = np.hypot(phi[0] + phi[4] * zz, phi[1] - phi[3] * zz)   # transverse speed / thd
        e += 0.5 * _RHO * _SPAR_D * _SPAR_CD * (8.0 / 3.0) / w * thd**3 * np.trapezoid(vb**3, zz)
    if plate:
        e += 0.5 * _RHO * _CD_N * (8.0 / 3.0) / w * (thd * abs(phi[4]))**3 * _INT_X3   # normal
        a_c = np.hypot(phi[0] + phi[4] * _PLATE_Z, phi[1] - phi[3] * _PLATE_Z)         # edge-on lever
        e += 0.5 * _RHO * _CD_T * (_T * 2.0 * _A) * (8.0 / 3.0) / w * (thd * a_c)**3   # rim
    return e / (4.0 * np.pi * 0.5 * i_eff * thd**2)


def _decay(lhs, kernel, phi, ma, theta0, force, n_peaks=6):  # type: ignore[no-untyped-def]
    r = integrate_cummins(lhs=lhs, kernel=kernel, xi0=phi * theta0, xi_dot0=np.zeros(6),
                          duration=sc.DURATION, dt=sc.DT, state_force=force)
    q = (r.xi @ ma @ phi) / float(phi @ ma @ phi)      # modal coordinate
    pk, _ = find_peaks(q, height=0.0)
    a = q[pk][: n_peaks + 1]
    d = np.log(a[:-1] / a[1:]); d = d[np.isfinite(d) & (d > 0)]
    return float(np.mean(d[:2]) / (2 * np.pi)), float(np.mean(a[:3]))   # zeta, representative amp


def main():  # type: ignore[no-untyped-def]
    sys.stdout.reconfigure(encoding="utf-8")
    hdb = sc.load_hdb()
    lhs, kernel = sc.build_lhs(hdb), sc.build_kernel(hdb)
    phi, w_p, i_eff, ma = _pitch_mode(lhs)
    T_p = 2 * np.pi / w_p
    print(f"PITCH mode: T={T_p:.3f}s  w={w_p:.3f} rad/s  surge/pitch(beta)={phi[0]:+.3f}  "
          f"rotation centre z={-phi[0] / phi[4]:+.3f} m  i_eff={i_eff:.2f} kg·m²\n")

    f_all = _drag_force()
    f_none = None
    print(f"{'theta(rad)':>10} {'KC_plate':>8} {'zeta_tot%':>9} {'zeta_rad%':>9} "
          f"{'zeta_drag%':>10} {'pred%':>8} {'meas/pred':>9}")
    thetas = [0.02, 0.05, 0.10, 0.15]
    meas_drag, pred_drag = [], []
    for th in thetas:
        zt, amp = _decay(lhs, kernel, phi, ma, th, f_all)
        zr, _ = _decay(lhs, kernel, phi, ma, th, f_none)
        zd = zt - zr
        zp = _zeta_pred(phi, w_p, i_eff, amp)
        kc = np.pi * amp                       # plate-edge KC ~ pi*theta
        meas_drag.append((amp, zd)); pred_drag.append((amp, zp))
        print(f"{th:10.2f} {kc:8.2f} {100 * zt:9.3f} {100 * zr:9.3f} {100 * zd:10.3f} "
              f"{100 * zp:8.3f} {zd / zp:9.2f}")

    print("\nSPAR vs PLATE split (predicted at theta=0.10, and measured single-element decays):")
    zp_s = _zeta_pred(phi, w_p, i_eff, 0.10, spar=True, plate=False)
    zp_p = _zeta_pred(phi, w_p, i_eff, 0.10, spar=False, plate=True)
    zs, a_s = _decay(lhs, kernel, phi, ma, 0.10, _drag_force(plate=False))
    zpl, a_p = _decay(lhs, kernel, phi, ma, 0.10, _drag_force(spar=False))
    zr10, _ = _decay(lhs, kernel, phi, ma, 0.10, f_none)
    print(f"  predicted: spar {100 * zp_s:.3f}%  plate {100 * zp_p:.3f}%  "
          f"(spar {100 * zp_s / (zp_s + zp_p):.0f}% / plate {100 * zp_p / (zp_s + zp_p):.0f}%)")
    print(f"  measured : spar {100 * (zs - zr10):.3f}%  plate {100 * (zpl - zr10):.3f}%")

    print("\nSPAR discretization convergence (measured zeta_drag at theta=0.10):")
    zr, _ = _decay(lhs, kernel, phi, ma, 0.10, f_none)
    for n in (1, 2, 4, 10, 16):
        zt, _ = _decay(lhs, kernel, phi, ma, 0.10, _drag_force(n_seg=n, plate=False))
        print(f"  n_seg={n:2d}: spar zeta_drag = {100 * (zt - zr):.3f}%")

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ma_a = np.array(meas_drag); pa = np.array(pred_drag)
    th_fine = np.linspace(0.01, 0.16, 60)
    ax.plot(th_fine, [100 * _zeta_pred(phi, w_p, i_eff, t) for t in th_fine], "-",
            color="#0c8b96", lw=2, label="prediction (moment integral)")
    ax.plot(ma_a[:, 0], 100 * ma_a[:, 1], "o", color="#d1543a", ms=9, label="measured (free-decay)")
    ax.set_xlabel("pitch amplitude θ (rad)"); ax.set_ylabel("drag damping ratio ζ_drag (%)")
    ax.set_title(f"Single-buoy PITCH damping: model vs first-principles\n"
                 f"T_pitch={T_p:.2f}s · 0.215 m plate (Cd_n=5) + spar (Cd=1.2) · quadratic ⇒ ζ∝θ",
                 fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend()
    out = _HERE / "fin_study" / "pitch_damping_verify.png"
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
