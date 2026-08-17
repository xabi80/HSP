"""Constrained frequency-domain forced response + heave mode shape, 12 vs 16 buoy
(no-fin), to explain the non-monotonic buoy RAO by the coupled mode participation.

For each model: solve the saddle system [Z Gᵀ; G 0][xi;lam] = [F;0] over a frequency
sweep, with Z(w) = -w^2 (M + A(w)) + i w B(w) + C (radiation damping only; A(w)/B(w)
the coupled BEM, G the yaw-locked joint Jacobian at equilibrium), F = coupled wave
excitation RAO (unit amplitude). Report the peak linear buoy-7 heave and the heave
mode shape: platform heave, mean/RMS buoy heave, buoy-7 heave, and the articulation
amplification (buoy heave beyond the platform).
"""
import sys
import warnings
from pathlib import Path

import numpy as np

_R = Path("studies")
sys.path.insert(0, str(_R / "platform-12buoy"))
sys.path.insert(0, str(_R / "platform-16buoy"))
sys.path.insert(0, str(_R / "cluster-3buoy-rigid"))
warnings.simplefilter("ignore")


def fd_sweep(hdb, setup, hydro_dof, N, buoy_hv, plat_hv, periods):  # type: ignore[no-untyped-def]
    w = np.asarray(hdb.omega)
    ndof = setup.lhs.M_plus_Ainf.shape[0]
    Ainf = np.zeros((ndof, ndof)); Ainf[np.ix_(hydro_dof, hydro_dof)] = np.asarray(hdb.A_inf)
    M = setup.lhs.M_plus_Ainf - Ainf
    C = setup.lhs.C
    G = setup.constraints.jacobian(setup.xi0)
    m = G.shape[0]
    A = np.asarray(hdb.A); B = np.asarray(hdb.B); RAO = np.asarray(hdb.RAO)
    out = []
    for T in periods:
        wn = 2 * np.pi / T
        i = int(np.argmin(np.abs(w - wn)))
        Ag = np.zeros((ndof, ndof)); Ag[np.ix_(hydro_dof, hydro_dof)] = A[:, :, i]
        Bg = np.zeros((ndof, ndof)); Bg[np.ix_(hydro_dof, hydro_dof)] = B[:, :, i]
        Z = -wn**2 * (M + Ag) + 1j * wn * Bg + C
        F = np.zeros(ndof, complex); F[hydro_dof] = RAO[:, i, 0]
        K = np.zeros((ndof + m, ndof + m), complex)
        K[:ndof, :ndof] = Z; K[:ndof, ndof:] = G.T; K[ndof:, :ndof] = G
        rhs = np.concatenate([F, np.zeros(m)])
        xi = np.linalg.solve(K, rhs)[:ndof]
        bh = np.abs(xi[buoy_hv])                 # per-buoy heave amplitude
        out.append((T, np.abs(xi[buoy_hv[6]]), np.abs(xi[plat_hv]), bh.mean(),
                    np.sqrt((bh**2).mean())))
    return out  # (T, buoy7_heave, plat_heave, mean_buoy_heave, rms_buoy_heave)


def build12():  # type: ignore[no-untyped-def]
    import platform_fin_fan as pff, platform_rao_pilot as prp
    hdb = pff._hdb("none"); setup = pff._build(pff._R_SPAR, 5.0, hdb)
    hydro = prp._hydro_dof(prp._deck_with_drag())
    buoy_hv = [6 * prp._buoy_body_index(k) + 2 for k in range(12)]
    plat_hv = 6 * prp._buoy_body_index_platform() + 2
    return hdb, setup, hydro, 12, buoy_hv, plat_hv


def build16():  # type: ignore[no-untyped-def]
    import platform16_fin_fan as pff, platform16_rao as prp
    hdb = pff._hdb("none"); setup = pff._build(pff._R_SPAR, 5.0, hdb)
    hydro = prp._hydro_dof(prp._deck_with_drag())
    buoy_hv = [6 * prp._buoy_body_index(k) + 2 for k in range(16)]
    plat_hv = 6 * prp._buoy_body_index_platform() + 2
    return hdb, setup, hydro, 16, buoy_hv, plat_hv


periods = np.round(np.linspace(2.0, 3.4, 29), 3)
for name, builder in [("12-buoy", build12), ("16-buoy", build16)]:
    hdb, setup, hydro, N, buoy_hv, plat_hv = builder()
    res = fd_sweep(hdb, setup, hydro, N, buoy_hv, plat_hv, periods)
    Tpk, b7, ph, mb, rb = max(res, key=lambda r: r[1])
    print(f"\n=== {name} (linear constrained FD, radiation damping only) ===")
    print("  T     buoy7   platform  meanbuoy  buoy7/plat")
    for T, x7, xp, xm, xr in res:
        mark = " <-- peak" if T == Tpk else ""
        if x7 > 0.15 * b7:  # only near-resonance rows
            print(f"  {T:.2f}  {x7:7.2f}  {xp:8.2f}  {xm:8.2f}  {x7 / xp:8.1f}{mark}")
    print(f"  PEAK: buoy-7 {b7:.2f} @T={Tpk}s | platform {ph:.2f} mean-buoy {mb:.2f} "
          f"-> buoy moves {'WITH platform (collective)' if ph > 0.3 * b7 else 'AGAINST a ~still platform (buoy-relative)'}")
