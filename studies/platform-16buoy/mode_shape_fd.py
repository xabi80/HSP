"""Constrained frequency-domain heave mode shape, 12 vs 16 buoy (no-fin).

For each model, solve the saddle system ``[Z Gᵀ; G 0][ξ;λ] = [F;0]`` over a period
sweep, with ``Z(ω) = -ω²(M + A(ω)) + i ω B(ω) + C`` (RADIATION damping only; A(ω)/
B(ω) the coupled BEM, G the yaw-locked joint Jacobian at equilibrium), F the coupled
wave-excitation RAO. Report per-period the buoy-7 heave, platform heave and their
ratio, so the mode STRUCTURE can be compared at the *same* period for both models.

WHY the same-period comparison matters (correction, 2026-08-16). An earlier version
of this script picked each model's own FD peak and labelled it collective vs
buoy-relative. That was misleading: the 16-buoy FD peaks at T = 2.35 s in a
razor-sharp, near-undamped INTERNAL mode (buoy/platform ≈ 280, platform ≈ 0) that
quadratic Morison drag suppresses entirely — it does NOT appear in the measured fan
(16-buoy goes 2.30→1.39, 2.40→1.97; no 2.35 s spike). The measured resonance is at
T = 2.50 s, where BOTH models heave collectively (buoy/platform ≈ 1). Read at that
shared period the radiation-only FD gives 16 < 12 — the same direction as measured
(2.80 < 4.27), i.e. the reversal is NOT a change of excited mode shape. The measured
response is drag-limited (drag ≈ 20× radiation), so the full quantitative mechanism
needs an equivalent-linearized-drag FD, not this radiation-only sweep. See
FIN-SENSITIVITY.md ("Mechanism — what is settled and what is still open").
"""
import sys
import warnings
from pathlib import Path

import numpy as np

_R = Path("studies")
for _p in ("platform-12buoy", "platform-16buoy", "cluster-3buoy-rigid"):
    sys.path.insert(0, str(_R / _p))
sys.path.insert(0, "floatsim")
warnings.simplefilter("ignore")

_T_MEAS = 2.50  # measured no-fin heave resonance (12- and 16-buoy), from the fan CSVs


def fd_sweep(hdb, setup, hydro_dof, buoy_hv, plat_hv, periods):  # type: ignore[no-untyped-def]
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
        xi = np.linalg.solve(K, np.concatenate([F, np.zeros(m)]))[:ndof]
        bh = np.abs(xi[buoy_hv])
        out.append((T, np.abs(xi[buoy_hv[6]]), np.abs(xi[plat_hv]), bh.mean()))
    return out  # (T, buoy7_heave, plat_heave, mean_buoy_heave)


def build12():  # type: ignore[no-untyped-def]
    import platform_fin_fan as pff, platform_rao_pilot as prp
    hdb = pff._hdb("none"); setup = pff._build(pff._R_SPAR, 5.0, hdb)
    hydro = prp._hydro_dof(prp._deck_with_drag())
    buoy_hv = [6 * prp._buoy_body_index(k) + 2 for k in range(12)]
    plat_hv = 6 * prp._buoy_body_index_platform() + 2
    return hdb, setup, hydro, buoy_hv, plat_hv


def build16():  # type: ignore[no-untyped-def]
    import platform16_fin_fan as pff, platform16_rao as prp
    hdb = pff._hdb("none"); setup = pff._build(pff._R_SPAR, 5.0, hdb)
    hydro = prp._hydro_dof(prp._deck_with_drag())
    buoy_hv = [6 * prp._buoy_body_index(k) + 2 for k in range(16)]
    plat_hv = 6 * prp._buoy_body_index_platform() + 2
    return hdb, setup, hydro, buoy_hv, plat_hv


def main() -> None:
    periods = np.round(np.arange(2.20, 2.86, 0.05), 3)
    for name, builder in [("12-buoy", build12), ("16-buoy", build16)]:
        hdb, setup, hydro, buoy_hv, plat_hv = builder()
        res = fd_sweep(hdb, setup, hydro, buoy_hv, plat_hv, periods)
        Tpk = max(res, key=lambda r: r[1])[0]
        Tm, x7m, xpm, xmm = next(r for r in res if abs(r[0] - _T_MEAS) < 1e-6)
        print(f"\n=== {name} (linear constrained FD, radiation damping only) ===")
        print("   T     buoy7   platform  meanbuoy  buoy7/plat")
        for T, x7, xp, xm in res:
            mark = "  <-- FD peak" if T == Tpk else ""
            star = "  [measured peak]" if abs(T - _T_MEAS) < 1e-6 else ""
            print(f"  {T:.2f}  {x7:7.2f}  {xp:8.3f}  {xm:8.2f}  {x7 / xp:8.1f}{mark}{star}")
        struct = "COLLECTIVE (buoys+platform together)" if x7m < 2 * xpm else "buoy-relative"
        print(f"  At the MEASURED resonance T={_T_MEAS}s: buoy7 {x7m:.1f} / platform {xpm:.1f} "
              f"-> ratio {x7m / xpm:.1f} => {struct}")
    print("\nBoth models are collective at the measured 2.50 s resonance; read there the "
          "radiation-only\nFD gives 16 < 12 (same direction as measured). The 16-buoy FD "
          "'peak' at 2.35 s is a\nnear-undamped internal mode that drag suppresses -- absent "
          "from the measured fan. The\ndrag-limited magnitude reversal needs an "
          "equivalent-linearized-drag FD (still open).")


if __name__ == "__main__":
    main()
