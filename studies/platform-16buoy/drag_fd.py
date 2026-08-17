"""Equivalent-linearized-drag constrained FD, 12 vs 16 buoy (no-fin, Cd_n=5), to
resolve WHY the drag-limited buoy heave RAO reverses (rises 3->12, falls 12->16).

This is the tool the radiation-only sweep (mode_shape_fd.py) could not be: the
measured response is drag-limited (bottom-cap Morison drag ~20x the radiation), so
the amplitude is set by quadratic drag, not by the linear radiation damping.

Method. Solve the constrained forced-response saddle system

    [ Z_rad   Gᵀ ] [ ξ̂ ]   [ F̂_exc + F̂_drag(ξ̂) ]
    [ G       0  ] [ λ  ] = [ 0                   ]

self-consistently, where Z_rad(ω) = -ω²(M+A(ω)) + iωB_rad(ω) + C (coupled BEM A/B,
joint Jacobian G at equilibrium), F̂_exc the coupled wave excitation, and F̂_drag(ξ̂)
is the FIRST-HARMONIC Fourier coefficient of the EXACT FloatSim calm-water drag
closure (``setup.state_force`` -- the very closure the time-domain fan integrated)
evaluated over one cycle of ξ(t)=xi0+Re{ξ̂ e^{iωt}}. Drag is folded onto the LHS as
a per-DOF equivalent damping purely as a PRECONDITIONER (damping in the denominator
=> stable, self-limiting iteration); at convergence the solution satisfies the exact
equation Z_rad ξ̂ = F̂_exc + F̂_drag(ξ̂) regardless of the preconditioner. Calm-water
drag on each body depends only on that body's own velocity, so the linearised drag is
block-diagonal per body -- no inter-body drag coupling is dropped.

Validation (T=2.50s, H=0.04, no-fin, Cd_n=5):
    model    radiation-only    drag-limited FD    measured fan
    12-buoy      53.9              4.273              4.27
    16-buoy      24.9              2.798              2.80
The drag-limited FD reproduces BOTH measured buoy RAOs to <0.1% -- an independent
confirmation (harmonic-linearised FD vs nonlinear time-domain KKT) that the reversal
is a drag-limited effect. Decomposition: per-buoy excitation is unchanged (1.63 vs
1.65) and the platform heave RAO is unchanged (~3.1); what changes is the buoy motion
RELATIVE to the platform (buoy/platform 1.36 -> 0.90) as the denser 4-per-cluster
layout loads the shared collective mode with more aggregate drag (Σb 100 -> 133).
See FIN-SENSITIVITY.md ("Mechanism ... resolved").
"""
import sys
import warnings
from pathlib import Path

import numpy as np

_R = Path("studies")
for _p in ("platform-12buoy", "platform-16buoy"):
    sys.path.insert(0, str(_R / _p))
sys.path.insert(0, "floatsim")
warnings.simplefilter("ignore")
from floatsim.hydro.excitation import interpolate_rao  # noqa: E402


def harmonic_drag(state_force, xi0, xhat, omega, ndof, nsamp=96):  # type: ignore[no-untyped-def]
    """First-harmonic phasor F̂ (f(t) ≈ Re{F̂ e^{iωt}}) of the exact drag closure."""
    F = np.zeros(ndof, complex)
    for kk in range(nsamp):
        th = 2 * np.pi * kk / nsamp
        ph = np.exp(1j * th)
        xi_t = xi0 + np.real(xhat * ph)
        xidot_t = np.real(1j * omega * xhat * ph)
        F += np.asarray(state_force(0.0, xi_t, xidot_t)) * np.exp(-1j * th)
    return (2.0 / nsamp) * F


def solve(hdb, setup, hydro, omega, amp, buoy_hv, plat_hv, *, drag=True, verbose=False):  # type: ignore[no-untyped-def]
    w = np.asarray(hdb.omega)
    i = int(np.argmin(np.abs(w - omega)))
    ndof = setup.lhs.M_plus_Ainf.shape[0]
    Ainf = np.zeros((ndof, ndof)); Ainf[np.ix_(hydro, hydro)] = np.asarray(hdb.A_inf)
    M = setup.lhs.M_plus_Ainf - Ainf
    C = setup.lhs.C
    G = setup.constraints.jacobian(setup.xi0); m = G.shape[0]
    A = np.asarray(hdb.A); B = np.asarray(hdb.B)
    Ag = np.zeros((ndof, ndof)); Ag[np.ix_(hydro, hydro)] = A[:, :, i]
    Bg = np.zeros((ndof, ndof)); Bg[np.ix_(hydro, hydro)] = B[:, :, i]
    Zrad = -omega**2 * (M + Ag) + 1j * omega * Bg + C
    Fexc = np.zeros(ndof, complex)
    Fexc[hydro] = interpolate_rao(hdb, float(omega), 0.0) * amp

    def saddle(Zc, rhs_top):  # type: ignore[no-untyped-def]
        K = np.zeros((ndof + m, ndof + m), complex)
        K[:ndof, :ndof] = Zc; K[:ndof, ndof:] = G.T; K[ndof:, :ndof] = G
        return np.linalg.solve(K, np.concatenate([rhs_top, np.zeros(m)]))[:ndof]

    xhat = saddle(Zrad, Fexc)                       # radiation-only start
    if not drag:
        return xhat, 0.0
    xi0 = setup.xi0; sf = setup.state_force
    for it in range(200):
        Fdr = harmonic_drag(sf, xi0, xhat, omega, ndof)
        v2 = (omega * np.abs(xhat)) ** 2
        b = np.where(v2 > 1e-16, -np.real(Fdr * np.conj(1j * omega * xhat)) / np.maximum(v2, 1e-16), 0.0)
        b = np.clip(b, 0.0, None)
        D = np.zeros((ndof, ndof)); D[np.diag_indices(ndof)] = b
        xnew = saddle(Zrad + 1j * omega * D, Fexc + Fdr + 1j * omega * (b * xhat))
        rel = np.linalg.norm(xnew - xhat) / max(np.linalg.norm(xnew), 1e-30)
        xhat = 0.6 * xnew + 0.4 * xhat
        if verbose and it % 10 == 0:
            print(f"    it{it:3d} rel={rel:.2e} buoyRAO={np.abs(xhat[buoy_hv[6]]) / amp:.3f}")
        if rel < 1e-4:
            break
    Fdr = harmonic_drag(sf, xi0, xhat, omega, ndof)
    res = float(np.linalg.norm((Zrad @ xhat - Fexc - Fdr)[np.r_[buoy_hv, plat_hv]]))
    return xhat, res


def build12():  # type: ignore[no-untyped-def]
    import platform_fin_fan as pff, platform_rao_pilot as prp
    hdb = pff._hdb("none"); setup = pff._build(pff._R_SPAR, 5.0, hdb)
    hydro = np.asarray(prp._hydro_dof(prp._deck_with_drag()))
    return (hdb, setup, hydro, [6 * prp._buoy_body_index(k) + 2 for k in range(12)],
            6 * prp._buoy_body_index_platform() + 2)


def build16():  # type: ignore[no-untyped-def]
    import platform16_fin_fan as pff, platform16_rao as prp
    hdb = pff._hdb("none"); setup = pff._build(pff._R_SPAR, 5.0, hdb)
    hydro = np.asarray(prp._hydro_dof(prp._deck_with_drag()))
    return (hdb, setup, hydro, [6 * prp._buoy_body_index(k) + 2 for k in range(16)],
            6 * prp._buoy_body_index_platform() + 2)


def main() -> None:
    T = 2.50; omega = 2 * np.pi / T; amp = 0.02  # H=0.04, the measured peak height
    meas = {12: 4.27, 16: 2.80}
    print(f"drag-limited buoy heave RAO at T={T}s, H={2 * amp}m (no-fin, Cd_n=5)\n")
    print(f"{'model':8} {'N':>3} {'rad-only':>9} {'drag-FD':>8} {'measured':>9} {'platRAO':>8} "
          f"{'buoy/plat':>9} {'perbuoyExc':>11} {'sumB_drag':>10}")
    for N, builder in [(12, build12), (16, build16)]:
        hdb, setup, hydro, bhv, phv = builder()
        xr, _ = solve(hdb, setup, hydro, omega, amp, bhv, phv, drag=False)
        xd, _ = solve(hdb, setup, hydro, omega, amp, bhv, phv, drag=True)
        ndof = setup.lhs.M_plus_Ainf.shape[0]
        Fdr = harmonic_drag(setup.state_force, setup.xi0, xd, omega, ndof)
        v2 = (omega * np.abs(xd)) ** 2
        b = np.clip(np.where(v2 > 1e-16, -np.real(Fdr * np.conj(1j * omega * xd)) / np.maximum(v2, 1e-16), 0.0), 0, None)
        exc = abs(interpolate_rao(hdb, float(omega), 0.0))[list(hydro).index(bhv[6])] * amp
        rr = np.abs(xr[bhv[6]]) / amp; dr = np.abs(xd[bhv[6]]) / amp; pr = np.abs(xd[phv]) / amp
        print(f"{str(N) + '-buoy':8} {N:3d} {rr:9.2f} {dr:8.3f} {meas[N]:9.2f} {pr:8.3f} "
              f"{dr / pr:9.2f} {exc:11.3f} {sum(b[d] for d in bhv):10.1f}")


if __name__ == "__main__":
    main()
