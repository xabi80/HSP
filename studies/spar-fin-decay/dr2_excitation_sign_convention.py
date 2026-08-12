"""DR2 — excitation sign-convention test on a SINGLE isolated spar-fin buoy.

**PHASE IS THE ENTIRE TEST; AMPLITUDE PROVES NOTHING.** A 180° excitation sign
error leaves |RAO| completely unchanged, so an amplitude comparison would happily
pass a flipped convention. This test compares the RESPONSE PHASE (relative to the
wave elevation at the body) two ways and against a physical anchor.

Why one isolated body: the excitation sign convention is a property of how force
is applied to ONE body, so a single 6-DOF buoy isolates it and the
frequency-domain solution is a closed-form 6×6 — no assembly, no joints, so the
test probes the convention and not the multi-body duplication.

Conditions that make it exact rather than approximate:
  - DRAG OFF (``state_force=None``) — linear FD vs linear TD, no
    equivalent-linearisation error in between.
  - OFF-RESONANCE, well clear of the heave mode (T_n ≈ 2.97 s) — phase is flat in
    ω there, so a small frequency mismatch cannot masquerade as a phase error.
  - SURGE and HEAVE — a sign error may be axis-specific.

Two independent references:
  (1) FD closed form:  Z(ω) = -ω²(M + A(ω)) + iω B(ω) + C ;  X_fd = Z⁻¹ (RAO·A),
      response x(t) = Re{X e^{+iωt}}. Computed here, NOT via the TD force closure,
      so a bug in ``make_regular_wave_force`` shows as TD≠FD.
  (2) Physical anchor: below the heave resonance (long waves) the buoy rides the
      wave, so heave must be IN PHASE with η (arg ≈ 0°). A flipped sign → 180°.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # ω/°/Δ in the report; Windows console is cp1252

import study_common as sc  # noqa: E402

from floatsim.hydro.excitation import interpolate_rao, make_regular_wave_force  # noqa: E402
from floatsim.solver.equilibrium import solve_static_equilibrium  # noqa: E402
from floatsim.solver.newmark import integrate_cummins  # noqa: E402
from floatsim.solver.ramp import HalfCosineRamp  # noqa: E402
from floatsim.waves.regular import RegularWave  # noqa: E402

_DOF = {"surge": 0, "heave": 2}
_A = 0.02  # wave amplitude (m); irrelevant to phase, small to stay linear


def _AB(hdb, omega):  # type: ignore[no-untyped-def]
    w = np.asarray(hdb.omega)
    i = int(max(0, min(np.searchsorted(w, omega) - 1, w.size - 2)))
    t = (omega - w[i]) / (w[i + 1] - w[i])
    A = np.asarray(hdb.A)
    B = np.asarray(hdb.B)
    return (1 - t) * A[:, :, i] + t * A[:, :, i + 1], (1 - t) * B[:, :, i] + t * B[:, :, i + 1]


def _fd(hdb, lhs, omega):  # type: ignore[no-untyped-def]
    Aw, Bw = _AB(hdb, omega)
    M = lhs.M_plus_Ainf - np.asarray(hdb.A_inf)
    Z = -(omega**2) * (M + Aw) + 1j * omega * Bw + lhs.C
    Fhat = interpolate_rao(hdb, omega, 0.0) * _A  # A_wave = A at origin (phi=0)
    return np.linalg.solve(Z, Fhat)  # (6,) complex; x(t)=Re{X e^{+iwt}}


def _td(lhs, kernel, hdb, xi_eq, omega, dur=280.0, dt=0.01):  # type: ignore[no-untyped-def]
    wave = RegularWave(amplitude=_A, omega=omega, heading_deg=0.0)
    f6 = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0),
                                 ramp=HalfCosineRamp(duration=20.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = integrate_cummins(lhs=lhs, kernel=kernel, xi0=xi_eq.copy(),
                                xi_dot0=np.zeros(6), duration=dur, dt=dt, rho_inf=0.8,
                                external_force=f6, state_force=None)  # DRAG OFF
    t = res.t
    m = t >= dur - 120.0  # long window; lstsq at omega rejects any free-decay ringing
    D = np.column_stack([np.cos(omega * t[m]), np.sin(omega * t[m]), np.ones(int(m.sum()))])
    X = np.zeros(6, complex)
    for d in range(6):
        c, *_ = np.linalg.lstsq(D, res.xi[m, d] - xi_eq[d], rcond=None)
        X[d] = c[0] - 1j * c[1]  # Re{X e^{+iwt}} = a cos - b sin? no: = Xr cos - Xi sin => X=a - i b
    return X


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hdb = sc.load_hdb()
        lhs = sc.build_lhs(hdb)
        kernel = sc.build_kernel(hdb)
        eq = solve_static_equilibrium(lhs=lhs, state_force=None)
    tn = 2.0 * np.pi * np.sqrt(lhs.M_plus_Ainf[2, 2] / lhs.C[2, 2])
    print(f"single spar-fin buoy: heave T_n = {tn:.3f} s.  DRAG OFF.  A = {_A} m.")
    print("PHASE IS THE ENTIRE TEST; amplitude (|X|) is a sanity check only.\n")

    for T in (6.0, 4.5, 2.0):
        omega = 2.0 * np.pi / T
        reg = "below" if T > tn else "above"
        Xfd, Xtd = _fd(hdb, lhs, omega), _td(lhs, kernel, hdb, eq.xi_eq, omega)
        print(f"=== T = {T} s  (ω = {omega:.3f}, {reg} the {tn:.2f} s heave resonance) ===")
        print(f"  {'DOF':6} {'FD phase':>10} {'TD phase':>10} {'Δphase':>9} "
              f"{'|X|_FD':>10} {'|X|_TD':>10}")
        for name, d in _DOF.items():
            pf, pt = np.degrees(np.angle(Xfd[d])), np.degrees(np.angle(Xtd[d]))
            print(f"  {name:6} {pf:9.1f}° {pt:9.1f}° {_wrap(pt - pf):8.1f}° "
                  f"{abs(Xfd[d]):10.3e} {abs(Xtd[d]):10.3e}")
        if T > tn:  # physical anchor: long wave -> heave rides the wave (in phase)
            ph = np.degrees(np.angle(Xtd[2]))
            verdict = "IN PHASE (convention OK)" if abs(_wrap(ph)) < 30 else \
                ("ANTI-PHASE (SIGN FLIP!)" if abs(_wrap(ph - 180)) < 30 else "ambiguous")
            print(f"  physical anchor: heave should ride the long wave -> arg≈0°; "
                  f"measured {ph:+.1f}° -> {verdict}")
        print()


if __name__ == "__main__":
    main()
