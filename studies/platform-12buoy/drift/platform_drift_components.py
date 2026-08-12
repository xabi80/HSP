"""Component-resolved surge-drift force balance (FS2 revised).

The single "Morison drag" row hides opposing contributions (FloatFEA's Cd_n
sweep shows the drift reverses sign at Cd_n/10). Split the drag by component and
add the radiation memory row, all in one steady-state balance:

    exc + spar + plate_normal + plate_tangential + radiation  ==  (M+A_inf)a + Cx  (~0)

Isolation: plate_normal = PlateDragElement(Cd_t=0); plate_tangential =
PlateDragElement(Cd_n=0) -- the two act on independent velocity components and
add. Note the model builds plate-normal DISTRIBUTED over n_radial*n_azimuthal
patches (rim carries pitch velocity) but plate-tangential LUMPED at the disc
centre (pitch contribution discarded -- the non-distribution-resolved term).

Radiation is measured independently of the balance: mean radiation force
(surge) = -v0*kappa, kappa = sum_{i,j in surge} integral K_ij ds (linear brake).

Case: fin 0.215, T=3.141 s, H=0.08 m, ramp 20 s.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

_STUDY = Path(__file__).resolve().parent.parent  # studies/platform-12buoy (this file: drift/)
sys.path.insert(0, str(_STUDY / "fin_study"))
sys.path.insert(0, str(_STUDY))
sys.path.insert(0, str(_STUDY.parent / "cluster-3buoy-rigid"))

import platform_fin_fan as pff  # noqa: E402
import platform_rao_pilot as prp  # noqa: E402
from floatsim.hydro.excitation import make_regular_wave_force  # noqa: E402
from floatsim.hydro.morison import (  # noqa: E402
    MorisonElement,
    PlateDragElement,
    make_morison_state_force,
)
from floatsim.io.deck import PlateMember  # noqa: E402
from floatsim.solver.newmark import integrate_cummins  # noqa: E402
from floatsim.solver.ramp import HalfCosineRamp  # noqa: E402
from floatsim.waves.regular import RegularWave  # noqa: E402

_PLAT = 16
_T, _H, _DT, _RAMP, _DUR = 3.141, 0.08, 0.01, 20.0, 180.0
_SIDX = [6 * b for b in range(17)]


def _forces(deck, n_dof, rho):  # type: ignore[no-untyped-def]
    """spar, plate-normal (Cd_t=0), plate-tangential (Cd_n=0) Morison sub-forces."""
    spar, pn, pt = [], [], []
    for k, body in enumerate(deck.bodies):
        for e in body.drag_elements:
            if isinstance(e, PlateMember):
                base = dict(body_index=k, center_body=np.asarray(e.center, float),
                            normal_body=np.asarray(e.normal, float), radius=e.radius,
                            thickness=e.thickness, n_radial=e.n_radial, n_azimuthal=e.n_azimuthal)
                pn.append(PlateDragElement(Cd_n=e.Cd_n, Cd_t=0.0, **base))
                pt.append(PlateDragElement(Cd_n=0.0, Cd_t=e.Cd_t, **base))
            else:
                spar.append(MorisonElement(
                    body_index=k, node_a_body=np.asarray(e.node_a, float),
                    node_b_body=np.asarray(e.node_b, float), diameter=e.diameter,
                    Cd=e.Cd, Ca=e.Ca, include_inertia=False))
    calm = np.zeros(3)

    def fl(_p, _t):  # type: ignore[no-untyped-def]
        return calm
    mk = lambda els: make_morison_state_force(els, n_dof=n_dof, fluid_velocity_fn=fl, rho=rho)
    return mk(spar), mk(pn), mk(pt)


def main() -> None:
    deck = prp._deck_with_drag()
    hdb = pff._hdb("0215")
    hydro_dof = prp._hydro_dof(deck)
    setup = pff._build(0.215, 5.0, hdb)
    spar_f, pn_f, pt_f = _forces(deck, 102, deck.environment.water_density)

    omega = 2.0 * np.pi / _T
    wave = RegularWave(amplitude=0.5 * _H, omega=omega, heading_deg=0.0)
    f72 = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0),
                                  ramp=HalfCosineRamp(duration=_RAMP))

    def ext(t):  # type: ignore[no-untyped-def]
        f = np.zeros(102)
        f[hydro_dof] = f72(t)
        return f

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = integrate_cummins(
            lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0, xi_dot0=setup.xi_dot0,
            duration=_DUR, dt=_DT, rho_inf=0.8, constraints=setup.constraints,
            external_force=ext, state_force=setup.state_force, projection_interval=1)

    T = 2 * np.pi / omega
    m = r.t >= (r.t[-1] - 18 * T)
    idx = np.where(m)[0]
    v0 = float(np.mean(r.xi_dot[m][:, 6 * _PLAT]))

    Ksub = setup.kernel.K[np.ix_(_SIDX, _SIDX)]
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x renamed trapz
    kappa = float(_trapz(Ksub, dx=setup.kernel.dt, axis=2).sum())
    rad = -v0 * kappa

    def msum(fn):  # type: ignore[no-untyped-def]
        return float(np.mean([fn(r.t[n], r.xi[n], r.xi_dot[n])[_SIDX].sum() for n in idx]))
    e_ = float(np.mean([ext(r.t[n])[_SIDX].sum() for n in idx]))
    sp, pnn, ptt = msum(spar_f), msum(pn_f), msum(pt_f)
    ma = float(np.mean([(setup.lhs.M_plus_Ainf @ r.xi_ddot[n])[_SIDX].sum() for n in idx]))
    cx = float(np.mean([(setup.lhs.C @ r.xi[n])[_SIDX].sum() for n in idx]))

    # A3: separate pure rectification from drift-resistance. Each drag row is the
    # oscillation-rectified force PLUS the quadratic resistance to the -x drift,
    # inseparable at one drift state. Re-evaluate with the mean drift velocity
    # removed from the surge DOFs (= zero drift, exact since v0 << v_osc): what
    # remains is pure rectification; the difference is the drift-resistance.
    def msum0(fn):  # type: ignore[no-untyped-def]
        acc = 0.0
        for n in idx:
            xd = r.xi_dot[n].copy()
            xd[_SIDX] -= v0
            acc += fn(r.t[n], r.xi[n], xd)[_SIDX].sum()
        return acc / len(idx)
    sp0, pn0, pt0 = msum0(spar_f), msum0(pn_f), msum0(pt_f)

    print(f"Case fin 0.215 T={_T}s H={_H}m ramp{_RAMP:g}s  |  drift v0 = {v0*1000:.4f} mm/s  "
          f"(-x)\nkappa(surge radiation damping) = {kappa:.3f} N*s/m\n")
    print("Mean SYSTEM surge force over the steady window (N)   [+x downwave / -x upwave]:")
    rows = [("wave excitation", e_), ("drag  spar cylinder", sp),
            ("drag  plate-NORMAL  (Cd_n=5.0, distributed)", pnn),
            ("drag  plate-TANGENTIAL (Cd_t=1.5, centre-lumped)", ptt),
            ("radiation memory (-v0*kappa, linear)", rad)]
    for lab, val in rows:
        print(f"  {lab:48}: {val:+.4e}  {'(-x)' if val < 0 else '(+x)'}")
    rhs = e_ + sp + pnn + ptt + rad
    print(f"  {'-'*66}")
    print(f"  {'SUM of forces (RHS)':48}: {rhs:+.4e}")
    print(f"  {'(M+A_inf)a + C x  (target ~0)':48}: {ma+cx:+.4e}")
    print(f"  {'CLOSURE residual':48}: {rhs-(ma+cx):+.4e} N")

    drivers = [(l, v) for l, v in rows if v < 0 and l != "wave excitation"]
    brakes = [(l, v) for l, v in rows if v > 0]
    print("\nOpposing contributions (the net -0.0136 N drag hid these):")
    print("  DRIVING (-x):  " + ", ".join(f"{l.split()[1]}={v:+.4f}" for l, v in drivers))
    print("  BRAKING (+x):  " + ", ".join(f"{l.split()[1]}={v:+.4f}" for l, v in brakes))
    print(f"\n  excitation mean = {e_:+.4e} N vs {0.5*np.ptp([ext(r.t[n])[_SIDX].sum() for n in idx]):.1f} N "
          "swing  (DR2: suggestive, sign convention still formally untested)")

    print("\nA3 -- pure rectification (drift velocity removed) vs drift-resistance (N):")
    for lab, act, rect in [("spar", sp, sp0), ("plate-NORMAL", pnn, pn0),
                           ("plate-TANG", ptt, pt0)]:
        print(f"  {lab:13}: total {act:+.4e} = rectification {rect:+.4e} + "
              f"drift-resistance {act - rect:+.4e}")
    net_rect = e_ + sp0 + pn0 + pt0
    print(f"  net rectification at v0=0 (the true driver): {net_rect:+.4e} N "
          f"{'(-x, drives the drift)' if net_rect < 0 else '(+x)'}")
    print("  A3 check: a pure brake cannot reverse a drift -> the Cd_n/10 sign flip needs a")
    print(f"  genuine Cd_n-independent +x rectification; spar rectification = {sp0:+.4e} N "
          f"{'supplies it' if sp0 > 0 else 'does NOT'}.")


if __name__ == "__main__":
    main()
