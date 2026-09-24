"""Reach of the drag-kinematics gap on the M11b PR8 RAO fan (study-side; no floatsim/ change).

The driver wires the deck's Morison drag against CALM water (floatsim/driver.py:469-474:
``fluid_velocity_fn=_calm_fluid``), so the PR8 fan (platform_rao_pilot.run_case ->
setup.state_force) has body-velocity drag and no drag excitation. OrcaFlex uses relative-velocity
Morison. This script re-runs selected fan cases with the SAME FloatSim model (same deck, BEM,
kernel, joints, integrator, adaptive settle) and three drag variants:

  calm   the fan as recorded, under the current code (baseline; separates the post-PR8
         trapezoidal-convolution fix 9fb5b33 from the drag effect)
  rel    relative-velocity drag, wired study-side through FloatSim's own
         ``make_morison_state_force`` / ``MorisonElement`` / ``PlateDragElement``, with two
         corrections the fluid path needs (both found in this check, both latent in floatsim/):
           (i)  absolute sampling point: ``make_morison_state_force`` samples the fluid at
                ``xi[0:3] + R @ arm`` (morison.py:798, :817), but build_system's xi is the
                DISPLACEMENT from each body's reference_point, so the body's reference point is
                added back per body (one closure per body);
           (ii) vertical orbital-velocity sign: ``airy_velocity`` returns u_z = +A w e^kz sin(psi)
                (kinematics.py:108), which is -d(eta)/dt against RegularWave.elevation (fails the
                kinematic free-surface BC, continuity and irrotationality by exactly 2x); the
                standard deep-water field (the module's own citation: Newman 6.3 / Faltinsen 2.34)
                has u_z = -A w e^kz sin(psi), so u_z is negated.
         The fluid velocity is ramped with the same HalfCosineRamp as the excitation.
  driver the FloatSim wiring (STEP 5 PR1 commit c): ``build_system(drag_wave=,
         drag_wave_ramp=)``. Gate: must reproduce "rel" exactly.
  naive  what wiring ``airy_velocity`` straight into the driver's drag builder would give
         (as-shipped vertical sign, displacement-relative sampling point) -- the STEP 5 PR1
         outcome without the two corrections.

Wiring check (run first): with a zero fluid field, the per-body composition reproduces the
driver's drag state force to machine precision at random states.

Usage: python pr8_reldrag_check.py <H> <T> <variant> [--history]
                                                         (writes pr8_reldrag_out/<case>.json;
                                                          --history also saves the full run)
       python pr8_reldrag_check.py check                (wiring check only)
       python pr8_reldrag_check.py summary              (table vs the recorded fan / sweep)
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import platform_rao_pilot as prp  # noqa: E402

from floatsim.driver import build_system  # noqa: E402
from floatsim.hydro.morison import (  # noqa: E402
    MorisonElement,
    PlateDragElement,
    make_morison_state_force,
)
from floatsim.hydro.readers.capytaine import read_capytaine  # noqa: E402
from floatsim.io.deck import PlateMember  # noqa: E402
from floatsim.solver.ramp import HalfCosineRamp  # noqa: E402
from floatsim.waves.kinematics import airy_velocity  # noqa: E402
from floatsim.waves.regular import RegularWave  # noqa: E402

OUT = _HERE / "pr8_reldrag_out"


def _airy_sign_fixed() -> bool:
    """True once floatsim's airy_velocity satisfies w = d(eta)/dt at the MWL (STEP 5 PR1
    commit a). Before it, "rel" negates u_z; after it, "naive" negates u_z to reproduce the
    as-shipped field. Either way the variants mean the same physics as in the record."""
    w = RegularWave(amplitude=1.0, omega=1.0, heading_deg=0.0)
    t, h = 0.3, 1.0e-6
    deta = (float(w.elevation(t + h)) - float(w.elevation(t - h))) / (2 * h)
    return bool(np.sign(airy_velocity(w, np.zeros(3), t)[2]) == np.sign(deta))


_SIGN_FIXED = _airy_sign_fixed()
RAMP_S, CAP_S = 20.0, 450.0
RECORDED = {  # platform-heave RAO as committed (fan: pr8_fan_out/rao_summary.csv;
    # T = 6.0: pr8_pilot_out/period_sweep_ext_H0p30.csv, 958c19f)
    (0.05, 3.141): 1.3934951158349806,
    (1.0, 3.141): 0.27666476923662364,
    (0.3, 2.0): 0.0018547565028258109,
    (0.3, 6.0): 0.991469645757887,
}


def _elements_by_body(deck) -> dict[int, list]:  # type: ignore[no-untyped-def]
    """The deck's drag elements, constructed exactly as floatsim/driver.py:431-466 does."""
    out: dict[int, list] = {}
    for k, body in enumerate(deck.bodies):
        for e in body.drag_elements:
            if isinstance(e, PlateMember):
                el = PlateDragElement(
                    body_index=k, center_body=np.asarray(e.center, dtype=np.float64),
                    normal_body=np.asarray(e.normal, dtype=np.float64), radius=e.radius,
                    thickness=e.thickness, Cd_n=e.Cd_n, Cd_t=e.Cd_t, n_radial=e.n_radial,
                    n_azimuthal=e.n_azimuthal)
            else:
                assert not e.include_inertia
                el = MorisonElement(
                    body_index=k, node_a_body=np.asarray(e.node_a, dtype=np.float64),
                    node_b_body=np.asarray(e.node_b, dtype=np.float64), diameter=e.diameter,
                    Cd=e.Cd, Ca=e.Ca, include_inertia=False)
            out.setdefault(k, []).append(el)
    return out


def _sum(forces):  # type: ignore[no-untyped-def]
    def f(t, xi, xd):  # type: ignore[no-untyped-def]
        tot = forces[0](t, xi, xd)
        for g in forces[1:]:
            tot = tot + g(t, xi, xd)
        return tot
    return f


def drag_state_force(deck, n_dof: int, rho: float, variant: str, wave=None):  # type: ignore[no-untyped-def]
    ramp = HalfCosineRamp(duration=RAMP_S)
    els = _elements_by_body(deck)
    if variant == "zero":
        zero = np.zeros(3)
        return _sum([make_morison_state_force(e, n_dof=n_dof, fluid_velocity_fn=lambda p, t: zero,
                                              rho=rho) for e in els.values()])
    if variant == "naive":
        flat = [e for es in els.values() for e in es]

        def shipped(p, t):  # type: ignore[no-untyped-def]
            u = airy_velocity(wave, p, t)
            if _SIGN_FIXED:          # reproduce the pre-fix (as-shipped) field
                u = u.copy()
                u[2] = -u[2]
            return ramp.value(t) * u
        return make_morison_state_force(flat, n_dof=n_dof, rho=rho, fluid_velocity_fn=shipped)
    assert variant == "rel"
    forces = []
    for k, es in els.items():
        ref = np.asarray(deck.bodies[k].reference_point, dtype=np.float64)

        def fluid(p, t, ref=ref):  # type: ignore[no-untyped-def]
            u = airy_velocity(wave, p + ref, t)          # (i) absolute sampling point
            if not _SIGN_FIXED:
                u[2] = -u[2]                              # (ii) u_z = d(eta)/dt at the MWL
            return ramp.value(t) * u
        forces.append(make_morison_state_force(es, n_dof=n_dof, fluid_velocity_fn=fluid, rho=rho))
    return _sum(forces)


def _setup(**drag_wave):  # type: ignore[no-untyped-def]
    deck = prp._deck_with_drag()
    # the fan deck has no connectors / catenaries: drag is its only state force
    assert not deck.connections
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        setup = build_system(deck, bem_databases={}, dt=0.01, t_max_kernel=30.0,
                             solve_equilibrium=False,
                             shared_hydro_database=read_capytaine(prp._PLAT_NC),
                             asymptote_check_override=prp._ASYMPTOTE_OVR,
                             kernel_decay_floor_override=prp._KERNEL_EXEMPT, **drag_wave)
    return deck, setup


def wiring_check() -> float:
    deck, setup = _setup()
    n = setup.lhs.n_dof
    mine = drag_state_force(deck, n, deck.environment.water_density, "zero")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(50):
        xi = rng.normal(0, 0.05, n)
        xd = rng.normal(0, 0.3, n)
        a, b = setup.state_force(1.0, xi, xd), mine(1.0, xi, xd)
        worst = max(worst, float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)))
    print(f"wiring check: max relative |driver drag - per-body composition| = {worst:.2e}")
    return worst


def run(H: float, T: float, variant: str, history: bool = False) -> None:
    OUT.mkdir(exist_ok=True)
    if variant == "driver":
        wave = RegularWave(amplitude=0.5 * H, omega=2 * np.pi / T, heading_deg=0.0)
        deck, setup = _setup(drag_wave=wave, drag_wave_ramp=HalfCosineRamp(duration=RAMP_S))
    else:
        deck, setup = _setup()
    n = setup.lhs.n_dof
    if variant not in ("calm", "driver"):
        wave = RegularWave(amplitude=0.5 * H, omega=2 * np.pi / T, heading_deg=0.0)
        setup = dataclasses.replace(setup, state_force=drag_state_force(
            deck, n, deck.environment.water_density, variant, wave))
    captured: dict = {}  # type: ignore[type-arg]
    orig = prp.integrate_cummins

    def _capture(*a, **k):  # type: ignore[no-untyped-def]
        captured["r"] = orig(*a, **k)
        return captured["r"]

    prp.integrate_cummins = _capture
    t0 = time.perf_counter()
    c = prp.run_case(setup, read_capytaine(prp._PLAT_NC), prp._hydro_dof(deck), height_m=H,
                     period_s=T, ramp_s=RAMP_S, cap_settle_s=CAP_S,
                     window_periods=8.0 if T < 2.6 else 6.0, dt=0.01)
    buoys = [c["rao"][f"buoy{k}_heave"] for k in range(1, 13)]
    rec = {"H": H, "T": T, "variant": variant, "rao_platform_heave": c["rao"]["platform_heave"],
           "rao_buoy_heave_min": min(buoys), "rao_buoy_heave_max": max(buoys),
           "rao_buoy_heave": buoys, "settled": c["settled"], "settle_ratio": c["settle_ratio"],
           "duration_s": c["duration_s"], "wall_s": time.perf_counter() - t0}
    stem = f"H{H:g}_T{T:g}_{variant}".replace(".", "p")
    (OUT / f"{stem}.json").write_text(json.dumps(rec, indent=1))
    if history:        # full run for the response analysis (not committed: ~15 MB)
        r = captured["r"]
        b = [prp._buoy_body_index(k) for k in range(12)]
        pdof = 6 * prp._buoy_body_index_platform()
        np.savez_compressed(OUT / f"history_{stem}.npz", t=r.t, platform=r.xi[:, pdof:pdof + 6],
                            buoy_heave=r.xi[:, [6 * k + 2 for k in b]],
                            buoy_pitch=r.xi[:, [6 * k + 4 for k in b]],
                            buoy_roll=r.xi[:, [6 * k + 3 for k in b]], omega=2 * np.pi / T, H=H)
    print(f"H {H} T {T} {variant}: platform-heave RAO {rec['rao_platform_heave']:.4f} "
          f"(settled {rec['settled']}, {rec['duration_s']:.0f} s sim, {rec['wall_s']:.0f} s wall)")


def summary() -> None:
    rows = []
    for (H, T), rec_v in RECORDED.items():
        r = {"H": H, "T": T, "recorded": rec_v}
        for v in ("calm", "rel", "naive"):
            f = OUT / (f"H{H:g}_T{T:g}_{v}".replace(".", "p") + ".json")
            r[v] = json.loads(f.read_text()) if f.exists() else None
        rows.append(r)
    print(f"{'H':>5} {'T':>6} {'recorded':>9} {'calm':>9} {'rel':>9} {'naive':>9}  "
          f"{'rel/calm':>8} {'naive/calm':>10}  settled(c,r,n)")
    for r in rows:
        g = {v: (r[v]["rao_platform_heave"] if r[v] else float("nan"))
             for v in ("calm", "rel", "naive")}
        st = ",".join(str(r[v]["settled"])[0] if r[v] else "-" for v in ("calm", "rel", "naive"))
        print(f"{r['H']:>5} {r['T']:>6} {r['recorded']:>9.4f} {g['calm']:>9.4f} {g['rel']:>9.4f} "
              f"{g['naive']:>9.4f}  {g['rel'] / g['calm']:>8.3f} "
              f"{g['naive'] / g['calm']:>10.3f}  {st}")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    if sys.argv[1] == "check":
        wiring_check()
    elif sys.argv[1] == "summary":
        summary()
    else:
        run(float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], "--history" in sys.argv)
