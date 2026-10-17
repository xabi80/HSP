"""FloatSim in regular waves: the payload cases at their tilted equilibria (Xabier, 2026-10-16).

Cases (tilted_cases.json):
- the platform with 50 kg on the deck and the cluster with 12.5 kg on the hub;
- the payload shifted along +x until the deck / hub tilts 5 deg and 10 deg;
- for comparison, the same payload centred (level) and the unloaded article.
Waves: regular, H = 0.1 m, travelling along +x (along the shift), T = 1.5 / 2.0 / 2.35 / 2.65 /
3.5 / 5.0 s.

Each case is FloatSim's own articulated model, built about the case's static equilibrium.
- The records' decks (articulated_wall.deck, mooring_verify._cluster_deck) are rebuilt with every
  body where the large-angle statics put it:
  - each buoy at its own draft, still hanging vertically;
  - the hubs level;
  - the deck or hub tilted.
- The payload is folded into the deck (platform) or hub (cluster) body: combined mass, CG and
  full inertia tensor. The tilt enters through the products of inertia.
- The joints attach at the equilibrium pin and hub positions.
- The drag elements are the records', with each spar wetted up to its own waterline.
- The BEM is tilted_bem.py's coupled Capytaine run at that geometry: each buoy has its own draft
  and its own hydrostatic matrix.
- FloatSim's static solve has no weight term, so it keeps this geometry as the equilibrium.
- Drag acts relative to the wave's orbital velocity: the same wave and ramp as the BEM excitation
  (build_system drag_wave). The records' older runs used still-water drag.
- Free-floating, as the statics, except for a very soft horizontal hold on the deck / hub
  (surge period ~45 s, 10-20x the longest wave). Without it, the start-up transient lets the
  free article drift ~0.5 m over a run, and FloatSim's drag follows the drifting body while the
  BEM excitation stays at the mean position. The hold changes the wave-frequency response by
  ~1 % at T = 5 s and less below.
FloatSim's linear dynamics about a tilted equilibrium are valid for small wave motions, whatever
the static tilt.

Not in FloatSim's linear model (no weight term), reported alongside, not added:
- each pin load's tipping of its buoy, -P L (tilted_cases.json gives P L against the buoy's C55);
- the deck / payload weight's m g h on the tilt.

Per run, from the last 30 s:
- amplitudes of the deck / hub centre (heave, surge, pitch);
- the largest gimbal angle (buoy pitch relative to its hub);
- each pin's motion relative to the local water surface, and the smallest pin freeboard.
Per case: the natural periods of heave, deck / hub pitch and the buoys' in-phase tilt
(mooring_verify mode tracking).

Writes tilted_dynamics/<article>_<case>.json; `summary` collects them into tilted_dynamics.json.
Run: python tilted_dynamics.py <article> <case>   |   python tilted_dynamics.py summary
"""

# ruff: noqa: E402, E702  -- platform environment and sys.path set before the record imports;
# compact lines, as the study scripts
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import platform_statics as ps  # sets PLAT_SQUARE_M / PLAT_ROT_DEG before the record imports

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "flume-mooring"))
warnings.simplefilter("ignore")

import articulated_wall as aw
import mooring_verify as mv

from floatsim.driver import build_system
from floatsim.hydro.excitation import make_regular_wave_force
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import (
    Body,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    Output,
    PlateMember,
    Simulation,
    YawLockedJoint,
    distributed_cylinder_drag,
)
from floatsim.io.deck import RegularWave as DeckWave
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave

PERIODS_S = (1.5, 2.0, 2.35, 2.65, 3.5, 5.0)
WAVE_H = 0.1
DT, DURATION, RAMP_S, FIT_S = 0.01, 90.0, 15.0, 30.0       # articulated_wall.run_rao settings
OUT = HERE / "tilted_dynamics"
HOLD_N_PER_M = {"cluster": 2.0, "platform": 8.0}   # soft horizontal hold, surge period ~45 s
BEM = HERE / "tilted_bem"


def _inertia(J: list[list[float]]) -> Inertia:
    return Inertia(Ixx=J[0][0], Iyy=J[1][1], Izz=J[2][2], Ixy=J[0][1], Ixz=J[0][2], Iyz=J[1][2])


def _buoy(k: int, r: dict) -> Body:
    spar = distributed_cylinder_drag(z_bottom=aw._SPAR_BOT_B, z_top=-r["z_cog"],  # own waterline
                                     diameter=aw.SPAR_D, cd=aw.SPAR_CD, n_segments=10)
    plate = PlateMember(type="plate", center=[0.0, 0.0, aw._PLATE_B], normal=[0.0, 0.0, 1.0],
                        radius=aw.PLATE_R, thickness=aw.PLATE_T, Cd_n=aw.PLATE_CDN,
                        Cd_t=aw.PLATE_CDT)
    return Body(name=f"buoy{k + 1}", reference_point=[r["x"], r["y"], r["z_cog"]], mass=aw.M_BUOY,
                inertia=Inertia(Ixx=aw.IXX, Iyy=aw.IYY, Izz=aw.IZZ),
                hydro_body_label=f"buoy{k + 1}", initial_conditions=InitialConditions(),
                drag_elements=[*spar, plate])


def _gimbal(k: int, r: dict, owner: str, owner_ref: np.ndarray) -> YawLockedJoint:
    pin = np.asarray(r["pin"])
    return YawLockedJoint(type="yaw_locked", body_a=f"buoy{k + 1}", body_b=owner,
                          attach_a_body=[0.0, 0.0, float(pin[2] - r["z_cog"])],
                          attach_b_body=(pin - owner_ref).tolist(), axis=[0, 0, 1.0])


def make_deck(case: dict) -> tuple[Deck, dict]:
    """FloatSim deck at the case's equilibrium, plus where to read the results: the top body
    (hub or deck), its reference centre, each pin's owner body and offset."""
    top = case["top"]; cg = np.asarray(top["cg"])
    bodies: list = []; joints: list = []; pins: list = []
    if case["article"] == "cluster":
        for k, r in enumerate(case["buoys"]):
            bodies.append(_buoy(k, r)); joints.append(_gimbal(k, r, "hub", cg))
            pins.append((4, np.asarray(r["pin"]) - cg, np.asarray(r["pin"])))
        bodies.append(Body(name="hub", reference_point=cg.tolist(), mass=top["mass_kg"],
                           inertia=_inertia(top["inertia"]), structural=True))
        centre = np.asarray(case["hubs"][0]["centre"])
        owner = [4] * 4
    else:
        owner = []
        for c, h in enumerate(case["hubs"]):
            hc = np.asarray(h["centre"])
            for b in range(4):
                k = 4 * c + b; r = case["buoys"][k]
                bodies.append(_buoy(k, r)); joints.append(_gimbal(k, r, f"hub{c + 1}", hc))
                pins.append((5 * c + 4, np.asarray(r["pin"]) - hc, np.asarray(r["pin"])))
                owner.append(5 * c + 4)
            bodies.append(Body(name=f"hub{c + 1}", reference_point=hc.tolist(), mass=ps.M_HUB,
                               inertia=Inertia(Ixx=0.3, Iyy=0.3, Izz=0.6), structural=True))
            joints.append(YawLockedJoint(type="yaw_locked", body_a=f"hub{c + 1}", body_b="platform",
                                         attach_a_body=[0.0, 0.0, 0.0],
                                         attach_b_body=(hc - cg).tolist(), axis=[0, 0, 1.0]))
        bodies.append(Body(name="platform", reference_point=cg.tolist(), mass=top["mass_kg"],
                           inertia=_inertia(top["inertia"]), structural=True))
        centre = np.array([0.0, 0.0, ps.S_TOP + case["q"][0]])   # deck's pin-plane centre
    dk = Deck(simulation=Simulation(duration=10.0, dt=DT),
              environment=Environment(water_depth=200.0, water_density=aw.RHO, gravity=aw.G),
              waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
              bodies=bodies, joints=joints,
              shared_hydro_database=HydroDatabaseRef(format="capytaine", path="placeholder.nc"),
              output=Output(file="o.h5", channels=["heave"], sample_rate=10.0))
    return dk, {"article": case["article"], "top": len(bodies) - 1, "cg": cg, "centre": centre,
                "pins": pins, "owner": owner}


def fit(t: np.ndarray, x: np.ndarray, w: float) -> float:
    """Amplitude at w; the constant and a linear drift (free surge) are fitted out."""
    D = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t), t - t.mean()])
    c, *_ = np.linalg.lstsq(D, x, rcond=None)
    return float(np.hypot(c[0], c[1]))


def point_z(xi: np.ndarray, body: int, d: np.ndarray) -> np.ndarray:
    """Vertical displacement of a point at offset d from a body's reference (small rotations)."""
    b = 6 * body
    return xi[:, b + 2] + xi[:, b + 3] * d[1] - xi[:, b + 4] * d[0]


def run_wave(dk: Deck, info: dict, hdb, T: float) -> dict:  # type: ignore[no-untyped-def]
    w = 2 * np.pi / T
    wave = RegularWave(amplitude=WAVE_H / 2, omega=w, heading_deg=0.0)
    ramp = HalfCosineRamp(duration=RAMP_S)
    setup = build_system(dk, bem_databases={}, dt=DT, t_max_kernel=30.0, solve_equilibrium=True,
                         shared_hydro_database=hdb, asymptote_check_override=aw.OVR,
                         kernel_decay_floor_override=aw.OVR, drag_wave=wave, drag_wave_ramp=ramp)
    n = setup.lhs.M_plus_Ainf.shape[0]; hd = aw.hydro_dof(dk); tb = info["top"]
    k = HOLD_N_PER_M[info["article"]]
    lhs = replace(setup.lhs, C=setup.lhs.C + mv.point_k(n, tb, (0.0, 0.0, 0.0), (k, k, 0.0)))
    f_exc = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0), ramp=ramp)

    def ext(t: float) -> np.ndarray:
        f = np.zeros(n); f[hd] = f_exc(t); return f

    r = integrate_cummins(lhs=lhs, kernel=setup.kernel, xi0=setup.xi0,
                          xi_dot0=setup.xi_dot0, duration=DURATION, dt=DT, rho_inf=0.8,
                          constraints=setup.constraints, external_force=ext,
                          state_force=setup.state_force, projection_interval=1)
    m = r.t >= r.t[-1] - FIT_S; t = r.t[m]; xi = r.xi[m] - setup.xi0
    dc = info["centre"] - info["cg"]
    out = {"T_s": T, "heave_m": fit(t, point_z(xi, tb, dc), w),
           "surge_m": fit(t, xi[:, 6 * tb], w),
           "pitch_deg": float(np.degrees(fit(t, xi[:, 6 * tb + 4], w)))}
    gimbal = [float(np.degrees(fit(t, xi[:, 6 * k + 4] - xi[:, 6 * o + 4], w)))
              for k, o in enumerate(info["owner"])]
    out["gimbal_deg_max"] = max(gimbal); out["gimbal_deg"] = gimbal
    rel, fb = [], []
    for body, d, pin in info["pins"]:
        x_pin = pin[0] + xi[:, 6 * body]                      # the water surface where the pin is
        dz = point_z(xi, body, d) - np.asarray(wave.elevation(t, x_pin, pin[1]))
        rel.append(fit(t, dz, w)); fb.append(float(pin[2] + dz.min()))
    out["pin_rel_m"] = rel; out["pin_freeboard_min_m"] = fb
    out["min_freeboard_m"] = min(fb)
    out["max_abs_xi"] = float(np.max(np.abs(xi)))
    return out


def geometric_stiffness(case: dict, info: dict, n: int) -> np.ndarray:
    """Rotational stiffness of the constant vertical loads FloatSim's linear model has no weight
    term for. It is an estimate, used for the natural periods only. A load F at an arm a from a
    body's reference adds (a . F) I - (a F^T + F a^T) / 2 to that body's rotations. The loads:
    - on each buoy: its pin load P pushing down at the pin, L above its CoG (the -P L term);
    - on the hub or hubs: the buoys' pin loads pushing up at the pins;
    - on the deck (platform): each hub's net load pushing up at the hub centre, about the
      deck + payload CG (the -m g h term).
    Every body's weight acts at its own reference (its CG), so it adds nothing; buoyancy is in C."""
    K = np.zeros((n, n))

    def add(body: int, a: np.ndarray, Fz: float) -> None:
        F = np.array([0.0, 0.0, Fz])
        K[6 * body + 3:6 * body + 6, 6 * body + 3:6 * body + 6] += (
            (a @ F) * np.eye(3) - 0.5 * (np.outer(a, F) + np.outer(F, a)))

    top, cg = info["top"], info["cg"]
    for k, r in enumerate(case["buoys"]):
        pin, P, owner = np.asarray(r["pin"]), r["pin_load_N"], info["owner"][k]
        bi = k if case["article"] == "cluster" else 5 * (k // 4) + k % 4
        add(bi, np.array([0.0, 0.0, pin[2] - r["z_cog"]]), -P)
        ref = cg if owner == top else np.asarray(case["hubs"][k // 4]["centre"])
        add(owner, pin - ref, +P)
    if case["article"] == "platform":
        for c, h in enumerate(case["hubs"]):
            Fc = sum(r["pin_load_N"] for r in case["buoys"][4 * c:4 * c + 4]) - ps.M_HUB * ps.G_ACC
            add(top, np.asarray(h["centre"]) - cg, +Fc)
    return K


def natural_periods(dk: Deck, nc: Path, top: int, K_extra: np.ndarray | None = None) -> dict:
    art = mv._articulated("case", dk, nc, top, ps.S_TOP, [])
    n = art["n"]
    K0 = art["C"] + mv.roll_split(n) + mv.point_k(n, top, (0.0, 0.0, 0.0), (1e-6, 1e-6, 0.0), 1e-6)
    if K_extra is not None:
        K0 = K0 + K_extra
    out = {}
    for name, pick, T0 in (("heave", mv.by_mac(mv.rigid_pattern(art, "heave")), 2.5),
                           ("pitch", mv.by_dof(6 * top + 4), 2.6),
                           ("buoy_tilt", mv.by_mac(mv.rigid_pattern(art, "tilt")), 2.9)):
        T, _, score = mv.track(art, K0, pick, T0)
        out[name] = {"T_s": float(T), "score": float(score)}
    return out


def periods(case: dict, dk: Deck, info: dict, nc: Path) -> dict:
    """FloatSim's natural periods, and the same with the pin-load tipping added (estimate)."""
    n = 6 * len(dk.bodies)
    out = {"periods": natural_periods(dk, nc, info["top"]),
           "periods_pin_tipping": natural_periods(dk, nc, info["top"],
                                                  geometric_stiffness(case, info, n))}
    for key, lab in (("periods", "FloatSim"), ("periods_pin_tipping", "+ pin-load tipping")):
        print(f"{case['article']} {case['case']}: {lab:18s} "
              + ", ".join(f"{k} {v['T_s']:.3f} s" for k, v in out[key].items()), flush=True)
    return out


def load_case(article: str, name: str) -> tuple[dict, Path]:
    cases = json.loads((HERE / "tilted_cases.json").read_text(encoding="utf-8"))["cases"]
    case = next(c for c in cases if c["article"] == article and c["case"] == name)
    return case, BEM / f"{article}_{name}_psd.nc"


def update_periods(article: str, name: str) -> None:
    """Recompute the natural periods of a finished case into its JSON."""
    case, nc = load_case(article, name)
    dk, info = make_deck(case)
    p = OUT / f"{article}_{name}.json"
    res = json.loads(p.read_text(encoding="utf-8"))
    res.update(periods(case, dk, info, nc))
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")


def run_case(article: str, name: str) -> None:
    case, nc = load_case(article, name)
    dk, info = make_deck(case)
    hdb = read_capytaine(nc)
    t0 = time.perf_counter()
    res = {"article": article, "case": name, "m_kg": case["m_kg"], "offset_m": case["offset_m"],
           "tilt_deg": case["tilt_deg"], "wave_H_m": WAVE_H,
           "static_freeboard_m": [float(p[2][2]) for p in info["pins"]],
           **periods(case, dk, info, nc), "runs": []}
    for T in PERIODS_S:
        o = run_wave(dk, info, hdb, T)
        res["runs"].append(o)
        print(f"  T {T:4.2f} s: heave {1e3 * o['heave_m']:6.1f} mm, "
              f"pitch {o['pitch_deg']:5.2f} deg, gimbal {o['gimbal_deg_max']:5.2f} deg, "
              f"pin rel {1e3 * max(o['pin_rel_m']):6.1f} mm, "
              f"min freeboard {1e3 * o['min_freeboard_m']:5.0f} mm "
              f"({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{article}_{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def summary() -> None:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*.json"))]
    (HERE / "tilted_dynamics.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote tilted_dynamics.json ({len(rows)} cases)")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    if sys.argv[1] == "summary":
        summary()
    elif sys.argv[1] == "periods":
        update_periods(sys.argv[2], sys.argv[3])
    else:
        run_case(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
