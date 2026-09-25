"""FloatSim decks for the three flume articles, moored or free-floating.

Everything here is FloatSim: the bodies are deck ``Body`` entries, the hydrodynamics come from
the Capytaine databases through ``build_system``, drag is the deck's Morison ``drag_elements``
(FloatSim's still-water drag), and the station-keeping mooring is four FloatSim ``Catenary``
lines per article (Irvine elastic catenary, body to earth). Motions are integrated with
``integrate_cummins`` under ``make_regular_wave_force`` -- the same composition every FloatSim
study uses.

Articles
    1 buoy        one buoy, reference point = CoG (0, 0, -0.907); BEM single_osu_open (built by
                  ``bem_cluster.py single``: same hull, mesh, omega grid and coupled FloatSim
                  path as the cluster and platform; computed about the CoG, which is the deck
                  convention: reference point = body origin = CoG = BEM origin).
    1 cluster     4 buoys + hub (yaw-locked pins), 45 deg; coupled BEM cluster_osu_open_rot45.
    4x4 platform  16 buoys + 4 hubs + deck (45 deg); coupled BEM from flume-wall-effect.

Mooring lines (design point of mooring_sizing: T_surge = 15 s, wall anchors +-5 m, at the SWL)
    per line: axial stiffness k = EA / L0 and pretension T0 from ``mooring_sizing.xspread``;
    unstretched length L0 = chord - T0 / k; near-neutral line weight 0.02 N/m (light rope +
    spring in water). Buoy: 4 lines to the spar at the SWL; cluster: one line per spar;
    platform: each line's 2-leg bridle is two FloatSim lines from the same wall anchor to the two
    spars of the half-row (k/2, T0/2 each) -- FloatSim catenaries are body-to-earth only.

FloatSim note (anchor frame): the deck anchors are the TRUE inertial wall anchors. Since
flume-mooring Phase C2, ``build_system`` passes each body's reference point to
``make_catenary_state_force``, which places the fairlead at reference point + displacement + arm
(``build_single`` does the same). Before C2 the fairlead sat at displacement + arm, and this
module gave each anchor relative to the moored body's reference point instead (same force).

FloatSim note (moored articulated equilibrium): ``solve_static_equilibrium`` solves
C xi = F_state body by body and ignores the joints, so it cannot balance a line pull that one
pinned buoy passes to the hub: hybr walks that buoy toward its anchor until the catenary goes
slack and the line solver raises. ``moored_equilibrium`` lets FloatSim's own constrained
integrator find the state instead: start from FloatSim's unmoored equilibrium, settle SETTLE_S in
calm water (``integrate_cummins`` with the joints), take the mean over the last SETTLE_AVG_S, and
accept it only if the static residual projected on the joint-feasible subspace null(G) is within
FloatSim's own equilibrium tolerance (``_EQUILIBRIUM_TOL_N``). Cached in moored_equilibrium.json.

FloatSim note (single small body): a one-body database takes the driver's per-body path (M8 Q2
lock), whose retardation kernel runs the high-frequency asymptote gate with no small-body
override -- the override (ITEM25-SMALL-BODY-APPLICABILITY) exists only on the coupled path. The
lone OSU buoy (L ~ 1.7 m) cannot pass that gate, so ``build_single`` assembles it with the
driver's own per-body functions (_per_body_lhs, _materialise_catenary, _build_drag_state_force,
_compose_state_force, solve_static_equilibrium), identical to build_system's per-body branch
except that the kernel receives the same small-body override the coupled path accepts.
"""
# ruff: noqa: E402, E702  -- sys.path bootstrap first; compact lines
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PLAT_ROT_DEG", "45")

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE.parent.parent, HERE.parent / "platform-12buoy" / "flume-wall-effect", HERE):
    sys.path.insert(0, str(_p))

import articulated_wall as aw
import mooring_sizing as ms
import mooring_verify as mv

import floatsim.driver as fsd
from floatsim.driver import build_system
from floatsim.hydro.excitation import make_regular_wave_force
from floatsim.hydro.readers.capytaine import read_capytaine
from floatsim.io.deck import (
    Body,
    Catenary,
    CatenaryLine,
    Deck,
    Environment,
    HydroDatabaseRef,
    Inertia,
    InitialConditions,
    Output,
    PlateMember,
    Simulation,
    distributed_cylinder_drag,
)
from floatsim.io.deck import RegularWave as DeckWave
from floatsim.solver.newmark import integrate_cummins
from floatsim.solver.ramp import HalfCosineRamp
from floatsim.waves.regular import RegularWave

BUOY_NC = HERE / "single_osu_open_psd.nc"      # bem_cluster.py single (same pipeline/mesh)
CLUSTER_NC = HERE / "cluster_osu_open_rot45_psd.nc"
PLATFORM_NC = (HERE.parent / "platform-12buoy" / "flume-wall-effect"
               / f"coupled_osu_open{aw.SUF}_psd.nc")
NAMES = {"buoy": "1 buoy", "cluster": "1 cluster (4 buoys)", "platform": "4x4 platform (45°)"}
DT, T_KERNEL = 0.01, 30.0
W_LINE = 0.02                        # N/m, near-neutral line in water
WL_B = np.array([0.0, 0.0, -aw.ZB])  # spar at the SWL in a buoy's (CoG) frame
OVR = aw.OVR
EQ_CACHE = HERE / "moored_equilibrium.json"
SETTLE_S, SETTLE_AVG_S = 240.0, 30.0


def _drag(cd_scale: float = 1.0) -> list:
    spar = distributed_cylinder_drag(z_bottom=aw._SPAR_BOT_B, z_top=aw._WL_B, diameter=aw.SPAR_D,
                                     cd=aw.SPAR_CD * cd_scale, n_segments=10)
    plate = PlateMember(type="plate", center=[0.0, 0.0, aw._PLATE_B], normal=[0.0, 0.0, 1.0],
                        radius=aw.PLATE_R, thickness=aw.PLATE_T, Cd_n=aw.PLATE_CDN * cd_scale,
                        Cd_t=aw.PLATE_CDT * cd_scale)
    return [*spar, plate]


def _buoy_deck(drag: list, pitch0: float = 0.0) -> Deck:
    body = Body(name="buoy", reference_point=[0.0, 0.0, aw.ZB], mass=aw.M_BUOY,
                inertia=Inertia(Ixx=aw.IXX, Iyy=aw.IYY, Izz=aw.IZZ),
                hydro_database=HydroDatabaseRef(format="capytaine", path=str(BUOY_NC)),
                initial_conditions=InitialConditions(position=[0, 0, 0, 0, pitch0, 0]),
                drag_elements=drag)
    return Deck(simulation=Simulation(duration=10.0, dt=DT),
                environment=Environment(water_depth=200.0, water_density=aw.RHO, gravity=aw.G),
                waves=DeckWave(type="regular", height=1.0, period=10.0, heading=0.0),
                bodies=[body], output=Output(file="o.h5", channels=["heave"], sample_rate=10.0))


def _strip_drag(dk: Deck) -> Deck:
    return dk.model_copy(update={"bodies": [b.model_copy(update={"drag_elements": []})
                                            for b in dk.bodies]})


def _scale_drag(dk: Deck, s: float) -> Deck:
    if s == 1.0:
        return dk
    return dk.model_copy(update={"bodies": [b.model_copy(update={"drag_elements": _drag(s)})
                                            if b.drag_elements else b for b in dk.bodies]})


def line_design(article: str, n_spar: int, H_design: float = 0.5) -> dict:
    """mooring_sizing's X-spread line: Kx from T_surge; T0 keeps the slack side taut under the
    drift offset + wave amplitude of the design wave height H_design (+20 %)."""
    a = ms.ARTICLES[NAMES[article]]
    f = max(sum(ms.drift_per_spar(H_design, T)[:2]) for T in ms.T_WAVE)
    Kx = (a["M"] + a["A"]) * (2 * np.pi / mv.T_SURGE) ** 2
    xs = ms.xspread(Kx, n_spar * f / Kx, 0.5 * H_design, mv.L_ANCHOR)
    return {"Kx": float(Kx), "k_line": float(xs["kl"]), "T0": float(xs["T0"])}


def mooring_lines(dk: Deck, article: str, *, fairlead: np.ndarray = WL_B, anchor_z: float = 0.0,
                  H_design: float = 0.5, balanced: bool = False, T0: float | None = None,
                  collar_r: float = 0.0, collar: str = "radial",
                  k_scale: float = 1.0) -> list[dict]:
    """The FloatSim lines: body, fairlead (body frame), true anchor, stiffness k and pretension.

    Defaults = the documented design (fairlead at the spar SWL, anchors at the SWL, T0 for
    H = 0.5 m). ``balanced``: every moored spar gets the line to the anchor at the other end of
    the flume on its side too, with k and T0 halved, so the axial pretension cancels on the spar
    (the single-buoy collar, generalized); total Kx is unchanged.

    ``T0``: per-line pretension override (the single-buoy redesign, Phase C item 2); k is kept.
    ``collar_r``: each line's fairlead moves ``collar_r`` from ``fairlead`` in plan -- towards
    its anchor (``collar="radial"``), or at right angles to it (``"pinwheel"``: the line leaves
    tangentially, the two diagonals turning opposite ways so the pretension moments cancel).
    ``k_scale``: scales every line's axial stiffness at the same T0 (soft-element sensitivity)."""
    B = dk.bodies
    buoys = [k for k, b in enumerate(B) if (b.hydro_body_label or b.hydro_database)]
    des = line_design(article, len(buoys), H_design)
    out = []
    for sx in (-1, 1):
        for sy in (1, -1):
            anchor = np.array([sx * mv.L_ANCHOR, sy * ms.W_FLUME / 2, anchor_z])
            if article == "buoy":
                legs, share = [buoys[0]], 1.0
            elif article == "cluster":
                legs = [k for k in buoys if np.sign(B[k].reference_point[0]) == sx
                        and np.sign(B[k].reference_point[1]) == sy]; share = 1.0
            else:
                xr = max(abs(B[k].reference_point[0]) for k in buoys)
                legs = [k for k in buoys if np.isclose(B[k].reference_point[0], sx * xr)
                        and np.sign(B[k].reference_point[1]) == sy]; share = 1.0 / len(legs)
            t0 = des["T0"] if T0 is None else T0
            for k in legs:
                ends = [anchor, anchor * np.array([-1.0, 1.0, 1.0])] if balanced else [anchor]
                for an in ends:
                    fl = np.asarray(fairlead, dtype=float).copy()
                    if collar_r:
                        u = an[:2] - np.asarray(B[k].reference_point, dtype=float)[:2]
                        u /= np.linalg.norm(u)
                        if collar == "pinwheel":
                            u = sx * sy * np.array([-u[1], u[0]])
                        fl[:2] += collar_r * u
                    out.append({"body": k, "name": B[k].name, "fairlead": fl,
                                "anchor": an,
                                "k": k_scale * des["k_line"] * share / len(ends),
                                "T0": t0 * share / len(ends), "group": len(out)})
    return out


def moored(dk: Deck, article: str, w_line: float = W_LINE,  # type: ignore[no-untyped-def]
           **opts) -> tuple[Deck, list[dict]]:
    """``w_line``: the line weight per unit length the FloatSim catenary uses (it has one
    uniform value and no air/water split: pass the submerged weight for a line in water, the
    DRY weight for a line wholly in air)."""
    lines = mooring_lines(dk, article, **opts)
    conns = []
    for ln in lines:
        ref = np.asarray(dk.bodies[ln["body"]].reference_point, float)
        chord = float(np.linalg.norm(ln["anchor"] - (ref + ln["fairlead"])))
        L0 = chord - ln["T0"] / ln["k"]
        conns.append(Catenary(type="catenary", body_a=ln["name"], body_b="earth",
                              attach_a_body=ln["fairlead"].tolist(),
                              attach_b_body=ln["anchor"].tolist(),
                              line=CatenaryLine(length=L0, weight_per_length=w_line,
                                                EA=ln["k"] * L0)))
        ln.update(chord=chord, L0=L0, w=w_line)
    return dk.model_copy(update={"connections": conns}), lines


def deck(article: str, *, drag: bool = True, cd_scale: float = 1.0, pitch0: float = 0.0) -> Deck:
    if article == "buoy":
        dk = _buoy_deck(_drag(cd_scale), pitch0)
    else:
        dk = mv._cluster_deck() if article == "cluster" else aw.deck()
        dk = _scale_drag(dk, cd_scale)
    return dk if drag else _strip_drag(dk)


def hdbs(article: str) -> dict:
    if article == "buoy":
        return {"bem_databases": {"buoy": read_capytaine(BUOY_NC)}}
    nc = CLUSTER_NC if article == "cluster" else PLATFORM_NC
    return {"bem_databases": {}, "shared_hydro_database": read_capytaine(nc),
            "asymptote_check_override": OVR, "kernel_decay_floor_override": OVR}


STATIC_RESIDUAL_MAX_N = 1.0e-4   # explicit check on every static solve (Phase D decision 6)


def _stiffness(state_force, lhs, xi: np.ndarray, h: float = 1.0e-5) -> np.ndarray:  # type: ignore[no-untyped-def]
    """C minus the central-difference Jacobian of FloatSim's state force at xi."""
    n = xi.size
    z = np.zeros(n)
    K = np.empty((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = h
        K[:, j] = -(state_force(0.0, xi + e, z) - state_force(0.0, xi - e, z)) / (2 * h)
    return np.asarray(lhs.C) + K


def checked_static_equilibrium(lhs, state_force, xi0: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    """FloatSim's static solve with an explicit residual check (tracker
    STATIC-SOLVE-FALSE-CONVERGENCE: hybr can report success without moving). First FloatSim's
    default solve; if its residual |C xi - F(xi)| exceeds STATIC_RESIDUAL_MAX_N, retry from a
    Newton step on FloatSim's linearised stiffness, tight tolerance, no Tikhonov term; raise if
    that fails too."""
    z = np.zeros(xi0.size)

    def res(x: np.ndarray) -> np.ndarray:
        return np.asarray(lhs.C @ x - state_force(0.0, x, z))

    x = np.asarray(fsd.solve_static_equilibrium(lhs=lhs, state_force=state_force, xi0=xi0,
                                                tol=fsd._EQUILIBRIUM_TOL_N,
                                                allow_failure=True).xi_eq)
    if np.abs(res(x)).max() <= STATIC_RESIDUAL_MAX_N:
        return x
    x_start = x - np.linalg.lstsq(_stiffness(state_force, lhs, x), res(x), rcond=None)[0]
    x = np.asarray(fsd.solve_static_equilibrium(lhs=lhs, state_force=state_force, xi0=x_start,
                                                tol=1.0e-9, regularization=0.0,
                                                allow_failure=True).xi_eq)
    r = float(np.abs(res(x)).max())
    if r > STATIC_RESIDUAL_MAX_N:
        raise RuntimeError(f"static equilibrium: residual {r:.2e} N after the checked retry")
    return x


def build_single(dk: Deck, hdb, solve_equilibrium: bool, wave: RegularWave | None = None,  # type: ignore[no-untyped-def]
                 ramp: HalfCosineRamp | None = None, dt: float | None = None):
    """build_system's per-body branch for the one-buoy deck (see the module note). ``wave`` /
    ``ramp``: wave-relative Morison drag, as build_system(drag_wave=, drag_wave_ramp=). ``dt``:
    the kernel's time step (default DT). The static solve is residual-checked."""
    name_to_index = fsd._validate_body_names(dk)
    body = dk.bodies[0]
    lhs = fsd.assemble_global_lhs([fsd._per_body_lhs(body, hdb, gravity=dk.environment.gravity)])
    kernel = fsd.assemble_global_kernel([fsd.compute_retardation_kernel(
        hdb, t_max=T_KERNEL, dt=DT if dt is None else dt, asymptote_check_override=OVR,
        kernel_decay_floor_override=OVR)])
    n = lhs.n_dof
    cats = [fsd._materialise_catenary(c, name_to_index) for c in dk.connections]
    refs = np.array([b.reference_point for b in dk.bodies], dtype=np.float64)
    cat_force = (fsd.make_catenary_state_force(cats, n_dof=n, body_reference_points=refs)
                 if cats else None)
    drag_force = fsd._build_drag_state_force(dk, n, rho=dk.environment.water_density,
                                             wave=wave, ramp=ramp)
    state = fsd._compose_state_force(None, cat_force, drag_force, n)
    xi0 = fsd.pack_state([np.asarray(b.initial_conditions.position, float) for b in dk.bodies])
    xd0 = fsd.pack_state([np.asarray(b.initial_conditions.velocity, float) for b in dk.bodies])
    if solve_equilibrium:
        xi0 = checked_static_equilibrium(lhs, state, xi0)
    return fsd.SimulationSetup(lhs=lhs, kernel=kernel, state_force=state, xi0=xi0, xi_dot0=xd0,
                               body_name_to_index=name_to_index, constraints=None)


def with_positions(dk: Deck, xi: np.ndarray) -> Deck:
    """The deck with each body's initial position set from the packed state xi."""
    return dk.model_copy(update={"bodies": [b.model_copy(update={
        "initial_conditions": b.initial_conditions.model_copy(
            update={"position": [float(v) for v in xi[6 * k:6 * k + 6]]})})
        for k, b in enumerate(dk.bodies)]})


def joint_residual(setup, xi: np.ndarray) -> float:  # type: ignore[no-untyped-def]
    """inf-norm of the static residual C xi - F_state(0, xi, 0) on the joint-feasible subspace
    null(G(xi)): zero exactly when the residual is a pure joint reaction G^T lambda."""
    n = setup.lhs.n_dof
    r = setup.lhs.C @ xi - setup.state_force(0.0, xi, np.zeros(n))
    if setup.constraints is None:
        return float(np.abs(r).max())
    g = np.asarray(setup.constraints.jacobian(xi))
    _u, sv, vh = np.linalg.svd(g)
    rank = int((sv > max(g.shape) * np.finfo(np.float64).eps * sv[0]).sum())
    return float(np.abs(vh[rank:] @ r).max())


def moored_equilibrium(article: str, tag: str | None = None, **opts) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Static equilibrium of a moored articulated deck, found by FloatSim (see the module note).
    ``opts`` go to ``mooring_lines`` (attachment variants); the result is cached under ``tag``
    (default: the article = the documented design)."""
    tag = tag or article
    cache = json.loads(EQ_CACHE.read_text()) if EQ_CACHE.exists() else {}
    dk0 = deck(article)
    dkm, lines = moored(dk0, article, **opts)
    key = [[round(ln["T0"], 6), round(ln["k"], 6), round(ln["L0"], 6)] for ln in lines]
    if opts.get("w_line", W_LINE) != W_LINE:
        key.append([round(opts["w_line"], 6)])
    if tag in cache and cache[tag]["lines"] == key:
        return np.asarray(cache[tag]["xi"])
    hd = hdbs(article)
    s0 = build_system(dk0, dt=DT, t_max_kernel=T_KERNEL, solve_equilibrium=True, **hd)
    s1 = build_system(with_positions(dkm, s0.xi0), dt=DT, t_max_kernel=T_KERNEL,
                      solve_equilibrium=False, **hd)
    r = integrate_cummins(lhs=s1.lhs, kernel=s1.kernel, xi0=s1.xi0, xi_dot0=s1.xi_dot0,
                          duration=SETTLE_S, dt=DT, rho_inf=0.8, constraints=s1.constraints,
                          state_force=s1.state_force, projection_interval=1)
    tail = r.t >= r.t[-1] - SETTLE_AVG_S
    xi = r.xi[tail].mean(axis=0)
    res = joint_residual(s1, xi)
    buoys = [k for k, b in enumerate(dkm.bodies) if b.hydro_body_label or b.hydro_database]
    tilt = [float(np.degrees(np.hypot(xi[6 * k + 3], xi[6 * k + 4]))) for k in buoys]
    rec = {"xi": xi.tolist(), "lines": key, "settle_s": SETTLE_S, "avg_s": SETTLE_AVG_S,
           "joint_residual_N": res, "residual_at_free_eq_N": joint_residual(s1, s0.xi0),
           "tail_max_speed_m_s": float(np.abs(r.xi_dot[tail]).max()),
           "tail_osc_amp": float(np.abs(r.xi[tail] - xi).max()),
           "max_buoy_tilt_deg": max(tilt), "mean_buoy_tilt_deg": float(np.mean(tilt)),
           "free_eq_max_abs": float(np.abs(s0.xi0).max())}
    print(f"{tag} moored equilibrium: joint residual {res:.3f} N, buoy tilt "
          f"{min(tilt):.2f}-{max(tilt):.2f} deg, tail speed {rec['tail_max_speed_m_s']:.1e} m/s",
          flush=True)
    if res > fsd._EQUILIBRIUM_TOL_N:
        raise RuntimeError(f"{tag}: settled state misses FloatSim's equilibrium tolerance "
                           f"({res:.3f} N > {fsd._EQUILIBRIUM_TOL_N} N)")
    lock = EQ_CACHE.with_suffix(".lock")          # parallel settles share the cache file
    for _ in range(600):
        try:
            fdl = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            time.sleep(0.1)
    try:
        cache = json.loads(EQ_CACHE.read_text()) if EQ_CACHE.exists() else {}
        cache[tag] = rec
        EQ_CACHE.write_text(json.dumps(cache, indent=1))
    finally:
        os.close(fdl)
        lock.unlink(missing_ok=True)
    return xi


def build(dk: Deck, article: str, *, solve_equilibrium: bool = True, hd: dict | None = None):  # type: ignore[no-untyped-def]
    hd = hd or hdbs(article)
    if article == "buoy":
        return build_single(dk, hd["bem_databases"]["buoy"], solve_equilibrium), hd
    if solve_equilibrium and dk.connections:
        dk = with_positions(dk, moored_equilibrium(article))
        solve_equilibrium = False
    return build_system(dk, dt=DT, t_max_kernel=T_KERNEL, solve_equilibrium=solve_equilibrium,
                        **hd), hd


def hydro_dof(dk: Deck) -> np.ndarray:
    idx = []
    for k, b in enumerate(dk.bodies):
        if b.hydro_body_label is not None or b.hydro_database is not None:
            idx.extend(range(6 * k, 6 * k + 6))
    return np.asarray(idx, dtype=int)


RAMP_S = 15.0


def wave_setup(dk: Deck, article: str, T: float, H: float, *, xi_eq: np.ndarray | None = None,  # type: ignore[no-untyped-def]
               hd: dict | None = None, dt: float | None = None):
    """FloatSim setup for a regular wave (heading 0) with WAVE-RELATIVE Morison drag (STEP 5
    PR1): the drag samples the same wave and ramp as the excitation. ``xi_eq``: the start state
    (the moored settle for an articulated article); None solves FloatSim's static equilibrium.
    Returns (setup, hd, wave, ramp)."""
    hd = hd or hdbs(article)
    wave = RegularWave(amplitude=0.5 * H, omega=2 * np.pi / T, heading_deg=0.0)
    ramp = HalfCosineRamp(duration=RAMP_S)
    d = dk if xi_eq is None else with_positions(dk, xi_eq)
    if article == "buoy":
        s = build_single(d, hd["bem_databases"]["buoy"], xi_eq is None, wave=wave, ramp=ramp,
                         dt=dt)
    else:
        s = build_system(d, dt=DT if dt is None else dt, t_max_kernel=T_KERNEL,
                         solve_equilibrium=xi_eq is None,
                         drag_wave=wave, drag_wave_ramp=ramp, **hd)
    return s, hd, wave, ramp


def drift_force(dk: Deck, F_spar: float, ramp: HalfCosineRamp | None = None):  # type: ignore[no-untyped-def]
    """The recorded drift bound APPLIED IN-RUN: a steady +x force ``F_spar`` at each spar's calm
    waterline (body point WL_B, small-angle arm), times the excitation ramp when given."""
    n = 6 * len(dk.bodies)
    spars = [k for k, b in enumerate(dk.bodies) if b.hydro_body_label or b.hydro_database]
    arm = np.asarray(WL_B, dtype=float)
    F3 = np.array([F_spar, 0.0, 0.0])

    def f(t: float, xi: np.ndarray, _xd: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        s = 1.0 if ramp is None else ramp.value(t)
        for k in spars:
            out[6 * k] = s * F_spar
            out[6 * k + 3:6 * k + 6] = s * np.cross(arm + np.cross(xi[6 * k + 3:6 * k + 6], arm),
                                                    F3)
        return out

    return f


def run_case(setup, hd: dict, dk: Deck, wave: RegularWave, ramp: HalfCosineRamp,  # type: ignore[no-untyped-def]
             n_settle: float, n_keep: int, extra_force=None, dt: float | None = None):
    """Excitation + FloatSim state force (+ ``extra_force``: applied drift, restraint) through
    the Cummins integrator; lasts ramp + n_settle + n_keep periods."""
    hdb = hd.get("shared_hydro_database") or hd["bem_databases"]["buoy"]
    f_wave = make_regular_wave_force(hdb=hdb, wave=wave, body_position=(0.0, 0.0, 0.0),
                                     ramp=ramp)
    idx = hydro_dof(dk); n = setup.lhs.M_plus_Ainf.shape[0]

    def ext(t):
        f = np.zeros(n); f[idx] = f_wave(t); return f

    sf = setup.state_force if extra_force is None else (
        lambda t, x, v: setup.state_force(t, x, v) + extra_force(t, x, v))
    return integrate_cummins(lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0,
                             xi_dot0=setup.xi_dot0,
                             duration=RAMP_S + n_settle + n_keep * wave.period,
                             dt=DT if dt is None else dt, rho_inf=0.8,
                             constraints=setup.constraints, external_force=ext, state_force=sf,
                             projection_interval=1)


def run_wave(setup, hd: dict, dk: Deck, T: float, H: float, n_settle: float, n_keep: int):  # type: ignore[no-untyped-def]
    """Regular wave (heading 0) through FloatSim's excitation + Cummins integrator."""
    hdb = hd.get("shared_hydro_database") or hd["bem_databases"]["buoy"]
    ramp = HalfCosineRamp(duration=15.0)
    f_wave = make_regular_wave_force(hdb=hdb, wave=RegularWave(amplitude=0.5 * H,
                                                               omega=2 * np.pi / T,
                                                               heading_deg=0.0),
                                     body_position=(0.0, 0.0, 0.0), ramp=ramp)
    idx = hydro_dof(dk); n = setup.lhs.M_plus_Ainf.shape[0]

    def ext(t):
        f = np.zeros(n); f[idx] = f_wave(t); return f

    return integrate_cummins(lhs=setup.lhs, kernel=setup.kernel, xi0=setup.xi0,
                             xi_dot0=setup.xi_dot0, duration=15.0 + n_settle + n_keep * T,
                             dt=DT, rho_inf=0.8, constraints=setup.constraints,
                             external_force=ext, state_force=setup.state_force,
                             projection_interval=1)
