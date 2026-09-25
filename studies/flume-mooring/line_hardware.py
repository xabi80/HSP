"""Soft-line hardware for the flume mooring (flume-mooring, decision 4).

The design lines are soft (k = 2.0 / 8.3 / 16.6 N/m per line or leg) and pretensioned (T0 = 4.0 /
9.0 / 18.0 N): each must stretch T0/k at rest and more under the drift and the waves. Two
realizable soft elements are checked against every criterion.

1. STROKE (FloatSim). Pre-stretch T0/k at the calm equilibrium, and the maximum stretch
   T_max/k in FloatSim runs of the governing extreme cases: H = 0.5 m at T = 2.35 s (the
   largest drift of the confirmed matrix) and T = 2.65 s (the tilt/heave resonance), drift bound
   applied in-run, wave-relative drag, the buoy on its r = 0.2 m collar at dt = 0.005 s
   (buoy_yaw_collar.py). Each run starts from the pull-curve offset for its mean drift
   (tank_rows/pull_*.json) and the drift is applied from t = 0, so the ~16 s surge mode need not
   settle from zero; 180 s of settling, then statistics over 30 s + 4 wave periods. Also per run:
   mean offset and surge range (field of view -0.3 / +1.0 m), minimum tension / T0, line
   clearance to a 0.6 H crest, and the largest yaw / sway / roll (symmetry).

2. ELASTIC SHOCK CORD, modelled by FloatSim's catenary as a linear line of axial stiffness k.
   A real cord is nonlinear (stiff at small strain, a softer plateau, stiffening near its rated
   elongation) and hysteretic. FloatSim represents neither: its catenary is linear elastic and
   quasi-static. The nonlinearity is bounded here by running the design with the secant stiffness
   k x 0.7 / 1.0 / 1.3 at the same T0 (pull curves, periods, design states); hysteresis is left
   out (it adds damping to the slow modes: decays in the tank will then settle faster than
   predicted, and the mean offset carries the loading/unloading gap measured in a pull test).

3. PULLEY + COUNTERWEIGHT (constant tension). The line runs over a wall pulley to a hanging
   weight W = T0: tension stays T0 whatever the offset, so the axial stiffness is ~0 and all
   restoring is geometric. FloatSim's catenary CANNOT represent it -- its unstretched length
   L0 = chord - T0/k must stay positive, so its axial stiffness cannot fall below T0/chord (the
   zero-length-spring limit), and there is no constant-tension connector. It is screened here in
   closed form (exact for the ideal element): the X-spread's restoring from line directions only,
   with FloatSim's article masses (surge/sway mass from the item-4 decays and pull stiffness).

Writes line_hardware.json.  Run: python line_hardware.py [stroke|cord|counterweight|all]
"""

# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import floatsim_decks as fd
import mooring_sizing as ms
import resonance_bandwidth as rb
import tank_predictions as tp

import floatsim.driver as fsd
from floatsim.mooring.catenary_analytic import make_catenary_state_force

OUT = HERE / "line_hardware.json"
ARTICLES = ("buoy", "cluster", "platform")
N_SPAR = {"buoy": 1, "cluster": 4, "platform": 16}
EXTREME = ((0.5, 2.35), (0.5, 2.65))
SETTLE, AVG = 180.0, 30.0
DT_ART = {"buoy": 0.0025, "cluster": None, "platform": None}  # buoy: collar yaw numerical
# growth (K dt / 2I) must stay below round-off amplification over the 234 s extreme runs
K_SCALES = (0.7, 1.0, 1.3)
FOV = (-0.3, 1.0)
CREST = 0.6


def _drift(H: float, T: float) -> float:
    return float(sum(ms.drift_per_spar(H, T, steep=ms.STEEP_MATRIX)[:2]))


def design_opts(article: str, t0_scale: float = 1.0) -> dict:
    """The design mooring options, with every line's pretension scaled by ``t0_scale`` (k kept)."""
    opts = rb.mooring_opts(article)
    if t0_scale != 1.0:
        base = opts.get("T0", fd.line_design(article, N_SPAR[article])["T0"])
        opts["T0"] = t0_scale * base
    return opts


def _setup(article: str, opts: dict | None = None, t0_scale: float = 1.0,  # type: ignore[no-untyped-def]
           tag: str | None = None, reuse_eq: bool = False):
    """(deck, calm equilibrium, per-line closures, lines) of the moored design.

    With ``opts`` and ``tag`` (an articulated article), the calm equilibrium is FloatSim's settle
    of THOSE lines, cached under ``tag`` (attachment_sweep.py: submerged attachments tilt the
    articles, so the pin-level settle does not apply). With ``reuse_eq`` the cached settle under
    ``tag`` (another line set: same geometry and T0, other k) is reused as it stands and accepted
    only if FloatSim's joint-projected static residual with THESE lines stays within its
    equilibrium tolerance (1 N).

    With ``t0_scale`` the lines carry a scaled pretension (same k). The articulated articles then
    reuse the cached settle of the unscaled design: pin-level lines act through the pins, so
    pretension makes no moment, and the settle is accepted only if FloatSim's joint-projected
    static residual with the SCALED lines stays within its equilibrium tolerance (1 N)."""
    opts = opts or design_opts(article, t0_scale)
    dk, lines = fd.moored(fd.deck(article), article, **opts)
    if article == "buoy":
        s = fd.build_single(dk, fd.hdbs("buoy")["bem_databases"]["buoy"], True)
        xi_eq = np.asarray(s.xi0)
    elif tag is not None and reuse_eq:
        xi_eq = np.asarray(json.loads(fd.EQ_CACHE.read_text())[tag]["xi"])
        s = fsd.build_system(fd.with_positions(dk, xi_eq), dt=fd.DT, t_max_kernel=fd.T_KERNEL,
                             solve_equilibrium=False, **fd.hdbs(article))
        res = fd.joint_residual(s, xi_eq)
        if res > fsd._EQUILIBRIUM_TOL_N:
            raise RuntimeError(f"{article}: settle {tag} not valid for these lines: {res:.3f} N")
    elif tag is not None:
        xi_eq = fd.moored_equilibrium(article, tag=tag, **opts)
    else:
        xi_eq = fd.moored_equilibrium(
            article, tag=f"{article}:pin_level_w{rb.W_AIR:g}", **rb.mooring_opts(article)
        )
        if t0_scale != 1.0:
            s = fsd.build_system(fd.with_positions(dk, xi_eq), dt=fd.DT, t_max_kernel=fd.T_KERNEL,
                                 solve_equilibrium=False, **fd.hdbs(article))
            res = fd.joint_residual(s, xi_eq)
            if res > fsd._EQUILIBRIUM_TOL_N:
                raise RuntimeError(f"{article}: settle not valid with T0 x{t0_scale}: {res:.3f} N")
    return dk, xi_eq, tp._line_forces(dk), lines


def line_states(dk, xi: np.ndarray, n_pts: int = 0) -> list[dict]:  # type: ignore[no-untyped-def]
    """Per line: FloatSim tension and lowest point (Irvine closed form at FloatSim's H, V_A);
    with ``n_pts``, also the line's profile (inertial x, y, z at n_pts points, anchor to
    fairlead), the same closed form, for the clearance to the local free surface."""
    ref = np.array([b.reference_point for b in dk.bodies], dtype=float)
    names = fsd._validate_body_names(dk)
    n = 6 * len(dk.bodies)
    out = []
    for c in dk.connections:
        att = fsd._materialise_catenary(c, names)
        b = att.body_index
        F = make_catenary_state_force([att], n_dof=n, body_reference_points=ref)(
            0.0, xi, np.zeros(n)
        )[6 * b : 6 * b + 3]
        L, w, EA = att.line.length, att.line.weight_per_length, att.line.EA
        H, VF = float(np.hypot(F[0], F[1])), float(-F[2])
        VA = VF - w * L
        z_low = float(att.anchor_global[2])
        if VA < 0.0 < VF:
            s_low = -VA / w
            z_low += (H - np.hypot(H, VA)) / w + (VA * s_low + 0.5 * w * s_low**2) / EA
        th = xi[6 * b + 3 : 6 * b + 6]
        fair = ref[b] + xi[6 * b : 6 * b + 3] + att.fairlead_body + np.cross(th, att.fairlead_body)
        row = {"T": float(np.hypot(H, VF)), "k": EA / L, "z_low": min(z_low, float(fair[2]))}
        if n_pts:
            sv = np.linspace(0.0, L, n_pts)
            xs = H * sv / EA + (H / w) * (np.arcsinh((VA + w * sv) / H) - np.arcsinh(VA / H))
            zs = (np.hypot(H, VA + w * sv) - np.hypot(H, VA)) / w + (VA * sv + 0.5 * w * sv**2) / EA
            a = np.asarray(att.anchor_global, dtype=float)
            u = (fair[:2] - a[:2]) / np.linalg.norm(fair[:2] - a[:2])
            row["pts"] = np.column_stack([a[0] + xs * u[0], a[1] + xs * u[1], a[2] + zs])
        out.append(row)
    return out


def _pull_offset(article: str, F_total: float) -> float:
    c = json.loads((HERE / "tank_rows" / f"pull_{article}.json").read_text())["surge"]["curve"]
    off = np.array([r["offset"] for r in c])
    F = np.array([r["F"] for r in c])
    return float(np.interp(F_total, F, off))


def extreme_run(args: tuple) -> dict:
    """One extreme case. ``drift=False`` is D0: no applied drift, from the calm equilibrium --
    the mean offset is then FloatSim's own (wave-relative drag on the moving body).

    Optional 6th element ``design``: {"opts", "tag", "K_surge"} -- another mooring (the
    attachment-height design): its lines, its settle, and a start offset F / K_surge."""
    article, H, T, drift = args[:4]
    t0_scale = args[4] if len(args) > 4 else 1.0
    design = args[5] if len(args) > 5 else None
    t0 = time.perf_counter()
    if design is None:
        dk, xi_eq, _lf, _lines = _setup(article, t0_scale=t0_scale)
    else:
        dk, xi_eq, _lf, _lines = _setup(article, opts=design["opts"], tag=design["tag"],
                                        reuse_eq=design.get("reuse_eq", False))
    F_spar = _drift(H, T) if drift else 0.0
    off = (N_SPAR[article] * F_spar / design["K_surge"] if design is not None
           else _pull_offset(article, N_SPAR[article] * F_spar))
    xi_start = tp.rigid(dk, xi_eq, "surge", off) if drift else xi_eq
    dt = DT_ART[article]
    s, hd, wave, ramp = fd.wave_setup(dk, article, T, H, xi_eq=xi_start, dt=dt)
    r = fd.run_case(
        s,
        hd,
        dk,
        wave,
        ramp,
        SETTLE + AVG,
        4,
        extra_force=fd.drift_force(dk, F_spar, ramp=None) if drift else None,
        dt=dt,
    )
    keep = r.t >= r.t[-1] - AVG - 4 * T
    idx = np.flatnonzero(keep)[::5]
    T0 = [ln["T"] for ln in line_states(dk, xi_eq)]
    tens, clr, emerge = [], [], []
    for i in idx:  # criterion 5: every line point above the LOCAL incident surface
        ls = line_states(dk, r.xi[i], n_pts=21)
        tens.append([ln["T"] for ln in ls])
        eta = ramp.value(float(r.t[i]))
        above = [
            ln["pts"][:, 2]
            - eta * np.asarray(wave.elevation(float(r.t[i]), ln["pts"][:, 0], ln["pts"][:, 1]))
            for ln in ls
        ]
        clr.append(min(float(np.min(a)) for a in above))
        emerge.append(max(float(np.max(a)) for a in above))  # > 0: a submerged line breaks surface
    tens = np.asarray(tens)
    k = np.array([ln["k"] for ln in line_states(dk, xi_eq)])
    surge = np.mean([r.xi[keep, 6 * b] - xi_eq[6 * b] for b in range(len(dk.bodies))], axis=0)
    anti = {
        nm: float(np.abs(r.xi[:, j::6] - xi_eq[j::6]).max())
        for nm, j in (("sway", 1), ("roll", 3), ("yaw", 5))
    }
    spars = [b for b, bd in enumerate(dk.bodies) if bd.hydro_body_label or bd.hydro_database]
    tilt = max(float(np.degrees(np.hypot(r.xi[keep, 6 * b + 3], r.xi[keep, 6 * b + 4])).max())
               for b in spars)
    row = {
        "article": article,
        "H": H,
        "T": T,
        "applied_drift": drift,
        "t0_scale": t0_scale,
        "tilt_max_deg": tilt,
        "drift_N_per_spar": F_spar,
        "T0_N": T0,
        "k_N_per_m": k.tolist(),
        "prestretch_m": (np.asarray(T0) / k).tolist(),
        "T_max_N": tens.max(axis=0).tolist(),
        "T_min_N": tens.min(axis=0).tolist(),
        "stretch_max_m": float((tens.max(axis=0) / k).max()),
        "T_min_ratio": float((tens.min(axis=0) / np.asarray(T0)).min()),
        "mean_offset_m": float(surge.mean()),
        "surge_min_m": float(surge.min()),
        "surge_max_m": float(surge.max()),
        "clearance_local_surface_min_m": float(min(clr)),
        "line_above_local_surface_max_m": float(max(emerge)),
        "tag": design["tag"] if design is not None else None,
        **({"_t": r.t[idx], "_xi": r.xi[idx]} if design is not None and design.get("history")
           else {}),                   # the sampled window (idx) for callers' own analysis
        "antisymmetric_max": anti,
        "wall_min": (time.perf_counter() - t0) / 60,
    }
    print(
        f"{article:8s} H {H} T {T} drift {drift}: stretch max {row['stretch_max_m']:.2f} m (pre "
        f"{max(row['prestretch_m']):.2f}), T {min(row['T_min_N']):.2f}-{max(row['T_max_N']):.2f}"
        f" N (min/T0 {row['T_min_ratio']:.2f}), surge {row['surge_min_m']:.3f}.."
        f"{row['surge_max_m']:.3f} m (mean {row['mean_offset_m']:.3f}), clr "
        f"{row['clearance_local_surface_min_m']:+.3f}, tilt {tilt:.1f} deg, yaw "
        f"{np.degrees(anti['yaw']):.1e} deg (T0 x{t0_scale}) "
        f"[{row['wall_min']:.1f} min]",
        flush=True,
    )
    return row


def cord(article: str) -> list[dict]:
    """Secant-stiffness band of an elastic cord: pull curves, periods, design-state offset."""
    dec = json.loads((HERE / "tank_rows" / f"decay_{article}_surge.json").read_text())
    decw = json.loads((HERE / "tank_rows" / f"decay_{article}_sway.json").read_text())
    base = json.loads((HERE / "tank_rows" / f"pull_{article}.json").read_text())
    Mx = base["surge"]["K0"] * (dec["T_s"] / (2 * np.pi)) ** 2  # effective surge mass
    My = base["sway"]["K0"] * (decw["T_s"] / (2 * np.pi)) ** 2
    rows = []
    for ks in K_SCALES:
        opts = {**rb.mooring_opts(article), "k_scale": ks}
        dk, xi_eq, lf, lines = _setup(article, opts)
        z = np.zeros(xi_eq.size)

        def fx(d: float, mode: str = "surge", lf=lf, dk=dk, xi_eq=xi_eq, z=z) -> float:  # type: ignore[no-untyped-def]
            xi = tp.rigid(dk, xi_eq, mode, d)
            j = 0 if mode == "surge" else 1
            return float(tp._resultant(dk, xi, np.sum([f(0.0, xi, z) for f in lf], axis=0))[j])

        Kx = -(fx(1e-3) - fx(-1e-3)) / 2e-3
        Ky = -(fx(1e-3, "sway") - fx(-1e-3, "sway")) / 2e-3
        F0 = fx(0.0)
        states = []
        for H, T in ((0.12, 1.4), (0.35, 1.85), (0.5, 2.35)):
            Ft = N_SPAR[article] * _drift(H, T)
            lo, hi = 0.0, 3.0  # rigid offset where the lines balance Ft
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if F0 - fx(mid) < Ft else (lo, mid)
            dmean = 0.5 * (lo + hi)
            ls = line_states(dk, tp.rigid(dk, xi_eq, "surge", dmean + 0.5 * H))
            T0 = [ln["T"] for ln in line_states(dk, xi_eq)]
            states.append(
                {
                    "H": H,
                    "T": T,
                    "mean_offset_m": dmean,
                    "design_offset_m": dmean + 0.5 * H,
                    "T_min_ratio": min(a["T"] / b for a, b in zip(ls, T0, strict=True)),
                    "stretch_max_m": max(a["T"] / a["k"] for a in ls),
                    "clearance_m": min(a["z_low"] for a in ls) - CREST * H,
                }
            )
        row = {
            "article": article,
            "k_scale": ks,
            "k_line": lines[0]["k"],
            "Kx": Kx,
            "Ky": Ky,
            "T_surge_s": 2 * np.pi * np.sqrt(Mx / Kx),
            "T_sway_s": 2 * np.pi * np.sqrt(My / Ky),
            "states": states,
        }
        rows.append(row)
        print(
            f"{article:8s} k x{ks}: Kx {Kx:.2f} Ky {Ky:.2f} N/m, surge {row['T_surge_s']:.1f} s,"
            f" sway {row['T_sway_s']:.1f} s | "
            + "  ".join(
                f"H{st['H']}: off {st['design_offset_m']:.2f} Tmin/T0 {st['T_min_ratio']:.2f}"
                f" stretch {st['stretch_max_m']:.2f} clr {st['clearance_m']:+.2f}"
                for st in states
            ),
            flush=True,
        )
    return rows


def counterweight(article: str) -> dict:
    """Constant-tension X-spread, closed form: each line pulls T_c towards its wall pulley."""
    dk, xi_eq, _lf, lines = _setup(article)
    T_c = [float(ln["T0"]) for ln in lines]
    ref = np.array([b.reference_point for b in dk.bodies], dtype=float)
    fair0 = np.array(
        [
            ref[ln["body"]] + xi_eq[6 * ln["body"] : 6 * ln["body"] + 3] + ln["fairlead"]
            for ln in lines
        ]
    )
    anch = np.array([ln["anchor"] for ln in lines])

    def force(d: np.ndarray) -> np.ndarray:  # net horizontal force at a rigid offset d
        v = anch[:, :2] - (fair0[:, :2] + d)
        return np.sum(np.asarray(T_c)[:, None] * v / np.linalg.norm(v, axis=1)[:, None], axis=0)

    Kx = -(force(np.array([1e-4, 0]))[0] - force(np.array([-1e-4, 0]))[0]) / 2e-4
    Ky = -(force(np.array([0, 1e-4]))[1] - force(np.array([0, -1e-4]))[1]) / 2e-4
    dec = json.loads((HERE / "tank_rows" / f"decay_{article}_surge.json").read_text())
    decw = json.loads((HERE / "tank_rows" / f"decay_{article}_sway.json").read_text())
    base = json.loads((HERE / "tank_rows" / f"pull_{article}.json").read_text())
    m_cw = sum(T_c) / 9.806  # counterweights ride with the line ends
    Mx = base["surge"]["K0"] * (dec["T_s"] / (2 * np.pi)) ** 2 + m_cw * 0.88
    My = base["sway"]["K0"] * (decw["T_s"] / (2 * np.pi)) ** 2 + m_cw * 0.12
    offs = []
    for H, T in ((0.04, 1.4), (0.08, 1.4), (0.12, 1.4), (0.2, 1.4), (0.35, 1.85), (0.5, 2.35)):
        Ft = N_SPAR[article] * _drift(H, T)
        lo, hi = 0.0, 4.9
        if force(np.array([hi, 0.0]))[0] + Ft > 0:
            offs.append(
                {
                    "H": H,
                    "drift_total_N": Ft,
                    "mean_offset_m": None,
                    "note": "no equilibrium inside the flume length",
                }
            )
            continue
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if force(np.array([mid, 0.0]))[0] + Ft > 0 else (lo, mid)
        d = 0.5 * (lo + hi)
        offs.append(
            {
                "H": H,
                "drift_total_N": Ft,
                "mean_offset_m": d,
                "design_offset_m": d + 0.5 * H,
                "in_field_of_view": d + 0.5 * H <= FOV[1],
            }
        )
    out = {
        "article": article,
        "T_c_N": T_c,
        "Kx": Kx,
        "Ky": Ky,
        "counterweight_kg": m_cw,
        "T_surge_s": 2 * np.pi * np.sqrt(Mx / Kx),
        "T_sway_s": 2 * np.pi * np.sqrt(My / Ky),
        "offsets": offs,
    }
    print(
        f"{article:8s} counterweight: Kx {Kx:.3f} Ky {Ky:.3f} N/m, surge "
        f"{out['T_surge_s']:.0f} s, sway {out['T_sway_s']:.0f} s | "
        + "  ".join(
            f"H{o['H']}: {'none' if o['mean_offset_m'] is None else round(o['design_offset_m'], 2)}"
            for o in offs
        ),
        flush=True,
    )
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    if step in ("counterweight", "all"):
        res["counterweight"] = {a: counterweight(a) for a in ARTICLES}
        OUT.write_text(json.dumps(res, indent=1, default=float))
    if step in ("cord", "all"):
        res["cord"] = {a: cord(a) for a in ARTICLES}
        OUT.write_text(json.dumps(res, indent=1, default=float))
    if step in ("stroke", "all"):
        cases = [(a, H, T, d) for a in ARTICLES for H, T in EXTREME for d in (True, False)]
        with ProcessPoolExecutor(max_workers=12) as ex:
            res["stroke"] = list(ex.map(extreme_run, cases))
        OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
