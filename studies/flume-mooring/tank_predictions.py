"""FloatSim predictions for the tank's mooring checks (flume-mooring Phase C, item 4): surge, sway
and yaw free decays, and static pull curves, for each article.

Design configuration: pin-level lines of 0.3 N/m (the single buoy with its redesigned
pretension, single_buoy_redesign.json), calm water, the deck's Morison drag.

1. PULL CURVES: the lines' restoring force (FloatSim catenary state force, one closure per line)
   against a RIGID offset of the whole article from its moored equilibrium -- surge Fx(dx), sway
   Fy(dy), yaw Mz(psi) about the article centre. Hydrostatics add nothing to these three rigid
   motions, so this is the article's whole restoring. A tank pull through the article centre at
   the pin plane makes no moment about the pins, so the article translates without tilting and
   this is the curve the test measures (to the small heave from the lines' changing vertical
   load, reported as dFz). Per point: every line's tension, and whether one falls below
   0.15 T0 (criterion 3) or goes slack.
2. FREE DECAYS from a rigid initial offset (surge 0.3 m, sway 0.1 m, yaw 5 deg), released at
   rest: period from successive up-crossings of the article's mean displacement, LOW-PASSED
   below 0.2 Hz (a rigid yaw or surge of the articulated articles also swings the pinned buoys,
   whose 2.6-2.9 s tilt modes would otherwise set the crossings; the mooring modes are >= 8 s),
   damping ratio per cycle from the log decrement of successive peaks. The yaw decay also
   settles the cluster's suspect yaw eigenmode (design_basis.py: lambda = -0.018 in the coupled
   eigen-analysis, +2.18 N m/rad as a rigid Rayleigh quotient; both omit the joint-reaction
   geometric stiffness that the pull curve includes).

Run:
    python tank_predictions.py pull <buoy|cluster|platform>
    python tank_predictions.py decay <article> <surge|sway|yaw> [--collar radial:0.12]
    python tank_predictions.py report                 -> tank_predictions.json
Rows are cached in tank_rows/.
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import floatsim_decks as fd
import resonance_bandwidth as rb
import single_buoy_redesign as sb

import floatsim.driver as fsd
from floatsim.driver import build_system
from floatsim.mooring.catenary_analytic import make_catenary_state_force
from floatsim.solver.newmark import integrate_cummins

ROWS = HERE / "tank_rows"
OUT = HERE / "tank_predictions.json"
AMP = {"surge": 0.3, "sway": 0.1, "yaw": np.radians(5.0)}
DURATION = {"surge": 100.0, "sway": 140.0, "yaw": 140.0}
F_LOWPASS_HZ = 0.2
PULL = {"surge": np.round(np.arange(-0.5, 1.5001, 0.05), 3),
        "sway": np.round(np.arange(-0.4, 0.4001, 0.05), 3),
        "yaw": np.round(np.radians(np.arange(-20.0, 20.001, 2.5)), 6)}


def _opts(article: str, collar: str | None) -> dict:
    opts = rb.mooring_opts(article)
    if collar:
        kind, r = collar.split(":")
        opts.update(collar=kind, collar_r=float(r))
    return opts


def moored(article: str, collar: str | None = None, drag: bool = True):  # type: ignore[no-untyped-def]
    """(setup, deck, equilibrium state) of the moored article."""
    opts = _opts(article, collar)
    dk, _ = fd.moored(fd.deck(article, drag=drag), article, **opts)
    hd = fd.hdbs(article)
    if article == "buoy":                         # the tight calm solve of the redesign study
        s = fd.build_single(dk, hd["bem_databases"]["buoy"], True)
        s = dataclasses.replace(s, xi0=sb.mean_offset(s, 0.0))
        return s, dk, np.asarray(s.xi0)
    tag = f"{article}:pin_level_w{rb.W_AIR:g}"
    xi_eq = fd.moored_equilibrium(article, tag=tag, **rb.mooring_opts(article))
    s = build_system(fd.with_positions(dk, xi_eq), dt=fd.DT, t_max_kernel=fd.T_KERNEL,
                     solve_equilibrium=False, **hd)
    return s, dk, xi_eq


def rigid(dk, xi: np.ndarray, mode: str, amp: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    """xi + a rigid surge / sway / yaw (about the article centre, the inertial origin)."""
    out = xi.copy()
    for k, b in enumerate(dk.bodies):
        if mode == "surge":
            out[6 * k] += amp
        elif mode == "sway":
            out[6 * k + 1] += amp
        else:
            p = np.asarray(b.reference_point, dtype=float)[:2] + xi[6 * k:6 * k + 2]
            c, s = np.cos(amp), np.sin(amp)
            out[6 * k:6 * k + 2] += np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]]) - p
            out[6 * k + 5] += amp
    return out


def _line_forces(dk):  # type: ignore[no-untyped-def]
    ref = np.array([b.reference_point for b in dk.bodies], dtype=float)
    names = fsd._validate_body_names(dk)
    n = 6 * len(dk.bodies)
    return [make_catenary_state_force([fsd._materialise_catenary(c, names)], n_dof=n,
                                      body_reference_points=ref) for c in dk.connections]


def _resultant(dk, xi: np.ndarray, F: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    """(Fx, Fy, Fz, Mz about the inertial origin) of a generalised force on all bodies."""
    out = np.zeros(4)
    for k, b in enumerate(dk.bodies):
        p = np.asarray(b.reference_point, dtype=float) + xi[6 * k:6 * k + 3]
        f = F[6 * k:6 * k + 6]
        out[:3] += f[:3]
        out[3] += f[5] + p[0] * f[1] - p[1] * f[0]
    return out


def pull(article: str, collar: str | None = None) -> dict:
    _s, dk, xi_eq = moored(article, collar, drag=False)
    lines = _line_forces(dk)
    n = xi_eq.size
    z = np.zeros(n)

    def state(xi: np.ndarray) -> tuple[np.ndarray, list[float]]:
        per = [f(0.0, xi, z) for f in lines]
        tens = []
        for F in per:
            nz = np.flatnonzero(np.abs(F) > 0)
            k = int(nz[0]) // 6
            tens.append(float(np.linalg.norm(F[6 * k:6 * k + 3])))
        return _resultant(dk, xi, np.sum(per, axis=0)), tens

    R0, T0 = state(xi_eq)
    out: dict = {"article": article, "collar": collar, "T0_lines_N": T0}
    for mode, grid in PULL.items():
        rows = []
        for a in grid:
            R, tens = state(rigid(dk, xi_eq, mode, float(a)))
            d = R - R0
            j = {"surge": 0, "sway": 1, "yaw": 3}[mode]
            rows.append({"offset": float(a), "F": float(-d[j]),
                         "dFz": float(d[2]), "T_min": min(tens), "T_max": max(tens),
                         "T_min_ratio": min(t / t0 for t, t0 in zip(tens, T0, strict=True))})
        # linear stiffness at the equilibrium (central difference, 1 mm / 0.1 deg)
        h = 1e-3 if mode != "yaw" else np.radians(0.1)
        j = {"surge": 0, "sway": 1, "yaw": 3}[mode]
        Kp = -(state(rigid(dk, xi_eq, mode, h))[0][j] - state(rigid(dk, xi_eq, mode, -h))[0][j])
        out[mode] = {"K0": float(Kp / (2 * h)), "curve": rows}
        print(f"{article} {mode}: K0 {Kp / (2 * h):.4f} "
              f"({'N m/rad' if mode == 'yaw' else 'N/m'})", flush=True)
    ROWS.mkdir(exist_ok=True)
    (ROWS / f"pull_{article}{'_' + collar.replace(':', '') if collar else ''}.json").write_text(
        json.dumps(out, indent=1, default=float))
    return out


def decay(article: str, mode: str, collar: str | None = None, amp: float | None = None) -> dict:
    t0 = time.perf_counter()
    amp = AMP[mode] if amp is None else amp
    s, dk, xi_eq = moored(article, collar, drag=True)
    xi0 = rigid(dk, xi_eq, mode, amp)
    r = integrate_cummins(lhs=s.lhs, kernel=s.kernel, xi0=xi0, xi_dot0=np.zeros_like(xi0),
                          duration=DURATION[mode], dt=fd.DT, rho_inf=0.8,
                          constraints=s.constraints, state_force=s.state_force,
                          projection_interval=1)
    j = {"surge": 0, "sway": 1, "yaw": 5}[mode]
    y_raw = np.mean([r.xi[:, 6 * k + j] - xi_eq[6 * k + j] for k in range(len(dk.bodies))],
                    axis=0)
    bb, aa = butter(4, F_LOWPASS_HZ, btype="low", fs=1.0 / fd.DT)
    y = filtfilt(bb, aa, y_raw)
    spec_f = np.fft.rfftfreq(y_raw.size, fd.DT)
    spec = np.abs(np.fft.rfft(y_raw - y_raw.mean()))
    up = np.flatnonzero((y[:-1] < 0.0) & (y[1:] >= 0.0))
    t_up = r.t[up] - y[up] * (r.t[up + 1] - r.t[up]) / (y[up + 1] - y[up])
    periods = np.diff(t_up)
    pk = [int(np.argmax(y[a:b])) + a for a, b in itertools.pairwise(up)]
    peaks = y[pk] if pk else np.array([])
    dec = np.log(peaks[:-1] / peaks[1:]) if peaks.size > 1 else np.array([])
    zeta = dec / np.sqrt(4 * np.pi**2 + dec**2)
    other = {m: float(np.abs(np.mean([r.xi[:, 6 * k + jj] - xi_eq[6 * k + jj]
                                      for k in range(len(dk.bodies))], axis=0)).max())
             for m, jj in (("surge", 0), ("sway", 1), ("heave", 2), ("yaw", 5)) if m != mode}
    band = spec_f > 1.0 / DURATION[mode]
    peaks_f = spec_f[band][np.argsort(spec[band])[::-1][:3]]
    row = {"article": article, "mode": mode, "collar": collar, "amp0": amp,
           "spectral_peaks_s": (1.0 / peaks_f).tolist(),
           "raw_abs_max_after_5s": float(np.abs(y_raw[r.t > 5.0]).max()),
           "duration_s": DURATION[mode], "periods_s": periods.tolist(),
           "T_s": float(np.mean(periods)) if periods.size else None,
           "peaks": peaks.tolist(), "zeta_per_cycle": zeta.tolist(),
           "final_abs_max": float(np.abs(y[r.t > r.t[-1] - 20.0]).max()),
           "max_cross_coupling": other, "wall_min": (time.perf_counter() - t0) / 60}
    ROWS.mkdir(exist_ok=True)
    tag = ("_" + collar.replace(":", "") if collar else "") + (f"_a{amp:g}" if amp != AMP[mode]
                                                               else "")
    (ROWS / f"decay_{article}_{mode}{tag}.json").write_text(
        json.dumps(row, indent=1, default=float))
    print(f"{article} {mode} decay: T {row['T_s']} s over {periods.size} cycles, zeta "
          f"{np.round(zeta, 4).tolist()}, final |y| {row['final_abs_max']:.2e}, "
          f"{row['wall_min']:.1f} min", flush=True)
    return row


def report() -> None:
    res = {p.stem: json.loads(p.read_text()) for p in sorted(ROWS.glob("*.json"))}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    for k, v in res.items():
        if k.startswith("decay_"):
            print(f"{k}: T {v['T_s']} s, zeta {np.round(v['zeta_per_cycle'], 4).tolist()}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pull", "decay", "report"])
    ap.add_argument("article", nargs="?", choices=["buoy", "cluster", "platform"])
    ap.add_argument("mode", nargs="?", choices=["surge", "sway", "yaw"])
    ap.add_argument("--collar", default=None, help="e.g. radial:0.12 (single buoy)")
    ap.add_argument("--amp", type=float, default=None, help="initial offset (m or rad)")
    a = ap.parse_args()
    if a.step == "pull":
        pull(a.article, a.collar)
    elif a.step == "decay":
        decay(a.article, a.mode, a.collar, a.amp)
    else:
        report()


if __name__ == "__main__":
    main()
