"""Half-power bandwidth of the in-band resonances, with drag, and the fine period step it sets
(flume-mooring Phase C, item 3).

FloatSim regular-wave sweeps (heading 0) of the MOORED articles in the design configuration:
pin-level lines of 0.3 N/m (the single buoy with its redesigned pretension,
single_buoy_redesign.json), WAVE-RELATIVE Morison drag (STEP 5 PR1: ``floatsim_decks.wave_setup``),
no applied drift (at H = 0.04 m the bound is 0.011 N per spar).

Drag damping grows with amplitude, so the NARROWEST resonance is at the SMALLEST operational
height: H = 0.04 m sets the step, for all three articles. H = 0.12 m (top of the operational
band) runs for the buoy and cluster to show the widening.

Per case: 120 s of settling after the 15 s ramp (a resonance with zeta ~ 2 % decays with a time
constant T/(2 pi zeta) ~ 20 s), then the wave-frequency amplitude (least-squares sinusoid) over
the last 4 periods; stationarity = the same over the 4 periods before. Channels: heave and pitch
of the buoy; for the cluster and platform the mean heave and mean pitch (tilt) of the buoys.

Half-power band per channel: near an SDOF resonance 1/a^2 is quadratic in omega, minimum at
omega_n, doubling at omega_n +- zeta omega_n. A parabola is fitted to 1/a^2 over the swept points
with a >= a_max/2. Step rule: the worst-case detuning is half a step, where the amplitude is
1/sqrt(1 + (step/dT_hp)^2) of the peak; >= 95 % needs step <= 0.33 dT_hp.

Rows are cached in bandwidth_rows/.  Run:
    python resonance_bandwidth.py run <buoy|cluster|platform> [--H 0.04] [--workers 16]
    python resonance_bandwidth.py report          -> resonance_bandwidth.json
"""
# ruff: noqa: E402  -- sys.path bootstrap first
from __future__ import annotations

import argparse
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

import design_basis as db
import floatsim_decks as fd

ROWS = HERE / "bandwidth_rows"
OUT = HERE / "resonance_bandwidth.json"
PERIODS = [round(2.30 + 0.05 * i, 2) for i in range(17)]     # 2.30 ... 3.10 s
N_SETTLE, N_KEEP = 120.0, 4
W_AIR = 0.3
STEP_FRACTION = 0.33


BUOY_COLLAR = {"collar": "radial", "collar_r": 0.2}   # decided 2026-09-24 (yaw ~0.86 s)


def mooring_opts(article: str, collar: bool = True) -> dict:
    """The design mooring: pin level, 0.3 N/m; the single buoy at its redesigned T0 and, unless
    ``collar=False``, on the decided r = 0.2 m radial collar. (The committed bandwidth_rows were
    computed on the spar axis, before the collar decision; buoy_yaw_collar.py compares.)"""
    opts = {**db._pin_opts(W_AIR)}
    if article == "buoy":
        sel = json.loads((HERE / "single_buoy_redesign.json").read_text())["notes"]
        opts["T0"] = sel["T0_selected"]
        if collar:
            opts.update(BUOY_COLLAR)
    return opts


def moored_deck(article: str):  # type: ignore[no-untyped-def]
    opts = mooring_opts(article)
    dk, _ = fd.moored(fd.deck(article), article, **opts)
    xi_eq = None if article == "buoy" else fd.moored_equilibrium(
        article, tag=f"{article}:pin_level_w{W_AIR:g}", **opts)
    return dk, xi_eq


def _amp(t: np.ndarray, y: np.ndarray, w: float) -> float:
    X = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.hypot(c[0], c[1]))


def run_one(article: str, T: float, H: float) -> dict:
    jp = ROWS / (f"{article}_H{H:g}_T{T:g}".replace(".", "p") + ".json")
    if jp.exists():
        return json.loads(jp.read_text())
    t0 = time.perf_counter()
    dk, xi_eq = moored_deck(article)
    s, hd, wave, ramp = fd.wave_setup(dk, article, T, H, xi_eq=xi_eq)
    r = fd.run_case(s, hd, dk, wave, ramp, N_SETTLE, N_KEEP)
    w = 2 * np.pi / T
    spars = [k for k, b in enumerate(dk.bodies) if b.hydro_body_label or b.hydro_database]
    heave = np.mean([r.xi[:, 6 * k + 2] for k in spars], axis=0)
    pitch = np.mean([r.xi[:, 6 * k + 4] for k in spars], axis=0)
    t_end = r.t[-1]
    last = r.t >= t_end - N_KEEP * T
    prev = (r.t >= t_end - 2 * N_KEEP * T) & ~last
    row = {"article": article, "T": T, "H": H, "wall_min": (time.perf_counter() - t0) / 60}
    for name, y in (("heave", heave), ("pitch", pitch)):
        a1, a0 = _amp(r.t[last], y[last], w), _amp(r.t[prev], y[prev], w)
        row[name] = a1 / (0.5 * H) if name == "heave" else float(np.degrees(a1))
        row[f"{name}_stationarity"] = a1 / a0 - 1.0
    ROWS.mkdir(exist_ok=True)
    jp.write_text(json.dumps(row))
    print(f"{article} H {H} T {T}: heave RAO {row['heave']:.4f} "
          f"({row['heave_stationarity']:+.2%}), pitch {row['pitch']:.3f} deg "
          f"({row['pitch_stationarity']:+.2%}), {row['wall_min']:.1f} min", flush=True)
    return row


def half_power(T: np.ndarray, a: np.ndarray) -> dict:
    """Parabola in omega through 1/a^2 over the points with a >= a_max/2."""
    om = 2 * np.pi / T
    m = a >= 0.5 * a.max()
    if m.sum() < 3:
        return {"ok": False}
    c2, c1, c0 = np.polyfit(om[m], 1.0 / a[m] ** 2, 2)
    om_n = -c1 / (2 * c2)
    q_min = c0 - c1 * c1 / (4 * c2)
    half = np.sqrt(q_min / c2)                     # 1/a^2 doubles at om_n +- half
    T_lo, T_hi = 2 * np.pi / (om_n + half), 2 * np.pi / (om_n - half)
    i = int(np.argmax(a))
    lo = hi = None                                  # direct interpolation, for comparison
    lvl = a.max() / np.sqrt(2)
    for j in range(i, 0, -1):
        if a[j - 1] < lvl <= a[j]:
            lo = float(np.interp(lvl, [a[j - 1], a[j]], [T[j - 1], T[j]]))
            break
    for j in range(i, a.size - 1):
        if a[j + 1] < lvl <= a[j]:
            hi = float(np.interp(lvl, [a[j + 1], a[j]], [T[j + 1], T[j]]))
            break
    return {"ok": True, "T_n": float(2 * np.pi / om_n), "zeta": float(half / om_n),
            "T_lo": float(T_lo), "T_hi": float(T_hi), "dT_hp": float(T_hi - T_lo),
            "a_peak_fit": float(1 / np.sqrt(q_min)), "a_max_swept": float(a.max()),
            "T_max_swept": float(T[i]), "dT_direct": (hi - lo) if lo and hi else None,
            "n_fit": int(m.sum())}


def report() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(ROWS.glob("*.json"))]
    res: dict = {"cases": {}, "step_rule": f"step <= {STEP_FRACTION} x dT_hp (peak >= 95 %)"}
    steps = []
    for key in sorted({(r["article"], r["H"]) for r in rows}):
        rs = sorted([r for r in rows if (r["article"], r["H"]) == key], key=lambda r: r["T"])
        T = np.array([r["T"] for r in rs])
        out = {"n": len(rs), "max_nonstationarity": max(
            max(abs(r["heave_stationarity"]), abs(r["pitch_stationarity"])) for r in rs)}
        for ch in ("heave", "pitch"):
            hp = half_power(T, np.array([r[ch] for r in rs]))
            out[ch] = hp
            if hp.get("ok"):
                steps.append((hp["dT_hp"] * STEP_FRACTION, key, ch))
                print(f"{key[0]:8s} H {key[1]:.2f} {ch:5s}: T_n {hp['T_n']:.3f} s, zeta "
                      f"{hp['zeta']:.3f}, band {hp['T_lo']:.3f}-{hp['T_hi']:.3f} s "
                      f"(dT {hp['dT_hp']:.3f}; direct {hp['dT_direct']}), peak "
                      f"{hp['a_max_swept']:.3f} at {hp['T_max_swept']} s", flush=True)
        res["cases"][f"{key[0]}@H{key[1]:g}"] = out
    s_min = min(steps)
    res["step_s"] = float(np.floor(s_min[0] * 100) / 100)
    res["step_set_by"] = {"value": s_min[0], "article": s_min[1][0], "H": s_min[1][1],
                          "channel": s_min[2]}
    print(f"fine step {res['step_s']:.2f} s (set by {s_min[1][0]} {s_min[2]} at H "
          f"{s_min[1][1]}: {s_min[0]:.4f} s)")
    OUT.write_text(json.dumps(res, indent=1, default=float))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "report"])
    ap.add_argument("article", nargs="?", choices=["buoy", "cluster", "platform"])
    ap.add_argument("--H", type=float, default=0.04)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.step == "report":
        report()
        return
    moored_deck(a.article)                          # settle once (cached) before the fan-out
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(run_one, [a.article] * len(PERIODS), PERIODS, [a.H] * len(PERIODS)))


if __name__ == "__main__":
    main()
